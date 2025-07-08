def detect_cellwise_missforest(df, n_folds=5, error_threshold_percentile=95, n_estimators=50, random_state=42):
    """
    Detects potentially manipulated cells using cross-validated missForest imputation,
    returning a boolean DataFrame mask with True for detected suspicious cells.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to check for manipulation
    n_folds : int
        Number of cross-validation folds (default: 5)
    error_threshold_percentile : float
        Percentile threshold for flagging high errors as suspicious (default: 95)
    n_estimators : int
        Number of trees in Random Forest (default: 50)
    random_state : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    mask : pandas.DataFrame
        Boolean mask with True for suspicious cells
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings('ignore')
    
    # Select only numeric columns
    df_num = df.select_dtypes(include=[np.number]).copy()
    if df_num.empty:
        return pd.DataFrame(False, index=df.index, columns=df.columns)
    
    # Clean the data
    # Replace infinity values with NaN first
    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    
    # Convert all columns to float to avoid integer/float issues
    for col in df_num.columns:
        df_num[col] = pd.to_numeric(df_num[col], errors='coerce').astype(float)
    
    # Remove rows that are all NaN
    df_num = df_num.dropna(how='all')
    if df_num.empty:
        return pd.DataFrame(False, index=df.index, columns=df.columns)
    
    # Remove columns that are all NaN
    df_num = df_num.dropna(axis=1, how='all')
    if df_num.empty:
        return pd.DataFrame(False, index=df.index, columns=df.columns)
    
        # Store scalers for each column
    scalers = {}
    df_scaled = df_num.copy()

    for col in df_num.columns:
        # Only scale if values are very large
        if df_num[col].max() > 1e9 or df_num[col].min() < -1e9:
            scaler = StandardScaler()
            df_scaled[col] = scaler.fit_transform(df_num[[col]])
            scalers[col] = scaler
            print(f"Scaled column '{col}' due to large values")

    # Use df_scaled instead of df_num for the rest of the function
    df_num = df_scaled
    
    # Store original indices
    original_index = df_num.index
    df_num = df_num.reset_index(drop=True)
    
    n_records, n_features = df_num.shape
    
    # Check if we have enough data
    if n_records < 10:
        print(f"Warning: Only {n_records} records available. Results may be unreliable.")
    
    if n_records < n_folds:
        n_folds = min(n_records, max(2, n_records // 2))
        print(f"Adjusted folds to {n_folds} due to small dataset")
    
    # Initialize error storage
    cell_errors = np.zeros((n_records, n_features))
    cell_counts = np.zeros((n_records, n_features))
    
    # Create missForest imputer with more robust settings
    def get_imputer():
        return IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=n_estimators, 
                random_state=random_state,
                max_depth=5  # Limit depth to prevent overfitting on small data
            ),
            max_iter=10,
            random_state=random_state,
            initial_strategy='median',  # More robust than mean
            imputation_order='ascending'
        )
    
    # Cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_count = 0
    for train_idx, test_idx in kf.split(df_num):
        fold_count += 1
        
        # Split data
        train_data = df_num.iloc[train_idx].values.astype(float)
        test_data = df_num.iloc[test_idx].values.astype(float)
        
        # For each test record
        for i, test_record_idx in enumerate(test_idx):
            # For each feature
            for j in range(n_features):
                # Skip if value is already missing
                if np.isnan(test_data[i, j]):
                    continue
                
                # Create dataset with this cell masked
                train_plus_test = np.vstack([train_data, test_data[i:i+1, :]])
                original_value = float(train_plus_test[-1, j])
                train_plus_test[-1, j] = np.nan
                
                try:
                    # Impute
                    imputer = get_imputer()
                    imputed_data = imputer.fit_transform(train_plus_test)
                    imputed_value = float(imputed_data[-1, j])
                    
                    # Calculate normalized error
                    col_std = np.nanstd(train_data[:, j])
                    if col_std > 0 and not np.isnan(col_std):
                        error = abs(original_value - imputed_value) / col_std
                    else:
                        # Use range-based normalization if std is 0
                        col_range = np.nanmax(train_data[:, j]) - np.nanmin(train_data[:, j])
                        if col_range > 0:
                            error = abs(original_value - imputed_value) / col_range
                        else:
                            error = abs(original_value - imputed_value)
                    
                    # Store error (cap at reasonable value to avoid outliers skewing results)
                    error = min(error, 10.0)
                    cell_errors[test_record_idx, j] += error
                    cell_counts[test_record_idx, j] += 1
                    
                except Exception as e:
                    # Skip this cell if imputation fails
                    continue
    
    # Average errors across folds
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_errors = np.divide(cell_errors, cell_counts)
        avg_errors[cell_counts == 0] = 0
    
    # Create error dataframe
    error_df = pd.DataFrame(avg_errors, columns=df_num.columns)

    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    # Determine threshold for each column
    for idx, col in enumerate(df_num.columns):
        # Get the original column name (before any transformations)
        original_col = df.select_dtypes(include=[np.number]).columns[idx]
        
        col_errors = error_df[col]
        col_errors_valid = col_errors[col_errors > 0]
        
        if len(col_errors_valid) > 0:
            threshold = np.percentile(col_errors_valid, error_threshold_percentile)
            # Map back to original indices
            suspicious_mask = col_errors > threshold
            
            # Use integer indexing to avoid issues
            for i, is_suspicious in enumerate(suspicious_mask):
                if is_suspicious:
                    original_idx = original_index[i]
                    mask.loc[original_idx, original_col] = True

    return mask


def diagnose_data_issues(df):
    """
    Diagnose potential data issues that could cause missForest to fail
    """
    import numpy as np
    import pandas as pd
    
    issues = []
    
    # Check numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Check for infinity values
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            issues.append(f"Column '{col}' has {inf_count} infinity values")
        
        # Check for very large values
        max_val = df[col].max()
        if max_val > 1e10:
            issues.append(f"Column '{col}' has very large values (max: {max_val:.2e})")
        
        # Check if column is actually numeric
        try:
            pd.to_numeric(df[col], errors='coerce')
        except:
            issues.append(f"Column '{col}' has non-numeric values despite being numeric type")
        
        # Check for constant columns
        if df[col].nunique() == 1:
            issues.append(f"Column '{col}' has only one unique value")
    
    return issues