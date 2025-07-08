


def detect_cellwise_missforest(df, n_folds=5, error_threshold_percentile=95, 
                               n_estimators=50, random_state=42, auto_optimize=True):
    """
    Detects potentially manipulated cells using cross-validated missForest imputation,
    with automatic parameter optimization based on dataset size.
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
    
    df_num = df.select_dtypes(include=[np.number]).copy()
    if df_num.empty:
        return pd.DataFrame(False, index=df.index, columns=df.columns)
    
    n_rows, n_cols = df_num.shape
    total_cells = n_rows * n_cols
    
    if auto_optimize:
        if total_cells > 100000:  
            n_folds = 2
            n_estimators = 10
            max_iter = 3
            print(f"Very large dataset ({n_rows}×{n_cols}). Using: {n_folds} folds, {n_estimators} trees")
        elif total_cells > 50000:  
            n_folds = 3
            n_estimators = 20
            max_iter = 5
            print(f"Large dataset ({n_rows}×{n_cols}). Using: {n_folds} folds, {n_estimators} trees")
        elif total_cells > 10000:  
            n_folds = min(n_folds, 3)
            n_estimators = min(n_estimators, 30)
            max_iter = 7
            print(f"Medium dataset ({n_rows}×{n_cols}). Using: {n_folds} folds, {n_estimators} trees")
        else:  
            max_iter = 10
            print(f"Small dataset ({n_rows}×{n_cols}). Using default parameters")
    else:
        max_iter = 10

    if n_cols > 50:
        max_depth = 3
    elif n_cols > 20:
        max_depth = 4
    else:
        max_depth = 5
    
    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    
    for col in df_num.columns:
        df_num[col] = pd.to_numeric(df_num[col], errors='coerce').astype(float)
    
    df_num = df_num.dropna(how='all')
    if df_num.empty:
        return pd.DataFrame(False, index=df.index, columns=df.columns)
    
    df_num = df_num.dropna(axis=1, how='all')
    if df_num.empty:
        return pd.DataFrame(False, index=df.index, columns=df.columns)
    
    scalers = {}
    df_scaled = df_num.copy()
    for col in df_num.columns:
        if df_num[col].max() > 1e9 or df_num[col].min() < -1e9:
            scaler = StandardScaler()
            df_scaled[col] = scaler.fit_transform(df_num[[col]])
            scalers[col] = scaler
    
    df_num = df_scaled
    original_index = df_num.index
    df_num = df_num.reset_index(drop=True)
    
    n_records, n_features = df_num.shape
    
    if n_records < n_folds:
        n_folds = max(2, n_records // 2)
        print(f"Adjusted folds to {n_folds} due to small number of records")
    
    # Initialize error storage
    cell_errors = np.zeros((n_records, n_features))
    cell_counts = np.zeros((n_records, n_features))
    
    def get_imputer():
        return IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=n_estimators, 
                random_state=random_state,
                max_depth=max_depth,
                n_jobs=-1 
            ),
            max_iter=max_iter,
            random_state=random_state,
            initial_strategy='median',
            imputation_order='ascending'
        )

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_count = 0
    for train_idx, test_idx in kf.split(df_num):
        fold_count += 1
        
        train_data = df_num.iloc[train_idx].values.astype(float)
        test_data = df_num.iloc[test_idx].values.astype(float)
        
        for i, test_record_idx in enumerate(test_idx):
            for j in range(n_features):
                if np.isnan(test_data[i, j]):
                    continue
                
                train_plus_test = np.vstack([train_data, test_data[i:i+1, :]])
                original_value = float(train_plus_test[-1, j])
                train_plus_test[-1, j] = np.nan
                
                try:
                    imputer = get_imputer()
                    imputed_data = imputer.fit_transform(train_plus_test)
                    imputed_value = float(imputed_data[-1, j])
                    
                    col_std = np.nanstd(train_data[:, j])
                    if col_std > 0 and not np.isnan(col_std):
                        error = abs(original_value - imputed_value) / col_std
                    else:
                        col_range = np.nanmax(train_data[:, j]) - np.nanmin(train_data[:, j])
                        if col_range > 0:
                            error = abs(original_value - imputed_value) / col_range
                        else:
                            error = abs(original_value - imputed_value)

                    error = min(error, 10.0)
                    cell_errors[test_record_idx, j] += error
                    cell_counts[test_record_idx, j] += 1
                    
                except Exception as e:
                    continue
    
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_errors = np.divide(cell_errors, cell_counts)
        avg_errors[cell_counts == 0] = 0
    
    # error dataframe
    error_df = pd.DataFrame(avg_errors, columns=df_num.columns)

    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for idx, col in enumerate(df_num.columns):
        original_col = df.select_dtypes(include=[np.number]).columns[idx]
        
        col_errors = error_df[col]
        col_errors_valid = col_errors[col_errors > 0]
        
        if len(col_errors_valid) > 0:
            threshold = np.percentile(col_errors_valid, error_threshold_percentile)
            suspicious_mask = col_errors > threshold
            
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