


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
            batch_size = 50
        elif total_cells > 50000:  
            n_folds = 3
            n_estimators = 15
            max_iter = 5
            batch_size = 30
        elif total_cells > 10000:  
            n_folds = 3
            n_estimators = 20
            max_iter = 5
            batch_size = 20
        else:  
            n_folds = min(n_folds, 3)
            n_estimators = min(n_estimators, 30)
            max_iter = 7
            batch_size = 10
    else:
        max_iter = 10
        batch_size = 10
    
    max_depth = 3 if n_cols > 30 else 4

    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    for col in df_num.columns:
        df_num[col] = pd.to_numeric(df_num[col], errors='coerce').astype(float)
    
    df_num = df_num.dropna(how='all').dropna(axis=1, how='all')
    if df_num.empty:
        return pd.DataFrame(False, index=df.index, columns=df.columns)
    
    scalers = {}
    df_scaled = df_num.copy()
    for col in df_num.columns:
        if abs(df_num[col].max()) > 1e9 or abs(df_num[col].min()) > 1e9:
            scaler = StandardScaler()
            df_scaled[col] = scaler.fit_transform(df_num[[col]])
            scalers[col] = scaler
    
    df_num = df_scaled
    original_index = df_num.index
    df_num = df_num.reset_index(drop=True)
    
    n_records, n_features = df_num.shape
    
    if n_records < n_folds * 2:
        n_folds = max(2, n_records // 2)
    
    cell_errors = np.zeros((n_records, n_features))
    cell_counts = np.zeros((n_records, n_features))
    
    imputer_config = {
        'estimator': RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        ),
        'max_iter': max_iter,
        'random_state': random_state,
        'initial_strategy': 'median'
    }
    
    print(f"Running optimized missForest: {n_folds} folds, {n_estimators} trees, batch_size={batch_size}")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df_num)):
        print(f"Processing fold {fold_idx + 1}/{n_folds}")
        
        train_data = df_num.iloc[train_idx].values.astype(float)
        test_data = df_num.iloc[test_idx].values.astype(float)
        
        imputer = IterativeImputer(**imputer_config)
        
        # Fit on training data with some artificial missing values
        train_with_missing = train_data.copy()
        # Add 10% random missing values to help imputer learn patterns
        mask = np.random.random(train_with_missing.shape) < 0.1
        train_with_missing[mask] = np.nan
        
        try:
            imputer.fit(train_with_missing)
        except:
            continue
        
        for batch_start in range(0, len(test_idx), batch_size):
            batch_end = min(batch_start + batch_size, len(test_idx))
            batch_test_indices = test_idx[batch_start:batch_end]
            batch_test_data = test_data[batch_start:batch_end]
            
            for feature_idx in range(n_features):
                feature_values = batch_test_data[:, feature_idx]
                if np.all(np.isnan(feature_values)):
                    continue
                
                batch_masked = batch_test_data.copy()
                original_values = batch_masked[:, feature_idx].copy()
                batch_masked[:, feature_idx] = np.nan
                
                try:
                    imputed_batch = imputer.transform(batch_masked)
                    imputed_values = imputed_batch[:, feature_idx]
                    
                    col_std = np.nanstd(train_data[:, feature_idx])
                    if col_std > 0:
                        errors = np.abs(original_values - imputed_values) / col_std
                    else:
                        col_range = np.nanmax(train_data[:, feature_idx]) - np.nanmin(train_data[:, feature_idx])
                        if col_range > 0:
                            errors = np.abs(original_values - imputed_values) / col_range
                        else:
                            errors = np.abs(original_values - imputed_values)
              
                    for i, (error, test_idx_val) in enumerate(zip(errors, batch_test_indices)):
                        if not np.isnan(error) and not np.isnan(original_values[i]):
                            cell_errors[test_idx_val, feature_idx] += min(error, 10.0)
                            cell_counts[test_idx_val, feature_idx] += 1
                
                except:
                    continue
    
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_errors = np.divide(cell_errors, cell_counts)
        avg_errors[cell_counts == 0] = 0
    
    error_df = pd.DataFrame(avg_errors, columns=df_num.columns)
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    for idx, col in enumerate(df_num.columns):
        original_col = df.select_dtypes(include=[np.number]).columns[idx]
        col_errors = error_df[col]
        col_errors_valid = col_errors[col_errors > 0]
        
        if len(col_errors_valid) > 0:
            threshold = np.percentile(col_errors_valid, error_threshold_percentile)
            suspicious_indices = original_index[col_errors > threshold]
            mask.loc[suspicious_indices, original_col] = True
    
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