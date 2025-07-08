
# Cellwise isolation forest

def detect_cellwise_isoforest(df, contamination=0.1, random_state=0):
    """
    Detects outliers in each numeric column using IsolationForest,
    returning a boolean DataFrame mask with True for detected outlier cells.
    """
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest

    df_num = df.select_dtypes(include=[np.number])
    mask = pd.DataFrame(False, index=df_num.index, columns=df_num.columns)
    for col in df_num.columns:
        col_data = df_num[[col]].dropna()  # IsolationForest in 2D
        if len(col_data) < 3: #if not enought data
            continue  
        iso = IsolationForest(contamination=contamination, random_state=random_state)
        preds = iso.fit_predict(col_data)

        # Outlier if it is -1
        mask.loc[col_data.index, col] = (preds == -1)
        
    return mask