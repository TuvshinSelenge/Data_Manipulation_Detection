
# Cellwise Local Outlier Factor
def detect_cellwise_lof(df, contamination=0.1, n_neighbors=10):
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import LocalOutlierFactor

    df_num = df.select_dtypes(include=[np.number])
    mask = pd.DataFrame(False, index=df_num.index, columns=df_num.columns)
    for col in df_num.columns:
        col_data = df_num[[col]].dropna()  # 2D DataFrame
        if len(col_data) < n_neighbors + 1:  # LOF needs more points than n_neighbors
            continue
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        preds = lof.fit_predict(col_data)
        # Outlier if -1
        mask.loc[col_data.index, col] = (preds == -1)
    return mask
