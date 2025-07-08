def detect_cellwise_outliers(df):
    import numpy as np
    import pandas as pd
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()

    df_num = df.select_dtypes(include=[np.number])
    r_df = pandas2ri.py2rpy(df_num)

    # Loading R package and starting DDC
    robjects.r('library(cellWise)')
    robjects.globalenv['df_r'] = r_df
    ddc_result = robjects.r('DDC(df_r)')

    #Getting Outlier-Indizes (1-based, columnwise)
    indcells = np.array(ddc_result.rx2('indcells')).astype(int)
    rows, cols = df_num.shape

    # Mask to mark outlier cells
    cell_flags = np.zeros(df_num.shape, dtype=bool)
    for idx in indcells:
        if 1 <= idx <= rows * cols:
            idx0 = idx - 1  # 0-based
            row = idx0 % rows
            col = idx0 // rows
            cell_flags[row, col] = True

    mask_df = pd.DataFrame(cell_flags, columns=df_num.columns, index=df_num.index)
    return mask_df
