
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
from cellwise_ddc import detect_cellwise_outliers
from cellwise_isoforest import detect_cellwise_isoforest
from cellwise_lof import detect_cellwise_lof
from cellwise_missForest import detect_cellwise_missforest

st.set_page_config(layout="wide")

st.title("Data Manipulation Detection: Multiple Methods")
st.markdown("""
This app applies several approaches to detect manipulated/outlier data cells:
- **1. Statistical (z-score)**
- **2. Cellwise detection (R: cellWise::DDC)**
- **3. Isolation Forest**
- **4. Local Outlier Factor (LOF)**
- **5. missForest with K-Crossvalidation**
""")

st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file with numeric data to begin analysis.")
    st.stop() 

def convert_boolean_columns(df):
    """Convert columns with TRUE/FALSE, YES/NO values to numeric 1/0"""
    df_converted = df.copy()
    
    for col in df_converted.columns:
        # Check if column contains boolean-like values
        unique_values = df_converted[col].dropna().astype(str).str.upper().unique()
        
        if set(unique_values).issubset({'TRUE', 'FALSE', 'T', 'F'}):
            st.info(f"Converting column '{col}' from TRUE/FALSE to 1/0")
            df_converted[col] = df_converted[col].astype(str).str.upper().map({
                'TRUE': 1, 'T': 1, 'FALSE': 0, 'F': 0
            })
        
        elif set(unique_values).issubset({'YES', 'NO', 'Y', 'N'}):
            st.info(f"Converting column '{col}' from YES/NO to 1/0")
            df_converted[col] = df_converted[col].astype(str).str.upper().map({
                'YES': 1, 'Y': 1, 'NO': 0, 'N': 0
            })
        
        elif set(unique_values).issubset({'1', '0'}):
            df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
        
    return df_converted

df = convert_boolean_columns(df)

with st.expander("Data Type Information"):
    st.write("**Column data types after conversion:**")
    dtype_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str)
    })
    st.dataframe(dtype_df)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
df_num = df[num_cols].copy()

if not num_cols:
    st.error("No numeric columns found in your CSV. Please upload numeric data.")
    st.stop()

st.write("### Preview of Data")
st.dataframe(df_num.head())

def highlight_cells(val, manip):
    return 'background-color: red' if manip else ''

# Statistical Approach (Z-Score)
st.header("1. Statistical Detection (Z-score)")
zscores = np.abs(zscore(df_num, nan_policy='omit'))
z_thresh = st.slider("Z-score threshold (for outlier)", 2.0, 5.0, 5.0)
z_manip = (zscores > z_thresh)
z_manip_df = pd.DataFrame(z_manip, columns=df_num.columns, index=df_num.index)

st.write(f"**Cells flagged as manipulated (z-score > {z_thresh}):**")
styled = df_num.style.apply(
    lambda col: [highlight_cells(val, z_manip_df[col.name][i]) for i, val in enumerate(col)],
    axis=0
)
st.dataframe(styled)

# Cellwise Detection (R DDC)
st.header("2. Cellwise Detection (R: cellWise::DDC)")
try:
    mask_df = detect_cellwise_outliers(df_num)
    if mask_df.sum().sum() == 0:
        st.warning("⚠️ R/DDC method may not be available in this deployment. Showing placeholder results.")
    st.write("**Cells flagged as manipulated by DDC (red):**")
    styled_ddc = df_num.style.apply(
        lambda col: [highlight_cells(val, mask_df[col.name][i]) for i, val in enumerate(col)],
        axis=0
    )
    st.dataframe(styled_ddc)
except Exception as e:
    st.warning(f"Cellwise DDC detection not available: R package required")

# Isolation Forest
st.header("3. Isolation Forest")

cellwise_iso_mask = detect_cellwise_isoforest(df_num, contamination=0.1, random_state=0)
st.write("**Cells flagged as manipulated (red) by Isolation Forest (per column):**")
styled_iso = df_num.style.apply(
    lambda col: ['background-color: red' if cellwise_iso_mask[col.name][i] else '' for i in range(len(col))],
    axis=0
)
st.dataframe(styled_iso)

# Local Outlier Factor
st.header("4. Local Outlier Factor")

cellwise_lof_mask = detect_cellwise_lof(df_num, contamination=0.1, n_neighbors=10)
st.write("**Cells flagged as manipulated (red) by LOF :**")
styled_lof = df_num.style.apply(
    lambda col: ['background-color: red' if cellwise_lof_mask[col.name][i] else '' for i in range(len(col))],
    axis=0
)
st.dataframe(styled_lof)

# missForest
st.header("5. missForest with K-Crossvalidation")

df_num_clean = df_num.copy()
for col in df_num_clean.columns:
    df_num_clean[col] = pd.to_numeric(df_num_clean[col], errors='coerce')

df_num_clean = df_num_clean.reset_index(drop=True)

try:
    with st.spinner('Running missForest cross-validation... This may take a moment.'):
        missForest = detect_cellwise_missforest(
            df_num_clean, 
            n_folds=5,
            error_threshold_percentile=90,
            n_estimators=50
        )
    
    st.write("**Cells flagged as manipulated (red) by missForest:**")
    styled_missForest = df_num.style.apply(
        lambda col: ['background-color: red' if i < len(missForest) and col.name in missForest.columns and missForest.loc[i, col.name] else '' 
                     for i in range(len(col))],
        axis=0
    )
    st.dataframe(styled_missForest)
    
    # Show summary statistics
    suspicious_count = missForest.sum().sum()
    if suspicious_count > 0:
        st.success(f"✅ Analysis complete! Found {suspicious_count} suspicious cells.")
        with st.expander("Detailed Results"):
            st.write("**Suspicious cells by column:**")
            for col in missForest.columns:
                count = missForest[col].sum()
                if count > 0:
                    st.write(f"  - {col}: {count} cells")
    
except Exception as e:
    st.error(f"missForest detection failed: {str(e)}")
    st.info("Try reducing the number of folds or adjusting parameters if the dataset is small.")

st.header("Summary: Manipulated Data Detected")
st.markdown("""
- **Red cells** are flagged as manipulated or outlier by each respective method.
- Boolean values (TRUE/FALSE, YES/NO) are automatically converted to 1/0.
- Use the sliders to adjust sensitivity for statistical and cellwise detection.
""")

st.write("**Methods compared:**")
st.markdown("""
- **Statistical (z-score):** Individual cells flagged if their z-score exceeds threshold.
- **Cellwise detection (R DDC):** Robust method from R (requires R installation).
- **Isolation Forest:** Anomaly detection per column.
- **LOF:** Local density-based outlier detection per column.
- **missForest:** Cross-validated imputation error - cells with high prediction errors when masked.
""")
