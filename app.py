import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
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

def create_visualization(df_data, mask_df, method_name, container):
    """Create visualizations for detected outliers using matplotlib"""
    with container:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_col = st.selectbox(f"X-axis column ({method_name})", df_data.columns, key=f"x_{method_name}")
        
        with col2:
            y_col = st.selectbox(f"Y-axis column ({method_name})", 
                               [col for col in df_data.columns if col != x_col], 
                               key=f"y_{method_name}")
        
        with col3:
            plot_type = st.selectbox(f"Plot type ({method_name})", 
                                   ["Scatter Plot", "Box Plot", "Bar Plot (Flagged Count)", "Heatmap"],
                                   key=f"plot_{method_name}")
        
        #creating figures
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == "Scatter Plot":
            flagged_mask = mask_df[x_col] | mask_df[y_col]
            normal_mask = ~flagged_mask
            
            ax.scatter(df_data.loc[normal_mask, x_col], 
                      df_data.loc[normal_mask, y_col], 
                      c='blue', label='Normal', alpha=0.6, s=50)
            ax.scatter(df_data.loc[flagged_mask, x_col], 
                      df_data.loc[flagged_mask, y_col], 
                      c='red', label='Flagged', alpha=0.8, s=50)
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{method_name}: {x_col} vs {y_col}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        elif plot_type == "Box Plot":
            data_to_plot = []
            labels = []
            colors = []
            
            for col in [x_col, y_col]:
                normal_data = df_data.loc[~mask_df[col], col].dropna()
                flagged_data = df_data.loc[mask_df[col], col].dropna()
                
                if len(normal_data) > 0:
                    data_to_plot.append(normal_data)
                    labels.append(f"{col}\n(Normal)")
                    colors.append('lightblue')
                
                if len(flagged_data) > 0:
                    data_to_plot.append(flagged_data)
                    labels.append(f"{col}\n(Flagged)")
                    colors.append('lightcoral')
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(f"{method_name}: Distribution of {x_col} and {y_col}")
            ax.grid(True, alpha=0.3)
            
        elif plot_type == "Bar Plot (Flagged Count)":
            flagged_counts = mask_df.sum()
            total_counts = len(df_data)
            normal_counts = total_counts - flagged_counts
            
            x = np.arange(len(flagged_counts))
            width = 0.35
            
            ax.bar(x, normal_counts, width, label='Normal', color='lightblue')
            ax.bar(x, flagged_counts, width, bottom=normal_counts, label='Flagged', color='lightcoral')
            
            ax.set_xlabel('Columns')
            ax.set_ylabel('Count')
            ax.set_title(f"{method_name}: Flagged Cells per Column")
            ax.set_xticks(x)
            ax.set_xticklabels(flagged_counts.index, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:
            heatmap_data = mask_df.astype(int)
            sns.heatmap(heatmap_data, 
                       cmap=['white', 'red'], 
                       cbar_kws={'label': 'Flagged (1) / Normal (0)'},
                       ax=ax,
                       vmin=0, vmax=1)
            
            ax.set_title(f"{method_name}: Heatmap of Flagged Cells")
            ax.set_xlabel("Columns")
            ax.set_ylabel("Row Index")
        
        plt.tight_layout()
        st.pyplot(fig)

# Statistical Approach (Z-Score)
st.header("1. Statistical Detection (Z-score)")
zscores = np.abs(zscore(df_num, nan_policy='omit'))
z_thresh = st.slider("Z-score threshold (for outlier)", 2.0, 5.0, 3.0)
z_manip = (zscores > z_thresh)
z_manip_df = pd.DataFrame(z_manip, columns=df_num.columns, index=df_num.index)

st.write(f"**Cells flagged as manipulated (z-score > {z_thresh}):**")
styled = df_num.style.apply(
    lambda col: [highlight_cells(val, z_manip_df[col.name][i]) for i, val in enumerate(col)],
    axis=0
)
st.dataframe(styled)

# Z-score Visualization
with st.expander("📊 Z-score Visualization"):
    create_visualization(df_num, z_manip_df, "Z-score", st.container())

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
    
    # DDC Visualization
    with st.expander("📊 DDC Visualization"):
        create_visualization(df_num, mask_df, "DDC", st.container())
        
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

# Isolation Forest Visualization
with st.expander("📊 Isolation Forest Visualization"):
    create_visualization(df_num, cellwise_iso_mask, "Isolation Forest", st.container())

# Local Outlier Factor
st.header("4. Local Outlier Factor")

cellwise_lof_mask = detect_cellwise_lof(df_num, contamination=0.1, n_neighbors=10)
st.write("**Cells flagged as manipulated (red) by LOF :**")
styled_lof = df_num.style.apply(
    lambda col: ['background-color: red' if cellwise_lof_mask[col.name][i] else '' for i in range(len(col))],
    axis=0
)
st.dataframe(styled_lof)

# LOF Visualization
with st.expander("📊 LOF Visualization"):
    create_visualization(df_num, cellwise_lof_mask, "LOF", st.container())

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
    
    # missForest Visualization
    with st.expander("📊 missForest Visualization"):
        create_visualization(df_num_clean, missForest, "missForest", st.container())
    
    suspicious_count = missForest.sum().sum()
    if suspicious_count > 0:
        st.success(f"✅ Analysis complete! Found {suspicious_count} suspicious cells using the missForest method.")
        with st.expander("Detailed Results"):
            st.write("**Suspicious cells by column:**")
            for col in missForest.columns:
                count = missForest[col].sum()
                if count > 0:
                    st.write(f"  - {col}: {count} cells")
    
except Exception as e:
    st.error(f"missForest detection failed: {str(e)}")
    st.info("Reduce the number of folds or adjusting parameters if the dataset is small.")

# Summary Comparison
st.header("📊 Summary: Method Comparison")
methods = []
total_flagged = []
all_masks = {}

if 'z_manip_df' in locals():
    methods.append("Z-score")
    total_flagged.append(z_manip_df.sum().sum())
    all_masks["Z-score"] = z_manip_df

if 'mask_df' in locals():
    methods.append("DDC")
    total_flagged.append(mask_df.sum().sum())
    all_masks["DDC"] = mask_df

if 'cellwise_iso_mask' in locals():
    methods.append("Isolation Forest")
    total_flagged.append(cellwise_iso_mask.sum().sum())
    all_masks["Isolation Forest"] = cellwise_iso_mask

if 'cellwise_lof_mask' in locals():
    methods.append("LOF")
    total_flagged.append(cellwise_lof_mask.sum().sum())
    all_masks["LOF"] = cellwise_lof_mask

if 'missForest' in locals():
    methods.append("missForest")
    total_flagged.append(missForest.sum().sum())
    all_masks["missForest"] = missForest

if methods:
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(methods, total_flagged, color='skyblue')
        ax.set_xlabel('Method')
        ax.set_ylabel('Total Flagged Cells')
        ax.set_title('Total Flagged Cells by Method')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        if len(all_masks) > 1:
            consensus_matrix = None
            for method, mask in all_masks.items():
                if consensus_matrix is None:
                    consensus_matrix = mask.astype(int)
                else:
                    # Ensure same shape
                    if mask.shape == consensus_matrix.shape:
                        consensus_matrix += mask.astype(int)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(consensus_matrix, 
                       cmap='Reds', 
                       cbar_kws={'label': 'Number of Methods Flagging'},
                       ax=ax)
            ax.set_title('Consensus: Number of Methods Flagging Each Cell')
            ax.set_xlabel("Columns")
            ax.set_ylabel("Row Index")
            plt.tight_layout()
            st.pyplot(fig)

st.header("Summary: Manipulated Data Detected")
st.markdown("""
- **Red cells** are flagged as manipulated or outlier by each respective method.
- **Visualizations** help identify patterns and relationships in the flagged data.
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