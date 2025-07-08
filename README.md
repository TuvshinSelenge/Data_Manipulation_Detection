# Data Manipulation Detection

An interdisciplinary project for detecting data manipulation and identifying potential manipulation methods.

## Overview

This project tackles data manipulation detection from multiple perspectives, implementing five different detection methods to identify suspicious or manipulated data points in datasets.

## Detection Methods

1. **Statistical Approach (Z-Score)**
   - Uses z-score analysis to identify statistical outliers
   - Implementation can be found in `app.py`

2. **Cellwise Outlier Detection (DDC)**
   - Utilizes the R package `cellWise` created by Jakob Raymaekers and Peter J. Rousseeuw
   - Detects outliers at the cell level rather than row level
   - Implementation in `cellwise_ddc.py`

3. **Isolation Forest**
   - Anomaly detection algorithm from scikit-learn
   - Identifies outliers by isolating anomalies in the feature space
   - Implementation in `cellwise_isoforest.py`

4. **Local Outlier Factor (LOF)**
   - Density-based anomaly detection from scikit-learn
   - Identifies outliers based on local density deviation
   - Implementation in `cellwise_lof.py`

5. **missForest with K-Fold Cross-Validation**
   - Supervised learning approach using Random Forest imputation
   - Detects manipulation by identifying cells that are difficult to predict
   - Implementation in `cellwise_missForest.py`
   - Note: This method requires more computation time than others

## Usage

### Quick Start

The project includes a Streamlit web interface for easy testing and visualization:

```bash
# Install dependencies
pip install streamlit pandas numpy scipy scikit-learn rpy2

# Run the application
streamlit run app.py


