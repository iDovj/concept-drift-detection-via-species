# concept-drift-detection-species# concept-drift-detection-via-species

# Concept Drift Detection via Species Estimation Metrics

This repository implements a simple, interpretable pipeline for **sudden** concept-drift detection in business-process event logs.  
Rather than heavy ML models, we segment the log into fixed windows, extract “species” (directly-follows pairs), compute biodiversity metrics (Chao1, Hill numbers, completeness, coverage), engineer delta/z/relative features, and apply an MAD-threshold to flag drift points.

**Key advantages**  
- No training required  
- Minimal parameters (just `N`, number of windows)  
- Fast, works in real-time settings  
- Fully transparent and easy to integrate into existing process-mining toolchains  

## Dependencies

All code was developed and tested with **Python 3.8+**.

Core libraries:

- [pm4py]
- pandas  
- matplotlib  
- seaborn  
- special4pm  
- cachetools  
- mpmath  


## Repository structure

- `README.md`
- `requirements.txt` ‒ pinned dependencies
- `preprocessing.py` ‒ log segmentation & bigram extraction
- `metrics.py` ‒ compute Chao1, Hill numbers, completeness, coverage
- `evaluate_metrics.py` ‒ feature engineering & MAD-based thresholding
- `evaluate_metrics_is_drift.py` ‒ window- and latency-aware performance evaluation
- `threshold_tuning_plot.py` ‒ threshold tuning & performance curves
- `manual_window_plot.py` ‒ generate illustrative metric-trend figures
- `exp_points.py` ‒ latency-aware drift-point matching
- `figures/` ‒ folder for generated PNGs used in the report

## Data

All experiments use the **CDLG** test event logs (with synthetic sudden drifts).  

https://huggingface.co/pm-science/cv4cdd_project/tree/main/datasets/cdlg/zipped_test


## 📖 Overview

1. **Preprocessing**  
   - Partition each XES log into _N_ equal‐sized windows  
   - **Select only logs with sudden drift** (using `select_sudden_logs.py`)  
   - Extract “species” as activity bigrams per window  

2. **Metrics Computation**  
   - Compute Chao1, Hill numbers (q=0,1,2), completeness, and coverage  
   - Implemented in `metrics.py`  

3. **Feature Engineering & Thresholding**  
   - Derive Δ-features (absolute differences), z-scores, and relative deltas  
   - Apply median + 2×MAD thresholding to flag drift windows  
   - Implemented in `evaluate_metrics.py`  

4. **Evaluation**  
   - Window‐level precision/recall/F1 and latency‐aware matching  
   - Scripts in `evaluate_metrics_is_drift.py` and `exp_points.py`  

5. **Visualization & Tuning**  
   - Plot example metric trends (`manual_window_plot.py`)  
   - Tune and display thresholds (`threshold_tuning_plot.py`)  

