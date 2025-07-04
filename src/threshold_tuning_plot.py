import os
import glob
import ast
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from pm4py.objects.log.importer.xes import importer as xes_importer
from preprocessing import segment_log_fixed_n
from metrics import compute_dfg, dfg_to_species_list, compute_chao1, compute_hill
from special4pm.estimation.metrics import completeness, coverage

# === Paths
LOG_FOLDER = "../data2/test/test/"
SUDDEN_LOGS_PATH = "../data/sudden_logs.csv"
GOLD_STANDARD_PATH = "../data/gold_standard.csv"
RESULTS_FOLDER = "../results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# === Load meta files
df_sudden = pd.read_csv(SUDDEN_LOGS_PATH)
df_gold = pd.read_csv(GOLD_STANDARD_PATH)

summary_results = []

# === Process last 40 logs
for log_number in df_sudden['log_name'].str.extract(r'log_(\d+)_')[0].unique()[100:400]:
    print(f"Processing log number: {log_number}")
    selected_log_name = df_sudden[df_sudden['log_name'].str.startswith(f"log_{log_number}_")]['log_name'].iloc[0]
    xes_files = glob.glob(os.path.join(LOG_FOLDER, f"log_{log_number}_*.xes"))
    if not xes_files:
        print(f"No XES file found for log number {log_number}")
        continue
    log = xes_importer.apply(xes_files[0])

    row_gold = df_gold[df_gold['log_name'] == selected_log_name]
    if row_gold.empty:
        print(f"No gold standard found for log {selected_log_name}")
        continue
    change_points = ast.literal_eval(row_gold.iloc[0]['change_point'])

    # === Segmentation
    windows = segment_log_fixed_n(log, n_windows=15)
    window_trace_bounds = []
    trace_idx = 0
    for window in windows:
        n = len(window)
        window_trace_bounds.append((trace_idx + 1, trace_idx + n))
        trace_idx += n

    # === Ground truth drift windows (±1 tolerance)
    drift_windows = set()
    for cp in change_points:
        for i, (start, end) in enumerate(window_trace_bounds):
            if start - 1 <= cp <= end + 1:
                drift_windows.add(i + 1)

    # === Compute metrics
    results = []
    for i, traces in enumerate(windows):
        dfg = compute_dfg(traces)
        species = dfg_to_species_list(dfg)
        species_counts = {j: count for j, count in enumerate(species)}
        sample_size = sum(species_counts.values())

        results.append({
            "window": i + 1,
            "n_traces": len(traces),
            "chao1": compute_chao1(species),
            "hill_q1": compute_hill(species, q=1),
            "hill_q2": compute_hill(species, q=2),
            "completeness": completeness(species_counts),
            "coverage": coverage(species_counts, sample_size)
        })

    df = pd.DataFrame(results)

    # === Directional deltas
    for col in ["chao1", "completeness", "coverage"]:
        df[f"delta_{col}"] = df[col].diff().abs().fillna(0)
    for col in ["hill_q1", "hill_q2"]:
        df[f"delta_{col}"] = df[col].diff().clip(lower=0).fillna(0)

    # === Z-scores and relative deltas
    for col in ["chao1", "hill_q1", "hill_q2", "completeness", "coverage"]:
        df[f"zscore_{col}"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
        df[f"reldelta_{col}"] = df[col].diff() / (df[col].shift(1) + 1e-9)

    df["is_drift"] = df["window"].isin(drift_windows)

    # === Evaluate all variants
    variant_cols = [
        "delta_chao1", "delta_hill_q1", "delta_hill_q2", "delta_completeness", "delta_coverage",
        "zscore_chao1", "zscore_hill_q1", "zscore_hill_q2",
        "reldelta_chao1", "reldelta_hill_q1", "reldelta_hill_q2"
    ]

    for col in variant_cols:
        df_filtered = df[df["n_traces"] >= 1]
        nonzero = df_filtered[col][df_filtered[col] > 0]

        if nonzero.empty:
            threshold = float("inf")
        else:
            median = nonzero.median()
            mad = np.median(np.abs(nonzero - median))
            base_threshold = median + 2 * mad
            max_delta = nonzero.max()
            ratio = max_delta / base_threshold if base_threshold > 0 else float("inf")
            threshold = max_delta * 0.8 if ratio < 1.2 and max_delta > 0.5 else base_threshold

        pred_col = f"predicted_drift_{col}"
        df[pred_col] = df[col] > threshold

        y_true = df["is_drift"]
        y_pred = df[pred_col]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        summary_results.append({
            "log_number": log_number,
            "metric": col,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

# === Save
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(f"{RESULTS_FOLDER}/summary_fscore_per_metric_15w_v1.csv", index=False)
print(summary_df)

# === Binned recall/f1
bins = [0.0, 0.5, 0.75, 0.9999, 1.0]
labels = ["0–0.5", "0.5–0.75", "0.75–1.0", "1.0"]

summary_df["recall_bin"] = pd.cut(summary_df["recall"], bins=bins, labels=labels, include_lowest=True)
summary_df["f1_bin"] = pd.cut(summary_df["f1"], bins=bins, labels=labels, include_lowest=True)
summary_df["precision_bin"] = pd.cut(summary_df["precision"], bins=bins, labels=labels, include_lowest=True)

recall_stats = summary_df.groupby(["metric", "recall_bin"]).size().unstack(fill_value=0)
f1_stats = summary_df.groupby(["metric", "f1_bin"]).size().unstack(fill_value=0)
precision_stats = summary_df.groupby(["metric", "precision_bin"]).size().unstack(fill_value=0)

combined_stats = recall_stats.join(f1_stats, lsuffix="_recall", rsuffix="_f1").join(precision_stats, rsuffix="_precision")
combined_stats.to_csv(f"{RESULTS_FOLDER}/recall_f1_precision_distribution_15w_v1.csv")

print("\n=== Distribution by recall, f1, and precision bins ===")
print(combined_stats)
print("\nProcessing complete.")
