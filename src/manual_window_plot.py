# src/main.py

import os
import glob
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

from pm4py.objects.log.importer.xes import importer as xes_importer
from preprocessing import segment_log_fixed_n
from metrics import compute_dfg, dfg_to_species_list, compute_chao1, compute_hill
from special4pm.estimation.metrics import completeness, coverage

# === Параметры
LOG_NUMBER = 1739
LOG_FOLDER = "../data/test/test/"
SUDDEN_LOGS_PATH = "../data/sudden_logs.csv"
GOLD_STANDARD_PATH = "../data/gold_standard.csv"

# === Загрузка данных
df_sudden = pd.read_csv(SUDDEN_LOGS_PATH)
matching_logs = [ln for ln in df_sudden['log_name'] if ln.startswith(f"log_{LOG_NUMBER}_")]
if not matching_logs:
    raise ValueError(f"No sudden log found for LOG_NUMBER={LOG_NUMBER}")
selected_log_name = matching_logs[0]

xes_path = glob.glob(os.path.join(LOG_FOLDER, f"log_{LOG_NUMBER}_*.xes"))[0]
log = xes_importer.apply(xes_path)

df_gold = pd.read_csv(GOLD_STANDARD_PATH)
row_gold = df_gold[df_gold['log_name'] == selected_log_name]
change_points = ast.literal_eval(row_gold.iloc[0]['change_point'])

# === Сегментация
windows = segment_log_fixed_n(log, n_windows=20)
window_trace_bounds = []
trace_idx = 0
for window in windows:
    n = len(window)
    window_trace_bounds.append((trace_idx + 1, trace_idx + n))
    trace_idx += n

# === Дрифт-окна
drift_windows = []
for trace_num in change_points:
    window_idx = next(i + 1 for i, (start, end) in enumerate(window_trace_bounds) if start <= trace_num <= end)
    drift_windows.append(window_idx)

# === Метрики по окнам
results = []
for i, traces in enumerate(windows):
    dfg = compute_dfg(traces)
    species = dfg_to_species_list(dfg)
    species_counts = {j: count for j, count in enumerate(species)}
    sample_size = sum(species_counts.values())

    results.append({
        "window": i + 1,
        "n_traces": len(traces),
        "hill_q0": compute_chao1(species),
        "hill_q1": compute_hill(species, q=1),
        "hill_q2": compute_hill(species, q=2),
        "completeness": completeness(species_counts),
        "coverage": coverage(species_counts, sample_size)
    })

df = pd.DataFrame(results)

# === Фичи
for col in ["hill_q0", "completeness", "coverage"]:
    df[f"delta_{col}"] = df[col].diff().abs().fillna(0)
for col in ["hill_q1", "hill_q2"]:
    df[f"delta_{col}"] = df[col].diff().clip(lower=0).fillna(0)
for col in ["hill_q0", "hill_q1", "hill_q2", "completeness", "coverage"]:
    df[f"zscore_{col}"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
    df[f"reldelta_{col}"] = df[col].diff() / (df[col].shift(1) + 1e-9)

df["is_drift"] = df["window"].isin(drift_windows)

# === Оценка по всем метрикам
variant_cols = [
    "delta_hill_q0", "delta_hill_q1", "delta_hill_q2",
    "delta_completeness", "delta_coverage",
    "zscore_hill_q0", "zscore_hill_q1", "zscore_hill_q2",
    "reldelta_hill_q0", "reldelta_hill_q1", "reldelta_hill_q2"
]

print(f"\n=== DRIFT DETECTION RESULTS FOR LOG {LOG_NUMBER} ===\n")

for col in variant_cols:
    df_filtered = df[df["n_traces"] > 0]
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

    print(f"{col:25s} | threshold: {threshold:.4f} | precision: {precision:.3f} | recall: {recall:.3f} | f1: {f1:.3f}")

# === Визуализация
plt.figure(figsize=(14, 6))
for col in ["hill_q0", "hill_q1", "hill_q2", "completeness", "coverage"]:
    plt.plot(df["window"], df[col], marker='o', label=col)

for dw in drift_windows:
    plt.axvline(x=dw, color='red', linestyle='--', label='Drift' if dw == drift_windows[0] else "")

plt.xlabel("Window")
plt.ylabel("Metric Value")
plt.title(f"Metric Trends — Log {LOG_NUMBER}")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"../figures/metric_trends_log_w20_{LOG_NUMBER}.png", dpi=300)
plt.show()
