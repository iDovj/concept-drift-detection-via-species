import os
import glob
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer

# === Параметры
LOG_FOLDER = "../data/test/test/"
SUDDEN_LOGS_PATH = "../data/sudden_logs.csv"
RESULTS_FOLDER = "../results"

os.makedirs(RESULTS_FOLDER, exist_ok=True)

# === Загрузить список sudden логов
df_sudden = pd.read_csv(SUDDEN_LOGS_PATH)
log_ids = df_sudden['log_name'].str.extract(r'log_(\d+)_')[0].unique()

# === Собрать данные
log_info = []

for log_id in log_ids:
    xes_files = glob.glob(os.path.join(LOG_FOLDER, f"log_{log_id}_*.xes"))
    if not xes_files:
        print(f"No XES file found for log number {log_id}")
        continue

    filepath = xes_files[0]
    try:
        log = xes_importer.apply(filepath)
        n_traces = len(log)
        log_name = os.path.basename(filepath)

        log_info.append({
            "log_number": log_id,
            "log_name": log_name,
            "num_traces": n_traces
        })

        print(f"{log_name}: {n_traces} traces")

    except Exception as e:
        print(f"Error reading {filepath}: {e}")

# === Сохранить в CSV
log_df = pd.DataFrame(log_info)
log_df.to_csv(os.path.join(RESULTS_FOLDER, "log_trace_counts.csv"), index=False)
print("Saved log trace counts to CSV.")
