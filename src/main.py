# src/main.py

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import segment_log
from metrics import compute_dfg, dfg_to_species_list, compute_chao1, compute_hill
import pandas as pd
import glob
import os

# Путь к sudden_logs.csv
SUDDEN_LOGS_PATH = "../data/sudden_logs.csv"

# Папка с XES логами
LOG_FOLDER = "../data/test/test/"

# Какой лог анализируем:
LOG_NUMBER = 1003

# Читаем список sudden logs
df_sudden = pd.read_csv(SUDDEN_LOGS_PATH)
sudden_log_names = df_sudden['log_name'].tolist()

# Ищем log_name, который начинается с "log_1010_"
matching_logs = [ln for ln in sudden_log_names if ln.startswith(f"log_{LOG_NUMBER}_")]

if not matching_logs:
    raise ValueError(f"No sudden log found with LOG_NUMBER={LOG_NUMBER}")

# Берём первый подходящий log (у тебя по сути один для каждого номера)
selected_log_name = matching_logs[0]

print(f"\nSelected log: {selected_log_name}")

# Теперь ищем соответствующий XES-файл
# Он будет в LOG_FOLDER как log_xxx_xxx.xes
xes_files = glob.glob(os.path.join(LOG_FOLDER, f"log_{LOG_NUMBER}_*.xes"))

if not xes_files:
    raise ValueError(f"No XES file found for LOG_NUMBER={LOG_NUMBER}")

# Берём первый подходящий (у тебя один XES на номер)
xes_path = xes_files[0]

print(f"Selected LOG_NUMBER={LOG_NUMBER}")
print(f"XES path: {xes_path}\n")


# Шаг 1: Загружаем log
print("Loading log...")

log = xes_importer.apply(xes_path)

print(f"\nNumber of traces: {len(log)}")
total_events = sum(len(trace) for trace in log)
print(f"Total number of events: {total_events}")


# Шаг 3: Извлекаем Directly-Follows Pairs (DFP)
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery

print("\nExtracting DFP...")
dfg = dfg_discovery.apply(log)

print("\nDirectly-Follows Pairs and frequencies:")
for (a, b), freq in dfg.items():
    print(f"{a} -> {b}: {freq}")

# Плитуем DFG (необязательно, но красиво)
from pm4py.visualization.dfg import visualizer as dfg_visualization

gviz = dfg_visualization.apply(dfg, variant=dfg_visualization.Variants.FREQUENCY, parameters={"format": "svg"})
dfg_visualization.view(gviz)

# === Загружаем gold_standard.csv → вытаскиваем change_points для этого лога

GOLD_STANDARD_PATH = "../data/gold_standard.csv"

df_gold = pd.read_csv(GOLD_STANDARD_PATH)

# Находим нужную строку:
row_gold = df_gold[df_gold['log_name'] == selected_log_name]

if row_gold.empty:
    raise ValueError(f"Log {selected_log_name} not found in gold_standard.csv")

# Парсим change_points:
import ast
change_points = ast.literal_eval(row_gold.iloc[0]['change_point'])

print(f"Drift points (event indices): {change_points}")
# Segment log → windows
windows = segment_log(log, window_size=80)

results = []

for i, window_traces in enumerate(windows):
    print(f"\nProcessing window {i+1}/{len(windows)} ...")
    
    # 1. DFG
    dfg = compute_dfg(window_traces)
    
    # 2. Species list
    species_list = dfg_to_species_list(dfg)
    
    # 3. Chao1
    chao1_value = compute_chao1(species_list)
    
    # 4. Hill number (например q=1)
    hill_value = compute_hill(species_list, q=1)
    
    # Сохраняем результат
    results.append({
        "window": i+1,
        "chao1": chao1_value,
        "hill_q1": hill_value
    })

# В DataFrame
df_profile = pd.DataFrame(results)

# Сохраняем профили
df_profile.to_csv("../data/completeness_profile_log_41.csv", index=False)

# === Посчитаем cumulative number of events per window


window_trace_bounds = []  # (trace_start_idx, trace_end_idx) for each window

trace_idx = 0

for window_traces in windows:
    traces_in_window = len(window_traces)
    window_trace_bounds.append( (trace_idx +1, trace_idx + traces_in_window) )
    trace_idx += traces_in_window



# === Для каждого drift trace → определим в какое окно он попал

# !!! Тут важно: предполагаем что change_points → это номер trace (1-based)

drift_trace_indices = change_points  # трактуем как номера trace!

drift_windows_from_traces = []

for trace_num in drift_trace_indices:
    window_idx = next(i+1 for i, (start, end) in enumerate(window_trace_bounds) if start <= trace_num <= end)
    drift_windows_from_traces.append(window_idx)

print("\nDrift trace points map to windows:", drift_windows_from_traces)

# Строим график
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(df_profile["window"], df_profile["chao1"], label="Chao1", marker="o")
plt.plot(df_profile["window"], df_profile["hill_q1"], label="Hill q=1", marker="o")

# Добавляем красные линии для drift points:
for dw in drift_windows_from_traces:
    plt.axvline(x=dw, color='red', linestyle='--', label='Drift' if dw == drift_windows_from_traces[0] else "")

plt.xlabel("Window")
plt.ylabel("Completeness Estimate")
plt.title(f"Completeness Profile - LOG_NUMBER {LOG_NUMBER}")
plt.legend()
plt.grid()

plt.savefig(f"../data/completeness_profile_log_{LOG_NUMBER}.png", dpi=300)
plt.show()