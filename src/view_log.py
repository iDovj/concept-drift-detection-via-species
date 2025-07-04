# src/view_log.py

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd

# Путь к твоему log файлу
LOG_PATH = "../data/test/test/sudden_trace_noise0_1000_IRO.xes"

# Загружаем log
print("Loading log...")
log = xes_importer.apply(LOG_PATH)

# Функция: log → DataFrame
def log_to_dataframe(log):
    rows = []
    for trace in log:
        case_id = trace.attributes["concept:name"]
        for event in trace:
            row = {
                "case_id": case_id,
                "event_name": event["concept:name"],
                "timestamp": event["time:timestamp"],
                # Добавляем все остальные атрибуты события (если есть)
                **{k: v for k, v in event.items() if k not in ["concept:name", "time:timestamp"]}
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df

# Преобразуем
df_log = log_to_dataframe(log)

# Сохраняем в CSV
csv_path = LOG_PATH.split("/")[-1].replace(".xes", "_flat.csv")
output_path = f"../data/{csv_path}"
df_log.to_csv(output_path, index=False)

print(f"\nSaved log as CSV to {output_path}")

# Показываем первые 100 строк
# print("\nFirst 100 rows of the log:")
# print(df_log.head(100))
