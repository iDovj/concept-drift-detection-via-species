# extract_sudden_logs.py

import pandas as pd
import ast

GOLD_STANDARD_PATH = "../data/gold_standard.csv"
OUTPUT_PATH = "../data/sudden_logs.csv"

df_gold = pd.read_csv(GOLD_STANDARD_PATH)

# В функцию преобразуем чтобы было красивее:

def is_all_sudden(change_types_str):
    change_types = ast.literal_eval(change_types_str)
    return all(ct == 'sudden' for ct in change_types)

# Фильтруем → только те где все 'sudden'
df_sudden = df_gold[df_gold['change_type'].apply(is_all_sudden)].copy()

# Добавляем колонку num_drifts → считаем количество change_point'ов
df_sudden['num_drifts'] = df_sudden['change_point'].apply(lambda x: len(ast.literal_eval(x)))

# Сохраняем
df_sudden.to_csv(OUTPUT_PATH, index=False)

print(f"Saved sudden_logs.csv with {len(df_sudden)} logs → columns: log_name, ..., num_drifts")