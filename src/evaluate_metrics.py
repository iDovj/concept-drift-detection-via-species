import pandas as pd

# Путь к файлу с подробными результатами
INPUT_CSV = "../results/evaluation_latency_30w.csv"
# Путь для сохранения сводной таблицы
OUTPUT_CSV = "../results/summary_metrics_by_feature_latency_30w.csv"

# Считываем данные
df = pd.read_csv(INPUT_CSV)

# Группируем по признаку и latency, вычисляем по среднему precision, recall, f1
summary = (
    df
    .groupby(['feature', 'latency'])
    .agg(
        precision_mean = ('precision', 'mean'),
        recall_mean    = ('recall',    'mean'),
        f1_mean        = ('f1',        'mean'),
        precision_std  = ('precision', 'std'),
        recall_std     = ('recall',    'std'),
        f1_std         = ('f1',        'std'),
        count          = ('feature',   'count')
    )
    .reset_index()
)

# Для удобства можно свернуть в формат с latency по столбцам
pivot_precision = summary.pivot(index='feature', columns='latency', values='precision_mean')
pivot_recall    = summary.pivot(index='feature', columns='latency', values='recall_mean')
pivot_f1        = summary.pivot(index='feature', columns='latency', values='f1_mean')

# Собираем единый DataFrame
agg = pd.concat([
    pivot_precision.add_prefix('P@'),
    pivot_recall.add_prefix('R@'),
    pivot_f1.add_prefix('F1@')
], axis=1)
agg.index.name = 'feature'

# Сохраняем сводную таблицу
agg.to_csv(OUTPUT_CSV)

print("Saved summary metrics to", OUTPUT_CSV)
print(agg.round(3))
