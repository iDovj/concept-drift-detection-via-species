import pandas as pd

# Путь к CSV, полученному из основного скрипта
INPUT_CSV = "../results/summary_fscore_per_metric_20w_v1.csv"
# Куда сохранить общий свод
OUTPUT_CSV = "../results/overall_system_metrics_20w.csv"

# Считаем данные
df = pd.read_csv(INPUT_CSV)

# 1) Средние по всем метрикам вместе
overall = df[['precision', 'recall', 'f1']].mean().to_frame().T
overall.insert(0, 'system', 'all_features')
overall['count'] = len(df)

# 2) По каждому признаку отдельно
by_feature = (
    df
    .groupby('metric')[['precision','recall','f1']]
    .agg(['mean','std','count'])
)
# упростим названия
by_feature.columns = ['_'.join(col).strip() for col in by_feature.columns.values]
by_feature.reset_index(inplace=True)

# Соберём итоговый DataFrame
summary = pd.concat([overall, by_feature], axis=0, sort=False)
summary.to_csv(OUTPUT_CSV, index=False)

print("=== Overall system metrics (all features) ===")
print(overall.round(3))
print("\n=== Metrics by feature ===")
print(by_feature.round(3))
print(f"\nSaved overall summary to {OUTPUT_CSV}")
