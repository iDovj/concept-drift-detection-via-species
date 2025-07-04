import os
import glob
import ast
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from pm4py.objects.log.importer.xes import importer as xes_importer
from preprocessing import segment_log_fixed_n
from metrics import compute_dfg, dfg_to_species_list, compute_hill
from special4pm.estimation.metrics import completeness, coverage

def evaluate_points(pred_points, gold_points, total_traces, latencies=[0.01, 0.025, 0.05]):
    results = {}
    for latency in latencies:
        max_dist = latency * total_traces
        unmatched = set(gold_points)
        tp = 0
        for p in sorted(pred_points):
            # match to nearest gold within allowed latency
            candidates = [g for g in unmatched if abs(g - p) <= max_dist]
            if candidates:
                gmin = min(candidates, key=lambda g: abs(g - p))
                tp += 1
                unmatched.remove(gmin)
        fp = len(pred_points) - tp
        fn = len(gold_points) - tp
        precision = tp/(tp+fp) if tp+fp>0 else 0.0
        recall    = tp/(tp+fn) if tp+fn>0 else 0.0
        f1        = 2*precision*recall/(precision+recall) if precision+recall>0 else 0.0
        results[latency] = (precision, recall, f1)
    return results

# Paths
LOG_FOLDER = "../data/test/test/"
SUDDEN_LOGS_PATH = "../data/sudden_logs.csv"
GOLD_STANDARD_PATH = "../data/gold_standard.csv"
RESULTS_FOLDER = "../results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load metadata
df_sudden = pd.read_csv(SUDDEN_LOGS_PATH)
df_gold   = pd.read_csv(GOLD_STANDARD_PATH)

summary = []

# Process a subset of 40 logs
log_ids = df_sudden['log_name'].str.extract(r'log_(\d+)_')[0].unique()[:30]
for log_number in log_ids:
    selected = df_sudden[df_sudden['log_name'].str.startswith(f"log_{log_number}_")]['log_name'].iloc[0]
    xes_file = glob.glob(os.path.join(LOG_FOLDER, f"log_{log_number}_*.xes"))
    if not xes_file:
        continue
    log = xes_importer.apply(xes_file[0])

    # gold change points (only sudden)
    row_gold = df_gold[df_gold['log_name']==selected]
    cps  = ast.literal_eval(row_gold.iloc[0]['change_point'])
    types= ast.literal_eval(row_gold.iloc[0]['drift_type'])
    gold_points = [cp for cp,t in zip(cps, types) if t.lower()=="sudden"]
    if not gold_points:
        continue

    total_traces = len(log)

    # segmentation
    N = 30
    windows = segment_log_fixed_n(log, n_windows=N)

    # compute trace bounds for midpoints
    bounds = []
    idx = 0
    for w in windows:
        n = len(w)
        bounds.append((idx+1, idx+n))
        idx += n

    # compute base metrics
    rows = []
    for i,w in enumerate(windows):
        dfg = compute_dfg(w)
        species = dfg_to_species_list(dfg)
        counts = dict(enumerate(species))
        size = sum(species)
        rows.append({
            "window": i+1,
            "n_traces": len(w),
            "hill_q0": compute_hill(species,q=0),
            "hill_q1": compute_hill(species,q=1),
            "hill_q2": compute_hill(species,q=2),
            "completeness": completeness(counts),
            "coverage": coverage(counts,size)
        })
    df = pd.DataFrame(rows)

    # derive features
    base = ["hill_q0","hill_q1","hill_q2","completeness","coverage"]
    for col in base:
        df[f"delta_{col}"]  = df[col].diff().abs().fillna(0)
        df[f"zscore_{col}"] = (df[col]-df[col].mean())/ (df[col].std()+1e-9)
        df[f"reldelta_{col}"] = df[col].diff()/(df[col].shift(1)+1e-9)

    variant_cols = [c for c in df.columns if c.startswith(("delta_","zscore_","reldelta_"))]

    # for each feature, predict drift points and evaluate
    for feature in variant_cols:
        vals = df[feature]
        nonz = vals[vals>0]
        if nonz.empty:
            thresh = float("inf")
        else:
            med = nonz.median()
            mad = np.median(np.abs(nonz-med))
            thresh = med+2*mad

        # predicted windows
        pw = df.index[vals>thresh].tolist()
        # convert to trace midpoints
        pp = [ (bounds[i][0]+bounds[i][1])//2 for i in pw ]

        # evaluate at latencies
        res = evaluate_points(pp, gold_points, total_traces)
        for lat,(p,r,f) in res.items():
            summary.append({
                "log_number": log_number,
                "feature": feature,
                "latency": lat,
                "precision": p,
                "recall": r,
                "f1": f
            })

# save results
pd.DataFrame(summary).to_csv(
    os.path.join(RESULTS_FOLDER,"evaluation_latency_30w.csv"), index=False
)
