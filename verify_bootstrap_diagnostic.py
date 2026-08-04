"""
Diagnostic: verify whether bootstrap CI=[0,0] is a real result or a
silent-failure artifact. Also verifies the sign-flip claim on comparison 3.
"""

import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd

from src.data_loader import load_and_preprocess_data, split_groups, get_bucket_index
from src.sdecho import run_sdecho
from src.reweighting import select_predicate, compute_gap_decomposition
from src.evaluation import removal_baseline
from src.sequence_builder import build_sequence
from src.sdecho import sequence_distance

CONFIG = {
    "data_path": "data/stackoverflow2022.csv",
    "subgroup_col": "AgeGroup",
    "subgroup_val1": "35-44",
    "subgroup_val2": "45-54",
    "group_col": "YearsExpBucket",
    "measure_col": "ConvertedCompYearly",
    "agg_func": "mean",
    "candidate_attrs": ["EdLevel", "RemoteWork", "Country"],
    "max_order": 2,
    "sdecho_k": 10,
    "max_values_per_attr": 10,
    "sdecho_min_support": 20,
    "predicate_rank": 0,
    "min_cell_support": 5,
    "n_bootstrap": 100,
    "bootstrap_ci": 0.95,
}

df = load_and_preprocess_data(CONFIG["data_path"], CONFIG)
df_A, df_B = split_groups(df, CONFIG["subgroup_col"], CONFIG["subgroup_val1"], CONFIG["subgroup_val2"])
index = get_bucket_index(df, CONFIG["group_col"])

print(f"Group A (35-44): {len(df_A):,}, Group B (45-54): {len(df_B):,}")
print(f"Buckets: {index}")

# --- Step 1: verify the sign-flip (removal vs reweighting) ---
sdecho_results = run_sdecho(
    df_A, df_B,
    group_col=CONFIG["group_col"], measure_col=CONFIG["measure_col"],
    agg_func=CONFIG["agg_func"], index=index,
    candidate_attrs=CONFIG["candidate_attrs"], max_order=CONFIG["max_order"],
    k=CONFIG["sdecho_k"], max_values_per_attr=CONFIG["max_values_per_attr"],
    min_support=CONFIG["sdecho_min_support"],
)

top = sdecho_results[0]
pred = top.predicate
print("\n=== Comparison 3 (35-44 vs 45-54) ===")
print(f"Top predicate: {pred}")
print(f"SDEcho removal reduction: {(1 - top.dist_after / top.dist_before) * 100:.2f}%")

removal_reduction = removal_baseline(df_A, df_B, pred, CONFIG["group_col"], CONFIG["measure_col"], index)
print(f"removal_baseline() reduction: {removal_reduction*100:.2f}%")

result = compute_gap_decomposition(
    df_A, df_B, pred,
    group_col=CONFIG["group_col"], measure_col=CONFIG["measure_col"],
    index=index, min_cell_support=CONFIG["min_cell_support"],
)
print(f"Reweighting explained fraction: {result.explained_fraction*100:.2f}%")

# --- Step 2: instrument the bootstrap to see how many resamples succeed ---
from src.reweighting import compute_gap_decomposition as cgd

n_success = 0
n_fail = 0
examples = []
first_errors = []
for i in range(CONFIG["n_bootstrap"]):
    df_A_boot = df_A.sample(n=len(df_A), replace=True, random_state=None)
    df_B_boot = df_B.sample(n=len(df_B), replace=True, random_state=None)
    try:
        r = cgd(df_A_boot, df_B_boot, pred,
                group_col=CONFIG["group_col"], measure_col=CONFIG["measure_col"],
                index=index, min_cell_support=CONFIG["min_cell_support"])
        n_success += 1
        examples.append(r.explained_fraction)
    except Exception as e:
        n_fail += 1
        if len(first_errors) < 5:
            first_errors.append(f"{type(e).__name__}: {e}")

print(f"\n=== Bootstrap diagnostic (n={CONFIG['n_bootstrap']}) ===")
print(f"Successes: {n_success}, Failures: {n_fail}")
if first_errors:
    print("First errors:")
    for e in first_errors:
        print(f"  {e}")

if examples:
    arr = np.array(examples)
    print(f"\nExplained fractions: min={arr.min()*100:.2f}%, max={arr.max()*100:.2f}%, "
          f"mean={arr.mean()*100:.2f}%, std={arr.std()*100:.2f}%")
    print(f"All zero? {np.all(arr == 0)}")
    alpha = 1 - CONFIG["bootstrap_ci"]
    lo = np.percentile(arr, 100 * alpha / 2) * 100
    hi = np.percentile(arr, 100 * (1 - alpha / 2)) * 100
    print(f"Percentile CI from instrumented run: [{lo:.2f}%, {hi:.2f}%]")
    # fraction of resamples where d_orig==0 (degenerate resample)
    n_degenerate = 0
    for i in range(min(len(examples), 50)):
        df_A_boot = df_A.sample(n=len(df_A), replace=True, random_state=None)
        df_B_boot = df_B.sample(n=len(df_B), replace=True, random_state=None)
        try:
            sA = build_sequence(df_A_boot, CONFIG["group_col"], CONFIG["measure_col"], "mean", index)
            sB = build_sequence(df_B_boot, CONFIG["group_col"], CONFIG["measure_col"], "mean", index)
            if sequence_distance(sA, sB) == 0:
                n_degenerate += 1
        except Exception:
            pass
    print(f"(spot-check) degenerate d_orig==0 resamples: {n_degenerate}/50")