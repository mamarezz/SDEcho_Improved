"""
Test additional group comparisons for SDEcho pipeline.

This file allows you to test different subgroup pairs without modifying run_pipeline.py.
It runs the full pipeline for each pair and saves results to separate files.

Usage:
    python test_additional_groups.py

Configuration:
    Edit the SUBGROUP_PAIRS list below to test different age groups or other subgroups.
"""

import sys
sys.path.insert(0, 'src')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================

# List of subgroup pairs to test (format: (subgroup_col, val1, val2, label))
# Choose meaningful comparisons based on your research question
SUBGROUP_PAIRS = [
    # Age group comparisons (current)
    ("AgeGroup", "25-34", "35-44", "25-34_vs_35-44"),

    # Additional age group comparisons
    ("AgeGroup", "25-34", "45-54", "25-34_vs_45-54"),
    ("AgeGroup", "35-44", "45-54", "35-44_vs_45-54"),

    # Gender comparisons (if applicable)
    # ("Gender", "Man", "Woman", "Men_vs_Women"),

    # Country comparisons (if applicable)
    # ("Country", "United States of America", "Germany", "USA_vs_Germany"),
]

# Pipeline configuration (same as run_pipeline.py)
CONFIG = {
    "data_path": "data/stackoverflow2022.csv",
    "subgroup_col": None,  # Will be set per pair
    "subgroup_val1": None,  # Will be set per pair
    "subgroup_val2": None,  # Will be set per pair
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

# ============================================================
# IMPORTS
# ============================================================

from src.data_loader import load_and_preprocess_data, split_groups, get_bucket_index
from src.sdecho import run_sdecho
from src.reweighting import select_predicate, compute_gap_decomposition, sequential_gap_decomposition
from src.evaluation import removal_baseline, bootstrap_explained_fraction_ci
from src.visualization import (
    plot_sequence_comparison,
    plot_gap_decomposition_bar,
    render_diagnostics_table,
    render_sequential_decomposition_table,
)

# ============================================================
# MAIN FUNCTION
# ============================================================

def run_comparison(subgroup_col, subgroup_val1, subgroup_val2, label):
    """
    Run the full pipeline for a single subgroup pair.
    """
    print("\n" + "=" * 80)
    print(f"RUNNING COMPARISON: {label.upper()}")
    print("=" * 80)

    # Update config for this pair
    CONFIG["subgroup_col"] = subgroup_col
    CONFIG["subgroup_val1"] = subgroup_val1
    CONFIG["subgroup_val2"] = subgroup_val2

    # ============================================================
    # STEP 1: Load and preprocess data
    # ============================================================
    print("\n" + "-" * 80)
    print("STEP 1: Loading and preprocessing data...")
    print("-" * 80)

    df = load_and_preprocess_data(CONFIG["data_path"], CONFIG)
    print(f"  Total rows after preprocessing: {len(df):,}")

    # ============================================================
    # STEP 2: Split into comparison groups
    # ============================================================
    print("\n" + "-" * 80)
    print(f"STEP 2: Splitting into comparison groups")
    print(f"  Group A: {subgroup_val1}")
    print(f"  Group B: {subgroup_val2}")
    print("-" * 80)

    df_A, df_B = split_groups(
        df,
        CONFIG["subgroup_col"],
        CONFIG["subgroup_val1"],
        CONFIG["subgroup_val2"]
    )
    print(f"  Group A ({subgroup_val1}): {len(df_A):,} rows")
    print(f"  Group B ({subgroup_val2}): {len(df_B):,} rows")

    if len(df_A) == 0 or len(df_B) == 0:
        print("\n  ❌ ERROR: One or both groups are empty!")
        print(f"  Available values for {subgroup_col}:")
        print(df[subgroup_col].value_counts().to_string())
        return None

    # ============================================================
    # STEP 3: Get bucket index
    # ============================================================
    index = get_bucket_index(df, CONFIG["group_col"])
    print(f"\n  Buckets found: {index}")

    # ============================================================
    # STEP 4: Build original sequences and compute distance
    # ============================================================
    print("\n" + "-" * 80)
    print("STEP 3-4: Computing original aggregate sequences...")
    print("-" * 80)

    from src.sequence_builder import build_sequence
    from src.sdecho import sequence_distance

    seq_A_orig = build_sequence(df_A, CONFIG["group_col"], CONFIG["measure_col"], "mean", index)
    seq_B = build_sequence(df_B, CONFIG["group_col"], CONFIG["measure_col"], "mean", index)
    d_orig = sequence_distance(seq_A_orig, seq_B)

    print(f"  Original aggregate sequences:")
    for i, bucket in enumerate(index):
        print(f"    {bucket:>6}: A={seq_A_orig[i]:>10,.0f}  B={seq_B[i]:>10,.0f}  diff={seq_A_orig[i]-seq_B[i]:>+10,.0f}")
    print(f"\n  Original sequence distance (d_orig): {d_orig:.2f}")

    # ============================================================
    # STEP 5: Run SDEcho
    # ============================================================
    print("\n" + "-" * 80)
    print("STEP 5: Running SDEcho predicate search...")
    print("-" * 80)

    sdecho_results = run_sdecho(
        df_A, df_B,
        group_col=CONFIG["group_col"],
        measure_col=CONFIG["measure_col"],
        agg_func=CONFIG["agg_func"],
        index=index,
        candidate_attrs=CONFIG["candidate_attrs"],
        max_order=CONFIG["max_order"],
        k=CONFIG["sdecho_k"],
        max_values_per_attr=CONFIG["max_values_per_attr"],
        min_support=CONFIG["sdecho_min_support"],
    )

    print(f"  Found {len(sdecho_results)} explanatory predicates")

    if not sdecho_results:
        print("\n  ❌ ERROR: No predicates found!")
        print("  Try reducing 'sdecho_min_support' or expanding 'candidate_attrs'.")
        return None

    # Show top 5
    print(f"\n  Top 5 predicates:")
    for i, r in enumerate(sdecho_results[:5], 1):
        reduction_pct = (1 - r.dist_after / r.dist_before) * 100
        print(f"  #{i}: {r.predicate}")
        print(f"      gamma={r.gamma:.4f}, reduction={reduction_pct:.1f}%")

    # ============================================================
    # STEP 6-10: Reweighting and gap decomposition
    # ============================================================
    print("\n" + "-" * 80)
    print("STEP 6-10: Counterfactual Reweighting and Gap Decomposition")
    print("-" * 80)

    predicate = select_predicate(sdecho_results, rank=CONFIG["predicate_rank"])
    print(f"\n  Selected predicate: {predicate}")

    # Core contribution: gap decomposition
    result = compute_gap_decomposition(
        df_A, df_B, predicate,
        group_col=CONFIG["group_col"],
        measure_col=CONFIG["measure_col"],
        index=index,
        min_cell_support=CONFIG["min_cell_support"],
    )

    # Show which buckets are being reweighted
    reweighted_labels = [index[i] for i in result.reweighted_buckets]
    print(f"\n  Gap Decomposition Results:")
    print(f"    Original distance: {result.d_orig:.2f}")
    print(f"    Counterfactual distance: {result.d_cf:.2f}")
    print(f"    Explained fraction: {result.explained_fraction:.2%}")
    print(f"    Residual gap: {result.residual_gap:.2f}")
    print(f"    Reweighted buckets: {', '.join(reweighted_labels)}")

    # ============================================================
    # Compare with removal baseline
    # ============================================================
    print("\n" + "-" * 80)
    print("Comparison with Removal Baseline")
    print("-" * 80)

    removal_reduction = removal_baseline(
        df_A, df_B, predicate,
        group_col=CONFIG["group_col"],
        measure_col=CONFIG["measure_col"],
        index=index
    )

    print(f"  Reweighting explained fraction: {result.explained_fraction:.2%}")
    print(f"  Removal-based reduction: {removal_reduction:.2%}")

    # ============================================================
    # Bootstrap confidence interval
    # ============================================================
    print("\n" + "-" * 80)
    print(f"Bootstrap Confidence Interval ({CONFIG['n_bootstrap']} resamples)")
    print("-" * 80)

    ci_lower, ci_upper = bootstrap_explained_fraction_ci(
        df_A, df_B, predicate,
        group_col=CONFIG["group_col"],
        measure_col=CONFIG["measure_col"],
        index=index,
        n_bootstrap=CONFIG["n_bootstrap"],
        ci=CONFIG["bootstrap_ci"],
    )

    print(f"  Explained fraction: {result.explained_fraction:.2%}")
    print(f"  95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]")

    # ============================================================
    # Diagnostics
    # ============================================================
    print("\n" + "-" * 80)
    print("Reweighting Diagnostics")
    print("-" * 80)

    diag_table = render_diagnostics_table(result.diagnostics)
    for _, row in diag_table.iterrows():
        print(f"  {row['Metric']:<35} {row['Value']}")

    # ============================================================
    # STEP 11: Sequential Gap Decomposition (Top-3 Predicates)
    # ============================================================
    print("\n" + "-" * 80)
    print("STEP 11: Sequential Gap Decomposition (Top-3 Predicates)")
    print("-" * 80)

    n_predicates = min(3, len(sdecho_results))
    top_predicates = [r.predicate for r in sdecho_results[:n_predicates]]

    print(f"\n  Using top {n_predicates} predicates:")
    for i, pred in enumerate(top_predicates, 1):
        print(f"    #{i}: {pred}")

    # Compute sequential gap decomposition
    seq_result = sequential_gap_decomposition(
        df_A, df_B, top_predicates,
        group_col=CONFIG["group_col"],
        measure_col=CONFIG["measure_col"],
        index=index,
        min_cell_support=CONFIG["min_cell_support"],
    )

    # Display the decomposition table
    seq_table = render_sequential_decomposition_table(seq_result)
    print(f"\n  Sequential Decomposition Table:")
    print("  " + "-" * 85)
    for _, row in seq_table.iterrows():
        print(f"  {row['Step']:>4}  {row['Counterfactual Intervention']:<50} "
              f"{row['Cumulative Explained']:>15}")
    print("  " + "-" * 85)

    # ============================================================
    # Generate visualizations
    # ============================================================
    print("\n" + "-" * 80)
    print("Generating visualizations...")
    print("-" * 80)

    # Plot 1: Sequence comparison
    fig1 = plot_sequence_comparison(
        result, index,
        title=f"Salary by Experience: {subgroup_val1} vs {subgroup_val2}",
        group_a_name=subgroup_val1,
        group_b_name=subgroup_val2,
        sequential_result=seq_result,
    )
    filename1 = f"sequence_comparison_{label}.png"
    fig1.savefig(filename1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  ✓ Saved: {filename1}")

    # Plot 2: Gap decomposition bar
    fig2 = plot_gap_decomposition_bar(result)
    filename2 = f"gap_decomposition_{label}.png"
    fig2.savefig(filename2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  ✓ Saved: {filename2}")

    # ============================================================
    # Save text summary
    # ============================================================
    filename3 = f"results_{label}.txt"
    with open(filename3, 'w') as f:
        f.write(f"SDEcho Pipeline Results - {label.upper()}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Comparison: {subgroup_val1} vs {subgroup_val2}\n")
        f.write(f"Sample sizes: Group A={len(df_A):,}, Group B={len(df_B):,}\n\n")
        f.write("Original Sequences:\n")
        for i, bucket in enumerate(index):
            f.write(f"  {bucket}: A={seq_A_orig[i]:,.0f}, B={seq_B[i]:,.0f}, diff={seq_A_orig[i]-seq_B[i]:,.0f}\n")
        f.write(f"\nOriginal distance: {d_orig:.2f}\n")
        f.write(f"Explained fraction: {result.explained_fraction:.2%}\n")
        f.write(f"Residual gap: {result.residual_gap:.2f}\n")
        f.write(f"\nTop predicate: {predicate}\n")
        f.write(f"Removal baseline reduction: {removal_reduction:.2%}\n")
        f.write(f"95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]\n\n")
        f.write("Sequential Decomposition:\n")
        for _, row in seq_table.iterrows():
            f.write(f"  Step {row['Step']}: {row['Cumulative Explained']} cumulative\n")

    print(f"  ✓ Saved: {filename3}")

    # Return summary
    return {
        "label": label,
        "subgroup_col": subgroup_col,
        "subgroup_val1": subgroup_val1,
        "subgroup_val2": subgroup_val2,
        "d_orig": d_orig,
        "explained_fraction": result.explained_fraction,
        "residual_gap": result.residual_gap,
        "top_predicate": str(predicate),
        "sample_size_A": len(df_A),
        "sample_size_B": len(df_B),
    }

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SDECHO PIPELINE - ADDITIONAL GROUP COMPARISONS")
    print("=" * 80)
    print(f"\nTesting {len(SUBGROUP_PAIRS)} subgroup pair(s)...\n")

    all_results = []

    for subgroup_col, val1, val2, label in SUBGROUP_PAIRS:
        try:
            result = run_comparison(subgroup_col, val1, val2, label)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR running comparison {label}: {e}")
            import traceback
            traceback.print_exc()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL COMPARISONS")
    print("=" * 80)

    if all_results:
        print("\nComparison Summary:")
        print("-" * 80)
        print(f"{'Label':<25} {'Group A':<15} {'Group B':<15} {'Explained':<12} {'Residual':<12} {'Sample A':<12} {'Sample B':<12}")
        print("-" * 80)

        for r in all_results:
            print(f"{r['label']:<25} {r['subgroup_val1']:<15} {r['subgroup_val2']:<15} "
                  f"{r['explained_fraction']:.2%}        {r['residual_gap']:<12.0f} "
                  f"{r['sample_size_A']:<12,} {r['sample_size_B']:<12,}")

        print("-" * 80)

        # Find best result
        best = max(all_results, key=lambda x: x['explained_fraction'])
        print(f"\n🎯 Best result: {best['label']} ({best['explained_fraction']:.2%} explained)")
        print(f"   Top predicate: {best['top_predicate']}")
        print(f"   Groups: {best['subgroup_val1']} vs {best['subgroup_val2']}")
        print(f"   Sample sizes: {best['sample_size_A']:,} vs {best['sample_size_B']:,}")

    print("\n" + "=" * 80)
    print("All comparisons completed!")
    print("Files saved with '_<label>' suffix (e.g., sequence_comparison_25-34_vs_35-44.png)")
    print("=" * 80)