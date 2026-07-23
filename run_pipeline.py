"""
Complete end-to-end pipeline runner.
Runs the full thesis pipeline on Stack Overflow 2022 data.
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
# CONFIGURATION
# ============================================================
CONFIG = {
    "data_path": "data/stackoverflow2022.csv",
    "subgroup_col": "AgeGroup",
    "subgroup_val1": "25-34",
    "subgroup_val2": "35-44",
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
    plot_sequential_sequence_comparison
)


def main():
    print("=" * 70)
    print("COUNTERFACTUAL REWEIGHTING FOR AGGREGATE SEQUENCE EXPLANATION")
    print("=" * 70)
    
    # ============================================================
    # STEP 1: Load and preprocess data
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 1: Loading and preprocessing Stack Overflow 2022 data...")
    print("=" * 70)
    
    df = load_and_preprocess_data(CONFIG["data_path"], CONFIG)
    print(f"  Total rows after preprocessing: {len(df):,}")
    
    # ============================================================
    # STEP 2: Split into comparison groups
    # ============================================================
    print("\n" + "=" * 70)
    print(f"STEP 2: Splitting into comparison groups")
    print(f"  Group A: {CONFIG['subgroup_val1']}")
    print(f"  Group B: {CONFIG['subgroup_val2']}")
    print("=" * 70)
    
    df_A, df_B = split_groups(
        df,
        CONFIG["subgroup_col"],
        CONFIG["subgroup_val1"],
        CONFIG["subgroup_val2"]
    )
    print(f"  Group A ({CONFIG['subgroup_val1']}): {len(df_A):,} rows")
    print(f"  Group B ({CONFIG['subgroup_val2']}): {len(df_B):,} rows")
    
    if len(df_A) == 0 or len(df_B) == 0:
        print("\n  ❌ One or both groups are empty after filtering!")
        print("  Please check subgroup_col and subgroup_val values.")
        print(f"\n  Available AgeGroup values:\n    {df[CONFIG['subgroup_col']].value_counts().to_string()}")
        return
    
    # ============================================================
    # STEP 3: Get bucket index
    # ============================================================
    index = get_bucket_index(df, CONFIG["group_col"])
    print(f"\n  Buckets found: {index}")
    
    # ============================================================
    # STEP 4: Build original sequences and compute distance
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 3-4: Computing original aggregate sequences and distance")
    print("=" * 70)
    
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
    print("\n" + "=" * 70)
    print("STEP 5: Running SDEcho predicate search...")
    print("=" * 70)
    
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
    print(f"\n  Top predicates (lower gamma = stronger explanation):")
    for i, r in enumerate(sdecho_results[:5], 1):
        reduction_pct = (1 - r.dist_after / r.dist_before) * 100
        print(f"  #{i}: {r.predicate}")
        print(f"      gamma={r.gamma:.4f}, distance {r.dist_before:.2f} -> {r.dist_after:.2f} ({reduction_pct:.1f}% reduction)")
        print(f"      matching: A={r.n1:,}, B={r.n2:,}")
    
    if not sdecho_results:
        print("\n  ❌ No predicates found meeting the minimum support threshold.")
        print("  Try reducing 'sdecho_min_support' or expanding 'candidate_attrs'.")
        return
    
    # ============================================================
    # STEP 6-10: Reweighting and gap decomposition
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 6-10: Counterfactual Reweighting and Gap Decomposition")
    print("=" * 70)
    
    predicate = select_predicate(sdecho_results, rank=CONFIG["predicate_rank"])
    print(f"\n  Selected predicate: {predicate}")
    print(f"  Attributes used for reweighting: {predicate.attrs}")
    
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
    print(f"\n  ┌──────────────────────────────────────────────────────┐")
    print(f"  │              GAP DECOMPOSITION RESULTS               │")
    print(f"  ├──────────────────────────────────────────────────────┤")
    print(f"  │  Original distance (d_orig):      {result.d_orig:>9.2f}          │")
    print(f"  │  Counterfactual distance (d_cf):   {result.d_cf:>9.2f}          │")
    print(f"  │  Explained fraction:              {result.explained_fraction:>9.2%}          │")
    print(f"  │  Residual gap:                    {result.residual_gap:>9.2f}          │")
    print(f"  │  Reweighted buckets:              {', '.join(reweighted_labels):<22}   │")
    print(f"  └──────────────────────────────────────────────────────┘")
    
    # Counterfactual sequence
    print(f"\n  Counterfactual sequence (reweighted A):")
    for i, bucket in enumerate(index):
        reweighted = " (reweighted)" if i in result.reweighted_buckets else " (original)"
        diff_orig = seq_A_orig[i] - seq_B[i]
        diff_cf = result.s_source_cf[i] - seq_B[i]
        print(f"    {bucket:>6}: orig A={seq_A_orig[i]:>10,.0f}  cf A={result.s_source_cf[i]:>10,.0f}  B={seq_B[i]:>10,.0f}  "
              f"orig diff={diff_orig:>+10,.0f}  cf diff={diff_cf:>+10,.0f}{reweighted}")
    
    # ============================================================
    # Compare with removal baseline
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Reweighting vs SDEcho Removal Baseline")
    print("=" * 70)
    
    removal_reduction = removal_baseline(
        df_A, df_B, predicate,
        group_col=CONFIG["group_col"],
        measure_col=CONFIG["measure_col"],
        index=index
    )
    
    print(f"\n  Reweighting (this thesis):")
    print(f"    Explained fraction: {result.explained_fraction:.2%}")
    print(f"    (proportion of gap attributed to compositional differences)")
    print(f"\n  Removal (SDEcho baseline):")
    print(f"    Reduction: {removal_reduction:.2%}")
    print(f"    (proportion of gap reduced by deleting matching tuples)")
    
    # ============================================================
    # Bootstrap confidence interval
    # ============================================================
    print("\n" + "=" * 70)
    print(f"BOOTSTRAP CONFIDENCE INTERVAL ({CONFIG['n_bootstrap']} resamples)")
    print("=" * 70)
    
    ci_lower, ci_upper = bootstrap_explained_fraction_ci(
        df_A, df_B, predicate,
        group_col=CONFIG["group_col"],
        measure_col=CONFIG["measure_col"],
        index=index,
        n_bootstrap=CONFIG["n_bootstrap"],
        ci=CONFIG["bootstrap_ci"],
    )
    
    print(f"\n  Explained fraction: {result.explained_fraction:.2%}")
    print(f"  {CONFIG['bootstrap_ci']:.0%} Confidence Interval: [{ci_lower:.2%}, {ci_upper:.2%}]")
    
    # ============================================================
    # Diagnostics
    # ============================================================
    print("\n" + "=" * 70)
    print("REWEIGHTING DIAGNOSTICS")
    print("=" * 70)
    
    diag_table = render_diagnostics_table(result.diagnostics)
    for _, row in diag_table.iterrows():
        print(f"  {row['Metric']:<35} {row['Value']}")
    
    # ============================================================
    # STEP 11: Sequential Gap Decomposition (Top-5 Predicates)
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 11: SEQUENTIAL GAP DECOMPOSITION")
    print("=" * 70)
    
    # Take top 5 predicates for sequential decomposition
    n_predicates = min(5, len(sdecho_results))
    top_predicates = [r.predicate for r in sdecho_results[:n_predicates]]
    
    print(f"\n  Using top {n_predicates} predicates for sequential decomposition:")
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
    print(f"\n  SEQUENTIAL GAP DECOMPOSITION TABLE:")
    print("  " + "-" * 85)
    for _, row in seq_table.iterrows():
        print(f"  {row['Step']:>4}  {row['Counterfactual Intervention']:<45} "
              f"{row['% of Gap Explained']:>12}  {row['Cumulative Explained']:>15}  {row['Remaining Gap (%)']:>15}")
    print("  " + "-" * 85)
    
    # ============================================================
    # Generate visualizations
    # ============================================================
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)
    
    # Plot 1: Sequence comparison (use sequential decomposition results)
    fig1 = plot_sequence_comparison(
        result, index,
        title=f"Salary by Experience: {CONFIG['subgroup_val1']} vs {CONFIG['subgroup_val2']}",
        group_a_name=CONFIG['subgroup_val1'],
        group_b_name=CONFIG['subgroup_val2'],
        sequential_result=seq_result,
    )
    fig1.savefig("sequence_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("  ✓ Saved: sequence_comparison.png")

    # Plot 2: Gap decomposition bar
    fig2 = plot_gap_decomposition_bar(result)
    fig2.savefig("gap_decomposition.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("  ✓ Saved: gap_decomposition.png")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE - RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"""
    Dataset:        Stack Overflow Developer Survey 2022
    Comparison:     Age {CONFIG['subgroup_val1']} vs {CONFIG['subgroup_val2']}
    Outcome:        {CONFIG['measure_col']}
    Sequence:       Mean salary by {CONFIG['group_col']}
    
    ┌─────────────────────────────────────────────────────┐
    │  Original gap (d_orig):            {d_orig:>9.2f}          │
    │  Counterfactual gap (d_cf):         {result.d_cf:>9.2f}          │
    │  Explained fraction:               {result.explained_fraction:>9.2%}          │
    │  Residual gap:                     {result.residual_gap:>9.2f}          │
    │  Bootstrap CI ({CONFIG['bootstrap_ci']:.0%}):            [{ci_lower:.2%}, {ci_upper:.2%}]      │
    ├─────────────────────────────────────────────────────┤
    │  Best SDEcho predicate:            {str(predicate):<30}  │
    │  SDEcho removal reduction:          {removal_reduction:>9.2%}          │
    │  Reweighting explained fraction:    {result.explained_fraction:>9.2%}          │
    └─────────────────────────────────────────────────────┘
    
    Interpretation:
      The explained fraction ({result.explained_fraction:.1%}) represents the proportion of
      the salary gap between age groups {CONFIG['subgroup_val1']} and {CONFIG['subgroup_val2']}
      that can be attributed to differences in the distribution of
      {predicate.attrs} — after exact cell-based reweighting.
      
      This is a statistical decomposition, NOT a causal claim.
      See docs/00_PROJECT_CONCEPT.md for disclaimers.
    """)


if __name__ == "__main__":
    main()