# src/visualization.py

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for scripted use

from src.reweighting import GapDecompositionResult, ReweightingDiagnostics, SequentialDecompositionResult


def plot_sequence_comparison(
    result: GapDecompositionResult, index: list[str], title: str,
    group_a_name: str = "Group A", group_b_name: str = "Group B",
    sequential_result: SequentialDecompositionResult = None,
) -> matplotlib.figure.Figure:
    """
    Plot original vs counterfactual vs target aggregate sequences.

    Creates a line plot showing three sequences side by side:
    - Group A's original sequence
    - Group A's counterfactual sequence (after reweighting)
    - Group B's sequence

    Args:
        result: GapDecompositionResult from compute_gap_decomposition
        index: Ordered bucket labels (x-axis)
        title: Plot title
        group_a_name: Name for Group A (default: "Group A")
        group_b_name: Name for Group B (default: "Group B")
        sequential_result: SequentialDecompositionResult for showing cumulative fractions

    Returns:
        matplotlib Figure object

    Notes:
        - Useful for visualizing how reweighting changes the source sequence
        - Shows whether counterfactual moves toward or away from target
    """
    # Create figure with space for predicates at bottom
    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(index))

    # Plot Group A original sequence
    ax.plot(x, result.s_source_orig, marker='o', linewidth=2,
            label=f'{group_a_name} (original)', color='blue')

    # Plot counterfactual sequence
    ax.plot(x, result.s_source_cf, marker='s', linewidth=2,
            label=f'{group_a_name} (counterfactual)', color='green', linestyle='--')

    # Plot Group B sequence
    ax.plot(x, result.s_target, marker='^', linewidth=2,
            label=f'{group_b_name}', color='red')

    # Labels and formatting
    ax.set_xlabel('Experience Bucket', fontsize=12)
    ax.set_ylabel('Mean Salary', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(index, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotation with explained fraction and reweighted buckets info
    ef_pct = result.explained_fraction * 100
    reweight_buckets_str = ""
    if result.reweighted_buckets is not None:
        bucket_labels = [index[i] for i in result.reweighted_buckets]
        reweight_buckets_str = f"\nReweighted: {', '.join(bucket_labels)}"
    annotation_text = f'Explained: {ef_pct:.1f}%\nResidual: {result.residual_gap:.2f}{reweight_buckets_str}'
    ax.text(0.02, 0.98, annotation_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add sequential decomposition summary below plot
    if sequential_result and sequential_result.steps:
        # Build sequential explanation text from the actual sequential decomposition
        pred_lines = []
        for step in sequential_result.steps[1:]:  # Skip step 0 (original gap)
            step_num = step["step"]
            pred = step["predicate"]
            cumulative_frac = step["cumulative_explained"] * 100
            if pred is None:
                pred_lines.append(f"#{step_num} Original gap: {cumulative_frac:.1f}%")
            else:
                pred_lines.append(f"#{step_num} {pred}: {cumulative_frac:.1f}%")

        pred_text = "\n".join(pred_lines)
        fig.text(0.5, -0.15, f"Sequential Gap Decomposition:\n{pred_text}",
                fontsize=9, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    return fig


def plot_gap_decomposition_bar(
    result: GapDecompositionResult,
) -> matplotlib.figure.Figure:
    """
    Bar chart showing gap decomposition into explained and residual components.
    
    Args:
        result: GapDecompositionResult from compute_gap_decomposition
    
    Returns:
        matplotlib Figure object
    
    Notes:
        - Visualizes d_orig, explained portion, and residual gap
        - Useful for thesis figures showing decomposition results
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Original\nGap', 'Explained\nFraction', 'Residual\nGap']
    values = [
        result.d_orig,
        result.d_orig * result.explained_fraction,
        result.residual_gap,
    ]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add percentage annotation
    ef_pct = result.explained_fraction * 100
    ax.text(0.5, 0.95, f'Explained: {ef_pct:.1f}%',
            transform=ax.transAxes, fontsize=12,
            ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Gap Decomposition', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_weight_distribution(
    weights: pd.Series, diagnostics: ReweightingDiagnostics,
) -> matplotlib.figure.Figure:
    """
    Histogram of cell weights, useful for spotting extreme-weight cells.
    
    Args:
        weights: Series of reweighting factors (NaN = trimmed)
        diagnostics: ReweightingDiagnostics for context
    
    Returns:
        matplotlib Figure object
    
    Notes:
        - Excludes NaN weights (trimmed cells)
        - Log scale on y-axis to show long-tailed distribution
        - Highlights max/min weights
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out NaN weights
    valid_weights = weights.dropna()
    
    if len(valid_weights) == 0:
        ax.text(0.5, 0.5, 'No valid weights to display',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12)
        ax.set_title('Weight Distribution (No Valid Weights)', fontsize=14)
        return fig
    
    # Histogram
    ax.hist(valid_weights, bins=min(50, len(valid_weights)),
            edgecolor='black', alpha=0.7, color='steelblue')
    
    # Mark min/max
    ax.axvline(diagnostics.min_weight, color='green', linestyle='--',
               linewidth=2, label=f'Min: {diagnostics.min_weight:.3f}')
    ax.axvline(diagnostics.max_weight, color='red', linestyle='--',
               linewidth=2, label=f'Max: {diagnostics.max_weight:.3f}')
    
    ax.set_xlabel('Weight', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Reweighting Weight Distribution\n'
                 f'({diagnostics.n_dropped_rows} rows dropped, '
                 f'{diagnostics.pct_dropped_rows:.1f}% of total)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def render_diagnostics_table(
    diagnostics: ReweightingDiagnostics,
) -> pd.DataFrame:
    """
    Create thesis-ready diagnostics table as a DataFrame.
    
    Args:
        diagnostics: ReweightingDiagnostics from compute_cell_weights
    
    Returns:
        DataFrame formatted for display/thesis inclusion
    
    Notes:
        - Human-readable labels and formatting
        - Suitable for direct inclusion in thesis tables
    """
    data = {
        'Metric': [
            'Attributes Used',
            'Source Rows (Total)',
            'Dropped Rows (Invalid Cells)',
            'Dropped Rows (%)',
            'Cells in Source (Total)',
            'Cells No Target Overlap',
            'Cells Below Min Support',
            'Valid Cells (Used for Reweighting)',
            'Min Cell Support Threshold',
            'Max Weight',
            'Min Weight',
        ],
        'Value': [
            ', '.join(diagnostics.attrs),
            diagnostics.n_source_rows,
            diagnostics.n_dropped_rows,
            f'{diagnostics.pct_dropped_rows:.2f}%',
            diagnostics.n_cells_source_total,
            diagnostics.n_cells_no_target_overlap,
            diagnostics.n_cells_below_min_support,
            diagnostics.n_cells_valid,
            diagnostics.min_cell_support,
            f'{diagnostics.max_weight:.4f}',
            f'{diagnostics.min_weight:.4f}',
        ]
    }
    
    df_table = pd.DataFrame(data)
    return df_table


def render_sequential_decomposition_table(
    result: SequentialDecompositionResult,
) -> pd.DataFrame:
    """
    Create thesis-ready sequential gap decomposition table.

    Produces the step-by-step decomposition table showing how each
    predicate intervention incrementally explains the gap.

    Args:
        result: SequentialDecompositionResult from sequential_gap_decomposition

    Returns:
        DataFrame with columns:
            Step, Counterfactual Intervention, % of Gap Explained,
            Cumulative Explained, Remaining Gap (%)

    Notes:
        - Matches the requested table format for thesis inclusion
        - Step 0 shows the original gap (no intervention)
    """
    rows = []
    for step in result.steps:
        if step["step"] == 0:
            intervention = "Original gap"
        else:
            pred = step["predicate"]
            # Build a human-readable intervention description
            attr_values = []
            for attr, val in pred.conditions.items():
                if attr == val:
                    attr_values.append(f"Change {attr}")
                else:
                    attr_values.append(f"Change {attr}={val}")
            intervention = " → ".join(attr_values)

        rows.append({
            "Step": step["step"],
            "Counterfactual Intervention": intervention,
            "% of Gap Explained": f"{step['explained_fraction'] * 100:.1f}%",
            "Cumulative Explained": f"{step['cumulative_explained'] * 100:.1f}%",
            "Remaining Gap (%)": f"{step['remaining_gap'] * 100:.1f}%",
        })

    df_table = pd.DataFrame(rows)
    return df_table


def plot_sequential_sequence_comparison(
    result: SequentialDecompositionResult,
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    group_col: str,
    measure_col: str,
    index: list[str],
    title: str = "Sequential Counterfactual Reweighting",
) -> matplotlib.figure.Figure:
    """
    Plot original vs all sequential counterfactual vs target aggregate sequences.

    Creates a line plot showing the progression of counterfactual sequences
    as each predicate is applied sequentially.

    Args:
        result: SequentialDecompositionResult from sequential_gap_decomposition
        df_source: Original source DataFrame
        df_target: Target DataFrame
        group_col: Bucket column name
        measure_col: Outcome column name
        index: Ordered bucket labels (x-axis)
        title: Plot title

    Returns:
        matplotlib Figure object

    Notes:
        - Shows the original source, all intermediate counterfactuals, and target
        - Each step uses a progressively lighter shade of green
    """
    from src.reweighting import weighted_aggregate_sequence

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(index))

    # Plot target sequence (always the same)
    ax.plot(x, result.s_target, marker='^', linewidth=2.5,
            label='Target', color='red')

    # Plot original source sequence
    ax.plot(x, result.s_source_orig, marker='o', linewidth=2.5,
            label='Source (original)', color='blue')

    # Plot each sequential counterfactual with progressively lighter green
    greens = ['#004d00', '#1a8a1a', '#33cc33', '#66e64d', '#99ff66', '#ccff99']
    for i, step in enumerate(result.steps[1:], 1):  # skip step 0
        color = greens[min(i - 1, len(greens) - 1)]
        label = f'CF Step {step["step"]}: {step["cumulative_explained"] * 100:.0f}% explained'

        # Build counterfactual sequence for this step
        cf_seq = weighted_aggregate_sequence(
            df_source, step["weights"], group_col, measure_col, index
        )

        # Only modify buckets in the reweight set
        # Buckets NOT in the set keep original values
        reweighted_buckets = step.get("reweighted_buckets", result.reweighted_buckets)
        if reweighted_buckets is None:
            reweighted_buckets = list(range(len(index)))  # fallback: all buckets
        cf_seq = cf_seq.copy()
        for j in range(len(index)):
            if j not in reweighted_buckets:
                cf_seq[j] = result.s_source_orig[j]

        ax.plot(x, cf_seq, marker='s', linewidth=1.5,
                label=label, color=color, linestyle='--', alpha=0.7 + 0.3 * (1 - i/len(result.steps)))

    # Labels and formatting
    ax.set_xlabel('Experience Bucket', fontsize=12)
    ax.set_ylabel('Aggregate Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(index, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add annotation box with final decomposition
    final_step = result.steps[-1]
    ef_pct = final_step["cumulative_explained"] * 100
    ax.text(0.02, 0.98,
            f'Cumulative explained: {ef_pct:.1f}%\n'
            f'Remaining gap: {final_step["remaining_gap"] * 100:.1f}%',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig




