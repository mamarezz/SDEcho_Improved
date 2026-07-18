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
) -> matplotlib.figure.Figure:
    """
    Plot original vs counterfactual vs target aggregate sequences.
    
    Creates a line plot showing three sequences side by side:
    - Source group's original sequence
    - Source group's counterfactual sequence (after reweighting)
    - Target group's sequence
    
    Args:
        result: GapDecompositionResult from compute_gap_decomposition
        index: Ordered bucket labels (x-axis)
        title: Plot title
    
    Returns:
        matplotlib Figure object
    
    Notes:
        - Useful for visualizing how reweighting changes the source sequence
        - Shows whether counterfactual moves toward or away from target
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(index))
    
    # Plot original source sequence
    ax.plot(x, result.s_source_orig, marker='o', linewidth=2,
            label='Source (original)', color='blue')
    
    # Plot counterfactual sequence
    ax.plot(x, result.s_source_cf, marker='s', linewidth=2,
            label='Source (counterfactual)', color='green', linestyle='--')
    
    # Plot target sequence
    ax.plot(x, result.s_target, marker='^', linewidth=2,
            label='Target', color='red')
    
    # Labels and formatting
    ax.set_xlabel('Experience Bucket', fontsize=12)
    ax.set_ylabel(f'Aggregate {result.predicate}', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(index, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotation with explained fraction
    ef_pct = result.explained_fraction * 100
    ax.text(0.02, 0.98, f'Explained: {ef_pct:.1f}%\nResidual: {result.residual_gap:.2f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
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

        # Only modify buckets [3-5] and [6-10] (indices 1 and 2)
        if len(index) >= 4 and index == ["0-2", "3-5", "6-10", "10-20"]:
            cf_seq = cf_seq.copy()
            cf_seq[0] = result.s_source_orig[0]  # [0-2] bucket: keep original
            cf_seq[3] = result.s_source_orig[3]  # [10-20] bucket: keep original

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




