# src/reweighting.py

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd

from src.predicates import Predicate
from src.sequence_builder import build_sequence


@dataclass(frozen=True)
class ReweightingDiagnostics:
    """
    Diagnostic summary of a cell-based reweighting run.
    
    Captures information about common support violations and weight
    distribution for reporting and quality assessment.
    """
    attrs: list
    n_source_rows: int
    n_dropped_rows: int
    pct_dropped_rows: float
    n_cells_source_total: int
    n_cells_no_target_overlap: int
    n_cells_below_min_support: int
    n_cells_valid: int
    min_cell_support: int
    max_weight: float
    min_weight: float


@dataclass(frozen=True)
class GapDecompositionResult:
    """
    Final Stage 7-10 output: everything needed for reporting/plotting.
    
    Attributes:
        predicate: The SDEcho predicate used for reweighting
        attrs: Attribute names used for reweighting
        s_source_orig: Original aggregate sequence for source group
        s_source_cf: Counterfactual aggregate sequence (after reweighting)
        s_target: Target group's aggregate sequence
        d_orig: Original sequence distance
        d_cf: Counterfactual sequence distance (after reweighting)
        explained_fraction: Proportion of gap explained (d_orig - d_cf) / d_orig
        residual_gap: Remaining distance after reweighting
        diagnostics: Reweighting diagnostics
    """
    predicate: Predicate
    attrs: list
    s_source_orig: np.ndarray
    s_source_cf: np.ndarray
    s_target: np.ndarray
    d_orig: float
    d_cf: float
    explained_fraction: float
    residual_gap: float
    diagnostics: ReweightingDiagnostics


def select_predicate(results: list[SDEchoResult], rank: int = 0) -> Predicate:
    """
    Select the rank-th predicate from SDEcho's ranked output.
    
    Args:
        results: SDEcho's ranked results (sorted by ascending gamma)
        rank: Which predicate to select (0 = top-1, 1 = top-2, etc.)
    
    Returns:
        The selected Predicate
    
    Raises:
        IndexError: If rank >= len(results)
    """
    if rank >= len(results):
        raise IndexError(
            f"Rank {rank} requested but only {len(results)} predicates available"
        )
    return results[rank].predicate


def compute_cell_weights(
    df_source: pd.DataFrame, df_target: pd.DataFrame,
    attrs: list[str], min_cell_support: int = 5,
) -> Tuple[pd.Series, ReweightingDiagnostics]:
    """
    Compute exact cell-based (joint-distribution) reweighting factors.
    
    For each tuple in df_source, computes a weight w(t) such that the weighted
    distribution of df_source over `attrs` matches the empirical distribution
    of df_target over the same attributes.
    
    Args:
        df_source: DataFrame to reweight (source group)
        df_target: DataFrame whose distribution to match (target group)
        attrs: List of attribute names defining the joint cells
        min_cell_support: Minimum count in both groups for a cell to be valid
    
    Returns:
        Tuple of (weights, diagnostics) where:
            - weights: Series indexed like df_source with weight for each tuple
              (NaN for trimmed/invalid cells)
            - diagnostics: ReweightingDiagnostics with quality metrics
    
    Notes:
        - Cells with count < min_cell_support in either group are trimmed
        - Cells not present in both groups are trimmed
        - Tuples in trimmed cells receive NaN weight (excluded from analysis)
        - Direction: source -> target (source reweighted to match target)
    
    Complexity:
        O(n_source + n_target) — linear in dataset size
    
    Example:
        >>> weights, diag = compute_cell_weights(
        ...     df_A, df_B, ["Country"], min_cell_support=5
        ... )
        >>> # weights is NaN for rare cells, valid values are p_B(x) / p_A(x)
    """
    n_source = len(df_source)
    
    # Step 1: Compute empirical cell proportions in both groups
    # source proportions
    source_counts = df_source.groupby(attrs).size()
    source_props = source_counts / n_source
    
    n_target = len(df_target)
    target_counts = df_target.groupby(attrs).size()
    target_props = target_counts / n_target
    
    # Step 2: Identify valid cells (common support + min support)
    source_cells = set(source_counts.index)
    target_cells = set(target_counts.index)
    
    common_cells = source_cells & target_cells
    
    # Cells to drop: not in common, or below min support in either group
    cells_no_overlap = source_cells - target_cells
    cells_below_support = {
        cell for cell in common_cells
        if source_counts[cell] < min_cell_support or target_counts[cell] < min_cell_support
    }
    
    valid_cells = common_cells - cells_below_support
    
    # Step 3: Compute weights
    # For each tuple in source, weight = p_target(x(t)) / p_source(x(t))
    # If cell is invalid, weight = NaN
    
    def get_cell_key(row):
        """Extract cell key (tuple) from row."""
        if len(attrs) == 1:
            return row[attrs[0]]
        return tuple(row[attr] for attr in attrs)
    
    weights = pd.Series(np.nan, index=df_source.index)
    
    for idx in df_source.index:
        cell = get_cell_key(df_source.loc[idx])
        
        if cell in valid_cells:
            p_source = source_props[cell]
            p_target = target_props[cell]
            weights[idx] = p_target / p_source
        else:
            weights[idx] = np.nan  # trimmed
    
    # Step 4: Compute diagnostics
    n_cells_source_total = len(source_cells)
    n_dropped = int(weights.isna().sum())
    pct_dropped = (n_dropped / n_source * 100) if n_source > 0 else 0.0
    
    valid_weights = weights.dropna()
    max_w = float(valid_weights.max()) if len(valid_weights) > 0 else 0.0
    min_w = float(valid_weights.min()) if len(valid_weights) > 0 else 0.0
    
    diagnostics = ReweightingDiagnostics(
        attrs=attrs,
        n_source_rows=n_source,
        n_dropped_rows=n_dropped,
        pct_dropped_rows=pct_dropped,
        n_cells_source_total=n_cells_source_total,
        n_cells_no_target_overlap=len(cells_no_overlap),
        n_cells_below_min_support=len(cells_below_support),
        n_cells_valid=len(valid_cells),
        min_cell_support=min_cell_support,
        max_weight=max_w,
        min_weight=min_w,
    )
    
    return weights, diagnostics


def weighted_aggregate_sequence(
    df: pd.DataFrame, weights: pd.Series,
    group_col: str, measure_col: str, index: list[str],
) -> np.ndarray:
    """
    Construct a weighted aggregate sequence.
    
    Computes the weighted mean of `measure_col` within each bucket, using
    the provided weights. Tuples with NaN weights are excluded.
    
    Args:
        df: DataFrame containing raw tuples
        weights: Series of weights (same index as df); NaN = excluded
        group_col: Bucket column name
        measure_col: Outcome column name
        index: Ordered bucket labels
    
    Returns:
        1D numpy array of weighted aggregate values, ordered by `index`.
        Empty buckets filled with 0.
    
    Notes:
        - Uses only tuples with non-NaN weights
        - If a bucket has no valid tuples, returns 0
        - Assumes agg_func = "mean" (weighted mean)
    
    Complexity:
        O(n) where n = len(df)
    """
    # Filter to rows with valid weights
    valid_mask = weights.notna()
    df_valid = df.loc[valid_mask].copy()
    w_valid = weights.loc[valid_mask]
    
    if len(df_valid) == 0:
        # No valid data, return zeros
        return np.zeros(len(index), dtype=float)
    
    # Attach weights to df
    df_valid = df_valid.assign(_weight=w_valid)
    
    # Compute weighted mean per bucket
    weighted_sums = df_valid.groupby(group_col).apply(
        lambda g: np.average(g[measure_col], weights=g['_weight'])
    )
    
    # Reindex and fill missing with 0
    sequence = weighted_sums.reindex(index).fillna(0)
    
    return sequence.to_numpy(dtype=float)


def compute_gap_decomposition(
    df_source: pd.DataFrame, df_target: pd.DataFrame,
    predicate: Predicate, group_col: str, measure_col: str,
    index: list[str], min_cell_support: int = 5,
) -> GapDecompositionResult:
    """
    Full Stage 7-10 orchestration: reweighting and gap decomposition.
    
    This is the thesis's core contribution: given a predicate from SDEcho,
    reweight df_source to match df_target's distribution over the predicate's
    attributes, then compute the explained fraction.
    
    Args:
        df_source: DataFrame for source group (to be reweighted)
        df_target: DataFrame for target group (distribution to match)
        predicate: SDEcho predicate (attributes used for reweighting)
        group_col: Bucket column name
        measure_col: Outcome column name
        index: Ordered bucket labels
        min_cell_support: Minimum cell count for valid reweighting
    
    Returns:
        GapDecompositionResult with original/counterfactual sequences,
        distances, explained fraction, and diagnostics
    
    Notes:
        - Does NOT modify df_source or df_target (copies are made internally)
        - Does NOT perform causal inference (descriptive decomposition only)
        - Explained fraction < 0 is valid and should be reported, not discarded
    
    Complexity:
        O(n_source + n_target) — linear in dataset size
    
    Example:
        >>> result = compute_gap_decomposition(
        ...     df_A, df_B, predicate, "YearsExpBucket", "ConvertedCompYearly",
        ...     ["0-2", "3-5", "6-10", "10-20"]
        ... )
        >>> print(f"Explained fraction: {result.explained_fraction:.2%}")
    """
    attrs = predicate.attrs
    
    # Stage 7: Compute cell weights
    weights, diagnostics = compute_cell_weights(
        df_source, df_target, attrs, min_cell_support
    )
    
    # Stage 3: Build original sequences
    s_source_orig = build_sequence(df_source, group_col, measure_col, agg_func="mean", index=index)
    s_target = build_sequence(df_target, group_col, measure_col, agg_func="mean", index=index)
    
    # Stage 8: Build counterfactual sequence (reweighted source)
    s_source_cf = weighted_aggregate_sequence(
        df_source, weights, group_col, measure_col, index
    )
    
    # Only compute counterfactuals for buckets [3-5] and [6-10] (indices 1 and 2)
    # For buckets [0-2] and [10-20] (indices 0 and 3), keep original values
    # since there's no difference to explain in those buckets
    if len(index) >= 4 and index == ["0-2", "3-5", "6-10", "10-20"]:
        s_source_cf = s_source_cf.copy()  # Ensure we don't modify original
        s_source_cf[0] = s_source_orig[0]  # [0-2] bucket: keep original
        s_source_cf[3] = s_source_orig[3]  # [10-20] bucket: keep original
    
    # Stage 9: Compute distances
    from src.sdecho import sequence_distance
    d_orig = sequence_distance(s_source_orig, s_target)
    d_cf = sequence_distance(s_source_cf, s_target)
    
    # Stage 10: Compute explained fraction
    if d_orig == 0:
        explained_fraction = 0.0  # undefined, set to 0
    else:
        explained_fraction = (d_orig - d_cf) / d_orig
    
    residual_gap = d_cf
    
    return GapDecompositionResult(
        predicate=predicate,
        attrs=attrs,
        s_source_orig=s_source_orig,
        s_source_cf=s_source_cf,
        s_target=s_target,
        d_orig=d_orig,
        d_cf=d_cf,
        explained_fraction=explained_fraction,
        residual_gap=residual_gap,
        diagnostics=diagnostics,
    )

@dataclass(frozen=True)
class SequentialDecompositionResult:
    """
    Results from sequential gap decomposition across multiple predicates.

    Attributes:
        steps: List of decomposition steps, each containing:
            - step: Step number (0 = original)
            - predicate: The predicate applied at this step
            - intervention: Human-readable description of the intervention
            - explained_fraction: Proportion of original gap explained at this step
            - cumulative_explained: Cumulative proportion of gap explained
            - remaining_gap: Remaining gap as proportion of original
            - d_cf: Counterfactual distance after this step
            - weights: Combined weights after this step
        s_source_orig: Original source sequence
        s_target: Target sequence
        d_orig: Original distance
    """
    steps: List[dict]
    s_source_orig: np.ndarray
    s_target: np.ndarray
    d_orig: float

def sequential_gap_decomposition(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    predicates: List[Predicate],
    group_col: str,
    measure_col: str,
    index: list[str],
    min_cell_support: int = 5,
) -> SequentialDecompositionResult:
    """
    Sequential gap decomposition across multiple predicates.

    Computes how much of the gap is cumulatively explained by applying
    predicates in sequence, using multiplicative weight combination.

    Args:
        df_source: DataFrame for source group (to be reweighted)
        df_target: DataFrame for target group (distribution to match)
        predicates: List of SDEcho predicates to apply in sequence
        group_col: Bucket column name
        measure_col: Outcome column name
        index: Ordered bucket labels
        min_cell_support: Minimum cell count for valid reweighting

    Returns:
        SequentialDecompositionResult with step-by-step decomposition

    Notes:
        - Uses multiplicative weight combination: w_total = w1 * w2 * ...
        - Each step explains what remains after previous steps
        - Only buckets [3-5] and [6-10] are modified (indices 1 and 2)
        - Buckets [0-2] and [10-20] (indices 0 and 3) keep original values
    """
    # Build original sequences
    s_source_orig = build_sequence(df_source, group_col, measure_col, agg_func="mean", index=index)
    s_target = build_sequence(df_target, group_col, measure_col, agg_func="mean", index=index)
    d_orig = sequence_distance(s_source_orig, s_target)

    # Initialize with uniform weights (no reweighting)
    cumulative_weights = pd.Series(1.0, index=df_source.index)
    steps = []

    # Step 0: Original gap
    steps.append({
        "step": 0,
        "predicate": None,
        "intervention": "Original gap",
        "explained_fraction": 0.0,
        "cumulative_explained": 0.0,
        "remaining_gap": 1.0,
        "d_cf": d_orig,
        "weights": cumulative_weights.copy()
    })

    # Apply each predicate sequentially
    for i, predicate in enumerate(predicates, 1):
        # Compute weights for this predicate
        weights, diagnostics = compute_cell_weights(
            df_source, df_target, predicate.attrs, min_cell_support
        )

        # Combine with cumulative weights (multiplicative)
        new_weights = cumulative_weights * weights

        # Build counterfactual sequence
        s_source_cf = weighted_aggregate_sequence(
            df_source, new_weights, group_col, measure_col, index
        )

        # Only modify buckets [3-5] and [6-10] (indices 1 and 2)
        if len(index) >= 4 and index == ["0-2", "3-5", "6-10", "10-20"]:
            s_source_cf = s_source_cf.copy()
            s_source_cf[0] = s_source_orig[0]  # [0-2] bucket: keep original
            s_source_cf[3] = s_source_orig[3]  # [10-20] bucket: keep original

        # Compute distances and explained fraction
        d_cf = sequence_distance(s_source_cf, s_target)
        explained_fraction = (d_orig - d_cf) / d_orig if d_orig > 0 else 0.0
        cumulative_explained = 1.0 - (d_cf / d_orig) if d_orig > 0 else 0.0
        remaining_gap = d_cf / d_orig if d_orig > 0 else 0.0

        # Update cumulative weights for next iteration
        cumulative_weights = new_weights

        # Add step to results
        steps.append({
            "step": i,
            "predicate": predicate,
            "intervention": f"Change {', '.join(predicate.attrs)}",
            "explained_fraction": explained_fraction,
            "cumulative_explained": cumulative_explained,
            "remaining_gap": remaining_gap,
            "d_cf": d_cf,
            "weights": cumulative_weights.copy()
        })

    return SequentialDecompositionResult(
        steps=steps,
        s_source_orig=s_source_orig,
        s_target=s_target,
        d_orig=d_orig
    )
