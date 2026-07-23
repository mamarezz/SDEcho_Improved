# src/reweighting.py

from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
import numpy as np
import pandas as pd

from src.predicates import Predicate
from src.sequence_builder import build_sequence
from src.sdecho import SDEchoResult


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
        reweighted_buckets: List of bucket indices where counterfactual was applied
        bucket_threshold: The threshold used to determine which buckets to reweight
        bucket_selection_method: Method used for bucket selection
        bucket_details: Per-bucket details showing abs and pct differences
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
    reweighted_buckets: Optional[List[int]] = None
    bucket_threshold: Optional[float] = None
    bucket_selection_method: Optional[str] = None
    bucket_details: Optional[List[dict]] = None


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


def _compute_salary_scale(seq_source: np.ndarray, seq_target: np.ndarray) -> float:
    """
    Compute a representative salary scale from the sequences.
    
    Uses the mean of max-min ranges across both sequences to get a sense
    of what constitutes a 'large' vs 'small' difference for this comparison.
    
    Args:
        seq_source: Source group sequence
        seq_target: Target group sequence
    
    Returns:
        Representative scale value (e.g., mean of max-min ranges)
    """
    range_source = np.max(seq_source) - np.min(seq_source)
    range_target = np.max(seq_target) - np.min(seq_target)
    return (range_source + range_target) / 2


def identify_buckets_for_reweighting(
    seq_source: np.ndarray,
    seq_target: np.ndarray,
    index: list[str],
    method: str = "relative_and_absolute",
    threshold_factor: float = 0.15,
    min_absolute_difference: float = 10000,
    min_percentage_difference: float = 5.0,
) -> Tuple[List[int], float, List[dict]]:
    """
    Automatically identify which experience buckets require counterfactual computation.
    
    A bucket is considered to have a "meaningful difference" based on one of
    several criteria. This avoids reweighting buckets where the sequences are
    already similar.
    
    Args:
        seq_source: Original aggregate sequence for source group
        seq_target: Target group's aggregate sequence
        index: Ordered bucket labels
        method: Threshold method. Options:
            - "relative_and_absolute" (default): A bucket is significant if
              BOTH its absolute difference exceeds `min_absolute_difference` AND
              its percentage difference (relative to the mean) exceeds
              `min_percentage_difference`. This is the most robust approach
              because it filters out both small-absolute differences on
              large-salary buckets AND small-percentage differences.
            - "absolute": Bucket is significant if absolute difference
              exceeds threshold_factor * scale (where scale = mean of ranges).
            - "percentage": Bucket is significant if |diff| / mean * 100
              exceeds threshold_factor percent.
            - "relative_threshold": Bucket is significant if its difference
              exceeds threshold_factor * max_difference.
            - "percentile": Bucket is significant if its difference exceeds
              the given percentile of all differences.
            - "all": All buckets are significant (reweight all).
        threshold_factor: Used differently by each method:
            - For "absolute": fraction of salary scale (default 0.15 = 15%)
            - For "percentage": minimum percentage (default 5.0 = 5%)
            - For "relative_threshold": fraction of max diff (default 0.1 = 10%)
            - For "percentile": percentile value 0-1 (default ignored)
        min_absolute_difference: Minimum absolute difference to consider
            (default 10000, prevents including tiny diffs on large salaries).
        min_percentage_difference: Minimum percentage difference relative to
            the mean of the two groups (default 5.0 = 5%).
    
    Returns:
        Tuple of (reweighted_indices, threshold_value, bucket_details):
            - reweighted_indices: List of bucket indices to apply counterfactual
            - threshold_value: The actual threshold used (for reporting)
            - bucket_details: List of dicts with per-bucket analysis
    
    Examples:
        >>> seq_A = np.array([135156, 131692, 173414, 205397])  # 25-34
        >>> seq_B = np.array([123917, 194521, 212527, 197010])  # 35-44
        >>> indices, thresh, details = identify_buckets_for_reweighting(seq_A, seq_B,
        ...     ["0-2", "3-5", "6-10", "10-20"])
        >>> [index[i] for i in indices]  # Only 3-5 and 6-10
        ['3-5', '6-10']
    """
    # Compute absolute differences per bucket
    abs_diffs = np.abs(seq_source - seq_target)
    
    # Compute percentage differences per bucket
    # (relative to the mean of the two values)
    means = (np.abs(seq_source) + np.abs(seq_target)) / 2
    means = np.where(means == 0, 1.0, means)  # Avoid division by zero
    pct_diffs = (abs_diffs / means) * 100
    
    # Build per-bucket details for reporting
    bucket_details = []
    for i, bucket in enumerate(index):
        bucket_details.append({
            "index": i,
            "label": bucket,
            "abs_diff": float(abs_diffs[i]),
            "pct_diff": float(pct_diffs[i]),
            "value_source": float(seq_source[i]),
            "value_target": float(seq_target[i]),
        })
    
    if method == "all":
        return list(range(len(index))), 0.0, bucket_details
    
    if method == "absolute":
        # Threshold based on salary scale
        scale = _compute_salary_scale(seq_source, seq_target)
        threshold = scale * threshold_factor
        threshold = max(threshold, min_absolute_difference)
        reweighted_indices = [
            i for i in range(len(index))
            if abs_diffs[i] > threshold
        ]
        
    elif method == "percentage":
        # Threshold based on percentage difference
        threshold = threshold_factor  # e.g., 5.0 = 5%
        reweighted_indices = [
            i for i in range(len(index))
            if pct_diffs[i] > threshold
        ]
        
    elif method == "percentile":
        # Use a percentile of absolute differences as threshold
        threshold = float(np.percentile(abs_diffs, threshold_factor * 100))
        reweighted_indices = [
            i for i in range(len(index))
            if abs_diffs[i] > threshold
        ]
        
    elif method == "relative_threshold":
        # Old method: based on max difference
        max_diff = np.max(abs_diffs)
        if max_diff == 0:
            return list(range(len(index))), 0.0, bucket_details
        threshold = max_diff * threshold_factor
        threshold = max(threshold, min_absolute_difference)
        reweighted_indices = [
            i for i in range(len(index))
            if abs_diffs[i] > threshold
        ]
        
    else:
        # Default: "relative_and_absolute" - BOTH must be significant.
        # 
        # Rationale: A bucket should be reweighted only when the difference is
        # meaningful in BOTH absolute terms AND relative terms.
        # 
        # - Absolute check prevents reweighting tiny differences on large scales
        #   (e.g., $8k diff on $200k salary = 4% — not worth touching)
        # - Percentage check prevents reweighting large absolute differences 
        #   that are small relative to the values (e.g., $50k diff on $1M salary)
        #
        # Default thresholds:
        #   min_absolute_difference = $10,000
        #   min_percentage_difference = 5%
        
        scale = _compute_salary_scale(seq_source, seq_target)
        abs_threshold = max(scale * 0.10, min_absolute_difference)
        threshold = abs_threshold
        
        reweighted_indices = [
            i for i in range(len(index))
            if abs_diffs[i] > abs_threshold and pct_diffs[i] > min_percentage_difference
        ]
    
    # Fallback: if no buckets meet the threshold, include all buckets
    # (to avoid silently doing nothing - better to reweight everything than nothing)
    if len(reweighted_indices) == 0:
        reweighted_indices = list(range(len(index)))
    
    return reweighted_indices, threshold, bucket_details


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
    """
    n_source = len(df_source)
    
    # Step 1: Compute empirical cell proportions in both groups
    source_counts = df_source.groupby(attrs).size()
    source_props = source_counts / n_source
    
    n_target = len(df_target)
    target_counts = df_target.groupby(attrs).size()
    target_props = target_counts / n_target
    
    # Step 2: Identify valid cells (common support + min support)
    source_cells = set(source_counts.index)
    target_cells = set(target_counts.index)
    
    common_cells = source_cells & target_cells
    
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
    """
    valid_mask = weights.notna()
    df_valid = df.loc[valid_mask].copy()
    w_valid = weights.loc[valid_mask]
    
    if len(df_valid) == 0:
        return np.zeros(len(index), dtype=float)
    
    df_valid = df_valid.assign(_weight=w_valid)
    
    weighted_sums = df_valid.groupby(group_col).apply(
        lambda g: np.average(g[measure_col], weights=g['_weight'])
    )
    
    sequence = weighted_sums.reindex(index).fillna(0)
    
    return sequence.to_numpy(dtype=float)


def compute_gap_decomposition(
    df_source: pd.DataFrame, df_target: pd.DataFrame,
    predicate: Predicate, group_col: str, measure_col: str,
    index: list[str], min_cell_support: int = 5,
    reweight_method: str = "relative_and_absolute",
    reweight_threshold_factor: float = 0.15,
    reweight_min_difference: float = 10000,
    reweight_min_percentage: float = 5.0,
    reweight_buckets: Optional[Union[str, List[int]]] = "dynamic",
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
        reweight_method: Method for selecting which buckets to reweight.
            See identify_buckets_for_reweighting() for details.
        reweight_threshold_factor: Threshold factor (interpretation depends on method).
        reweight_min_difference: Minimum absolute difference to consider significant.
        reweight_min_percentage: Minimum percentage difference to consider significant.
        reweight_buckets: Override for bucket selection. Can be:
            - "dynamic" (default): Use automatic detection
            - A list of bucket indices (e.g., [1, 2] for [3-5] and [6-10])
            - "all": Reweight all buckets
    
    Returns:
        GapDecompositionResult with original/counterfactual sequences,
        distances, explained fraction, and diagnostics
    
    Notes:
        - Does NOT modify df_source or df_target (copies are made internally)
        - Does NOT perform causal inference (descriptive decomposition only)
        - Explained fraction < 0 is valid and should be reported, not discarded
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
    
    # Stage 8b: Determine which buckets to apply counterfactual reweighting
    if isinstance(reweight_buckets, list):
        buckets_to_reweight = reweight_buckets
        threshold_used = 0.0
        bucket_details = []
    elif reweight_buckets == "dynamic":
        buckets_to_reweight, threshold_used, bucket_details = identify_buckets_for_reweighting(
            s_source_orig, s_target, index,
            method=reweight_method,
            threshold_factor=reweight_threshold_factor,
            min_absolute_difference=reweight_min_difference,
            min_percentage_difference=reweight_min_percentage,
        )
    elif reweight_buckets == "all":
        buckets_to_reweight = list(range(len(index)))
        threshold_used = 0.0
        bucket_details = []
    else:
        raise ValueError(f"Unknown reweight_buckets value: {reweight_buckets}")
    
    # For buckets NOT in the reweight set, use original values
    s_source_cf = s_source_cf.copy()
    for i in range(len(index)):
        if i not in buckets_to_reweight:
            s_source_cf[i] = s_source_orig[i]
    
    # Stage 9: Compute distances
    from src.sdecho import sequence_distance
    d_orig = sequence_distance(s_source_orig, s_target)
    d_cf = sequence_distance(s_source_cf, s_target)
    
    # Stage 10: Compute explained fraction
    if d_orig == 0:
        explained_fraction = 0.0
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
        reweighted_buckets=buckets_to_reweight,
        bucket_threshold=threshold_used,
        bucket_selection_method=reweight_method,
        bucket_details=bucket_details,
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
            - reweighted_buckets: Bucket indices reweighted at this step
        s_source_orig: Original source sequence
        s_target: Target sequence
        d_orig: Original distance
        reweighted_buckets: List of bucket indices where counterfactual was applied
    """
    steps: List[dict]
    s_source_orig: np.ndarray
    s_target: np.ndarray
    d_orig: float
    reweighted_buckets: Optional[List[int]] = None


def sequential_gap_decomposition(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    predicates: List[Predicate],
    group_col: str,
    measure_col: str,
    index: list[str],
    min_cell_support: int = 5,
    reweight_method: str = "relative_and_absolute",
    reweight_threshold_factor: float = 0.15,
    reweight_min_difference: float = 10000,
    reweight_min_percentage: float = 5.0,
    reweight_buckets: Optional[Union[str, List[int]]] = "dynamic",
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
        reweight_method: Method for selecting which buckets to reweight
        reweight_threshold_factor: Threshold factor for bucket selection
        reweight_min_difference: Minimum absolute difference to consider significant
        reweight_min_percentage: Minimum percentage difference to consider significant
        reweight_buckets: Override for bucket selection:
            - "dynamic" (default): Use automatic detection
            - A list of bucket indices (e.g., [1, 2] for [3-5] and [6-10])
            - "all": Reweight all buckets

    Returns:
        SequentialDecompositionResult with step-by-step decomposition
    """
    # Build original sequences
    s_source_orig = build_sequence(df_source, group_col, measure_col, agg_func="mean", index=index)
    s_target = build_sequence(df_target, group_col, measure_col, agg_func="mean", index=index)
    from src.sdecho import sequence_distance
    d_orig = sequence_distance(s_source_orig, s_target)

    # Determine which buckets to apply counterfactual reweighting
    if isinstance(reweight_buckets, list):
        buckets_to_reweight = reweight_buckets
    elif reweight_buckets == "dynamic":
        buckets_to_reweight, _, _ = identify_buckets_for_reweighting(
            s_source_orig, s_target, index,
            method=reweight_method,
            threshold_factor=reweight_threshold_factor,
            min_absolute_difference=reweight_min_difference,
            min_percentage_difference=reweight_min_percentage,
        )
    elif reweight_buckets == "all":
        buckets_to_reweight = list(range(len(index)))
    else:
        raise ValueError(f"Unknown reweight_buckets value: {reweight_buckets}")

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
        "weights": cumulative_weights.copy(),
        "reweighted_buckets": buckets_to_reweight,
    })

    # Apply each predicate sequentially
    for i, predicate in enumerate(predicates, 1):
        weights, diagnostics = compute_cell_weights(
            df_source, df_target, predicate.attrs, min_cell_support
        )

        new_weights = cumulative_weights * weights

        s_source_cf = weighted_aggregate_sequence(
            df_source, weights, group_col, measure_col, index
        ).copy()

        # Only modify buckets in the reweight set
        for j in range(len(index)):
            if j not in buckets_to_reweight:
                s_source_cf[j] = s_source_orig[j]

        d_cf = sequence_distance(s_source_cf, s_target)
        explained_fraction = (d_orig - d_cf) / d_orig if d_orig > 0 else 0.0
        cumulative_explained = 1.0 - (d_cf / d_orig) if d_orig > 0 else 0.0
        remaining_gap = d_cf / d_orig if d_orig > 0 else 0.0

        cumulative_weights = new_weights

        steps.append({
            "step": i,
            "predicate": predicate,
            "intervention": f"Change {', '.join(predicate.attrs)}",
            "explained_fraction": explained_fraction,
            "cumulative_explained": cumulative_explained,
            "remaining_gap": remaining_gap,
            "d_cf": d_cf,
            "weights": cumulative_weights.copy(),
            "reweighted_buckets": buckets_to_reweight,
        })

    return SequentialDecompositionResult(
        steps=steps,
        s_source_orig=s_source_orig,
        s_target=s_target,
        d_orig=d_orig,
        reweighted_buckets=buckets_to_reweight,
    )