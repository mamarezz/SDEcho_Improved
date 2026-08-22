# src/sequence_builder.py

import warnings

import numpy as np
import pandas as pd


class EmptyBucketError(ValueError):
    """Raised when a weighted sequence has no positive mass in a bucket."""


def build_sequence(
    df: pd.DataFrame, group_col: str, measure_col: str,
    agg_func: str, index: list[str]
) -> np.ndarray:
    """
    Construct an aggregate sequence from a DataFrame.
    
    This is the single source of truth for turning a DataFrame into an
    aggregate sequence, used identically in the original (Stage 3) and
    counterfactual (Stage 8) paths.
    
    Args:
        df: DataFrame containing the raw tuples
        group_col: Column name defining the buckets (x-axis)
        measure_col: Column name of the numeric outcome to aggregate
        agg_func: Aggregation function ("mean", "sum", "count", etc.)
        index: Ordered list of bucket labels to use as the sequence index.
            Only buckets present in this list will appear in the output;
            missing buckets are filled with 0.
    
    Returns:
        1D numpy array of aggregated values, ordered according to `index`.
        Empty buckets (no rows in df for that bucket) are filled with 0.
    
    Example:
        >>> df = pd.DataFrame({
        ...     "YearsExpBucket": ["0-2", "0-2", "3-5"],
        ...     "ConvertedCompYearly": [50000, 60000, 80000]
        ... })
        >>> seq = build_sequence(df, "YearsExpBucket", "ConvertedCompYearly",
        ...                      "mean", ["0-2", "3-5", "6-10"])
        >>> seq  # array([55000., 80000., 0.])
    
    Notes:
        - Suppresses FutureWarning from pandas groupby operations
        - Uses fillna(0) for buckets with no data (consistent with SDEcho)
    """
    # Group by bucket and aggregate
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        aggregated = df.groupby(group_col)[measure_col].agg(agg_func)
    
    # Reindex to desired bucket order, fill missing with 0
    sequence = aggregated.reindex(index).fillna(0)
    
    return sequence.to_numpy(dtype=float)


def validate_weights(
    df: pd.DataFrame,
    weights: np.ndarray | pd.Series | None,
) -> np.ndarray:
    """Return finite, non-negative positional weights for ``df``."""
    if weights is None:
        return np.ones(len(df), dtype=float)

    if isinstance(weights, pd.Series) and not weights.index.equals(df.index):
        raise ValueError("a weight Series index must exactly match the DataFrame index")
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or len(values) != len(df):
        raise ValueError("weights must be one-dimensional and match the DataFrame length")
    if not np.all(np.isfinite(values)):
        raise ValueError("weights must contain only finite values")
    if np.any(values < 0):
        raise ValueError("weights must be non-negative")
    if float(values.sum()) <= 0:
        raise ValueError("weights must contain positive total mass")
    return values


def select_material_buckets(
    source_sequence: np.ndarray,
    target_sequence: np.ndarray,
    index: list[str],
    *,
    relative_to_max_gap: float = 0.18,
    min_absolute_gap: float = 10_000.0,
    min_percentage_gap: float = 5.0,
) -> list[str]:
    """Freeze buckets with materially different original sequence values.

    A bucket must exceed both an absolute threshold and a symmetric percentage
    threshold.  Selection is performed once on the original sequences; it is
    never recomputed adaptively during the iterative path.
    """
    source = np.asarray(source_sequence, dtype=float)
    target = np.asarray(target_sequence, dtype=float)
    if source.shape != target.shape or source.ndim != 1 or len(source) != len(index):
        raise ValueError("sequences and bucket index must have the same length")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("bucket selection requires finite sequence values")
    if relative_to_max_gap < 0 or min_absolute_gap < 0 or min_percentage_gap < 0:
        raise ValueError("bucket selection thresholds must be non-negative")

    absolute_gaps = np.abs(source - target)
    if len(absolute_gaps) == 0 or float(absolute_gaps.max()) == 0:
        return []
    absolute_threshold = max(
        float(absolute_gaps.max()) * relative_to_max_gap,
        min_absolute_gap,
    )
    symmetric_scale = (np.abs(source) + np.abs(target)) / 2.0
    percentage_gaps = np.divide(
        absolute_gaps * 100.0,
        symmetric_scale,
        out=np.full_like(absolute_gaps, np.inf),
        where=symmetric_scale > 0,
    )
    return [
        str(index[position])
        for position in range(len(index))
        if absolute_gaps[position] > absolute_threshold
        and percentage_gaps[position] > min_percentage_gap
    ]


def effective_sample_size(weights: np.ndarray | pd.Series) -> float:
    """Return Kish's effective sample size for non-negative weights."""
    values = np.asarray(weights, dtype=float)
    positive = values[values > 0]
    if len(positive) == 0:
        return 0.0
    denominator = float(np.square(positive).sum())
    return 0.0 if denominator == 0 else float(positive.sum() ** 2 / denominator)


def build_weighted_mean_sequence(
    df: pd.DataFrame,
    weights: np.ndarray | pd.Series | None,
    group_col: str,
    measure_col: str,
    index: list[str],
    *,
    require_all_buckets: bool = True,
) -> np.ndarray:
    """Build a weighted-mean sequence using positional row weights.

    Unlike the legacy unweighted helper, this function never represents an
    empty bucket as a numeric zero when ``require_all_buckets`` is true.  A
    missing weighted bucket makes a removal score undefined and is therefore
    reported explicitly.
    """
    if group_col not in df.columns or measure_col not in df.columns:
        raise KeyError(f"DataFrame must contain {group_col!r} and {measure_col!r}")

    row_weights = validate_weights(df, weights)
    bucket_values = df[group_col].astype(str).to_numpy()
    outcomes = pd.to_numeric(df[measure_col], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(outcomes)):
        raise ValueError(f"{measure_col!r} must contain only finite numeric values")

    sequence = []
    for bucket in index:
        mask = (bucket_values == str(bucket)) & (row_weights > 0)
        bucket_mass = float(row_weights[mask].sum())
        if bucket_mass <= 0:
            if require_all_buckets:
                raise EmptyBucketError(
                    f"bucket {bucket!r} has no positive-weight observations"
                )
            sequence.append(np.nan)
            continue
        sequence.append(float(np.average(outcomes[mask], weights=row_weights[mask])))

    return np.asarray(sequence, dtype=float)
