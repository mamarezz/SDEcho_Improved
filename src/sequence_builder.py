# src/sequence_builder.py

import warnings
import pandas as pd
import numpy as np


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
