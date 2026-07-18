# src/sdecho.py

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.predicates import Predicate, enumerate_predicates, predicate_mask
from src.sequence_builder import build_sequence


@dataclass(frozen=True)
class SDEchoResult:
    """
    One ranked candidate explanation from SDEcho's predicate search.
    
    Attributes:
        predicate: The discovered predicate
        gamma: Explanation score (lower = stronger explanation)
        dist_before: Original sequence distance
        dist_after: Distance after removing predicate-matching tuples
        n1: Number of matching tuples in group A
        n2: Number of matching tuples in group B
    """
    predicate: Predicate
    gamma: float
    dist_before: float
    dist_after: float
    n1: int          # matching tuple count in group A
    n2: int          # matching tuple count in group B


def sequence_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two aggregate sequences.
    
    Args:
        s1: First sequence vector
        s2: Second sequence vector
    
    Returns:
        Euclidean distance (L2 norm of difference)
    
    Complexity:
        O(len(s1)) where len(s1) = number of buckets
    """
    return float(np.sqrt(np.sum((s1 - s2) ** 2)))


def run_sdecho(
    df1: pd.DataFrame, df2: pd.DataFrame,
    group_col: str, measure_col: str, agg_func: str, index: list[str],
    candidate_attrs: list[str], max_order: int, k: int,
    max_values_per_attr: int, min_support: int,
) -> list[SDEchoResult]:
    """
    Brute-force SDEcho predicate search, ranked ascending by gamma.
    
    For each candidate predicate, computes gamma = (dist_after / dist_before) * penalty,
    where:
        - dist_after: distance after removing predicate-matching tuples
        - dist_before: original distance
        - penalty: 1 + (n1/N1) + (n2/N2) to penalize predicates matching too many tuples
    
    Args:
        df1: DataFrame for group A
        df2: DataFrame for group B
        group_col: Bucket column name
        measure_col: Outcome column name
        agg_func: Aggregation function ("mean", "sum", etc.)
        index: Ordered bucket labels
        candidate_attrs: Attributes to search for predicates
        max_order: Maximum predicate size (number of attributes)
        k: Number of top predicates to return
        max_values_per_attr: Limit values per attribute (None for all)
        min_support: Minimum combined matching tuples to consider a predicate
    
    Returns:
        List of top-k SDEchoResult objects, sorted by ascending gamma
        (lower gamma = stronger explanation)
    
    Complexity:
        O(N * M * P) where N = len(df1), M = len(df2), P = number of predicates.
        P is exponential in max_order and max_values_per_attr.
    
    Notes:
        - This is a brute-force reimplementation, not the optimized SDEcho algorithm
        - If dist_before == 0, raises ValueError (nothing to explain)
        - Predicates with support < min_support are filtered out
    """
    # Step 1: Build original aggregate sequences
    s1_orig = build_sequence(df1, group_col, measure_col, agg_func, index)
    s2_orig = build_sequence(df2, group_col, measure_col, agg_func, index)
    
    # Step 2: Compute original distance
    dist_before = sequence_distance(s1_orig, s2_orig)
    
    if dist_before == 0:
        raise ValueError(
            "Original sequences are identical (dist=0); nothing to explain."
        )
    
    # Step 3: Enumerate all candidate predicates
    predicates = enumerate_predicates(
        df1, df2, candidate_attrs, max_order, max_values_per_attr
    )
    
    n1_total = len(df1)
    n2_total = len(df2)
    
    # Step 4: Evaluate each predicate
    results = []
    for predicate in predicates:
        # Get masks for matching tuples
        mask1 = predicate_mask(df1, predicate)
        mask2 = predicate_mask(df2, predicate)
        
        n1_match = int(mask1.sum())
        n2_match = int(mask2.sum())
        
        # Support filter
        if n1_match + n2_match < min_support:
            continue
        
        # Skip if no matches in either group (removal has no effect)
        if n1_match == 0 and n2_match == 0:
            continue
        
        # Remove matching tuples and rebuild sequences
        df1_rest = df1.loc[~mask1]
        df2_rest = df2.loc[~mask2]
        
        s1_after = build_sequence(df1_rest, group_col, measure_col, agg_func, index)
        s2_after = build_sequence(df2_rest, group_col, measure_col, agg_func, index)
        
        dist_after = sequence_distance(s1_after, s2_after)
        
        # Compute gamma with penalty term
        penalty = 1.0 + (n1_match / n1_total) + (n2_match / n2_total)
        gamma = (dist_after / dist_before) * penalty
        
        results.append(SDEchoResult(
            predicate=predicate,
            gamma=gamma,
            dist_before=dist_before,
            dist_after=dist_after,
            n1=n1_match,
            n2=n2_match,
        ))
    
    # Step 5: Sort by gamma (ascending) and return top-k
    results.sort(key=lambda r: r.gamma)
    return results[:k]
