# src/predicates.py

from dataclasses import dataclass, field
from typing import Dict, Any
import itertools
import pandas as pd


@dataclass(frozen=True)
class Predicate:
    """
    A conjunctive predicate over one or more attributes.
    
    Example:
        Predicate({"Country": "United States of America"})  # single attribute
        Predicate({"Country": "USA", "EdLevel": "Bachelor"})  # conjunction
    """
    conditions: Dict[str, Any]  # {"Country": "United States of America"}

    @property
    def attrs(self) -> list:
        """Return list of attribute names in this predicate."""
        return list(self.conditions.keys())

    def __repr__(self) -> str:
        return " & ".join(f"{k}={v}" for k, v in self.conditions.items())


def enumerate_predicates(
    df1: pd.DataFrame, df2: pd.DataFrame,
    candidate_attrs: list[str], max_order: int,
    max_values_per_attr: int | None = None,
) -> list[Predicate]:
    """
    Generate all candidate predicates up to max_order attributes.
    
    For each combination of attributes (order 1 to max_order), enumerate all
    value combinations from the union of values observed in both groups.
    
    Args:
        df1: Source DataFrame for group A
        df2: Source DataFrame for group B
        candidate_attrs: List of attribute names to consider
        max_order: Maximum number of attributes in a predicate (1, 2, ...)
        max_values_per_attr: If set, only use the top-k most frequent values
            per attribute across both groups (reduces combinatorial explosion)
    
    Returns:
        List of Predicate objects (possibly empty if no valid combinations)
    
    Complexity:
        O(C * V^O) where C = len(candidate_attrs), V = average values per attr,
        O = max_order. Can be exponential in max_order.
    """
    # Build value map: for each attribute, collect values from both groups
    value_map = {}
    for attr in candidate_attrs:
        # Combine values from both groups, drop NaN
        combined = pd.concat([df1[attr], df2[attr]]).dropna()
        
        if max_values_per_attr is not None:
            # Keep only top-k most frequent values
            top_vals = combined.value_counts().nlargest(max_values_per_attr).index
            value_map[attr] = list(top_vals)
        else:
            value_map[attr] = list(combined.unique())
    
    # Generate predicates for each order (1 to max_order)
    predicates = []
    for order in range(1, max_order + 1):
        # All combinations of attributes at this order
        for attrs_combo in itertools.combinations(candidate_attrs, order):
            # Get value lists for these attributes
            value_lists = [value_map[a] for a in attrs_combo]
            
            # Cartesian product of values
            for value_combo in itertools.product(*value_lists):
                conditions = dict(zip(attrs_combo, value_combo))
                predicates.append(Predicate(conditions))
    
    return predicates


def predicate_mask(df: pd.DataFrame, predicate: Predicate) -> pd.Series:
    """
    Return a boolean mask of rows matching all conditions in the predicate.
    
    Args:
        df: DataFrame to mask
        predicate: Predicate specifying attribute=value conditions
    
    Returns:
        Boolean Series (same index as df) where True indicates the row
        satisfies all predicate conditions
    
    Example:
        >>> predicate = Predicate({"Country": "USA", "EdLevel": "Bachelor"})
        >>> mask = predicate_mask(df, predicate)
        >>> matching_rows = df[mask]
    """
    mask = pd.Series(True, index=df.index)
    
    for attr, val in predicate.conditions.items():
        mask &= (df[attr] == val)
    
    return mask

        