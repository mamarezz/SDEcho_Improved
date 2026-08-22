"""Weighted, brute-force SDEcho predicate discovery.

The published SDEcho score is defined on ordinary tuples.  The iterative
method needs the same removal question on a weighted pseudo-population, so
this module provides an explicit weighted extension.  Unit weights recover
the ordinary brute-force mean score for candidates satisfying the same support
and nonempty-bucket eligibility rules.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.predicates import Predicate, enumerate_predicates, predicate_mask
from src.sequence_builder import (
    EmptyBucketError,
    build_weighted_mean_sequence,
    effective_sample_size,
    validate_weights,
)


@dataclass(frozen=True)
class SDEchoResult:
    """One predicate evaluated on the current weighted populations."""

    predicate: Predicate
    gamma: float
    distance_before: float
    distance_after_removal: float
    source_support: int
    target_support: int
    source_active_support: int
    target_active_support: int
    source_weighted_share: float
    target_weighted_share: float
    source_match_ess: float
    target_match_ess: float

    @property
    def removal_reduction(self) -> float:
        """Signed fractional distance reduction caused by tuple removal."""
        if self.distance_before == 0:
            return 0.0
        return 1.0 - self.distance_after_removal / self.distance_before

    # Concise compatibility aliases used by older result readers.
    @property
    def dist_before(self) -> float:
        return self.distance_before

    @property
    def dist_after(self) -> float:
        return self.distance_after_removal

    @property
    def n1(self) -> int:
        return self.source_support

    @property
    def n2(self) -> int:
        return self.target_support


def sequence_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """Return Euclidean distance between equally shaped finite sequences."""
    left = np.asarray(s1, dtype=float)
    right = np.asarray(s2, dtype=float)
    if left.shape != right.shape:
        raise ValueError("sequence shapes must match")
    if left.ndim != 1 or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("sequences must be one-dimensional and finite")
    return float(np.linalg.norm(left - right))


def predicate_signature(predicate: Predicate) -> frozenset[tuple[str, object]]:
    """Return an order-independent, hashable predicate signature."""
    return frozenset(predicate.conditions.items())


def _weighted_share(mask: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return 0.0 if total == 0 else float(weights[mask].sum() / total)


def run_sdecho(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    *,
    group_col: str,
    measure_col: str,
    index: list[str],
    candidate_attrs: list[str],
    max_order: int,
    k: int,
    max_values_per_attr: int | None,
    min_support: int,
    source_weights: np.ndarray | pd.Series | None = None,
    target_weights: np.ndarray | pd.Series | None = None,
    min_support_per_group: int = 1,
    min_active_support_per_group: int = 1,
    min_match_ess: float = 1.0,
    excluded_predicates: Iterable[Predicate] = (),
    agg_func: str = "mean",
) -> list[SDEchoResult]:
    """Rank predicates by weighted SDEcho removal score.

    Removal is implemented by setting the matching rows' current weights to
    zero in both groups.  The current calibration weights are *not* refitted
    while a candidate is scored.  Raw support protects against tiny cells;
    weighted shares define the size penalty on the current pseudo-population.
    Distance, support, match ESS, and the size penalty are all restricted to
    rows in ``index`` so search and intervention can share one frozen scope.
    """
    if agg_func != "mean":
        raise ValueError("the iterative weighted implementation supports mean aggregation only")
    if k <= 0:
        return []
    if (
        min_support < 0
        or min_support_per_group < 0
        or min_active_support_per_group < 0
        or min_match_ess < 0
    ):
        raise ValueError("support thresholds must be non-negative")
    if not candidate_attrs:
        return []

    weights_source = validate_weights(df_source, source_weights)
    weights_target = validate_weights(df_target, target_weights)
    scope_values = {str(bucket) for bucket in index}
    source_scope = df_source[group_col].astype(str).isin(scope_values).to_numpy()
    target_scope = df_target[group_col].astype(str).isin(scope_values).to_numpy()

    source_sequence = build_weighted_mean_sequence(
        df_source, weights_source, group_col, measure_col, index
    )
    target_sequence = build_weighted_mean_sequence(
        df_target, weights_target, group_col, measure_col, index
    )
    distance_before = sequence_distance(source_sequence, target_sequence)
    if distance_before <= np.finfo(float).eps:
        return []

    excluded = {predicate_signature(predicate) for predicate in excluded_predicates}
    predicates = enumerate_predicates(
        df_source,
        df_target,
        candidate_attrs,
        max_order,
        max_values_per_attr,
    )

    results: list[SDEchoResult] = []
    for predicate in predicates:
        if predicate_signature(predicate) in excluded:
            continue

        source_mask = (
            predicate_mask(df_source, predicate).to_numpy(dtype=bool) & source_scope
        )
        target_mask = (
            predicate_mask(df_target, predicate).to_numpy(dtype=bool) & target_scope
        )
        source_support = int(source_mask.sum())
        target_support = int(target_mask.sum())
        if source_support + target_support < min_support:
            continue
        if source_support < min_support_per_group or target_support < min_support_per_group:
            continue
        source_active_mask = source_mask & (weights_source > 0)
        target_active_mask = target_mask & (weights_target > 0)
        source_active_support = int(source_active_mask.sum())
        target_active_support = int(target_active_mask.sum())
        source_match_ess = effective_sample_size(weights_source[source_mask])
        target_match_ess = effective_sample_size(weights_target[target_mask])
        if (
            source_active_support < min_active_support_per_group
            or target_active_support < min_active_support_per_group
            or source_match_ess < min_match_ess
            or target_match_ess < min_match_ess
        ):
            continue

        source_after = weights_source.copy()
        target_after = weights_target.copy()
        source_after[source_mask] = 0.0
        target_after[target_mask] = 0.0

        try:
            source_removed_sequence = build_weighted_mean_sequence(
                df_source, source_after, group_col, measure_col, index
            )
            target_removed_sequence = build_weighted_mean_sequence(
                df_target, target_after, group_col, measure_col, index
            )
        except (EmptyBucketError, ValueError):
            continue

        distance_after = sequence_distance(
            source_removed_sequence, target_removed_sequence
        )
        source_share = _weighted_share(
            source_mask[source_scope], weights_source[source_scope]
        )
        target_share = _weighted_share(
            target_mask[target_scope], weights_target[target_scope]
        )
        penalty = 1.0 + source_share + target_share
        gamma = distance_after / distance_before * penalty

        results.append(
            SDEchoResult(
                predicate=predicate,
                gamma=float(gamma),
                distance_before=distance_before,
                distance_after_removal=distance_after,
                source_support=source_support,
                target_support=target_support,
                source_active_support=source_active_support,
                target_active_support=target_active_support,
                source_weighted_share=source_share,
                target_weighted_share=target_share,
                source_match_ess=source_match_ess,
                target_match_ess=target_match_ess,
            )
        )

    results.sort(key=lambda result: (result.gamma, str(result.predicate)))
    return results[:k]
