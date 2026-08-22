"""Cumulative calibration for exact SDEcho predicate events."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.predicates import Predicate, predicate_mask
from src.sequence_builder import effective_sample_size, validate_weights


class BalanceError(ValueError):
    """Raised when requested predicate-event balance is not estimable."""


@dataclass(frozen=True)
class BalanceDiagnostics:
    """Quality diagnostics for one cumulative calibration run."""

    converged: bool
    iterations: int
    max_balance_error: float
    effective_sample_size: float
    effective_sample_size_ratio: float
    minimum_bucket_effective_sample_size_ratio: float
    min_positive_weight: float
    max_weight: float
    n_constraints: int


def _weighted_prevalence(mask: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        raise BalanceError("cannot compute prevalence with zero total weight")
    return float(weights[mask].sum() / total)


def _constraint_masks(
    df: pd.DataFrame,
    predicate: Predicate,
    group_col: str,
    bucket: str,
) -> tuple[np.ndarray, np.ndarray]:
    bucket_mask = df[group_col].astype(str).to_numpy() == str(bucket)
    event_mask = predicate_mask(df, predicate).to_numpy(dtype=bool)
    return bucket_mask, bucket_mask & event_mask


def predicate_balance_table(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    source_weights: np.ndarray | pd.Series,
    target_weights: np.ndarray | pd.Series | None,
    predicates: list[Predicate],
    group_col: str,
    index: list[str],
) -> pd.DataFrame:
    """Return before/after-ready prevalence diagnostics for constraints."""
    source_w = validate_weights(df_source, source_weights)
    target_w = validate_weights(df_target, target_weights)
    rows: list[dict[str, object]] = []
    for predicate in predicates:
        for bucket in index:
            source_bucket, source_event = _constraint_masks(
                df_source, predicate, group_col, bucket
            )
            target_bucket, target_event = _constraint_masks(
                df_target, predicate, group_col, bucket
            )
            source_prevalence = _weighted_prevalence(
                source_event[source_bucket], source_w[source_bucket]
            )
            target_prevalence = _weighted_prevalence(
                target_event[target_bucket], target_w[target_bucket]
            )
            rows.append(
                {
                    "predicate": str(predicate),
                    "bucket": str(bucket),
                    "source_prevalence": source_prevalence,
                    "target_prevalence": target_prevalence,
                    "absolute_error": abs(source_prevalence - target_prevalence),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "predicate",
            "bucket",
            "source_prevalence",
            "target_prevalence",
            "absolute_error",
        ],
    )


def predicate_support_table(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    source_weights: np.ndarray | pd.Series,
    target_weights: np.ndarray | pd.Series | None,
    predicates: list[Predicate],
    group_col: str,
    index: list[str],
) -> pd.DataFrame:
    """Return raw, active, and effective support for every event constraint."""
    source_w = validate_weights(df_source, source_weights)
    target_w = validate_weights(df_target, target_weights)
    rows: list[dict[str, object]] = []
    for predicate in predicates:
        for bucket in index:
            source_bucket, source_event = _constraint_masks(
                df_source, predicate, group_col, bucket
            )
            target_bucket, target_event = _constraint_masks(
                df_target, predicate, group_col, bucket
            )
            source_nonevent = source_bucket & ~source_event
            target_nonevent = target_bucket & ~target_event
            source_active_event = source_event & (source_w > 0)
            target_active_event = target_event & (target_w > 0)
            source_active_nonevent = source_nonevent & (source_w > 0)
            target_active_nonevent = target_nonevent & (target_w > 0)
            rows.append(
                {
                    "predicate": str(predicate),
                    "bucket": str(bucket),
                    "source_event_count": int(source_event.sum()),
                    "target_event_count": int(target_event.sum()),
                    "source_complement_count": int(source_nonevent.sum()),
                    "target_complement_count": int(target_nonevent.sum()),
                    "source_active_event_count": int(source_active_event.sum()),
                    "target_active_event_count": int(target_active_event.sum()),
                    "source_active_complement_count": int(
                        source_active_nonevent.sum()
                    ),
                    "target_active_complement_count": int(
                        target_active_nonevent.sum()
                    ),
                    "source_event_ess": effective_sample_size(source_w[source_event]),
                    "target_event_ess": effective_sample_size(target_w[target_event]),
                    "source_complement_ess": effective_sample_size(
                        source_w[source_nonevent]
                    ),
                    "target_complement_ess": effective_sample_size(
                        target_w[target_nonevent]
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "predicate",
            "bucket",
            "source_event_count",
            "target_event_count",
            "source_complement_count",
            "target_complement_count",
            "source_active_event_count",
            "target_active_event_count",
            "source_active_complement_count",
            "target_active_complement_count",
            "source_event_ess",
            "target_event_ess",
            "source_complement_ess",
            "target_complement_ess",
        ],
    )


def _minimum_bucket_ess_ratio(
    df_source: pd.DataFrame,
    weights: np.ndarray,
    group_col: str,
    index: list[str],
) -> float:
    """Return the smallest Kish-ESS/raw-count ratio in the requested buckets."""
    ratios: list[float] = []
    bucket_values = df_source[group_col].astype(str).to_numpy()
    for bucket in index:
        bucket_mask = bucket_values == str(bucket)
        count = int(bucket_mask.sum())
        if count == 0:
            raise BalanceError(f"bucket {bucket!r} is absent from the source group")
        ratios.append(effective_sample_size(weights[bucket_mask]) / count)
    return min(ratios, default=1.0)


def calibrate_predicate_events(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    predicates: list[Predicate],
    *,
    group_col: str,
    index: list[str],
    base_source_weights: np.ndarray | pd.Series | None = None,
    target_weights: np.ndarray | pd.Series | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
    min_event_count_per_bucket: int = 1,
    min_complement_count_per_bucket: int = 1,
    min_active_event_count_per_bucket: int = 1,
    min_active_complement_count_per_bucket: int = 1,
    min_event_ess_per_bucket: float = 1.0,
    min_complement_ess_per_bucket: float = 1.0,
) -> tuple[np.ndarray, BalanceDiagnostics]:
    """Balance all selected binary predicate events within every bucket.

    Weights are recomputed from the same base weights on every call.  Cyclic
    raking then satisfies every selected predicate-by-bucket marginal, so a
    later step does not silently discard the balance constraints from an
    earlier step.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    support_thresholds = (
        min_event_count_per_bucket,
        min_complement_count_per_bucket,
        min_active_event_count_per_bucket,
        min_active_complement_count_per_bucket,
        min_event_ess_per_bucket,
        min_complement_ess_per_bucket,
    )
    if any(value < 0 for value in support_thresholds):
        raise ValueError("per-bucket support thresholds must be non-negative")
    if len(index) != len(set(map(str, index))):
        raise ValueError("index must not contain duplicate buckets")

    base_weights = validate_weights(df_source, base_source_weights)
    fixed_target_weights = validate_weights(df_target, target_weights)
    base_weights = base_weights * (len(df_source) / float(base_weights.sum()))
    fixed_target_weights = fixed_target_weights * (
        len(df_target) / float(fixed_target_weights.sum())
    )
    if not predicates:
        ess = effective_sample_size(base_weights)
        positive = base_weights[base_weights > 0]
        return base_weights.copy(), BalanceDiagnostics(
            converged=True,
            iterations=0,
            max_balance_error=0.0,
            effective_sample_size=ess,
            effective_sample_size_ratio=ess / len(df_source) if len(df_source) else 0.0,
            minimum_bucket_effective_sample_size_ratio=_minimum_bucket_ess_ratio(
                df_source, base_weights, group_col, index
            ),
            min_positive_weight=float(positive.min()),
            max_weight=float(positive.max()),
            n_constraints=0,
        )

    signatures = [frozenset(predicate.conditions.items()) for predicate in predicates]
    if len(signatures) != len(set(signatures)):
        raise ValueError("predicates must not contain duplicates")

    constraints: list[tuple[np.ndarray, np.ndarray, float, str, Predicate]] = []
    for predicate in predicates:
        for bucket in index:
            source_bucket, source_event = _constraint_masks(
                df_source, predicate, group_col, bucket
            )
            target_bucket, target_event = _constraint_masks(
                df_target, predicate, group_col, bucket
            )
            if not source_bucket.any() or not target_bucket.any():
                raise BalanceError(f"bucket {bucket!r} is absent from one comparison group")

            source_event_count = int(source_event.sum())
            target_event_count = int(target_event.sum())
            source_complement_count = int((source_bucket & ~source_event).sum())
            target_complement_count = int((target_bucket & ~target_event).sum())
            if (
                source_event_count < min_event_count_per_bucket
                or target_event_count < min_event_count_per_bucket
            ):
                raise BalanceError(
                    f"{predicate} lacks event support in bucket {bucket}: "
                    f"source={source_event_count}, target={target_event_count}"
                )
            if (
                source_complement_count < min_complement_count_per_bucket
                or target_complement_count < min_complement_count_per_bucket
            ):
                raise BalanceError(
                    f"{predicate} lacks complement support in bucket {bucket}: "
                    f"source={source_complement_count}, target={target_complement_count}"
                )

            source_active_event_count = int(
                (source_event & (base_weights > 0)).sum()
            )
            target_active_event_count = int(
                (target_event & (fixed_target_weights > 0)).sum()
            )
            source_nonevent = source_bucket & ~source_event
            target_nonevent = target_bucket & ~target_event
            source_active_complement_count = int(
                (source_nonevent & (base_weights > 0)).sum()
            )
            target_active_complement_count = int(
                (target_nonevent & (fixed_target_weights > 0)).sum()
            )
            if (
                source_active_event_count < min_active_event_count_per_bucket
                or target_active_event_count < min_active_event_count_per_bucket
            ):
                raise BalanceError(
                    f"{predicate} lacks active event support in bucket {bucket}: "
                    f"source={source_active_event_count}, "
                    f"target={target_active_event_count}"
                )
            if (
                source_active_complement_count
                < min_active_complement_count_per_bucket
                or target_active_complement_count
                < min_active_complement_count_per_bucket
            ):
                raise BalanceError(
                    f"{predicate} lacks active complement support in bucket {bucket}: "
                    f"source={source_active_complement_count}, "
                    f"target={target_active_complement_count}"
                )

            source_event_ess = effective_sample_size(base_weights[source_event])
            target_event_ess = effective_sample_size(
                fixed_target_weights[target_event]
            )
            source_complement_ess = effective_sample_size(
                base_weights[source_nonevent]
            )
            target_complement_ess = effective_sample_size(
                fixed_target_weights[target_nonevent]
            )
            if (
                source_event_ess < min_event_ess_per_bucket
                or target_event_ess < min_event_ess_per_bucket
            ):
                raise BalanceError(
                    f"{predicate} has insufficient event ESS in bucket {bucket}: "
                    f"source={source_event_ess:.3f}, target={target_event_ess:.3f}"
                )
            if (
                source_complement_ess < min_complement_ess_per_bucket
                or target_complement_ess < min_complement_ess_per_bucket
            ):
                raise BalanceError(
                    f"{predicate} has insufficient complement ESS in bucket {bucket}: "
                    f"source={source_complement_ess:.3f}, "
                    f"target={target_complement_ess:.3f}"
                )

            target_prevalence = _weighted_prevalence(
                target_event[target_bucket], fixed_target_weights[target_bucket]
            )
            source_base_mass = float(base_weights[source_bucket].sum())
            source_event_mass = float(base_weights[source_event].sum())
            source_nonevent_mass = source_base_mass - source_event_mass
            if target_prevalence > tolerance and source_event_mass <= 0:
                raise BalanceError(
                    f"{predicate} has target mass but no source support in bucket {bucket}"
                )
            if target_prevalence < 1.0 - tolerance and source_nonevent_mass <= 0:
                raise BalanceError(
                    f"the complement of {predicate} has no source support in bucket {bucket}"
                )
            constraints.append(
                (source_bucket, source_event, target_prevalence, str(bucket), predicate)
            )

    weights = base_weights.copy()
    max_error = float("inf")
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        for bucket_mask, event_mask, target_prevalence, bucket, predicate in constraints:
            bucket_weights = weights[bucket_mask]
            bucket_total = float(bucket_weights.sum())
            if bucket_total <= 0:
                raise BalanceError(f"calibration removed all mass from bucket {bucket}")

            local_event = event_mask[bucket_mask]
            event_mass = float(bucket_weights[local_event].sum())
            nonevent_mass = bucket_total - event_mass

            if target_prevalence <= tolerance:
                weights[event_mask] = 0.0
            elif target_prevalence >= 1.0 - tolerance:
                weights[bucket_mask & ~event_mask] = 0.0
            else:
                if event_mass <= 0 or nonevent_mass <= 0:
                    raise BalanceError(
                        f"constraints are infeasible for {predicate} in bucket {bucket}"
                    )
                weights[event_mask] *= target_prevalence * bucket_total / event_mass
                weights[bucket_mask & ~event_mask] *= (
                    (1.0 - target_prevalence) * bucket_total / nonevent_mass
                )

        errors = []
        for bucket_mask, event_mask, target_prevalence, _, _ in constraints:
            current = _weighted_prevalence(
                event_mask[bucket_mask], weights[bucket_mask]
            )
            errors.append(abs(current - target_prevalence))
        max_error = max(errors, default=0.0)
        if max_error <= tolerance:
            converged = True
            break

    if not converged:
        raise BalanceError(
            f"calibration did not converge after {max_iterations} iterations; "
            f"maximum balance error is {max_error:.3g}"
        )

    positive = weights[weights > 0]
    ess = effective_sample_size(weights)
    diagnostics = BalanceDiagnostics(
        converged=True,
        iterations=iterations,
        max_balance_error=max_error,
        effective_sample_size=ess,
        effective_sample_size_ratio=ess / len(df_source) if len(df_source) else 0.0,
        minimum_bucket_effective_sample_size_ratio=_minimum_bucket_ess_ratio(
            df_source, weights, group_col, index
        ),
        min_positive_weight=float(positive.min()) if len(positive) else 0.0,
        max_weight=float(positive.max()) if len(positive) else 0.0,
        n_constraints=len(constraints),
    )
    return weights, diagnostics
