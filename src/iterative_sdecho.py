"""Sequential SDEcho discovery and cumulative predicate-event balancing."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.balancing import BalanceDiagnostics, BalanceError, calibrate_predicate_events
from src.predicates import Predicate, predicate_mask
from src.sdecho import SDEchoResult, run_sdecho, sequence_distance
from src.sequence_builder import build_weighted_mean_sequence, validate_weights


@dataclass(frozen=True)
class IterationStep:
    """One accepted discovery-and-balance step."""

    iteration: int
    sdecho_rank: int
    predicate: Predicate
    sdecho_result: SDEchoResult
    distance_before: float
    search_distance_before: float
    search_distance_after_removal: float
    distance_after_balancing: float
    conditional_balance_reduction: float
    original_scale_path_increment: float
    cumulative_reduction: float
    source_sequence_before: np.ndarray
    source_sequence_after: np.ndarray
    target_sequence: np.ndarray
    balance_diagnostics: BalanceDiagnostics

    @property
    def distance_after_removal(self) -> float:
        """Compatibility alias for removal distance on the frozen search buckets."""
        return self.search_distance_after_removal


@dataclass(frozen=True)
class IterativeSDEchoResult:
    """Complete ordered counterfactual explanation path."""

    initial_source_sequence: np.ndarray
    target_sequence: np.ndarray
    initial_distance: float
    steps: tuple[IterationStep, ...]
    final_source_sequence: np.ndarray
    final_distance: float
    final_source_weights: np.ndarray
    stopping_reason: str
    balance_index: tuple[str, ...]
    search_index: tuple[str, ...]
    candidate_pool_size: int
    rejections: tuple["CandidateRejection", ...]

    @property
    def predicates(self) -> tuple[Predicate, ...]:
        return tuple(step.predicate for step in self.steps)

    @property
    def cumulative_reduction(self) -> float:
        if self.initial_distance == 0:
            return 0.0
        return 1.0 - self.final_distance / self.initial_distance


@dataclass(frozen=True)
class CandidateRejection:
    """One SDEcho candidate not accepted as a balancing step."""

    iteration: int
    sdecho_rank: int
    predicate: Predicate
    reason: str


def _same_event_mask(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    left: Predicate,
    right: Predicate,
) -> bool:
    return bool(
        np.array_equal(
            predicate_mask(df_source, left).to_numpy(dtype=bool),
            predicate_mask(df_source, right).to_numpy(dtype=bool),
        )
        and np.array_equal(
            predicate_mask(df_target, left).to_numpy(dtype=bool),
            predicate_mask(df_target, right).to_numpy(dtype=bool),
        )
    )


def result_table(result: IterativeSDEchoResult) -> pd.DataFrame:
    """Return one reporting row per accepted iteration."""
    rows = []
    for step in result.steps:
        rows.append(
            {
                "iteration": step.iteration,
                "sdecho_rank": step.sdecho_rank,
                "predicate": str(step.predicate),
                "gamma": step.sdecho_result.gamma,
                "distance_before": step.distance_before,
                "search_distance_before": step.search_distance_before,
                "search_distance_after_removal": (
                    step.search_distance_after_removal
                ),
                "removal_reduction": step.sdecho_result.removal_reduction,
                "distance_after_balancing": step.distance_after_balancing,
                "conditional_balance_reduction": step.conditional_balance_reduction,
                "original_scale_path_increment": step.original_scale_path_increment,
                "cumulative_reduction": step.cumulative_reduction,
                "source_support": step.sdecho_result.source_support,
                "target_support": step.sdecho_result.target_support,
                "source_active_support": step.sdecho_result.source_active_support,
                "target_active_support": step.sdecho_result.target_active_support,
                "source_weighted_share": step.sdecho_result.source_weighted_share,
                "target_weighted_share": step.sdecho_result.target_weighted_share,
                "source_match_ess": step.sdecho_result.source_match_ess,
                "target_match_ess": step.sdecho_result.target_match_ess,
                "effective_sample_size": step.balance_diagnostics.effective_sample_size,
                "effective_sample_size_ratio": (
                    step.balance_diagnostics.effective_sample_size_ratio
                ),
                "minimum_bucket_effective_sample_size_ratio": (
                    step.balance_diagnostics.minimum_bucket_effective_sample_size_ratio
                ),
                "max_weight": step.balance_diagnostics.max_weight,
                "min_positive_weight": step.balance_diagnostics.min_positive_weight,
                "max_balance_error": step.balance_diagnostics.max_balance_error,
            }
        )
    return pd.DataFrame(rows)


def rejection_table(result: IterativeSDEchoResult) -> pd.DataFrame:
    """Return every candidate rejection retained during the greedy search."""
    return pd.DataFrame(
        [
            {
                "iteration": rejection.iteration,
                "sdecho_rank": rejection.sdecho_rank,
                "predicate": str(rejection.predicate),
                "reason": rejection.reason,
            }
            for rejection in result.rejections
        ],
        columns=["iteration", "sdecho_rank", "predicate", "reason"],
    )


def run_iterative_sdecho(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    *,
    group_col: str,
    measure_col: str,
    index: list[str],
    candidate_attrs: list[str],
    max_order: int = 2,
    max_iterations: int = 5,
    candidate_pool_size: int = 10,
    max_values_per_attr: int | None = 10,
    min_support: int = 20,
    min_support_per_group: int = 1,
    min_active_support_per_group: int = 1,
    min_match_ess: float = 1.0,
    min_removal_reduction: float = 0.0,
    min_balance_reduction: float = 0.0,
    distance_tolerance: float = 1e-9,
    min_effective_sample_size_ratio: float = 0.1,
    min_bucket_effective_sample_size_ratio: float = 0.1,
    max_weight: float = 25.0,
    calibration_tolerance: float = 1e-10,
    calibration_max_iterations: int = 10_000,
    min_event_count_per_bucket: int = 1,
    min_complement_count_per_bucket: int = 1,
    min_active_event_count_per_bucket: int = 1,
    min_active_complement_count_per_bucket: int = 1,
    min_event_ess_per_bucket: float = 1.0,
    min_complement_ess_per_bucket: float = 1.0,
    balance_index: list[str] | None = None,
    base_source_weights: np.ndarray | pd.Series | None = None,
    target_weights: np.ndarray | pd.Series | None = None,
) -> IterativeSDEchoResult:
    """Build a greedy, ordered explanation path.

    At each iteration weighted SDEcho supplies candidates in removal-score
    order.  The first candidate that is non-duplicate, passes the configured
    support checks, and produces a material positive balance update is
    accepted.  All selected
    predicate-by-bucket margins are recalibrated from the original base
    weights, preserving earlier constraints.
    """
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive")
    if not 0 <= min_effective_sample_size_ratio <= 1:
        raise ValueError("min_effective_sample_size_ratio must be between 0 and 1")
    if not 0 <= min_bucket_effective_sample_size_ratio <= 1:
        raise ValueError(
            "min_bucket_effective_sample_size_ratio must be between 0 and 1"
        )
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")

    base_weights = validate_weights(df_source, base_source_weights)
    fixed_target_weights = validate_weights(df_target, target_weights)
    base_weights = base_weights * (len(df_source) / float(base_weights.sum()))
    fixed_target_weights = fixed_target_weights * (
        len(df_target) / float(fixed_target_weights.sum())
    )
    balance_buckets = list(index if balance_index is None else balance_index)
    if not set(balance_buckets).issubset(set(index)):
        raise ValueError("balance_index must be a subset of the scored bucket index")
    if len(balance_buckets) != len(set(map(str, balance_buckets))):
        raise ValueError("balance_index must not contain duplicate buckets")
    search_buckets = balance_buckets.copy()
    current_weights = base_weights.copy()

    initial_source = build_weighted_mean_sequence(
        df_source, current_weights, group_col, measure_col, index
    )
    target_sequence = build_weighted_mean_sequence(
        df_target, fixed_target_weights, group_col, measure_col, index
    )
    initial_distance = sequence_distance(initial_source, target_sequence)
    current_sequence = initial_source.copy()
    current_distance = initial_distance
    selected: list[Predicate] = []
    steps: list[IterationStep] = []
    rejections: list[CandidateRejection] = []
    stopping_reason = "maximum iterations reached"

    if not balance_buckets:
        stopping_reason = "no buckets satisfy the frozen intervention policy"
    elif initial_distance <= distance_tolerance:
        stopping_reason = "initial sequences are already within tolerance"
    else:
        for iteration in range(1, max_iterations + 1):
            rankings = run_sdecho(
                df_source,
                df_target,
                group_col=group_col,
                measure_col=measure_col,
                index=search_buckets,
                candidate_attrs=candidate_attrs,
                max_order=max_order,
                k=candidate_pool_size,
                max_values_per_attr=max_values_per_attr,
                min_support=min_support,
                min_support_per_group=min_support_per_group,
                min_active_support_per_group=min_active_support_per_group,
                min_match_ess=min_match_ess,
                source_weights=current_weights,
                target_weights=fixed_target_weights,
                excluded_predicates=selected,
            )
            if not rankings:
                stopping_reason = "weighted SDEcho found no eligible predicate"
                break

            accepted: tuple[
                int,
                SDEchoResult,
                np.ndarray,
                np.ndarray,
                float,
                BalanceDiagnostics,
            ] | None = None
            for rank, candidate in enumerate(rankings, start=1):
                if candidate.removal_reduction <= min_removal_reduction:
                    rejections.append(
                        CandidateRejection(
                            iteration,
                            rank,
                            candidate.predicate,
                            "removal reduction is below the configured minimum",
                        )
                    )
                    continue
                if any(
                    _same_event_mask(
                        df_source, df_target, candidate.predicate, previous
                    )
                    for previous in selected
                ):
                    rejections.append(
                        CandidateRejection(
                            iteration,
                            rank,
                            candidate.predicate,
                            "event mask duplicates a previously selected predicate",
                        )
                    )
                    continue

                try:
                    candidate_weights, diagnostics = calibrate_predicate_events(
                        df_source,
                        df_target,
                        selected + [candidate.predicate],
                        group_col=group_col,
                        index=balance_buckets,
                        base_source_weights=base_weights,
                        target_weights=fixed_target_weights,
                        tolerance=calibration_tolerance,
                        max_iterations=calibration_max_iterations,
                        min_event_count_per_bucket=min_event_count_per_bucket,
                        min_complement_count_per_bucket=min_complement_count_per_bucket,
                        min_active_event_count_per_bucket=(
                            min_active_event_count_per_bucket
                        ),
                        min_active_complement_count_per_bucket=(
                            min_active_complement_count_per_bucket
                        ),
                        min_event_ess_per_bucket=min_event_ess_per_bucket,
                        min_complement_ess_per_bucket=min_complement_ess_per_bucket,
                    )
                    candidate_sequence = build_weighted_mean_sequence(
                        df_source,
                        candidate_weights,
                        group_col,
                        measure_col,
                        index,
                    )
                except (BalanceError, ValueError) as exc:
                    rejections.append(
                        CandidateRejection(
                            iteration,
                            rank,
                            candidate.predicate,
                            f"calibration rejected: {exc}",
                        )
                    )
                    continue

                candidate_distance = sequence_distance(
                    candidate_sequence, target_sequence
                )
                conditional_reduction = (
                    0.0
                    if current_distance == 0
                    else (current_distance - candidate_distance) / current_distance
                )
                if conditional_reduction <= min_balance_reduction:
                    rejections.append(
                        CandidateRejection(
                            iteration,
                            rank,
                            candidate.predicate,
                            "balance update is not a material positive reduction",
                        )
                    )
                    continue
                if (
                    diagnostics.effective_sample_size_ratio
                    < min_effective_sample_size_ratio
                    or diagnostics.minimum_bucket_effective_sample_size_ratio
                    < min_bucket_effective_sample_size_ratio
                    or diagnostics.max_weight > max_weight
                ):
                    rejections.append(
                        CandidateRejection(
                            iteration,
                            rank,
                            candidate.predicate,
                            "balance update fails effective-sample-size or weight limits",
                        )
                    )
                    continue

                accepted = (
                    rank,
                    candidate,
                    candidate_weights,
                    candidate_sequence,
                    candidate_distance,
                    diagnostics,
                )
                break

            if accepted is None:
                stopping_reason = (
                    f"no candidate among the top {candidate_pool_size} weighted "
                    "SDEcho results produced a feasible material balance reduction"
                )
                break

            rank, candidate, new_weights, new_sequence, new_distance, diagnostics = accepted
            original_scale_path_increment = (
                0.0
                if initial_distance == 0
                else (current_distance - new_distance) / initial_distance
            )
            cumulative_reduction = (
                0.0
                if initial_distance == 0
                else (initial_distance - new_distance) / initial_distance
            )
            conditional_reduction = (
                0.0
                if current_distance == 0
                else (current_distance - new_distance) / current_distance
            )
            steps.append(
                IterationStep(
                    iteration=iteration,
                    sdecho_rank=rank,
                    predicate=candidate.predicate,
                    sdecho_result=candidate,
                    distance_before=current_distance,
                    search_distance_before=candidate.distance_before,
                    search_distance_after_removal=candidate.distance_after_removal,
                    distance_after_balancing=new_distance,
                    conditional_balance_reduction=conditional_reduction,
                    original_scale_path_increment=original_scale_path_increment,
                    cumulative_reduction=cumulative_reduction,
                    source_sequence_before=current_sequence.copy(),
                    source_sequence_after=new_sequence.copy(),
                    target_sequence=target_sequence.copy(),
                    balance_diagnostics=diagnostics,
                )
            )
            selected.append(candidate.predicate)
            current_weights = new_weights
            current_sequence = new_sequence
            current_distance = new_distance

            if current_distance <= distance_tolerance:
                stopping_reason = "remaining distance is within tolerance"
                break

    return IterativeSDEchoResult(
        initial_source_sequence=initial_source,
        target_sequence=target_sequence,
        initial_distance=initial_distance,
        steps=tuple(steps),
        final_source_sequence=current_sequence,
        final_distance=current_distance,
        final_source_weights=current_weights,
        stopping_reason=stopping_reason,
        balance_index=tuple(balance_buckets),
        search_index=tuple(search_buckets),
        candidate_pool_size=candidate_pool_size,
        rejections=tuple(rejections),
    )
