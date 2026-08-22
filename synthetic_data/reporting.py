"""Ground-truth comparison tables for the iterative synthetic benchmark."""

import numpy as np
import pandas as pd

from src.iterative_sdecho import IterativeSDEchoResult
from src.visualization import sequence_path_table
from synthetic_data.generator import (
    BUCKETS,
    CANDIDATE_ATTRIBUTES,
    EXPECTED_BUCKET_GAP_PATH,
    EXPECTED_DISTANCE_PATH,
    EXPECTED_PREDICATES,
    GROUP_COL,
    split_synthetic_groups,
)


def order_distance_recovery_table(result: IterativeSDEchoResult) -> pd.DataFrame:
    """Compare expected and recovered predicates and full-vector distances."""
    recovered_distances = [result.initial_distance] + [
        step.distance_after_balancing for step in result.steps
    ]
    rows: list[dict[str, object]] = [
        {
            "step": 0,
            "expected_predicate": "Initial",
            "recovered_predicate": "Initial",
            "exact_predicate_match": True,
            "sdecho_rank": np.nan,
            "expected_distance": EXPECTED_DISTANCE_PATH[0],
            "recovered_distance": recovered_distances[0],
            "absolute_distance_error": abs(
                EXPECTED_DISTANCE_PATH[0] - recovered_distances[0]
            ),
            "original_scale_path_increment": 0.0,
            "cumulative_reduction": 0.0,
        }
    ]
    step_count = max(len(result.steps), len(EXPECTED_PREDICATES))
    for position in range(1, step_count + 1):
        step = result.steps[position - 1] if position <= len(result.steps) else None
        expected_predicate = (
            EXPECTED_PREDICATES[position - 1]
            if position <= len(EXPECTED_PREDICATES)
            else None
        )
        expected_distance = (
            EXPECTED_DISTANCE_PATH[position]
            if position < len(EXPECTED_DISTANCE_PATH)
            else np.nan
        )
        recovered_distance = (
            step.distance_after_balancing if step is not None else np.nan
        )
        rows.append(
            {
                "step": position,
                "expected_predicate": (
                    str(expected_predicate) if expected_predicate is not None else None
                ),
                "recovered_predicate": (
                    str(step.predicate) if step is not None else None
                ),
                "exact_predicate_match": bool(
                    step is not None
                    and expected_predicate is not None
                    and step.predicate == expected_predicate
                ),
                "sdecho_rank": step.sdecho_rank if step is not None else np.nan,
                "expected_distance": expected_distance,
                "recovered_distance": recovered_distance,
                "absolute_distance_error": (
                    abs(expected_distance - recovered_distance)
                    if np.isfinite(expected_distance)
                    and np.isfinite(recovered_distance)
                    else np.nan
                ),
                "original_scale_path_increment": (
                    step.original_scale_path_increment if step is not None else np.nan
                ),
                "cumulative_reduction": (
                    step.cumulative_reduction if step is not None else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def bucket_gap_recovery_table(result: IterativeSDEchoResult) -> pd.DataFrame:
    """Return signed source-minus-target gaps and their known oracle values."""
    table = sequence_path_table(result, BUCKETS)
    stages = list(dict.fromkeys(table["stage"]))
    stage_numbers = {stage: number for number, stage in enumerate(stages)}
    table["step"] = table["stage"].map(stage_numbers).astype(int)
    table["expected_signed_gap"] = table["step"].map(
        lambda step: (
            -EXPECTED_BUCKET_GAP_PATH[step]
            if step < len(EXPECTED_BUCKET_GAP_PATH)
            else np.nan
        )
    )
    table["absolute_gap_error"] = (
        table["gap"] - table["expected_signed_gap"]
    ).abs()
    return table[
        [
            "step",
            "stage",
            "bucket",
            "source_value",
            "target_value",
            "gap",
            "expected_signed_gap",
            "absolute_gap_error",
        ]
    ]


def final_weight_profile_table(
    dataset: pd.DataFrame,
    result: IterativeSDEchoResult,
) -> pd.DataFrame:
    """Compare recovered weights with exact target/source cell-share ratios."""
    source, target = split_synthetic_groups(dataset)
    profile_columns = [GROUP_COL, *CANDIDATE_ATTRIBUTES]
    weighted_source = source.copy()
    weighted_source["recovered_weight"] = result.final_source_weights

    source_profiles = (
        weighted_source.groupby(profile_columns, observed=True, sort=True)
        .agg(
            source_count=("RowID", "size"),
            recovered_mean_weight=("recovered_weight", "mean"),
            recovered_min_weight=("recovered_weight", "min"),
            recovered_max_weight=("recovered_weight", "max"),
            recovered_weighted_mass=("recovered_weight", "sum"),
        )
        .reset_index()
    )
    target_profiles = (
        target.groupby(profile_columns, observed=True, sort=True)
        .size()
        .rename("target_count")
        .reset_index()
    )
    profiles = source_profiles.merge(
        target_profiles, on=profile_columns, how="inner", validate="one_to_one"
    )
    source_bucket_count = profiles.groupby(GROUP_COL)["source_count"].transform(
        "sum"
    )
    target_bucket_count = profiles.groupby(GROUP_COL)["target_count"].transform(
        "sum"
    )
    profiles["expected_weight"] = (
        profiles["target_count"] / target_bucket_count
    ) / (profiles["source_count"] / source_bucket_count)
    profiles["absolute_weight_error"] = (
        profiles["recovered_mean_weight"] - profiles["expected_weight"]
    ).abs()
    profiles["profile"] = (
        profiles["Country"].astype(str)
        + " / "
        + profiles["Education"].astype(str)
        + " / "
        + profiles["RemoteWork"].astype(str)
    )
    return profiles[
        [
            GROUP_COL,
            "profile",
            *CANDIDATE_ATTRIBUTES,
            "source_count",
            "target_count",
            "expected_weight",
            "recovered_mean_weight",
            "recovered_min_weight",
            "recovered_max_weight",
            "recovered_weighted_mass",
            "absolute_weight_error",
        ]
    ]
