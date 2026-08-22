import numpy as np
import pandas as pd
import pytest

from src.balancing import calibrate_predicate_events
from src.sdecho import run_sdecho
from synthetic_data.generator import (
    BUCKETS,
    CANDIDATE_ATTRIBUTES,
    EXPECTED_PREDICATES,
    GROUP_COL,
    MEASURE_COL,
    generate_iterative_ground_truth,
    split_synthetic_groups,
)


def _search(source, target, source_weights=None, excluded=()):
    return run_sdecho(
        source,
        target,
        group_col=GROUP_COL,
        measure_col=MEASURE_COL,
        index=BUCKETS,
        candidate_attrs=CANDIDATE_ATTRIBUTES,
        max_order=1,
        k=6,
        max_values_per_attr=None,
        min_support=20,
        min_support_per_group=5,
        source_weights=source_weights,
        excluded_predicates=excluded,
    )


def test_unit_weight_sdecho_recovers_first_ground_truth_predicate():
    source, target = split_synthetic_groups(generate_iterative_ground_truth())
    result = _search(source, target)[0]

    assert result.predicate == EXPECTED_PREDICATES[0]
    assert result.distance_before == pytest.approx(50_000.0)
    assert result.distance_after_removal == pytest.approx(26_000.0)
    assert result.removal_reduction == pytest.approx(0.48)
    assert result.gamma == pytest.approx(0.728)


def test_weighted_rerun_discovers_second_predicate_without_mutation():
    source, target = split_synthetic_groups(generate_iterative_ground_truth())
    source_before = source.copy(deep=True)
    weights, _ = calibrate_predicate_events(
        source,
        target,
        [EXPECTED_PREDICATES[0]],
        group_col=GROUP_COL,
        index=BUCKETS,
    )

    result = _search(
        source,
        target,
        source_weights=weights,
        excluded=[EXPECTED_PREDICATES[0]],
    )[0]

    assert result.predicate == EXPECTED_PREDICATES[1]
    assert result.distance_before == pytest.approx(26_000.0)
    assert result.distance_after_removal == pytest.approx(12_000.0)
    assert np.isfinite(result.gamma)
    assert source.equals(source_before)


def test_search_support_and_penalty_are_restricted_to_search_buckets():
    source = pd.DataFrame(
        {
            "Bucket": ["inside"] * 4 + ["outside"] * 20,
            "Z": [1, 0, 0, 0] + [1] * 20,
            "Y": [10.0, 0.0, 0.0, 0.0] + [999.0] * 20,
        }
    )
    target = pd.DataFrame(
        {
            "Bucket": ["inside"] * 4 + ["outside"] * 20,
            "Z": [1, 1, 0, 0] + [0] * 20,
            "Y": [10.0, 10.0, 0.0, 0.0] + [-999.0] * 20,
        }
    )

    results = run_sdecho(
        source,
        target,
        group_col="Bucket",
        measure_col="Y",
        index=["inside"],
        candidate_attrs=["Z"],
        max_order=1,
        k=2,
        max_values_per_attr=None,
        min_support=1,
    )
    result = next(
        candidate
        for candidate in results
        if candidate.predicate.conditions["Z"] == 1
    )

    assert result.source_support == 1
    assert result.target_support == 2
    assert result.source_weighted_share == pytest.approx(0.25)
    assert result.target_weighted_share == pytest.approx(0.50)
