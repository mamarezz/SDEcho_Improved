import numpy as np
import pandas as pd
import pytest

from src.balancing import (
    BalanceError,
    calibrate_predicate_events,
    predicate_balance_table,
    predicate_support_table,
)
from src.predicates import Predicate
from synthetic_data.generator import (
    BUCKETS,
    EXPECTED_PREDICATES,
    GROUP_COL,
    generate_iterative_ground_truth,
    split_synthetic_groups,
)


def test_single_predicate_is_balanced_inside_every_bucket():
    source, target = split_synthetic_groups(generate_iterative_ground_truth())
    weights, diagnostics = calibrate_predicate_events(
        source,
        target,
        [EXPECTED_PREDICATES[0]],
        group_col=GROUP_COL,
        index=BUCKETS,
    )
    table = predicate_balance_table(
        source,
        target,
        weights,
        None,
        [EXPECTED_PREDICATES[0]],
        GROUP_COL,
        BUCKETS,
    )

    assert diagnostics.converged
    assert diagnostics.max_balance_error <= 1e-10
    assert table["absolute_error"].max() <= 1e-10
    np.testing.assert_allclose(table["source_prevalence"], 0.30)
    np.testing.assert_allclose(table["target_prevalence"], 0.30)


def _rows_from_joint_counts(counts):
    rows = []
    for (z1, z2), count in counts.items():
        rows.extend({"Bucket": "b", "Z1": z1, "Z2": z2} for _ in range(count))
    return pd.DataFrame(rows)


def test_cyclic_calibration_preserves_correlated_prior_constraints():
    source = _rows_from_joint_counts({(0, 0): 50, (0, 1): 10, (1, 0): 10, (1, 1): 30})
    target = _rows_from_joint_counts({(0, 0): 30, (0, 1): 20, (1, 0): 20, (1, 1): 30})
    predicates = [Predicate({"Z1": 1}), Predicate({"Z2": 1})]

    weights, diagnostics = calibrate_predicate_events(
        source,
        target,
        predicates,
        group_col="Bucket",
        index=["b"],
    )
    table = predicate_balance_table(
        source, target, weights, None, predicates, "Bucket", ["b"]
    )

    assert diagnostics.iterations > 1
    assert diagnostics.max_balance_error <= 1e-10
    assert weights.sum() == pytest.approx(len(source))
    np.testing.assert_allclose(table["source_prevalence"], [0.5, 0.5], atol=1e-10)


def test_per_bucket_support_rejects_zero_target_event_cell():
    source = pd.DataFrame({"Bucket": ["b"] * 4, "Z": [1, 1, 0, 0]})
    target = pd.DataFrame({"Bucket": ["b"] * 4, "Z": [0, 0, 0, 0]})

    with pytest.raises(BalanceError, match="lacks event support"):
        calibrate_predicate_events(
            source,
            target,
            [Predicate({"Z": 1})],
            group_col="Bucket",
            index=["b"],
            min_event_count_per_bucket=1,
        )


def test_weighted_support_rejects_effectively_single_row_event():
    source = pd.DataFrame({"Bucket": ["b"] * 8, "Z": [1] * 4 + [0] * 4})
    target = source.copy()
    concentrated = np.array([1.0, 1e-6, 1e-6, 1e-6, 1.0, 1.0, 1.0, 1.0])

    with pytest.raises(BalanceError, match="insufficient event ESS"):
        calibrate_predicate_events(
            source,
            target,
            [Predicate({"Z": 1})],
            group_col="Bucket",
            index=["b"],
            base_source_weights=concentrated,
            min_event_count_per_bucket=4,
            min_active_event_count_per_bucket=4,
            min_event_ess_per_bucket=2.0,
        )


def test_duplicate_balance_buckets_are_rejected():
    source = pd.DataFrame({"Bucket": ["b", "b"], "Z": [0, 1]})
    with pytest.raises(ValueError, match="duplicate buckets"):
        calibrate_predicate_events(
            source,
            source.copy(),
            [Predicate({"Z": 1})],
            group_col="Bucket",
            index=["b", "b"],
        )


def test_empty_diagnostic_tables_keep_stable_csv_schemas():
    source = pd.DataFrame({"Bucket": ["b"], "Z": [0]})

    balance = predicate_balance_table(
        source, source.copy(), None, None, [], "Bucket", ["b"]
    )
    support = predicate_support_table(
        source, source.copy(), None, None, [], "Bucket", ["b"]
    )

    assert balance.empty
    assert "bucket" in balance.columns
    assert support.empty
    assert {
        "source_active_complement_count",
        "target_active_complement_count",
        "source_complement_ess",
        "target_complement_ess",
    }.issubset(support.columns)
