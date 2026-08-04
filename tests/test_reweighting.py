import numpy as np
import pandas as pd

from src.predicates import Predicate
from src.reweighting import (
    GapDecompositionResult,
    ReweightingDiagnostics,
    audit_predicate_reweighting,
    compute_gap_decomposition,
    sequential_gap_decomposition,
    weighted_aggregate_sequence,
)


def _dummy_diagnostics():
    return ReweightingDiagnostics(
        attrs=["country"],
        n_source_rows=10,
        n_dropped_rows=0,
        pct_dropped_rows=0.0,
        n_cells_source_total=2,
        n_cells_no_target_overlap=0,
        n_cells_below_min_support=0,
        n_cells_valid=2,
        min_cell_support=1,
        max_weight=1.0,
        min_weight=1.0,
    )


def _gap_result(source_orig, source_cf, target, d_orig, d_cf):
    return GapDecompositionResult(
        predicate=Predicate({"country": "USA"}),
        attrs=["country"],
        s_source_orig=np.array(source_orig, dtype=float),
        s_source_cf=np.array(source_cf, dtype=float),
        s_target=np.array(target, dtype=float),
        d_orig=d_orig,
        d_cf=d_cf,
        explained_fraction=(d_orig - d_cf) / d_orig,
        residual_gap=d_cf,
        diagnostics=_dummy_diagnostics(),
    )


def test_gap_decomposition_reweights_all_buckets_by_default():
    df_source = pd.DataFrame(
        {
            "bucket": ["x", "x", "y", "y"],
            "country": ["a", "b", "a", "b"],
            "outcome": [0.0, 100.0, 0.0, 100.0],
        }
    )
    df_target = pd.DataFrame(
        {
            "bucket": ["x", "x", "y", "y"],
            "country": ["a", "b", "b", "b"],
            "outcome": [50.0, 50.0, 75.0, 75.0],
        }
    )

    result = compute_gap_decomposition(
        df_source,
        df_target,
        Predicate({"country": "b"}),
        group_col="bucket",
        measure_col="outcome",
        index=["x", "y"],
        min_cell_support=1,
    )

    assert result.reweighted_buckets == [0, 1]
    assert np.allclose(result.s_source_cf, [75.0, 75.0])


def test_audit_predicate_reweighting_classifies_compositional_result():
    result = _gap_result(
        source_orig=[0.0, 0.0],
        source_cf=[8.0, 8.0],
        target=[10.0, 10.0],
        d_orig=10.0,
        d_cf=2.0,
    )

    audit = audit_predicate_reweighting(result, ["a", "b"])

    assert audit.category == "compositional"
    assert audit.n_buckets_improved == 2
    assert audit.n_buckets_worsened == 0
    assert audit.bucket_diagnostics[0]["status"] == "improved"
    assert "causes" not in audit.explanation.lower()


def test_audit_predicate_reweighting_classifies_gap_amplifying_result():
    result = _gap_result(
        source_orig=[8.0, 8.0],
        source_cf=[0.0, 0.0],
        target=[10.0, 10.0],
        d_orig=2.0,
        d_cf=10.0,
    )

    audit = audit_predicate_reweighting(result, ["a", "b"])

    assert audit.category == "gap-amplifying"
    assert audit.explained_fraction < 0
    assert audit.n_buckets_worsened == 2


def test_audit_predicate_reweighting_classifies_bucket_specific_result():
    result = _gap_result(
        source_orig=[0.0, 10.0],
        source_cf=[4.0, 20.0],
        target=[10.0, 10.0],
        d_orig=10.0,
        d_cf=9.6,
    )

    audit = audit_predicate_reweighting(
        result,
        ["a", "b"],
        meaningful_ef_threshold=0.05,
    )

    assert audit.category == "bucket-specific"
    assert audit.n_buckets_improved == 1
    assert audit.n_buckets_worsened == 1
    assert [b["status"] for b in audit.bucket_diagnostics] == ["improved", "worsened"]


def test_sequential_gap_decomposition_uses_cumulative_weights():
    df_source = pd.DataFrame(
        {
            "bucket": ["x"] * 8,
            "a": ["a0", "a0", "a0", "a0", "a1", "a1", "a1", "a1"],
            "b": ["b0", "b0", "b1", "b1", "b0", "b0", "b1", "b1"],
            "outcome": [0.0, 0.0, 10.0, 10.0, 100.0, 100.0, 110.0, 110.0],
        }
    )
    df_target = pd.DataFrame(
        {
            "bucket": ["x"] * 10,
            "a": ["a0", "a1", "a1", "a1", "a1", "a1", "a1", "a1", "a1", "a1"],
            "b": ["b0", "b0", "b0", "b1", "b1", "b1", "b1", "b1", "b1", "b1"],
            "outcome": [105.0] * 10,
        }
    )
    predicates = [Predicate({"a": "a1"}), Predicate({"b": "b1"})]

    result = sequential_gap_decomposition(
        df_source,
        df_target,
        predicates,
        group_col="bucket",
        measure_col="outcome",
        index=["x"],
        min_cell_support=1,
    )

    final_weights = result.steps[-1]["weights"]
    expected_sequence = weighted_aggregate_sequence(
        df_source, final_weights, "bucket", "outcome", ["x"]
    )

    assert np.isclose(result.steps[-1]["d_cf"], abs(expected_sequence[0] - 105.0))
    assert not np.isclose(result.steps[-1]["d_cf"], result.steps[1]["d_cf"])
