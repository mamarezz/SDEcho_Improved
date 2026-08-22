import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.balancing import predicate_balance_table
from src.iterative_sdecho import rejection_table, result_table, run_iterative_sdecho
from src.predicates import Predicate
from src.sdecho import run_sdecho
from synthetic_data.benchmark import run_synthetic_benchmark, write_synthetic_artifacts
from synthetic_data.generator import (
    BUCKETS,
    EXPECTED_DISTANCE_PATH,
    EXPECTED_PREDICATES,
    GROUP_COL,
    CANDIDATE_ATTRIBUTES,
    MEASURE_COL,
    generate_iterative_ground_truth,
    split_synthetic_groups,
)


def test_iterative_method_recovers_exact_order_and_distance_path():
    benchmark = run_synthetic_benchmark()
    result = benchmark.result

    assert result.predicates == EXPECTED_PREDICATES
    actual_path = [result.initial_distance] + [
        step.distance_after_balancing for step in result.steps
    ]
    np.testing.assert_allclose(actual_path, EXPECTED_DISTANCE_PATH, atol=1e-8)
    gap_path = [
        result.initial_source_sequence - result.target_sequence,
        *[
            step.source_sequence_after - step.target_sequence
            for step in result.steps
        ],
    ]
    np.testing.assert_allclose(
        gap_path,
        [
            [-25_000.0] * 4,
            [-13_000.0] * 4,
            [-6_000.0] * 4,
            [-2_000.0] * 4,
        ],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        [step.original_scale_path_increment for step in result.steps],
        [0.48, 0.28, 0.16],
        atol=1e-12,
    )
    assert result.cumulative_reduction == pytest.approx(0.92)
    np.testing.assert_allclose(
        [step.distance_after_removal for step in result.steps],
        [26_000.0, 12_000.0, 4_000.0],
    )
    np.testing.assert_allclose(
        [step.sdecho_result.gamma for step in result.steps],
        [0.728, 0.7384615384615385, 0.5833333333333334],
    )
    assert result.stopping_reason == (
        "no candidate among the top 6 weighted SDEcho results produced a "
        "feasible material balance reduction"
    )


def test_final_weights_preserve_every_previous_bucket_constraint():
    benchmark = run_synthetic_benchmark()
    source, target = split_synthetic_groups(benchmark.dataset)
    table = predicate_balance_table(
        source,
        target,
        benchmark.result.final_source_weights,
        None,
        list(EXPECTED_PREDICATES),
        GROUP_COL,
        BUCKETS,
    )

    assert len(table) == len(BUCKETS) * len(EXPECTED_PREDICATES)
    assert table["absolute_error"].max() <= 1e-10
    assert benchmark.result.steps[-1].balance_diagnostics.max_weight == pytest.approx(12.0)
    assert (
        benchmark.result.steps[-1]
        .balance_diagnostics.minimum_bucket_effective_sample_size_ratio
        == pytest.approx(0.4153846153846154)
    )


def test_ground_truth_json_and_reporting_table_match_executable_specification():
    specification = json.loads(
        (Path("synthetic_data") / "ground_truth.json").read_text(encoding="utf-8")
    )
    benchmark = run_synthetic_benchmark()
    table = result_table(benchmark.result)

    assert specification["expected_euclidean_distance_path"] == list(
        EXPECTED_DISTANCE_PATH
    )
    assert specification["expected_bucket_gap_path"] == [25000.0, 13000.0, 6000.0, 2000.0]
    assert specification["expected_predicate_order"] == [
        dict(predicate.conditions) for predicate in EXPECTED_PREDICATES
    ]
    assert specification["rows_per_group_per_bucket"] == 200
    assert specification["number_of_buckets"] == len(BUCKETS)
    assert specification["irreducible_target_shift_per_bucket"] == 2_000.0
    counts = (
        benchmark.dataset.groupby(["ComparisonGroup", GROUP_COL], observed=True)
        .size()
        .to_numpy()
    )
    np.testing.assert_array_equal(counts, 200)
    assert table["predicate"].tolist() == [str(p) for p in EXPECTED_PREDICATES]
    assert table["cumulative_reduction"].iloc[-1] == pytest.approx(0.92)


def test_final_weighted_search_has_candidates_but_no_material_removal_gain():
    benchmark = run_synthetic_benchmark()
    source, target = split_synthetic_groups(benchmark.dataset)
    rankings = run_sdecho(
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
        source_weights=benchmark.result.final_source_weights,
        excluded_predicates=benchmark.result.predicates,
    )

    assert rankings
    assert max(abs(candidate.removal_reduction) for candidate in rankings) <= 1e-12

    terminal_rejections = rejection_table(benchmark.result).query("iteration == 4")
    assert len(terminal_rejections) == 3
    assert terminal_rejections["reason"].eq(
        "removal reduction is below the configured minimum"
    ).all()


def test_synthetic_benchmark_is_exactly_reproducible():
    first = run_synthetic_benchmark()
    second = run_synthetic_benchmark()

    pd.testing.assert_frame_equal(first.dataset, second.dataset)
    pd.testing.assert_frame_equal(result_table(first.result), result_table(second.result))
    pd.testing.assert_frame_equal(
        rejection_table(first.result), rejection_table(second.result)
    )
    np.testing.assert_array_equal(
        first.result.final_source_weights, second.result.final_source_weights
    )
    assert first.result.predicates == second.result.predicates
    assert first.result.stopping_reason == second.result.stopping_reason


def test_selective_bucket_path_searches_and_changes_only_frozen_buckets():
    source, target = split_synthetic_groups(generate_iterative_ground_truth())
    intervention_buckets = ["3-5", "6-10"]
    result = run_iterative_sdecho(
        source,
        target,
        group_col=GROUP_COL,
        measure_col=MEASURE_COL,
        index=BUCKETS,
        balance_index=intervention_buckets,
        candidate_attrs=CANDIDATE_ATTRIBUTES,
        max_order=1,
        max_iterations=3,
        candidate_pool_size=6,
        max_values_per_attr=None,
        min_support=10,
        min_support_per_group=5,
        min_active_support_per_group=5,
        min_match_ess=5.0,
        min_removal_reduction=0.001,
        min_balance_reduction=0.001,
        min_effective_sample_size_ratio=0.4,
        min_bucket_effective_sample_size_ratio=0.4,
        max_weight=15.0,
        min_event_count_per_bucket=5,
        min_complement_count_per_bucket=5,
    )

    assert result.search_index == tuple(intervention_buckets)
    assert result.balance_index == tuple(intervention_buckets)
    assert result.predicates == EXPECTED_PREDICATES
    np.testing.assert_array_equal(
        result.final_source_sequence[[0, 3]], result.initial_source_sequence[[0, 3]]
    )
    outside = ~source[GROUP_COL].isin(intervention_buckets).to_numpy()
    np.testing.assert_array_equal(result.final_source_weights[outside], 1.0)
    assert all(
        step.sdecho_result.distance_before
        == pytest.approx(
            np.linalg.norm(
                (
                    step.source_sequence_before - step.target_sequence
                )[[1, 2]]
            )
        )
        for step in result.steps
    )


def _fallback_fixture():
    rows_source = []
    rows_target = []
    source_counts = {(0, 0): 48, (0, 1): 32, (1, 0): 12, (1, 1): 8}
    target_counts = {(0, 0): 64, (0, 1): 16, (1, 0): 16, (1, 1): 4}
    for (z1, z2), count in source_counts.items():
        rows_source.extend(
            {"Bucket": "b", "Z1": z1, "Z2": z2, "Y": 100 * z1 + 100 * z2}
            for _ in range(count)
        )
    for (z1, z2), count in target_counts.items():
        rows_target.extend(
            {"Bucket": "b", "Z1": z1, "Z2": z2, "Y": 100 * z2}
            for _ in range(count)
        )
    return pd.DataFrame(rows_source), pd.DataFrame(rows_target)


def test_search_rejects_rank_one_zero_gain_and_accepts_rank_two():
    source, target = _fallback_fixture()
    result = run_iterative_sdecho(
        source,
        target,
        group_col="Bucket",
        measure_col="Y",
        index=["b"],
        candidate_attrs=["Z1", "Z2"],
        max_order=1,
        max_iterations=1,
        candidate_pool_size=4,
        max_values_per_attr=None,
        min_support=1,
        min_support_per_group=1,
        min_active_support_per_group=1,
        min_match_ess=1.0,
        min_removal_reduction=0.0,
        min_balance_reduction=0.001,
    )

    assert result.steps[0].predicate == Predicate({"Z2": 1})
    assert result.steps[0].sdecho_rank == 2
    assert result.rejections[0].predicate == Predicate({"Z1": 1})
    assert result.rejections[0].reason == (
        "balance update is not a material positive reduction"
    )
    assert result.initial_distance == pytest.approx(40.0)
    assert result.final_distance == pytest.approx(20.0)


def test_synthetic_artifact_writer_creates_tables_and_figures(tmp_path: Path):
    artifacts = write_synthetic_artifacts(tmp_path)

    assert set(artifacts) == {
        "dataset",
        "iterations",
        "rejections",
        "sequence_path",
        "final_balance",
        "final_support",
        "final_weights",
        "order_recovery",
        "bucket_gap_recovery",
        "weight_profiles",
        "summary",
        "distance_png",
        "distance_svg",
        "sequence_png",
        "sequence_svg",
        "recovery_png",
        "recovery_svg",
        "bucket_gap_png",
        "bucket_gap_svg",
        "weights_png",
        "weights_svg",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts.values())
    assert artifacts["recovery_png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    summary = json.loads(artifacts["summary"].read_text(encoding="utf-8"))
    assert summary["exact_order_recovered"] is True
    np.testing.assert_allclose(
        summary["recovered_distance_path"], EXPECTED_DISTANCE_PATH, atol=1e-8
    )
    order = pd.read_csv(artifacts["order_recovery"])
    gaps = pd.read_csv(artifacts["bucket_gap_recovery"])
    weights = pd.read_csv(artifacts["weight_profiles"])
    assert order["exact_predicate_match"].all()
    assert order["absolute_distance_error"].max() <= 1e-8
    assert gaps["absolute_gap_error"].max() <= 1e-8
    assert len(weights) == 32
    assert weights["absolute_weight_error"].max() <= 1e-10


def test_incomplete_recovery_is_reported_without_crashing_or_claiming_exactness(
    tmp_path: Path,
):
    partial = run_synthetic_benchmark(max_iterations=2)
    artifacts = write_synthetic_artifacts(tmp_path, partial)
    summary = json.loads(artifacts["summary"].read_text(encoding="utf-8"))
    order = pd.read_csv(artifacts["order_recovery"])

    assert summary["exact_order_recovered"] is False
    assert summary["distance_path_length_matches"] is False
    assert summary["exact_distance_path_recovered"] is False
    assert summary["maximum_absolute_distance_error"] is None
    assert len(order) == 4
    assert pd.isna(order.loc[order["step"] == 3, "recovered_predicate"]).all()
    assert artifacts["recovery_png"].stat().st_size > 0
