"""End-to-end runner for the ordered synthetic ground truth."""

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.balancing import predicate_balance_table, predicate_support_table
from src.iterative_sdecho import (
    IterativeSDEchoResult,
    rejection_table,
    result_table,
    run_iterative_sdecho,
)
from src.visualization import plot_distance_path, plot_sequence_path, sequence_path_table
from synthetic_data.generator import (
    BUCKETS,
    CANDIDATE_ATTRIBUTES,
    EXPECTED_DISTANCE_PATH,
    EXPECTED_PREDICATES,
    GROUP_COL,
    MEASURE_COL,
    generate_iterative_ground_truth,
    split_synthetic_groups,
)
from synthetic_data.reporting import (
    bucket_gap_recovery_table,
    final_weight_profile_table,
    order_distance_recovery_table,
)
from synthetic_data.visualization import (
    plot_bucket_gap_heatmap,
    plot_final_weight_profiles,
    plot_recovery_summary,
)


@dataclass(frozen=True)
class SyntheticBenchmark:
    dataset: pd.DataFrame
    result: IterativeSDEchoResult


def run_synthetic_benchmark(max_iterations: int = 5) -> SyntheticBenchmark:
    """Run the iterative method with the frozen ground-truth configuration."""
    dataset = generate_iterative_ground_truth()
    source, target = split_synthetic_groups(dataset)
    result = run_iterative_sdecho(
        source,
        target,
        group_col=GROUP_COL,
        measure_col=MEASURE_COL,
        index=BUCKETS,
        candidate_attrs=CANDIDATE_ATTRIBUTES,
        max_order=1,
        max_iterations=max_iterations,
        candidate_pool_size=6,
        max_values_per_attr=None,
        min_support=20,
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
        balance_index=BUCKETS,
    )
    return SyntheticBenchmark(dataset=dataset, result=result)


def write_synthetic_artifacts(
    output_dir: str | Path,
    benchmark: SyntheticBenchmark | None = None,
) -> dict[str, Path]:
    """Write data, diagnostics, and thesis-ready figures for the benchmark."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    benchmark = benchmark or run_synthetic_benchmark()
    result = benchmark.result
    source, target = split_synthetic_groups(benchmark.dataset)

    artifacts = {
        "dataset": destination / "iterative_ground_truth.csv",
        "iterations": destination / "iterative_recovery.csv",
        "rejections": destination / "candidate_rejections.csv",
        "sequence_path": destination / "sequence_path.csv",
        "final_balance": destination / "final_balance.csv",
        "final_support": destination / "final_support.csv",
        "final_weights": destination / "final_source_weights.csv",
        "order_recovery": destination / "order_distance_recovery.csv",
        "bucket_gap_recovery": destination / "bucket_gap_recovery.csv",
        "weight_profiles": destination / "final_weight_profiles.csv",
        "summary": destination / "summary.json",
        "distance_png": destination / "distance_path.png",
        "distance_svg": destination / "distance_path.svg",
        "sequence_png": destination / "sequence_path.png",
        "sequence_svg": destination / "sequence_path.svg",
        "recovery_png": destination / "recovery_summary.png",
        "recovery_svg": destination / "recovery_summary.svg",
        "bucket_gap_png": destination / "bucket_gap_heatmap.png",
        "bucket_gap_svg": destination / "bucket_gap_heatmap.svg",
        "weights_png": destination / "final_weight_profiles.png",
        "weights_svg": destination / "final_weight_profiles.svg",
    }

    benchmark.dataset.to_csv(artifacts["dataset"], index=False)
    result_table(result).to_csv(artifacts["iterations"], index=False)
    rejection_table(result).to_csv(artifacts["rejections"], index=False)
    sequence_path_table(result, BUCKETS).to_csv(
        artifacts["sequence_path"], index=False
    )
    predicate_balance_table(
        source,
        target,
        result.final_source_weights,
        None,
        list(result.predicates),
        GROUP_COL,
        BUCKETS,
    ).to_csv(artifacts["final_balance"], index=False)
    predicate_support_table(
        source,
        target,
        result.final_source_weights,
        None,
        list(result.predicates),
        GROUP_COL,
        BUCKETS,
    ).to_csv(artifacts["final_support"], index=False)
    pd.DataFrame(
        {
            "RowID": source["RowID"].to_numpy(),
            "final_source_weight": result.final_source_weights,
        }
    ).to_csv(artifacts["final_weights"], index=False)
    order_recovery = order_distance_recovery_table(result)
    bucket_recovery = bucket_gap_recovery_table(result)
    weight_profiles = final_weight_profile_table(benchmark.dataset, result)
    order_recovery.to_csv(artifacts["order_recovery"], index=False)
    bucket_recovery.to_csv(artifacts["bucket_gap_recovery"], index=False)
    weight_profiles.to_csv(artifacts["weight_profiles"], index=False)

    recovered_path = [result.initial_distance] + [
        step.distance_after_balancing for step in result.steps
    ]
    distance_path_length_matches = len(recovered_path) == len(EXPECTED_DISTANCE_PATH)
    maximum_distance_error = (
        max(
            abs(expected - recovered)
            for expected, recovered in zip(EXPECTED_DISTANCE_PATH, recovered_path)
        )
        if distance_path_length_matches
        else None
    )
    summary = {
        "expected_predicates": [str(predicate) for predicate in EXPECTED_PREDICATES],
        "recovered_predicates": [str(predicate) for predicate in result.predicates],
        "exact_order_recovered": result.predicates == EXPECTED_PREDICATES,
        "expected_distance_path": list(EXPECTED_DISTANCE_PATH),
        "recovered_distance_path": recovered_path,
        "distance_path_length_matches": distance_path_length_matches,
        "maximum_absolute_distance_error": maximum_distance_error,
        "exact_distance_path_recovered": bool(
            distance_path_length_matches
            and maximum_distance_error is not None
            and maximum_distance_error <= 1e-8
        ),
        "original_scale_path_increments": [
            step.original_scale_path_increment for step in result.steps
        ],
        "cumulative_reduction": result.cumulative_reduction,
        "remaining_residual_share": result.final_distance / result.initial_distance,
        "stopping_reason": result.stopping_reason,
        "rows": len(benchmark.dataset),
        "buckets": BUCKETS,
    }
    artifacts["summary"].write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figures = {
        "distance": plot_distance_path(result),
        "sequence": plot_sequence_path(
            result,
            BUCKETS,
            source_label="Synthetic source",
            target_label="Synthetic target",
        ),
        "recovery": plot_recovery_summary(
            result, EXPECTED_PREDICATES, EXPECTED_DISTANCE_PATH
        ),
        "bucket_gap": plot_bucket_gap_heatmap(bucket_recovery, BUCKETS),
        "weights": plot_final_weight_profiles(weight_profiles),
    }
    for name, figure in figures.items():
        figure.savefig(artifacts[f"{name}_png"], dpi=180)
        svg_path = artifacts[f"{name}_svg"]
        figure.savefig(svg_path)
        svg_text = svg_path.read_text(encoding="utf-8")
        svg_path.write_text(
            "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
            encoding="utf-8",
        )
        plt.close(figure)
    return artifacts


if __name__ == "__main__":
    benchmark = run_synthetic_benchmark()
    output_directory = Path(__file__).resolve().parent / "results"
    artifacts = write_synthetic_artifacts(output_directory, benchmark)
    print(result_table(benchmark.result).to_string(index=False))
    print(f"\nStopping reason: {benchmark.result.stopping_reason}")
    print(f"Results written to: {output_directory}")
    print(f"Main figure: {artifacts['recovery_png']}")
