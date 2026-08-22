"""Run Sequential Counterfactual SDEcho on Stack Overflow 2022 data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.balancing import predicate_balance_table, predicate_support_table
from src.data_loader import get_bucket_index, load_and_preprocess_data, split_groups
from src.iterative_sdecho import rejection_table, result_table, run_iterative_sdecho
from src.sequence_builder import build_weighted_mean_sequence, select_material_buckets
from src.visualization import plot_distance_path, plot_sequence_path, sequence_path_table


CONFIG = {
    "data_path": "data/stackoverflow2022.csv",
    "output_dir": "iterative_results",
    "subgroup_col": "AgeGroup",
    "source_group": "25-34",
    "target_group": "35-44",
    "group_col": "YearsExpBucket",
    "measure_col": "ConvertedCompYearly",
    "candidate_attrs": ["EdLevel", "RemoteWork", "Country"],
    "max_order": 2,
    "max_iterations": 5,
    "candidate_pool_size": 20,
    "max_values_per_attr": 10,
    "min_support": 20,
    "min_support_per_group": 5,
    "min_active_support_per_group": 5,
    "min_match_ess": 5.0,
    "min_event_count_per_bucket": 5,
    "min_complement_count_per_bucket": 5,
    "min_active_event_count_per_bucket": 5,
    "min_active_complement_count_per_bucket": 5,
    "min_event_ess_per_bucket": 5.0,
    "min_complement_ess_per_bucket": 5.0,
    "min_removal_reduction": 0.01,
    "min_balance_reduction": 0.005,
    "distance_tolerance": 1.0,
    "min_effective_sample_size_ratio": 0.20,
    "min_bucket_effective_sample_size_ratio": 0.20,
    "max_weight": 25.0,
    "balance_bucket_policy": "material",
    "bucket_relative_to_max_gap": 0.18,
    "bucket_min_absolute_gap": 10_000.0,
    "bucket_min_percentage_gap": 5.0,
}


def _print_result(result) -> None:
    print("\nSequential explanation path")
    print("-" * 90)
    if not result.steps:
        print("No predicate was accepted.")
    else:
        table = result_table(result)
        columns = [
            "iteration",
            "sdecho_rank",
            "predicate",
            "distance_before",
            "search_distance_after_removal",
            "distance_after_balancing",
            "original_scale_path_increment",
            "cumulative_reduction",
            "effective_sample_size_ratio",
            "minimum_bucket_effective_sample_size_ratio",
            "max_weight",
        ]
        print(table[columns].to_string(index=False))
    print(f"\nInitial distance: {result.initial_distance:,.2f}")
    print(f"Final distance:   {result.final_distance:,.2f}")
    print(f"Cumulative reduction: {result.cumulative_reduction:.2%}")
    print(f"Frozen balance buckets: {list(result.balance_index)}")
    print(f"Stopping reason: {result.stopping_reason}")
    print("Interpretation: ordered descriptive standardization, not a causal effect.")


def main(config: dict | None = None):
    """Execute the real-data iterative pipeline and write transparent artifacts."""
    settings = {**CONFIG, **(config or {})}
    dataset = load_and_preprocess_data(settings["data_path"], {
        "group_col": settings["group_col"],
        "subgroup_col": settings["subgroup_col"],
        "measure_col": settings["measure_col"],
    })
    source, target = split_groups(
        dataset,
        settings["subgroup_col"],
        settings["source_group"],
        settings["target_group"],
    )
    source = source.reset_index(drop=True)
    target = target.reset_index(drop=True)
    if source.empty or target.empty:
        raise ValueError("source and target groups must both contain rows")

    index = get_bucket_index(dataset, settings["group_col"])
    initial_source_sequence = build_weighted_mean_sequence(
        source, None, settings["group_col"], settings["measure_col"], index
    )
    target_sequence = build_weighted_mean_sequence(
        target, None, settings["group_col"], settings["measure_col"], index
    )
    if settings["balance_bucket_policy"] == "all":
        balance_index = list(index)
    elif settings["balance_bucket_policy"] == "material":
        balance_index = select_material_buckets(
            initial_source_sequence,
            target_sequence,
            index,
            relative_to_max_gap=settings["bucket_relative_to_max_gap"],
            min_absolute_gap=settings["bucket_min_absolute_gap"],
            min_percentage_gap=settings["bucket_min_percentage_gap"],
        )
    else:
        raise ValueError("balance_bucket_policy must be 'all' or 'material'")

    result = run_iterative_sdecho(
        source,
        target,
        group_col=settings["group_col"],
        measure_col=settings["measure_col"],
        index=index,
        candidate_attrs=settings["candidate_attrs"],
        max_order=settings["max_order"],
        max_iterations=settings["max_iterations"],
        candidate_pool_size=settings["candidate_pool_size"],
        max_values_per_attr=settings["max_values_per_attr"],
        min_support=settings["min_support"],
        min_support_per_group=settings["min_support_per_group"],
        min_active_support_per_group=settings["min_active_support_per_group"],
        min_match_ess=settings["min_match_ess"],
        min_removal_reduction=settings["min_removal_reduction"],
        min_balance_reduction=settings["min_balance_reduction"],
        distance_tolerance=settings["distance_tolerance"],
        min_effective_sample_size_ratio=settings[
            "min_effective_sample_size_ratio"
        ],
        min_bucket_effective_sample_size_ratio=settings[
            "min_bucket_effective_sample_size_ratio"
        ],
        max_weight=settings["max_weight"],
        min_event_count_per_bucket=settings["min_event_count_per_bucket"],
        min_complement_count_per_bucket=settings[
            "min_complement_count_per_bucket"
        ],
        min_active_event_count_per_bucket=settings[
            "min_active_event_count_per_bucket"
        ],
        min_active_complement_count_per_bucket=settings[
            "min_active_complement_count_per_bucket"
        ],
        min_event_ess_per_bucket=settings["min_event_ess_per_bucket"],
        min_complement_ess_per_bucket=settings[
            "min_complement_ess_per_bucket"
        ],
        balance_index=balance_index,
    )
    _print_result(result)

    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_table(result).to_csv(output_dir / "iterations.csv", index=False)
    rejection_table(result).to_csv(output_dir / "candidate_rejections.csv", index=False)
    sequence_path_table(result, index).to_csv(
        output_dir / "sequence_path.csv", index=False
    )
    balance = predicate_balance_table(
        source,
        target,
        result.final_source_weights,
        None,
        list(result.predicates),
        settings["group_col"],
        index,
    )
    balance["is_intervention_bucket"] = balance["bucket"].isin(
        result.balance_index
    )
    balance.to_csv(output_dir / "final_balance.csv", index=False)
    predicate_support_table(
        source,
        target,
        result.final_source_weights,
        None,
        list(result.predicates),
        settings["group_col"],
        list(result.balance_index),
    ).to_csv(output_dir / "final_support.csv", index=False)

    summary = {
        "source_group": settings["source_group"],
        "target_group": settings["target_group"],
        "buckets": index,
        "balance_buckets": list(result.balance_index),
        "search_buckets": list(result.search_index),
        "initial_distance": result.initial_distance,
        "final_distance": result.final_distance,
        "cumulative_reduction": result.cumulative_reduction,
        "predicates": [str(predicate) for predicate in result.predicates],
        "stopping_reason": result.stopping_reason,
        "interpretation": (
            "Ordered, path-dependent descriptive standardization; not a causal effect."
        ),
        "resolved_config": settings,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    distance_figure = plot_distance_path(result)
    distance_figure.savefig(output_dir / "distance_path.png", dpi=180)
    plt.close(distance_figure)
    sequence_figure = plot_sequence_path(
        result,
        index,
        source_label=settings["source_group"],
        target_label=settings["target_group"],
    )
    sequence_figure.savefig(output_dir / "sequence_path.png", dpi=180)
    plt.close(sequence_figure)
    return result


if __name__ == "__main__":
    main()
