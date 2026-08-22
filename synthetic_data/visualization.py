"""Figures that compare the ordered synthetic oracle with recovered results."""

import matplotlib.pyplot as plt
import numpy as np

from src.iterative_sdecho import IterativeSDEchoResult
from src.predicates import Predicate


def plot_recovery_summary(
    result: IterativeSDEchoResult,
    expected_predicates: tuple[Predicate, ...],
    expected_distance_path: tuple[float, ...],
):
    """Show oracle/recovered distance paths and original-scale contributions."""
    recovered_distances = [result.initial_distance] + [
        step.distance_after_balancing for step in result.steps
    ]
    expected = np.asarray(expected_distance_path, dtype=float)
    recovered = np.asarray(recovered_distances, dtype=float)
    stage_count = max(len(expected), len(recovered))
    positions = np.arange(stage_count)
    stage_labels = ["Initial"]
    for position in range(1, stage_count):
        expected_predicate = (
            expected_predicates[position - 1]
            if position <= len(expected_predicates)
            else None
        )
        recovered_predicate = (
            result.steps[position - 1].predicate
            if position <= len(result.steps)
            else None
        )
        if (
            expected_predicate == recovered_predicate
            and expected_predicate is not None
        ):
            detail = str(expected_predicate)
        elif expected_predicate is None:
            detail = f"Extra: {recovered_predicate}"
        elif recovered_predicate is None:
            detail = f"Expected: {expected_predicate}\nRecovered: missing"
        else:
            detail = (
                f"Expected: {expected_predicate}\nRecovered: {recovered_predicate}"
            )
        stage_labels.append(f"Step {position}\n{detail}")

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    distance_axis, contribution_axis = axes
    distance_axis.plot(
        np.arange(len(expected)),
        expected,
        marker="s",
        linestyle="--",
        linewidth=2.2,
        color="#d17a22",
        label="Ground truth",
    )
    distance_axis.plot(
        np.arange(len(recovered)),
        recovered,
        marker="o",
        linewidth=2.6,
        color="#2f6f9f",
        label="Recovered",
    )
    distance_axis.set_xticks(positions, stage_labels[:stage_count])
    distance_axis.set_ylabel("Remaining Euclidean distance")
    distance_axis.set_title("Expected vs recovered full-vector distance path")
    distance_axis.grid(axis="y", alpha=0.25)
    distance_axis.legend()
    for position, value in enumerate(recovered):
        distance_axis.annotate(
            f"{value:,.0f}",
            (position, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )

    expected_step_shares = (expected[:-1] - expected[1:]) / expected[0]
    recovered_step_shares = np.asarray(
        [step.original_scale_path_increment for step in result.steps], dtype=float
    )
    share_step_count = max(len(expected_step_shares), len(recovered_step_shares))
    expected_shares = np.full(share_step_count + 1, np.nan)
    recovered_shares = np.full(share_step_count + 1, np.nan)
    expected_shares[: len(expected_step_shares)] = expected_step_shares
    recovered_shares[: len(recovered_step_shares)] = recovered_step_shares
    expected_shares[-1] = expected[-1] / expected[0]
    recovered_shares[-1] = result.final_distance / result.initial_distance
    share_labels = [f"Step {i}" for i in range(1, share_step_count + 1)] + [
        "Residual"
    ]
    share_positions = np.arange(len(share_labels))
    width = 0.36
    contribution_axis.bar(
        share_positions - width / 2,
        expected_shares * 100,
        width,
        color="#d17a22",
        alpha=0.72,
        label="Ground truth",
    )
    contribution_axis.bar(
        share_positions + width / 2,
        recovered_shares * 100,
        width,
        color="#2f6f9f",
        alpha=0.86,
        label="Recovered",
    )
    contribution_axis.set_xticks(share_positions, share_labels)
    contribution_axis.set_ylabel("Share of original distance (%)")
    contribution_axis.set_title("Path increments and irreducible residual")
    contribution_axis.grid(axis="y", alpha=0.25)
    contribution_axis.legend()
    for position, value in enumerate(recovered_shares * 100):
        if not np.isfinite(value):
            continue
        contribution_axis.annotate(
            f"{value:.0f}%",
            (position + width / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
        )

    order_matches = result.predicates == expected_predicates
    distance_matches = len(expected) == len(recovered) and np.allclose(
        expected, recovered, atol=1e-8
    )
    figure.suptitle(
        "Iterative SDEcho synthetic validation — "
        f"exact oracle recovery: {'yes' if order_matches and distance_matches else 'no'}",
        fontsize=14,
    )
    figure.tight_layout()
    return figure


def plot_bucket_gap_heatmap(gap_table, buckets: list[str]):
    """Plot the recovered signed gap in every bucket after every step."""
    pivot = gap_table.pivot(index="step", columns="bucket", values="gap").reindex(
        columns=buckets
    )
    stage_labels = (
        gap_table[["step", "stage"]]
        .drop_duplicates()
        .sort_values("step")["stage"]
        .tolist()
    )
    values = pivot.to_numpy(dtype=float) / 1_000.0
    limit = float(np.max(np.abs(values)))
    figure, axis = plt.subplots(figsize=(9, 5.8))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(buckets)), buckets)
    axis.set_yticks(np.arange(len(stage_labels)), stage_labels)
    axis.set_xlabel("Sequence bucket")
    axis.set_title("Recovered source − target gap by iteration (thousands)")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{values[row, column]:,.0f}",
                ha="center",
                va="center",
                color="white" if abs(values[row, column]) > 0.55 * limit else "black",
                fontweight="bold",
            )
    figure.colorbar(image, ax=axis, label="Signed gap (thousands)")
    figure.tight_layout()
    return figure


def plot_final_weight_profiles(profile_table):
    """Plot expected and recovered final multipliers for each binary profile."""
    summary = (
        profile_table.groupby("profile", sort=True)
        .agg(
            expected_weight=("expected_weight", "mean"),
            recovered_weight=("recovered_mean_weight", "mean"),
        )
        .sort_values("expected_weight")
    )
    positions = np.arange(len(summary))
    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.barh(
        positions,
        summary["recovered_weight"],
        color="#2f6f9f",
        alpha=0.82,
        label="Recovered mean weight",
    )
    axis.scatter(
        summary["expected_weight"],
        positions,
        marker="D",
        s=55,
        facecolors="white",
        edgecolors="#d17a22",
        linewidths=2,
        label="Ground-truth multiplier",
        zorder=3,
    )
    axis.axvline(1.0, color="#555555", linestyle="--", linewidth=1.3)
    axis.set_xscale("log")
    axis.set_yticks(positions, summary.index)
    axis.set_xlabel("Final source weight (log scale)")
    axis.set_title(
        "Expected vs recovered target/source profile multipliers\n"
        "(joint recovery is specific to this independent positive control)"
    )
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure
