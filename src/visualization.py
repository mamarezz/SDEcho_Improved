"""Reporting helpers for an iterative explanation path."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.iterative_sdecho import IterativeSDEchoResult


def sequence_path_table(
    result: IterativeSDEchoResult,
    index: list[str],
) -> pd.DataFrame:
    """Return every source counterfactual sequence beside the fixed target."""
    rows: list[dict[str, object]] = []
    source_sequences = [result.initial_source_sequence] + [
        step.source_sequence_after for step in result.steps
    ]
    labels = ["Initial source"] + [
        f"After {step.iteration}: {step.predicate}" for step in result.steps
    ]
    for label, sequence in zip(labels, source_sequences):
        for bucket, value, target in zip(index, sequence, result.target_sequence):
            rows.append(
                {
                    "stage": label,
                    "bucket": bucket,
                    "source_value": float(value),
                    "target_value": float(target),
                    "gap": float(value - target),
                }
            )
    return pd.DataFrame(rows)


def plot_distance_path(result: IterativeSDEchoResult):
    """Plot residual distance after each accepted predicate balance."""
    distances = [result.initial_distance] + [
        step.distance_after_balancing for step in result.steps
    ]
    labels = ["Initial"] + [str(step.predicate) for step in result.steps]
    figure, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(distances))
    axis.plot(positions, distances, marker="o", linewidth=2.5, color="#2f6f9f")
    axis.fill_between(positions, distances, alpha=0.12, color="#2f6f9f")
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylabel("Remaining sequence distance")
    axis.set_title("Sequential Counterfactual SDEcho: residual distance")
    axis.grid(axis="y", alpha=0.25)
    for position, value in zip(positions, distances):
        axis.annotate(
            f"{value:,.0f}",
            (position, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    figure.tight_layout()
    return figure


def plot_sequence_path(
    result: IterativeSDEchoResult,
    index: list[str],
    *,
    source_label: str = "Source",
    target_label: str = "Target",
):
    """Plot the original, iterative counterfactual, and target sequences."""
    figure, axis = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(index))
    axis.plot(
        positions,
        result.target_sequence,
        marker="o",
        linewidth=3,
        color="#b23a48",
        label=target_label,
    )
    axis.plot(
        positions,
        result.initial_source_sequence,
        marker="o",
        linestyle="--",
        color="#777777",
        label=f"{source_label}: initial",
    )
    palette = plt.cm.viridis(np.linspace(0.25, 0.85, max(len(result.steps), 1)))
    for color, step in zip(palette, result.steps):
        axis.plot(
            positions,
            step.source_sequence_after,
            marker="o",
            linewidth=2,
            color=color,
            label=f"After {step.iteration}: {step.predicate}",
        )
    axis.set_xticks(positions, index)
    axis.set_xlabel("Sequence bucket")
    axis.set_ylabel("Weighted mean outcome")
    axis.set_title("Sequential predicate-event balancing")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure
