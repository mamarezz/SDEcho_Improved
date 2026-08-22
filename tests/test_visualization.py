import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.visualization import plot_distance_path, plot_sequence_path, sequence_path_table
from synthetic_data.benchmark import run_synthetic_benchmark
from synthetic_data.generator import BUCKETS


def test_iterative_reporting_contains_every_stage_and_bucket():
    result = run_synthetic_benchmark().result
    table = sequence_path_table(result, BUCKETS)

    assert len(table) == (len(result.steps) + 1) * len(BUCKETS)
    assert table["stage"].nunique() == len(result.steps) + 1

    distance_figure = plot_distance_path(result)
    sequence_figure = plot_sequence_path(result, BUCKETS)
    assert len(distance_figure.axes) == 1
    assert len(sequence_figure.axes) == 1
    plt.close(distance_figure)
    plt.close(sequence_figure)
