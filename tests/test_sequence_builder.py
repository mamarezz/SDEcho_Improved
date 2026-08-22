import numpy as np
import pandas as pd
import pytest

from src.sequence_builder import (
    EmptyBucketError,
    build_weighted_mean_sequence,
    effective_sample_size,
    select_material_buckets,
)


def test_weighted_mean_sequence_uses_positional_weights():
    frame = pd.DataFrame(
        {
            "Bucket": ["b1", "b1", "b2", "b2"],
            "Outcome": [10.0, 30.0, 20.0, 40.0],
        },
        index=[10, 20, 30, 40],
    )
    result = build_weighted_mean_sequence(
        frame,
        np.array([3.0, 1.0, 1.0, 3.0]),
        "Bucket",
        "Outcome",
        ["b1", "b2"],
    )
    np.testing.assert_allclose(result, [15.0, 35.0])


def test_weighted_mean_sequence_rejects_empty_bucket():
    frame = pd.DataFrame({"Bucket": ["b1", "b2"], "Outcome": [1.0, 2.0]})
    with pytest.raises(EmptyBucketError):
        build_weighted_mean_sequence(
            frame,
            np.array([1.0, 0.0]),
            "Bucket",
            "Outcome",
            ["b1", "b2"],
        )


def test_effective_sample_size_is_kish_ess():
    assert effective_sample_size(np.ones(4)) == pytest.approx(4.0)
    assert effective_sample_size(np.array([1.0, 3.0])) == pytest.approx(1.6)


def test_material_bucket_selection_freezes_only_main_meaningful_gaps():
    source = np.array([135156.0, 131692.0, 173414.0, 205397.0])
    target = np.array([123917.0, 194521.0, 212527.0, 197010.0])

    assert select_material_buckets(
        source, target, ["0-2", "3-5", "6-10", "10-20"]
    ) == ["3-5", "6-10"]
