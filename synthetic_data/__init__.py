"""Deterministic ground-truth data for iterative SDEcho validation."""

from synthetic_data.generator import (
    BUCKETS,
    CANDIDATE_ATTRIBUTES,
    EXPECTED_BUCKET_GAP_PATH,
    EXPECTED_DISTANCE_PATH,
    EXPECTED_PREDICATES,
    generate_iterative_ground_truth,
    split_synthetic_groups,
)

__all__ = [
    "BUCKETS",
    "CANDIDATE_ATTRIBUTES",
    "EXPECTED_BUCKET_GAP_PATH",
    "EXPECTED_DISTANCE_PATH",
    "EXPECTED_PREDICATES",
    "generate_iterative_ground_truth",
    "split_synthetic_groups",
]
