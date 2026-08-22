"""Sequential counterfactual SDEcho for aggregate-sequence explanation."""

from src.balancing import (
    BalanceDiagnostics,
    BalanceError,
    calibrate_predicate_events,
    predicate_balance_table,
    predicate_support_table,
)
from src.data_loader import get_bucket_index, load_and_preprocess_data, split_groups
from src.iterative_sdecho import (
    IterationStep,
    IterativeSDEchoResult,
    CandidateRejection,
    rejection_table,
    result_table,
    run_iterative_sdecho,
)
from src.predicates import Predicate, enumerate_predicates, predicate_mask
from src.sdecho import SDEchoResult, run_sdecho, sequence_distance
from src.sequence_builder import (
    EmptyBucketError,
    build_sequence,
    build_weighted_mean_sequence,
    effective_sample_size,
    select_material_buckets,
)
from src.visualization import plot_distance_path, plot_sequence_path, sequence_path_table

__version__ = "2.0.0"

__all__ = [
    "BalanceDiagnostics",
    "BalanceError",
    "CandidateRejection",
    "EmptyBucketError",
    "IterationStep",
    "IterativeSDEchoResult",
    "Predicate",
    "SDEchoResult",
    "build_sequence",
    "build_weighted_mean_sequence",
    "calibrate_predicate_events",
    "effective_sample_size",
    "enumerate_predicates",
    "get_bucket_index",
    "load_and_preprocess_data",
    "predicate_balance_table",
    "predicate_support_table",
    "predicate_mask",
    "plot_distance_path",
    "plot_sequence_path",
    "result_table",
    "rejection_table",
    "run_iterative_sdecho",
    "run_sdecho",
    "sequence_distance",
    "sequence_path_table",
    "select_material_buckets",
    "split_groups",
]
