from src.evaluation import bootstrap_explained_fraction_ci, generate_synthetic_dataset
from src.predicates import Predicate
from src.reweighting import compute_gap_decomposition


def test_synthetic_dataset_recovers_calibrated_explained_fraction():
    df_source, df_target, ground_truth = generate_synthetic_dataset(
        effect_size=0.5,
        n_per_group=20000,
        seed=7,
    )

    result = compute_gap_decomposition(
        df_source,
        df_target,
        Predicate({"covariate": 1}),
        group_col="bucket",
        measure_col="outcome",
        index=[0, 1],
        min_cell_support=5,
    )

    assert abs(result.explained_fraction - ground_truth) < 0.05


def test_bootstrap_handles_resampled_duplicate_indices():
    """Bootstrap resampling must not fail because replacement duplicates rows."""
    df_source, df_target, _ = generate_synthetic_dataset(
        effect_size=0.5,
        n_per_group=1_000,
        seed=11,
    )

    lower, upper, n_valid, n_failed = bootstrap_explained_fraction_ci(
        df_source,
        df_target,
        Predicate({"covariate": 1}),
        group_col="bucket",
        measure_col="outcome",
        index=[0, 1],
        n_bootstrap=20,
        min_cell_support=5,
        random_state=3,
        return_diagnostics=True,
    )

    assert n_valid == 20
    assert n_failed == 0
    assert lower <= upper
