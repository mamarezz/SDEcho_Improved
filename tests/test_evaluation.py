from src.evaluation import generate_synthetic_dataset
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
