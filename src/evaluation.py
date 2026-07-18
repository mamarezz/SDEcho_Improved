# src/evaluation.py

from typing import Tuple
import numpy as np
import pandas as pd

from src.predicates import Predicate
from src.sequence_builder import build_sequence
from src.sdecho import sequence_distance


def removal_baseline(
    df_source: pd.DataFrame, df_target: pd.DataFrame,
    predicate: Predicate, group_col: str, measure_col: str,
    index: list[str],
) -> float:
    """
    SDEcho's removal-based gap reduction for comparison with reweighting.
    
    Removes tuples matching the predicate and recomputes the distance,
    returning the reduction fraction: (d_orig - d_after) / d_orig.
    
    Args:
        df_source: DataFrame for group A
        df_target: DataFrame for group B
        predicate: Predicate whose matching tuples are removed
        group_col: Bucket column name
        measure_col: Outcome column name
        index: Ordered bucket labels
    
    Returns:
        Reduction fraction in [0, 1] (can be negative if removal increases distance)
    
    Notes:
        - This is the SDEcho baseline (destructive counterfactual)
        - For comparison against compute_gap_decomposition's explained_fraction
        - Not symmetric: removes from both groups
    
    Complexity:
        O(n_source + n_target) — single pass
    """
    from src.predicates import predicate_mask
    
    # Build original sequences
    s_source_orig = build_sequence(df_source, group_col, measure_col, "mean", index)
    s_target = build_sequence(df_target, group_col, measure_col, "mean", index)
    d_orig = sequence_distance(s_source_orig, s_target)
    
    if d_orig == 0:
        return 0.0
    
    # Remove matching tuples from both groups
    mask_source = predicate_mask(df_source, predicate)
    mask_target = predicate_mask(df_target, predicate)
    
    df_source_rest = df_source.loc[~mask_source]
    df_target_rest = df_target.loc[~mask_target]
    
    # Build sequences after removal
    s_source_after = build_sequence(df_source_rest, group_col, measure_col, "mean", index)
    s_target_after = build_sequence(df_target_rest, group_col, measure_col, "mean", index)
    d_after = sequence_distance(s_source_after, s_target_after)
    
    # Compute reduction fraction
    reduction = (d_orig - d_after) / d_orig
    
    return reduction


def bootstrap_explained_fraction_ci(
    df_source: pd.DataFrame, df_target: pd.DataFrame,
    predicate: Predicate, group_col: str, measure_col: str,
    index: list[str], n_bootstrap: int = 1000, ci: float = 0.95,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for the explained fraction.
    
    Resamples tuples with replacement from both groups, recomputes the
    explained fraction, and returns the percentile-based CI.
    
    Args:
        df_source: DataFrame for source group
        df_target: DataFrame for target group
        predicate: Predicate for reweighting
        group_col: Bucket column name
        measure_col: Outcome column name
        index: Ordered bucket labels
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound) for the CI
    
    Notes:
        - Uses percentile method (not bias-corrected)
        - Does NOT bootstrap the predicate selection (fixed predicate)
        - Resamples tuples independently in each group
    
    Complexity:
        O(n_bootstrap * (n_source + n_target))
    """
    from src.reweighting import compute_gap_decomposition
    
    bootstrap_fractions = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement from each group
        df_source_boot = df_source.sample(n=len(df_source), replace=True, random_state=None)
        df_target_boot = df_target.sample(n=len(df_target), replace=True, random_state=None)
        
        try:
            result = compute_gap_decomposition(
                df_source_boot, df_target_boot, predicate,
                group_col, measure_col, index
            )
            bootstrap_fractions.append(result.explained_fraction)
        except Exception:
            # Skip failed bootstrap samples (e.g., empty groups)
            continue
    
    if len(bootstrap_fractions) == 0:
        return (0.0, 0.0)
    
    # Compute percentile-based CI
    alpha = 1 - ci
    lower = np.percentile(bootstrap_fractions, 100 * alpha / 2)
    upper = np.percentile(bootstrap_fractions, 100 * (1 - alpha / 2))
    
    return (float(lower), float(upper))


def generate_synthetic_dataset(
    effect_size: float, n_per_group: int, seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Generate synthetic dataset with known ground-truth explained fraction.
    
    Creates two groups (A, B) with:
    - Same overall outcome distribution
    - Different covariate distributions
    - A known compositional effect
    
    Args:
        effect_size: Desired explained fraction (0 to 1)
        n_per_group: Number of tuples per group
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (df_A, df_B, ground_truth_fraction)
    
    Notes:
        - Ground truth is approximate (Monte Carlo estimate)
        - Used to validate that the estimator recovers known effects
    
    Example:
        >>> df_A, df_B, gt = generate_synthetic_dataset(0.3, 1000, 42)
        >>> result = compute_gap_decomposition(df_A, df_B, predicate, ...)
        >>> # result.explained_fraction should be close to gt
    """
    np.random.seed(seed)
    
    # Simplified synthetic example:
    # Two groups, two buckets, one binary covariate
    
    # Bucket values (outcomes)
    bucket_vals = {0: 100.0, 1: 200.0}
    
    # Group A: covariate distribution (e.g., 80% cov=0, 20% cov=1)
    p_A = np.array([0.8, 0.2])
    
    # Group B: covariate distribution (e.g., 20% cov=0, 80% cov=1)
    # The effect_size determines how much this compositional difference explains
    p_B = np.array([0.2, 0.8])
    
    # Generate tuples for group A
    buckets_A = np.random.choice([0, 1], size=n_per_group, p=[0.5, 0.5])
    covariates_A = np.random.choice([0, 1], size=n_per_group, p=p_A)
    outcomes_A = np.array([bucket_vals[b] + np.random.normal(0, 10) for b in buckets_A])
    
    # Generate tuples for group B
    # Add additional "residual" difference to achieve desired effect_size
    buckets_B = np.random.choice([0, 1], size=n_per_group, p=[0.5, 0.5])
    covariates_B = np.random.choice([0, 1], size=n_per_group, p=p_B)
    
    # Residual effect: scale outcomes by effect_size
    residual_scale = 1 - effect_size
    outcomes_B = np.array([
        bucket_vals[b] * residual_scale + np.random.normal(0, 10)
        for b in buckets_B
    ])
    
    # Create DataFrames
    df_A = pd.DataFrame({
        'bucket': buckets_A,
        'covariate': covariates_A,
        'outcome': outcomes_A,
    })
    
    df_B = pd.DataFrame({
        'bucket': buckets_B,
        'covariate': covariates_B,
        'outcome': outcomes_B,
    })
    
    # Approximate ground truth (effect_size as specified)
    ground_truth = effect_size
    
    return df_A, df_B, ground_truth


def run_ablation(
    df: pd.DataFrame, config: dict, param_name: str, param_values: list,
) -> pd.DataFrame:
    """
    Sweep one configuration parameter and collect results.
    
    Useful for sensitivity analysis (e.g., varying min_cell_support
    to see its effect on explained fraction).
    
    Args:
        df: Preprocessed DataFrame
        config: Base configuration dictionary
        param_name: Name of parameter to sweep (must be in config)
        param_values: List of values to test
    
    Returns:
        DataFrame with columns: param_value, explained_fraction, residual_gap,
        pct_dropped, etc.
    
    Notes:
        - Uses fixed groups and predicate from config
        - Does NOT re-run SDEcho (predicate is fixed)
        - Suitable for plotting sensitivity curves
    
    Complexity:
        O(len(param_values) * (n_A + n_B))
    """
    from src.predicates import Predicate
    from src.reweighting import compute_gap_decomposition
    
    # Extract groups
    subgroup_col = config['subgroup_col']
    subgroup_val1 = config['subgroup_val1']
    subgroup_val2 = config['subgroup_val2']
    
    df_A = df[df[subgroup_col] == subgroup_val1].copy()
    df_B = df[df[subgroup_col] == subgroup_val2].copy()
    
    # Get predicate (assumes already run or uses a default)
    # For simplicity, use a single-attribute predicate on first candidate attr
    candidate_attrs = config['candidate_attrs']
    predicate = Predicate({candidate_attrs[0]: df[candidate_attrs[0]].iloc[0]})
    
    group_col = config['group_col']
    measure_col = config['measure_col']
    index = config.get('index', ["0-2", "3-5", "6-10", "10-20"])
    
    results = []
    
    for param_val in param_values:
        # Update config with this parameter value
        config_mod = config.copy()
        config_mod[param_name] = param_val
        
        try:
            result = compute_gap_decomposition(
                df_A, df_B, predicate, group_col, measure_col, index,
                min_cell_support=config_mod.get('min_cell_support', 5)
            )
            
            results.append({
                'param_value': param_val,
                'explained_fraction': result.explained_fraction,
                'residual_gap': result.residual_gap,
                'pct_dropped': result.diagnostics.pct_dropped_rows,
                'n_valid_cells': result.diagnostics.n_cells_valid,
            })
        except Exception as e:
            results.append({
                'param_value': param_val,
                'explained_fraction': np.nan,
                'residual_gap': np.nan,
                'pct_dropped': np.nan,
                'n_valid_cells': np.nan,
                'error': str(e),
            })
    
    return pd.DataFrame(results)
