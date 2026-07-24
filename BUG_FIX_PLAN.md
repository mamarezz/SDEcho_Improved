# Bug Fix and Improvement Plan

Based on analysis of `src/reweighting.py`, `src/evaluation.py`, `src/visualization.py`, and `run_pipeline.py`.

## Bugs Found

### 1. Bootstrap CI returns degenerate [0, 0] (CRITICAL)
- **File**: `src/evaluation.py`, function `bootstrap_explained_fraction_ci`
- **Problem**: Silently catches exceptions with `continue`; no diagnostics printed. Many resamples may fail due to common support loss.
- **Fix**: Surface `n_valid`, `n_failed`, `pct_failed` in output. Increase default `n_bootstrap` to 1000. Don't let it silently return [0,0].

### 2. Sequential decomposition is NOT cumulative (CRITICAL)
- **File**: `src/reweighting.py`, function `sequential_gap_decomposition` (lines 646-680)
- **Problem**: Counterfactual sequence `s_source_cf` is computed from `weights` alone (the current predicate's weights), NOT from `cumulative_weights`. Each step independently reweights from scratch.
- **Fix**: Use `cumulative_weights` (after multiplication) to compute `s_source_cf`, so each step builds on the previous.

### 3. "United Statesof America" missing space
- **File**: `src/visualization.py`, `render_sequential_decomposition_table` line 288
- **Problem**: Predicate `__repr__` shows `Country=United States of America`, but the intervention text shows `Change Country=United Statesof America` — the `&` join is eating the space.
- **Fix**: Use proper join in `render_sequential_decomposition_table`.

## High-Value Improvements

### 4. Per-bucket explained fraction reporting
- Show how the aggregate 23.11% decomposes across individual buckets

### 5. Rank all predicates by reweighting-explained-fraction
- Compare SDEcho's gamma ranking vs actual reweighting-based explained fraction

### 6. Reverse direction robustness check
- Reweight B toward A as well as A toward B

### 7. Synthetic ground-truth validation
- Run validation to show estimator recovers known values

### 8. min_cell_support ablation
- Test support=5, 10, 20 to show sensitivity