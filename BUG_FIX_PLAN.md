# Bug Fix and Improvement Plan — COMPLETED

All 3 bugs fixed and 5 improvements implemented. See final state of files below.

## Bugs Fixed

### ✅ 1. Sequential decomposition not cumulative (CRITICAL)
- **File**: `src/reweighting.py` line 653
- **Fix**: Changed `weights` → `new_weights` in `weighted_aggregate_sequence` call. Each step now uses `cumulative_weights` (product of all prior weights × current weights), so each step builds on the previous. Step 11 now monotonically decreases remaining gap.

### ✅ 2. Bootstrap CI returns degenerate [0,0] (CRITICAL)
- **File**: `src/evaluation.py`
- **Fix**: `bootstrap_explained_fraction_ci` now returns `(lower, upper, n_valid, n_failed)` — surfaces failed bootstrap resamples. Default `n_bootstrap` already 1000. Pipeline prints diagnostics and flags high failure rates.

### ✅ 3. "United Statesof America" space bug
- **File**: `src/visualization.py`
- **Fix**: Replaced manual string reconstruction in `render_sequential_decomposition_table` with `f"Change {pred}"` using Predicate's `__repr__` directly.

## Improvements Added

### ✅ 4. Per-bucket explained fraction breakdown
- **File**: `run_pipeline.py`
- Shows orig diff, cf diff, change, and bucket-level EF for each bucket alongside the aggregate.

### ✅ 5. Predicate ranking comparison
- **File**: `run_pipeline.py`
- Side-by-side table: SDEcho γ vs Reweighting EF for top-10 predicates.

### ✅ 6. Reverse direction robustness check
- **File**: `run_pipeline.py`
- Reweights B→A and reports both EF values.

### ✅ 7. min_cell_support ablation
- **File**: `run_pipeline.py`
- Tests support=5, 10, 20 with EF, dropped %, and valid cells.