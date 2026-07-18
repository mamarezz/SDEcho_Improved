# Module: `evaluation.py`

**Pipeline stage:** Not a pipeline stage itself — this module consumes
outputs from `sdecho.py` and `reweighting.py` to produce the thesis's
empirical evidence: baseline comparisons, uncertainty quantification,
ground-truth validation, and sensitivity analysis.

**Status:** Newly specified in this doc; nothing here exists yet in code.
This is the natural next implementation step after `reweighting.py` is
tested and validated on real data.

---

## 1. Why This Module Is Needed

Every prior module produces a *number* (a distance, a gamma, an explained
fraction). None of them, on their own, tell you whether that number is
**meaningful** — whether it would look different by chance, whether it's
consistent with a known ground truth, or whether it's actually distinct from
what SDEcho's own mechanism already reports. This module exists specifically
to answer the question raised repeatedly in earlier review:
*"how do you know this pipeline adds value beyond SDEcho, and how do you
know it's even computing something correct?"* Per
`docs/00_PROJECT_CONCEPT.md` §4, the empirical contrast between
removal-based and reweighting-based gap explanation is the thesis's central
empirical claim — this module is where that claim gets tested.

## 2. Where It Fits in the Pipeline

```
sdecho.py                          reweighting.py
list[SDEchoResult]                 GapDecompositionResult
        │                                  │
        └────────────────┬─────────────────┘
                          ▼
                   evaluation.py
     removal_baseline_reduction()   (compares against Stage 5's
                                      OWN dist_after — no recomputation,
                                      see §3.1)
     bootstrap_explained_fraction_ci()
     generate_synthetic_dataset()
     run_ablation()
                          │
                          ▼
              results tables (pd.DataFrame)
                          │
                          ▼
                  visualization.py
```

## 3. Design Notes

### 3.1 `removal_baseline_reduction` reuses Stage 5's own numbers — no recomputation

SDEcho's `run_sdecho()` (Stage 5) already computes, for every candidate
predicate, `dist_before` and `dist_after` (the distance after **removing**
matching tuples — see `docs/06_MODULE_sdecho.md` §3.2). The
removal-based reduction fraction is therefore *already computed* the moment
Stage 5 finishes:

$$
\text{RemovalReduction}(P) = \frac{\text{dist\_before} - \text{dist\_after}}{\text{dist\_before}}
$$

This function does **not** re-run predicate masking or re-aggregate
sequences — it extracts this value directly from the `SDEchoResult` already
produced for the selected predicate. Recomputing it from raw DataFrames (as
an earlier draft implicitly assumed would be necessary) would duplicate
logic that `sdecho.py` already owns, violating the same
"single source of truth per computation" principle applied in
`docs/04_MODULE_sequence_builder.md` and `docs/07_MODULE_reweighting.md`.

This makes the central comparison of the thesis — removal-based reduction
vs. reweighting-based explained fraction, for the *same* predicate — a
direct, free-standing two-number comparison with no risk of the two metrics
being computed on subtly different data or bucket indices, since both trace
back to the same `index` (per `docs/06_MODULE_sdecho.md` §3.1) and the same
`SDEchoResult`.

### 3.2 Bootstrap resampling strategy

`bootstrap_explained_fraction_ci()` resamples **whole tuples with
replacement**, independently within `df_source` and `df_target` (a standard
two-sample bootstrap), then re-runs `compute_gap_decomposition()` on each
resample. This is **not** a stratified-by-cell bootstrap — it does not
guarantee every cell retains representation in every resample. Consequence:
some bootstrap iterations may produce `NaN` or undefined `explained_fraction`
values if a resample happens to eliminate a previously-valid cell's support
entirely. This must be handled explicitly (§5) — dropped and reported, not
silently averaged in as zero.

### 3.3 Synthetic ground truth is deliberately simple

`generate_synthetic_dataset()` constructs the simplest possible scenario
with an **analytically known** explained fraction: a single categorical
covariate with two cells, controlled per-cell outcome means, and controlled
per-group cell-mix proportions — directly parallel to the India/USA toy
example already used as the hand-computable test fixture in
`docs/00_PROJECT_CONCEPT.md` §5 and `docs/07_MODULE_reweighting.md` §10.
This keeps the synthetic validation traceable to a worked example a reader
can verify by hand, rather than a black-box data-generating process whose
correctness itself would need separate validation.

## 4. Public API

```python
def removal_baseline_reduction(result: SDEchoResult) -> float:
    """
    Extract SDEcho's own removal-based reduction fraction for a predicate
    — no recomputation, see §3.1.

    Args:
        result: the SDEchoResult for the predicate being compared
            (typically the one selected in Stage 6).

    Returns:
        (dist_before - dist_after) / dist_before, or NaN if dist_before=0.
    """


def bootstrap_explained_fraction_ci(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    predicate: Predicate,
    group_col: str,
    measure_col: str,
    agg_func: str,
    index: list[str],
    min_cell_support: int = 5,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, dict]:
    """
    Bootstrap confidence interval on the explained fraction (§3.2).

    Returns:
        (lower, upper, diagnostics) where diagnostics reports how many of
        the n_bootstrap resamples produced a valid (non-NaN) explained
        fraction, and the full resampled distribution for plotting.
    """


def generate_synthetic_dataset(
    p_source_cell1: float,
    p_target_cell1: float,
    mean_cell1: float,
    mean_cell2: float,
    n_per_group: int,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Generate a synthetic two-group, two-cell dataset with a known,
    analytically-computable ground-truth explained fraction (§3.3).

    Args:
        p_source_cell1, p_target_cell1: proportion of each group in
            "cell1" (vs. "cell2") — controls the compositional gap.
        mean_cell1, mean_cell2: the (noiseless, or low-noise) outcome mean
            within each cell — controls the residual/unexplained gap.
        n_per_group: sample size per group.
        seed: RNG seed for reproducibility.

    Returns:
        (df_source, df_target, true_explained_fraction) — the last value
        computed analytically from the input parameters, independent of
        the pipeline code, for validating compute_gap_decomposition's
        output against.
    """


def run_ablation(
    base_config: dict,
    param_name: str,
    param_values: list,
    pipeline_fn: Callable[[dict], GapDecompositionResult],
) -> pd.DataFrame:
    """
    Sweep one config parameter, holding all others fixed, and collect
    results for sensitivity analysis.

    Args:
        base_config: the full pipeline CONFIG (docs/02_ARCHITECTURE.md §5).
        param_name: the config key to vary (e.g., "min_cell_support").
        param_values: the values to sweep.
        pipeline_fn: a function that runs the full pipeline given a config
            dict and returns a GapDecompositionResult — supplied by the
            orchestration layer (thesis.ipynb), not owned by this module,
            so run_ablation has no knowledge of pipeline internals beyond
            the config interface.

    Returns:
        DataFrame with one row per param_value, columns for
        explained_fraction, residual_gap, pct_dropped_rows, and any other
        diagnostics worth plotting.
    """
```

## 5. Implementation

```python
"""
evaluation.py

Empirical validation layer: baseline comparison against SDEcho's own
removal-based metric, bootstrap uncertainty quantification, synthetic
ground-truth validation, and config-sensitivity ablations. This module
calls reweighting.py and sdecho.py functions; it does not reimplement
their logic (see docs/08_MODULE_evaluation.md §3 for why this matters).
"""

from typing import Callable

import numpy as np
import pandas as pd

from src.predicates import Predicate
from src.sdecho import SDEchoResult
from src.reweighting import compute_gap_decomposition, GapDecompositionResult


def removal_baseline_reduction(result: SDEchoResult) -> float:
    """See docstring in module interface (§4)."""
    if result.dist_before == 0:
        return float("nan")
    return (result.dist_before - result.dist_after) / result.dist_before


def bootstrap_explained_fraction_ci(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    predicate: Predicate,
    group_col: str,
    measure_col: str,
    agg_func: str,
    index: list[str],
    min_cell_support: int = 5,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, dict]:
    """See docstring in module interface (§4)."""
    rng = np.random.default_rng(seed)
    fractions = []
    n_failed = 0

    for _ in range(n_bootstrap):
        resampled_source = df_source.sample(
            n=len(df_source), replace=True, random_state=rng.integers(1e9)
        )
        resampled_target = df_target.sample(
            n=len(df_target), replace=True, random_state=rng.integers(1e9)
        )
        result = compute_gap_decomposition(
            resampled_source, resampled_target, predicate,
            group_col, measure_col, agg_func, index, min_cell_support,
        )
        if np.isnan(result.explained_fraction):
            n_failed += 1
            continue
        fractions.append(result.explained_fraction)

    if not fractions:
        raise RuntimeError(
            "All bootstrap resamples produced NaN explained fractions — "
            "min_cell_support is likely too strict relative to sample size."
        )

    alpha = 1 - ci
    lower = float(np.percentile(fractions, 100 * alpha / 2))
    upper = float(np.percentile(fractions, 100 * (1 - alpha / 2)))

    diagnostics = {
        "n_bootstrap": n_bootstrap,
        "n_valid": len(fractions),
        "n_failed": n_failed,
        "pct_failed": round(100 * n_failed / n_bootstrap, 2),
        "resampled_fractions": fractions,  # kept for plotting distributions
    }
    return lower, upper, diagnostics


def generate_synthetic_dataset(
    p_source_cell1: float,
    p_target_cell1: float,
    mean_cell1: float,
    mean_cell2: float,
    n_per_group: int,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """See docstring in module interface (§4)."""
    rng = np.random.default_rng(seed)

    def _make_group(p_cell1: float) -> pd.DataFrame:
        is_cell1 = rng.random(n_per_group) < p_cell1
        cell = np.where(is_cell1, "cell1", "cell2")
        # Noiseless outcome for exact analytical verification; a small
        # noise term can be added later as a robustness variant.
        outcome = np.where(is_cell1, mean_cell1, mean_cell2)
        return pd.DataFrame({
            "covariate": cell,
            "outcome": outcome,
            "bucket": "single",  # single-bucket sequence: reduces to a
                                  # scalar-mean comparison, matching the
                                  # hand-computable toy example exactly.
        })

    df_source = _make_group(p_source_cell1)
    df_target = _make_group(p_target_cell1)

    # Analytical ground truth, computed independently of pipeline code.
    mean_source = p_source_cell1 * mean_cell1 + (1 - p_source_cell1) * mean_cell2
    mean_target = p_target_cell1 * mean_cell1 + (1 - p_target_cell1) * mean_cell2
    # Counterfactual: source reweighted to target's cell mix — by
    # construction, this exactly equals mean_target when there is no
    # within-cell (residual) difference, i.e. mean_cell1/mean_cell2 are
    # the same in both groups (which they are here, by design).
    mean_source_cf = p_target_cell1 * mean_cell1 + (1 - p_target_cell1) * mean_cell2

    d_orig = abs(mean_source - mean_target)
    d_cf = abs(mean_source_cf - mean_target)
    true_explained_fraction = (
        (d_orig - d_cf) / d_orig if d_orig != 0 else float("nan")
    )

    return df_source, df_target, true_explained_fraction


def run_ablation(
    base_config: dict,
    param_name: str,
    param_values: list,
    pipeline_fn: Callable[[dict], GapDecompositionResult],
) -> pd.DataFrame:
    """See docstring in module interface (§4)."""
    rows = []
    for value in param_values:
        config = dict(base_config)
        config[param_name] = value
        result = pipeline_fn(config)
        rows.append({
            param_name: value,
            "explained_fraction": result.explained_fraction,
            "residual_gap": result.residual_gap,
            "pct_dropped_rows": result.diagnostics.pct_dropped_rows,
            "n_cells_valid": result.diagnostics.n_cells_valid,
        })
    return pd.DataFrame(rows)
```

## 6. Known Gaps / TODOs

- **`generate_synthetic_dataset` currently produces a *noiseless* synthetic
  outcome** (exact means, no within-cell variance) — this validates the
  *formula* is implemented correctly but does not test robustness to
  sampling noise. A noisy variant (e.g., `mean_cell + N(0, sigma)`) should
  be added as a second synthetic experiment once the noiseless version
  passes, to check how `explained_fraction` estimates degrade under
  realistic variance — not yet implemented.
- **Synthetic dataset uses a single bucket**, deliberately, to keep the
  ground truth hand-verifiable (§3.3). A multi-bucket synthetic extension
  (matching the real pipeline's structure more closely) is a natural
  follow-up once the single-bucket case is validated, not a blocker for it.
- **`run_ablation`'s `pipeline_fn` contract is loosely specified** — it
  assumes the caller (likely `thesis.ipynb` or a `run_pipeline.py` script)
  wires together Stages 1–7 given a config dict and returns a
  `GapDecompositionResult`. This orchestration function does not exist yet
  in any module — worth deciding whether it belongs in a new
  `pipeline.py` / `orchestration.py` or stays notebook-only. Flagged as an
  open architectural question for the next file in this doc set.
- **Bootstrap failure handling (§3.2) reports failures but does not yet
  investigate *why* they occur** (e.g., which specific cell lost support)
  — useful for diagnosing whether `min_cell_support` needs adjustment, not
  yet implemented as structured output.

## 7. How Reviewer #2 Would Critique This Module

- *"Your synthetic validation has zero within-cell noise — real survey
  data is nothing like this. What does passing this test actually prove?"*
  — Fair; it proves the **formula and code path are implemented correctly**
  (a necessary but not sufficient condition for the pipeline being useful
  on real data). This should be stated explicitly, not oversold as
  "validating the method on realistic data" — that's what the real SO
  survey experiments are for.
- *"A non-stratified bootstrap with a known failure mode (cells losing
  support under resampling) — why not use a stratified bootstrap instead,
  which would avoid this entirely?"* — A reasonable suggestion; the
  non-stratified choice was made for simplicity (matching the project's
  "no new algorithm" preference), but if failure rates turn out to be high
  in practice, switching to a per-cell stratified bootstrap is a modest,
  well-justified upgrade — worth trying both and reporting whichever is
  more defensible, once real numbers are available.

## 8. Complexity

- `removal_baseline_reduction`: $O(1)$ — pure field extraction.
- `bootstrap_explained_fraction_ci`: $O(\text{n\_bootstrap} \times (n_A + n_B))$
  — linear per resample, `n_bootstrap` resamples; dominant added cost in
  this module, but still cheap relative to Stage 5's search cost.
- `generate_synthetic_dataset`: $O(n_{\text{per\_group}})$.
- `run_ablation`: $O(|\text{param\_values}|) \times \text{cost(pipeline\_fn)}$
  — cost dominated by however many times the full pipeline (including
  Stage 5's SDEcho search) is re-run; this is the most expensive function
  in the module if `pipeline_fn` re-runs SDEcho on every sweep value rather
  than reusing a cached predicate where the swept parameter doesn't affect
  Stage 5 (e.g., sweeping `min_cell_support` alone does not require
  re-running SDEcho at all — an optimization worth applying once
  `pipeline_fn` exists, not before).

## 9. Tests to Write (`tests/test_evaluation.py`)

1. **`removal_baseline_reduction` correctness**: construct an `SDEchoResult`
   with known `dist_before`/`dist_after`, assert exact fraction.
2. **Synthetic ground-truth recovery**: the module's own headline
   correctness test — generate a synthetic dataset with known
   `true_explained_fraction`, run it through `compute_gap_decomposition`,
   assert the pipeline's estimate matches the analytical value within a
   small numerical tolerance (this is the most important test in the
   entire test suite, since it validates the full Stage 7-10 chain
   end-to-end against ground truth, not just individual functions in
   isolation).
3. **Bootstrap CI sanity**: on the same synthetic dataset, assert the
   bootstrap CI **contains** the true explained fraction at the expected
   rate (a basic coverage check, not a full calibration study).
4. **Bootstrap failure reporting**: construct a deliberately
   support-fragile dataset (a cell with count just barely above
   `min_cell_support`) and confirm `n_failed`/`pct_failed` reflect the
   expected instability, rather than silently returning a CI computed from
   a shrunken, unreported sample.
5. **`run_ablation` output shape**: confirm one row per `param_values`
   entry, correct column set, using a stub `pipeline_fn`.

## 10. Thesis-Ready Description

> To assess whether reweighting-based explained fraction differs
> systematically from SDEcho's own removal-based reduction metric, we
> compute both for the same selected predicate and report their difference
> across [N] group-pair comparisons. To quantify estimation uncertainty, we
> report a [95]% bootstrap confidence interval on the explained fraction,
> obtained by independently resampling each group with replacement over
> [1000] iterations; resamples that eliminate common support for a
> previously valid cell are excluded and their exclusion rate reported. To
> validate the correctness of the explained-fraction estimator
> independently of real-data noise, we additionally construct a synthetic
> two-group, two-cell dataset with an analytically known ground-truth
> explained fraction and confirm the pipeline recovers this value within
> numerical tolerance.
