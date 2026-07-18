# Module: `reweighting.py`

**Pipeline stage:** Stage 6 (predicate selection) through Stage 10 (gap
decomposition reporting). **This is the thesis's actual contribution** — see
`docs/00_PROJECT_CONCEPT.md` §4 for the exact novelty framing.

**Status:** Formalizes the working `compute_gap_decomposition` draft against
the module structure fixed in files 3–7, with two consolidations applied
(§3) that remove redundant code the earlier draft had duplicated.

---

## 1. Why This Module Is Needed

Every module before this one (`data_loader`, `sequence_builder`,
`predicates`, `sdecho`) is either infrastructure or a reused, unmodified
component of SDEcho. This module is where SDEcho's diagnostic output
(*"this predicate explains the divergence"*) is turned into this thesis's
counterfactual, compositional claim (*"here is how much of the divergence is
attributable to differing distributions on that predicate's attributes, and
here is what remains"*) — the distinction formalized in
`docs/01_METHODOLOGY.md` §6–8.

## 2. Where It Fits in the Pipeline

```
sdecho.py                    predicates.py         sequence_builder.py
list[SDEchoResult]            Predicate.attrs        build_sequence()
        │                          │                       │
        └──────────┬───────────────┘                       │
                    ▼                                       │
             reweighting.py                                 │
        select_predicate()  (Stage 6)                       │
        compute_cell_weights()  (Stage 7)                   │
        compute_gap_decomposition()  (Stage 9-10) ──────────┘
                    │              (calls build_sequence()
                    ▼               internally for both
       GapDecompositionResult       original AND counterfactual
                    │               sequences — see §3.1)
                    ▼
       evaluation.py / visualization.py
```

## 3. Consolidations Applied Relative to the Earlier Draft

### 3.1 `weighted_aggregate_sequence()` removed — superseded

The earlier working draft of this module defined its own
`weighted_aggregate_sequence()` function, duplicating aggregation logic that
`sequence_builder.build_sequence()` was later designed (in
`docs/04_MODULE_sequence_builder.md`, §5–6) to handle for **both** the
unweighted (Stage 3) and weighted (Stage 8) cases via a single `weights`
parameter. That module fix was made specifically to prevent Stage 3 and
Stage 8 from silently diverging (see that doc's §1). Keeping a second,
separate `weighted_aggregate_sequence()` here would reintroduce exactly the
duplication that fix was meant to eliminate.

**Consolidation:** this module now calls
`sequence_builder.build_sequence(df, ..., weights=weights)` directly for the
counterfactual sequence, and `build_sequence(df, ..., weights=None)` for the
original — the *only* difference between Stage 3/4 and Stage 8/9 is which
DataFrame and which `weights` argument are passed in, not which function is
used to aggregate.

### 3.2 `get_covariate_attrs()` removed — superseded

The earlier draft defined a standalone `get_covariate_attrs(predicate)`
function. Since `predicates.py` (file 6) already defines
`Predicate.attrs` as a property, this standalone function is now redundant;
callers use `predicate.attrs` directly.

## 4. Assumptions Introduced

Fully specified in `docs/01_METHODOLOGY.md` §6, §10 — summarized here for
convenience, not re-derived:

1. Categorical, discrete covariates only (cell-based reweighting requires
   enumerable joint cells).
2. Common support required: cells absent from either group, or below
   `min_cell_support`, are trimmed (§6.3 of the methodology doc) — and, per
   `docs/06_MODULE_sdecho.md` §4.1, this per-group support standard is
   **stricter** than SDEcho's own combined-count `min_support` threshold, a
   discrepancy still open for resolution.
3. Ignorability limited to the predicate's attribute(s) `X` only — no claim
   about unmeasured confounders.
4. Weights are static and unstabilized by default (no capping) — deferred
   to an empirical robustness check in `evaluation.py`.
5. Reweighting direction (`source → target`) is configuration-determined,
   not automatically inferred, and is a genuinely non-symmetric operation
   (methodology doc §6.4) — the thesis must report which direction was used
   for every result, not assume it's obvious from context.

## 5. Public API

```python
def select_predicate(
    results: list[SDEchoResult], rank: int = 0
) -> Predicate:
    """
    Select a predicate from SDEcho's ranked output (Stage 6).

    Args:
        results: ranked list from sdecho.run_sdecho(), ascending by gamma.
        rank: 0-indexed rank to select (0 = top-1, strongest explanation).

    Returns:
        The selected Predicate.

    Raises:
        IndexError: if rank >= len(results) — fails loudly rather than
            silently returning a wrong predicate.
    """


def compute_cell_weights(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    attrs: list[str],
    min_cell_support: int = 5,
) -> tuple[pd.Series, ReweightingDiagnostics]:
    """
    Exact cell-based (joint-distribution) reweighting factors — see
    docs/01_METHODOLOGY.md §6 for the full mathematical specification.

    Args:
        df_source: group being reweighted.
        df_target: group whose distribution df_source is aligned to.
        attrs: covariate attributes, typically predicate.attrs (§3.2).
        min_cell_support: minimum per-group cell count required (§4.2).

    Returns:
        (weights, diagnostics) — weights is a pd.Series aligned to
        df_source.index, NaN for trimmed (insufficient-support) rows.
    """


def compute_gap_decomposition(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    predicate: Predicate,
    group_col: str,
    measure_col: str,
    agg_func: str,
    index: list[str],
    min_cell_support: int = 5,
) -> GapDecompositionResult:
    """
    Full Stage 7-10 orchestration: reweight, recompute the counterfactual
    sequence (via sequence_builder.build_sequence — see §3.1), and report
    the gap decomposition.

    Args:
        df_source, df_target: the two comparison groups.
        predicate: the SDEcho predicate selected in Stage 6.
        group_col, measure_col, agg_func, index: as used throughout;
            `index` must be the SAME bucket index used in Stage 3-5 (see
            docs/06_MODULE_sdecho.md §3.1 for why this consistency matters).
        min_cell_support: passed through to compute_cell_weights.

    Returns:
        GapDecompositionResult with original/counterfactual sequences,
        distances, explained fraction, residual gap, and diagnostics.
    """
```

## 6. Implementation

```python
"""
reweighting.py

Stage 6-10 of the pipeline: predicate selection, exact cell-based
reweighting, counterfactual sequence recomputation, and gap decomposition
reporting. THIS MODULE IS THE THESIS'S ACTUAL CONTRIBUTION — see
docs/00_PROJECT_CONCEPT.md for the precise novelty claim.

This is a counterfactual STATISTICAL REWEIGHTING analysis, not a causal
intervention. See docs/00_PROJECT_CONCEPT.md §7 for the disclaimer that
must accompany every reported result.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.predicates import Predicate
from src.sdecho import SDEchoResult, sequence_distance
from src.sequence_builder import build_sequence


@dataclass(frozen=True)
class ReweightingDiagnostics:
    attrs: list
    n_source_rows: int
    n_dropped_rows: int
    pct_dropped_rows: float
    n_cells_source_total: int
    n_cells_no_target_overlap: int
    n_cells_below_min_support: int
    n_cells_valid: int
    min_cell_support: int
    max_weight: float
    min_weight: float


@dataclass(frozen=True)
class GapDecompositionResult:
    predicate: Predicate
    attrs: list
    s_source_orig: np.ndarray
    s_source_cf: np.ndarray
    s_target: np.ndarray
    d_orig: float
    d_cf: float
    explained_fraction: float
    residual_gap: float
    diagnostics: ReweightingDiagnostics


def select_predicate(results: list[SDEchoResult], rank: int = 0) -> Predicate:
    """See docstring in module interface (§5)."""
    if rank >= len(results):
        raise IndexError(
            f"Requested rank {rank} but only {len(results)} SDEcho "
            f"results are available."
        )
    return results[rank].predicate


def _build_joint_cell_key(df: pd.DataFrame, attrs: list[str]) -> pd.Series:
    """Combine one or more attributes into a single joint-cell key per row."""
    if len(attrs) == 1:
        return df[attrs[0]]
    return df[attrs].apply(tuple, axis=1)


def compute_cell_weights(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    attrs: list[str],
    min_cell_support: int = 5,
) -> tuple[pd.Series, ReweightingDiagnostics]:
    """See docstring in module interface (§5)."""
    source_cells = _build_joint_cell_key(df_source, attrs)
    target_cells = _build_joint_cell_key(df_target, attrs)

    n_source, n_target = len(df_source), len(df_target)
    source_counts = source_cells.value_counts()
    target_counts = target_cells.value_counts()
    source_props = source_counts / n_source
    target_props = target_counts / n_target

    cells_in_source = set(source_counts.index)
    cells_in_target = set(target_counts.index)
    common_cells = cells_in_source & cells_in_target

    valid_cells = {
        cell for cell in common_cells
        if source_counts[cell] >= min_cell_support
        and target_counts[cell] >= min_cell_support
    }

    weight_map = {
        cell: float(target_props[cell] / source_props[cell])
        for cell in valid_cells
    }
    weights = source_cells.map(weight_map)

    n_dropped = int(weights.isna().sum())
    diagnostics = ReweightingDiagnostics(
        attrs=attrs,
        n_source_rows=n_source,
        n_dropped_rows=n_dropped,
        pct_dropped_rows=round(100 * n_dropped / n_source, 2) if n_source else float("nan"),
        n_cells_source_total=len(source_counts),
        n_cells_no_target_overlap=len(cells_in_source - cells_in_target),
        n_cells_below_min_support=len(common_cells - valid_cells),
        n_cells_valid=len(valid_cells),
        min_cell_support=min_cell_support,
        max_weight=float(np.nanmax(list(weight_map.values()))) if weight_map else float("nan"),
        min_weight=float(np.nanmin(list(weight_map.values()))) if weight_map else float("nan"),
    )
    return weights, diagnostics


def compute_gap_decomposition(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    predicate: Predicate,
    group_col: str,
    measure_col: str,
    agg_func: str,
    index: list[str],
    min_cell_support: int = 5,
) -> GapDecompositionResult:
    """See docstring in module interface (§5)."""
    attrs = predicate.attrs  # §3.2 consolidation: no separate helper needed

    weights, diagnostics = compute_cell_weights(
        df_source, df_target, attrs, min_cell_support
    )

    # §3.1 consolidation: build_sequence() handles BOTH the unweighted
    # original and the weighted counterfactual — same function, same code
    # path, differing only in the weights argument.
    s_source_orig = build_sequence(df_source, group_col, measure_col, agg_func, index)
    s_target = build_sequence(df_target, group_col, measure_col, agg_func, index)
    s_source_cf = build_sequence(df_source, group_col, measure_col, agg_func, index, weights=weights)

    d_orig = sequence_distance(s_source_orig, s_target)
    d_cf = sequence_distance(s_source_cf, s_target)
    explained_fraction = (d_orig - d_cf) / d_orig if d_orig != 0 else float("nan")

    return GapDecompositionResult(
        predicate=predicate,
        attrs=attrs,
        s_source_orig=s_source_orig,
        s_source_cf=s_source_cf,
        s_target=s_target,
        d_orig=d_orig,
        d_cf=d_cf,
        explained_fraction=explained_fraction,
        residual_gap=d_cf,
        diagnostics=diagnostics,
    )
```

## 7. Known Gaps / TODOs

- **`min_support` (Stage 5) vs. `min_cell_support` (Stage 7) inconsistency**
  — inherited from `docs/06_MODULE_sdecho.md` §4.1/§7; unresolved, needs a
  decision before final results.
- **No weight capping yet** (§4.4) — deferred to `evaluation.py`'s
  robustness checks, per `docs/00_PROJECT_CONCEPT.md` §6.
- **No bootstrap CI yet** — `compute_gap_decomposition` returns a point
  estimate only; `evaluation.py`'s `bootstrap_explained_fraction_ci()`
  (specified in `docs/02_ARCHITECTURE.md` §4) will call this function
  repeatedly on resampled data, not duplicate its logic — confirming the
  consolidation principle from §3 extends to the evaluation layer too.
- **`explained_fraction` can be negative or exceed 1** (methodology doc §8)
  — currently no warning is raised when this occurs; consider adding one
  here (analogous to the bucket-exclusion warning added in
  `sequence_builder.py`) so it's surfaced immediately during experimentation
  rather than discovered later while writing up results.

## 8. How Reviewer #2 Would Critique This Module

- *"You reweight on the joint cell of the predicate's attributes, but
  Stage 5's predicate search only checked combined support across both
  groups — how many of your 'reweighted' predicates would have been
  rejected by Stage 7's own, stricter per-group standard, had it been
  applied at discovery time?"* — A real, checkable question: report how
  often Stage 5's top predicate gets **heavily trimmed** in Stage 7 (high
  `pct_dropped_rows`), since that indicates the predicate SDEcho considered
  "strong" may have poor common support for the compositional question this
  thesis actually asks.
- *"Direction of reweighting is a config choice with real consequences —
  did you check whether explained fraction differs meaningfully in the
  reverse direction?"* — Not yet tested; a cheap addition to
  `evaluation.py`'s ablations (§4 of the methodology doc already documents
  this asymmetry theoretically, but it hasn't been empirically shown).

## 9. Complexity

$O(n_A + n_B)$ overall for Stage 7-10 (methodology doc §9) — the module
adds effectively no asymptotic cost beyond what `sequence_builder.py` and
`sdecho.py` already contribute; SDEcho's brute-force search (Stage 5)
remains the dominant cost in the full pipeline.

## 10. Tests to Write (`tests/test_reweighting.py`)

1. **Hand-computable toy example** — the India/USA example from
   `docs/00_PROJECT_CONCEPT.md` §5, worked through exactly: assert
   `compute_cell_weights` produces weights `0.25` and `4.0` for the
   respective cells, and `compute_gap_decomposition` produces
   `d_orig=29000`, `d_cf=5000` (scalar-mean special case, single bucket),
   `explained_fraction≈0.828`. This is the single most important test in
   the module — it directly validates the formula against a number a human
   can verify by hand.
2. **`select_predicate` bounds check**: `IndexError` on out-of-range rank.
3. **Trimming correctness**: synthetic data with a deliberately
   zero-overlap cell (present in source, absent in target) — assert it's
   excluded and reflected in `diagnostics.n_cells_no_target_overlap`.
4. **`min_cell_support` boundary**: cell with count exactly at, one above,
   and one below the threshold — off-by-one regression test.
5. **Consolidation regression tests**: confirm `compute_gap_decomposition`'s
   `s_source_orig` exactly equals a direct call to
   `sequence_builder.build_sequence(df_source, ..., weights=None)` — a
   cheap way to guarantee the §3.1 consolidation hasn't silently
   reintroduced a divergent code path.
6. **Negative / >1 explained fraction edge case**: construct a synthetic
   example where reweighting increases the distance (methodology doc §8),
   confirm it's computed correctly (not clipped or hidden) and, once the
   §7 TODO warning is added, that it's raised.

## 11. Thesis-Ready Description

> For the predicate selected in Stage 6, the source group's tuples are
> partitioned into joint cells over the predicate's constituent attributes,
> and each tuple is assigned a weight equal to the ratio of the target
> group's to the source group's empirical proportion in that cell. Cells
> present in only one group, or with fewer than [`min_cell_support`]
> observations in either group, are excluded and their exclusion rate is
> reported (Table X). The source group's aggregate sequence is recomputed
> under these weights, yielding a counterfactual sequence representing "the
> source group's outcomes, had its distribution over the predicate's
> attributes matched the target group's." The reduction in sequence
> distance between the original and counterfactual comparison — the
> **explained fraction** — quantifies the portion of the divergence
> attributable to this compositional difference; the remainder — the
> **residual gap** — reflects differences not accounted for by this
> attribute set. This is reported as a descriptive statistical estimate
> under an explicit no-unmeasured-confounding-beyond-X assumption, not as a
> causal effect (see [Limitations, Section Z]).
