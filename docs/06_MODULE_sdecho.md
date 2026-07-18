# Module: `sdecho.py`

**Pipeline stage:** Stage 4 (sequence distance) + Stage 5 (SDEcho predicate
search).

**Status:** Adapted from `_dist()` and `compute_gamma()` in the original
notebook. Two corrections applied relative to the original, both explained
below (§3): (a) bucket-index duplication removed, (b) reimplementation
fidelity formally documented as an open validation item.

**Note on the `Predicate.conditions` hashability question raised in
`docs/05_MODULE_predicates.md`:** left as dict-based (unchanged) for this
pass, per instruction. This module's `SDEchoResult.predicate` field will
need the same eventual migration if/when that's applied — noted, not
blocking.

---

## 1. Why This Module Is Needed

This module contains the one piece of the pipeline that is **not this
thesis's contribution**: SDEcho's own predicate-discovery mechanism, reused
as a black box that produces the input to Stage 6–10. It exists as its own
module (rather than living inline in a notebook) so that (a) it can be
tested and validated independently of the reweighting logic that consumes
its output, and (b) the reimplementation can be clearly bounded and audited
against the original SDEcho paper, since — as flagged repeatedly in prior
review — this is a from-scratch brute-force reimplementation, not the
original authors' code.

## 2. Where It Fits in the Pipeline

```
sequence_builder.py          predicates.py
  build_sequence()         enumerate_predicates()
  determine_bucket_index()   predicate_mask()
        │                          │
        └────────────┬─────────────┘
                      ▼
                 sdecho.py
        sequence_distance()  (Stage 4)
        run_sdecho()         (Stage 5)
                      │
                      ▼
         list[SDEchoResult], ranked by gamma
                      │
                      ▼
              reweighting.py (Stage 6: select_predicate)
```

## 3. Corrections Applied Relative to the Original Notebook

### 3.1 Bucket-index duplication (fixed)

The original notebook computed `desired_order` **twice, independently**:
once inside `get_sequences_for_year()` (Stage 3) and again, separately,
inside `compute_gamma()` (Stage 5) — both hardcoded to the same truncated
list, but as *copy-pasted*, not shared, logic. This is exactly the
duplicated-code pattern `IMPLEMENTATION_RULES.md` warns against, and it is
also a latent correctness risk: if the two copies were ever edited
independently (e.g., one fixed to include `"20+"`, the other forgotten),
Stage 4's `d_orig` and Stage 5's `dist_before` would silently be computed
over *different* bucket sets, making them incomparable without either
function raising an error.

**Fix:** `run_sdecho()` now takes `index: list[str]` as an explicit
parameter — the same `index` produced once by
`sequence_builder.determine_bucket_index()` and reused for Stage 3, 4, 5,
and 8 alike. There is exactly one place in the entire pipeline where the
bucket index is computed. `run_sdecho()` internally recomputes `dist_before`
using `sequence_builder.build_sequence()` (not a re-implementation) purely
as a **consistency check** — see §5 implementation, where it asserts this
matches the value passed in from Stage 4, catching any future accidental
divergence immediately rather than silently.

### 3.2 Reimplementation fidelity (documented, not yet resolved)

`run_sdecho()` (via `compute_gamma`'s original logic) is a **brute-force
reimplementation** of SDEcho's search:

- The original SDEcho paper's contribution is an *efficient* search
  strategy (structured pruning over the predicate lattice); this
  reimplementation instead exhaustively enumerates all predicates up to
  `max_order` and scores each one — functionally equivalent in *output* for
  small search spaces, but **not the same algorithm**, and its efficiency
  claims (if any are made in the thesis) must not be attributed to SDEcho's
  original method.
- The `gamma` formula used here —

  $$\gamma(P) = \frac{d(s_{D_1 \setminus M_1(P)},\, s_{D_2 \setminus M_2(P)})}{d(s_{D_1}, s_{D_2})} \cdot \left(1 + \frac{|M_1(P)|}{n_1} + \frac{|M_2(P)|}{n_2}\right)$$

  (formalized in `docs/01_METHODOLOGY.md` §4) — was reconstructed from the
  original notebook's code, not verified line-by-line against the published
  paper's formula. **This is an open validation item**: before any thesis
  result depends on specific `gamma` values or predicate rankings, the
  formula should be checked against the SDEcho paper's stated definition
  (and, ideally, against a worked example from the paper itself, if one is
  provided) to confirm this reimplementation is faithful. Tracked in
  `docs/11_ASSUMPTIONS_AND_LIMITATIONS.md`.

## 4. Assumptions Introduced

1. **`min_support` filters on `|M_1(P)| + |M_2(P)|`** (combined matching
   count across both groups), not on each group individually — a predicate
   matching 19 tuples in group A and 1 in group B passes a
   `min_support=20` threshold despite having essentially no support in
   group B. This asymmetry is inherited unchanged from the original code
   and is worth flagging as a limitation, since Stage 7's reweighting
   (which does check *per-group* cell support, per
   `docs/01_METHODOLOGY.md` §6.3) uses a stricter, more defensible standard
   than SDEcho's own search does. This inconsistency between Stage 5's and
   Stage 7's support-checking philosophy should be explicitly noted in the
   thesis's limitations, not silently left as an implicit inconsistency.
2. **Ranking is purely by ascending `gamma`**, with no tie-breaking rule
   specified when multiple predicates have equal or near-equal `gamma`
   values — `pandas`/Python's sort is stable, so ties resolve by
   enumeration order (which itself depends on `itertools.combinations`
   order in `predicates.py`), i.e., an **arbitrary but deterministic**
   tie-break. Fine for reproducibility, but should not be over-interpreted
   as meaningful.
3. **Top-1 selection (Stage 6, in `reweighting.py`) assumes the top-ranked
   predicate is unambiguously "the" explanation** — no statistical test or
   confidence measure distinguishes the top predicate's `gamma` from the
   second-ranked predicate's. If they're close, presenting only the top-1
   as "the" explanation overstates certainty. Worth reporting the top-k
   (e.g., top-3) gammas alongside the selected one in thesis tables, so a
   reader can judge how dominant the top predicate actually is.

## 5. Public API

```python
def sequence_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Euclidean distance between two aggregate sequences.

    Identical metric to the one used throughout the pipeline (Stage 4,
    Stage 9) — see docs/01_METHODOLOGY.md §3.
    """


def run_sdecho(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    group_col: str,
    measure_col: str,
    agg_func: str,
    index: list[str],
    candidate_attrs: list[str],
    max_order: int,
    k: int,
    max_values_per_attr: int,
    min_support: int,
    expected_d_orig: float | None = None,
) -> list[SDEchoResult]:
    """
    Brute-force SDEcho predicate search, ranked ascending by gamma
    (lower gamma = stronger explanation).

    Args:
        df1, df2: the two comparison groups (group A = source, group B =
            target, matching the convention used throughout the pipeline).
        group_col, measure_col, agg_func, index: as used throughout;
            `index` MUST be the same bucket index produced by
            sequence_builder.determine_bucket_index() and used for the
            Stage 3/4 original-distance computation — see §3.1.
        candidate_attrs, max_order, max_values_per_attr: passed through to
            predicates.enumerate_predicates().
        k: number of top-ranked predicates to return.
        min_support: minimum combined (|M1| + |M2|) matching tuple count
            for a predicate to be reported — see §4.1 for its asymmetry
            caveat.
        expected_d_orig: if provided, the internally recomputed
            dist_before is asserted to match this value (within floating
            point tolerance) — a consistency check against Stage 4's
            already-computed d_orig, catching bucket-index or aggregation
            divergence immediately. See §3.1.

    Returns:
        List of up to k SDEchoResult objects, ranked ascending by gamma.

    Raises:
        ValueError: if the original sequences are identical (d_orig = 0) —
            nothing to explain, matching the original implementation's
            behavior.
        AssertionError: if expected_d_orig is provided and does not match
            the internally recomputed distance (see §3.1) — surfaces a
            pipeline consistency bug immediately rather than silently.
    """
```

## 6. Implementation

```python
"""
sdecho.py

Stage 4 (sequence distance) and Stage 5 (SDEcho brute-force predicate
search) — reused, unmodified in ALGORITHM, from the original SDEcho
reimplementation. This is NOT this thesis's contribution; see
docs/00_PROJECT_CONCEPT.md for what is and isn't novel here.

IMPORTANT: this is a brute-force reimplementation, not the original
SDEcho authors' code. See docs/06_MODULE_sdecho.md §3.2 for the fidelity
caveat this implies.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.predicates import Predicate, enumerate_predicates, predicate_mask
from src.sequence_builder import build_sequence


@dataclass(frozen=True)
class SDEchoResult:
    """One ranked candidate explanation from SDEcho's predicate search."""
    predicate: Predicate
    gamma: float
    dist_before: float
    dist_after: float
    n1: int
    n2: int


def sequence_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """See docstring in module interface (§5)."""
    return float(np.sqrt(np.sum((s1 - s2) ** 2)))


def run_sdecho(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    group_col: str,
    measure_col: str,
    agg_func: str,
    index: list[str],
    candidate_attrs: list[str],
    max_order: int,
    k: int,
    max_values_per_attr: int,
    min_support: int,
    expected_d_orig: float | None = None,
) -> list[SDEchoResult]:
    """See docstring in module interface (§5)."""
    s1 = build_sequence(df1, group_col, measure_col, agg_func, index)
    s2 = build_sequence(df2, group_col, measure_col, agg_func, index)
    d_orig = sequence_distance(s1, s2)

    if expected_d_orig is not None:
        assert math.isclose(d_orig, expected_d_orig, rel_tol=1e-9), (
            f"Bucket-index/aggregation inconsistency detected: run_sdecho's "
            f"internally computed d_orig ({d_orig}) does not match the "
            f"externally supplied expected_d_orig ({expected_d_orig}). "
            f"Check that the same `index` was used in both Stage 4 and "
            f"Stage 5 calls — see docs/06_MODULE_sdecho.md §3.1."
        )

    if d_orig == 0:
        raise ValueError(
            "Original sequences are identical (distance = 0); "
            "nothing to explain."
        )

    n1, n2 = len(df1), len(df2)
    candidates = enumerate_predicates(df1, df2, candidate_attrs, max_order, max_values_per_attr)

    results: list[SDEchoResult] = []
    for predicate in candidates:
        mask1 = predicate_mask(df1, predicate)
        mask2 = predicate_mask(df2, predicate)
        size1, size2 = int(mask1.sum()), int(mask2.sum())

        if size1 + size2 < min_support:
            continue  # see §4.1 for the asymmetry this threshold allows
        if size1 == 0 and size2 == 0:
            continue

        rest1, rest2 = df1.loc[~mask1], df2.loc[~mask2]
        s1_after = build_sequence(rest1, group_col, measure_col, agg_func, index)
        s2_after = build_sequence(rest2, group_col, measure_col, agg_func, index)
        dist_after = sequence_distance(s1_after, s2_after)

        penalty = 1.0 + size1 / n1 + size2 / n2
        gamma = (dist_after / d_orig) * penalty

        results.append(SDEchoResult(
            predicate=predicate, gamma=gamma,
            dist_before=d_orig, dist_after=dist_after,
            n1=size1, n2=size2,
        ))

    results.sort(key=lambda r: r.gamma)
    return results[:k]
```

## 7. Known Gaps / TODOs

- **Fidelity validation (§3.2) is the single most important open item in
  this module** — must be resolved (or at least explicitly attempted and
  documented as attempted-but-inconclusive) before final thesis results are
  reported, since every downstream number in Stages 6–10 depends on this
  module's correctness.
- `min_support` asymmetry (§4.1) — decide whether to tighten this to a
  per-group minimum (matching Stage 7's stricter standard) or explicitly
  justify keeping the looser combined threshold; currently just inherited,
  not decided.
- No confidence/stability measure on the top-ranked predicate (§4.3) —
  reporting top-k gammas alongside the selected predicate is a cheap fix,
  not yet implemented.

## 8. How Reviewer #2 Would Critique This Module

- *"You call this 'SDEcho' but it's a brute-force reimplementation with an
  unvalidated gamma formula — how do you know your results reflect SDEcho's
  actual method?"* — This is the most serious open risk in the entire
  thesis and needs a direct, honest answer in the thesis text: state
  clearly that this is a reimplementation, describe what was and wasn't
  validated against the original paper, and scope any claims accordingly
  (e.g., "we build on the *predicate-search paradigm* of SDEcho" rather
  than "we use SDEcho").
- *"Two different support thresholds (Stage 5's combined min_support vs.
  Stage 7's per-group min_cell_support) with no stated reason for the
  difference looks inconsistent."* — Fair; needs either alignment or
  explicit justification (§4.1, §7).

## 9. Complexity

See `docs/01_METHODOLOGY.md` §9 and `docs/05_MODULE_predicates.md` §8 —
this module's cost is dominated by `enumerate_predicates`'s combinatorial
predicate count multiplied by the per-predicate `build_sequence` +
`sequence_distance` cost, i.e., $O(|\mathcal{P}| \cdot (n_1 + n_2))$.

## 10. Tests to Write (`tests/test_sdecho.py`)

1. **`sequence_distance` correctness**: hand-computed Euclidean distance on
   a small known pair of vectors.
2. **`run_sdecho` consistency assertion**: deliberately pass a mismatched
   `expected_d_orig` and confirm `AssertionError` is raised — this is the
   regression test for the bug class fixed in §3.1.
3. **`ValueError` on identical sequences**: two groups with identical
   aggregate sequences should raise, not silently return an empty result.
4. **`min_support` filtering**: synthetic predicate with known
   `size1`/`size2` below and above the threshold — confirm correct
   inclusion/exclusion.
5. **End-to-end small example**: a tiny synthetic dataset (≤20 rows, 1
   candidate attribute, 2 values) with a hand-computable expected top
   predicate and gamma — the SDEcho analogue of the reweighting toy example
   already used in `docs/00_PROJECT_CONCEPT.md` §5 and referenced in
   `docs/04_MODULE_sequence_builder.md` §10, giving the thesis a matched
   pair of hand-verified examples spanning both stages.

## 11. Thesis-Ready Description

> Explanatory predicates are discovered via a brute-force reimplementation
> of SDEcho's predicate-search procedure: candidate predicates are enumerated
> as conjunctions of up to [`max_order`] attributes (see [Module: predicates]),
> and each is scored by the relative reduction in sequence distance achieved
> by removing its matching tuples, penalized by the fraction of each group
> it matches. We note that this is a reimplementation rather than the
> original authors' code, and that the original SDEcho paper's contribution
> — an efficient, pruned search strategy — is not reproduced here; we instead
> perform exhaustive search over a bounded predicate space, which is
> tractable at the scale of this thesis's search parameters
> ([`max_order`]=X, [`max_values_per_attr`]=Y) but is not intended as an
> efficiency contribution.
