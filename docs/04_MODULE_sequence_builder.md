# Module: `sequence_builder.py`

**Pipeline stage:** Stage 3 (construct aggregate sequences), reused
unmodified for Stage 8 (counterfactual aggregate sequence).

**Status:** Adapted and **corrected** from `_aggregate_sequence()` and
`get_sequences_for_year()` in the original notebook. See §3 for the specific
bug fix this module introduces relative to the original code.

**Correction to `docs/02_ARCHITECTURE.md`:** the data-flow diagram in that
file shows `split_groups()` inside this module; it is actually implemented
in `data_loader.py` (per `docs/03_MODULE_data_loader.md`). This module is
scoped to sequence construction only. The architecture diagram should be
corrected to reflect this the next time that file is revised.

---

## 1. Why This Module Is Needed

This module is the **single source of truth** for turning a group's
DataFrame into an aggregate sequence — a fixed-length numeric vector indexed
by bucket. It is used twice in the pipeline: once for the *original*
sequences (Stage 3, on unweighted data) and once for the *counterfactual*
sequence (Stage 8, on reweighted data). Using the exact same aggregation
function in both places is essential — if Stage 3 and Stage 8 used even
slightly different aggregation logic, the comparison between
$d_{\text{orig}}$ and $d_{\text{cf}}$ (§8 of `01_METHODOLOGY.md`) would not
be measuring the effect of reweighting alone, but also measuring an
unintended implementation discrepancy.

## 2. Where It Fits in the Pipeline

```
data_loader.py           reweighting.py
   D_A, D_B    ─────┐        weights (Stage 7)
                     ▼            ▼
              sequence_builder.py
              build_sequence(D, weights=None, ...)
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  s_A, s_B (Stage 3)      s_A_cf (Stage 8, weights passed)
        │                         │
        └────────────┬────────────┘
                      ▼
                  sdecho.py / reweighting.py
             (sequence_distance, gap decomposition)
```

## 3. Bug Fix: Bucket Index Determination

### The problem in the original code

```python
# Original notebook — DO NOT carry this forward as-is:
desired_order = ["0-2", "3-5", "6-10", "10-20"]   # "20+" never appears
```

This hardcoded list silently excludes any respondent with 20+ years of
professional experience from **every** sequence, **every** distance
computation, and **every** SDEcho/reweighting result in the original
notebook — with no comment, no justification, and no report of how many
rows this affects. This is exactly the kind of undisclosed data-cleaning
decision flagged as a validity threat in earlier review (see
`docs/00_PROJECT_CONCEPT.md` — related discussion).

### The fix

The bucket index should be determined **programmatically**, as the ordered
intersection (or union — see decision below) of a canonical bucket ordering
with the buckets actually observed in the data, and the count of excluded
respondents (if any are excluded) must be reported.

```python
def determine_bucket_index(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    group_col: str,
    canonical_order: list[str],
    min_bucket_support: int = 1,
) -> tuple[list[str], dict]:
    """
    Determine the ordered bucket index used for aggregate sequences.

    Uses ALL buckets in `canonical_order` that are present (with at least
    `min_bucket_support` rows) in at least one of the two groups. This
    differs from the original notebook, which hardcoded a truncated list
    that silently dropped the "20+" bucket with no justification.

    Args:
        df_a, df_b: the two comparison groups.
        group_col: the bucketing attribute.
        canonical_order: the full intended bucket ordering (e.g., all five
            experience buckets, including "20+").
        min_bucket_support: minimum row count (summed across both groups)
            for a bucket to be included; buckets below this are excluded
            and reported, not silently dropped.

    Returns:
        (bucket_index, diagnostics) where diagnostics reports which
        canonical buckets were excluded and why (absent vs. low support).
    """
    counts_a = df_a[group_col].value_counts()
    counts_b = df_b[group_col].value_counts()

    included, excluded = [], []
    for bucket in canonical_order:
        total = counts_a.get(bucket, 0) + counts_b.get(bucket, 0)
        if total >= min_bucket_support:
            included.append(bucket)
        else:
            excluded.append((bucket, total))

    diagnostics = {
        "canonical_order": canonical_order,
        "included_buckets": included,
        "excluded_buckets": excluded,  # [(bucket_label, total_count), ...]
        "min_bucket_support": min_bucket_support,
    }
    return included, diagnostics
```

**Default `canonical_order` must include `"20+"`.** If, after running this
on real data, `"20+"` turns out to have genuinely low support and gets
excluded by the `min_bucket_support` rule, that is a *reported, justified*
exclusion — categorically different from the original silent hardcoding.

## 4. Assumptions Introduced

1. **Bucket index is computed once per (group A, group B) pair** and reused
   identically for both the original and counterfactual sequences — this is
   required for `d_orig` and `d_cf` to be comparable (see §2 above).
2. **Empty-but-included buckets are zero-filled**, per the convention
   already fixed in `docs/01_METHODOLOGY.md` §2 (`s_G[b] = 0` if
   $D_G^{(b)} = \emptyset$). This is a real modeling choice with a
   known risk: a zero-filled empty bucket is indistinguishable from a true
   zero-valued aggregate. Currently unresolved beyond "flagged as a
   limitation" — an alternative (exclude empty buckets from the distance
   sum entirely, e.g., only compare buckets present in *both* sequences)
   should be tested as a robustness check in `evaluation.py`.
3. **`min_bucket_support` is a new parameter this fix introduces** and needs
   a default value decision — recommend starting at the same value as
   `min_cell_support` (5) for consistency with the reweighting stage's
   support threshold, but this is a judgment call, not derived from theory.

## 5. Public API

```python
def determine_bucket_index(
    df_a: pd.DataFrame, df_b: pd.DataFrame, group_col: str,
    canonical_order: list[str], min_bucket_support: int = 5,
) -> tuple[list[str], dict]:
    """See §3 above."""


def build_sequence(
    df: pd.DataFrame,
    group_col: str,
    measure_col: str,
    agg_func: str,
    index: list[str],
    weights: pd.Series | None = None,
) -> np.ndarray:
    """
    Compute an aggregate sequence over the given bucket index.

    If `weights` is None, computes the unweighted aggregate (Stage 3).
    If `weights` is provided, computes the weighted aggregate (Stage 8),
    excluding rows with undefined (NaN) weight. This single function
    implements BOTH _aggregate_sequence (original) and
    weighted_aggregate_sequence (counterfactual) from earlier drafts,
    unified into one implementation so Stage 3 and Stage 8 cannot silently
    diverge (see §1 above for why this matters).

    Args:
        df: the group's DataFrame.
        group_col: the bucketing attribute.
        measure_col: the numeric outcome column.
        agg_func: currently only "mean" is supported (see
            docs/01_METHODOLOGY.md §7 for the sum/count extension note).
        index: the fixed, ordered bucket labels (from determine_bucket_index).
        weights: optional per-row weights (see docs/01_METHODOLOGY.md §6-7).

    Returns:
        np.ndarray of length len(index); 0.0 for buckets with no
        (weighted) observations.

    Raises:
        NotImplementedError: if agg_func != "mean".
    """
```

## 6. Implementation

```python
"""
sequence_builder.py

Stage 3 (original sequences) and Stage 8 (counterfactual sequences) share
this module's build_sequence() as their single implementation, so the two
cannot silently diverge in aggregation logic. See docs/04_MODULE_sequence_builder.md
for the rationale.
"""

import warnings
import numpy as np
import pandas as pd


def determine_bucket_index(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    group_col: str,
    canonical_order: list[str],
    min_bucket_support: int = 5,
) -> tuple[list[str], dict]:
    """See docstring in module interface (§5)."""
    counts_a = df_a[group_col].value_counts()
    counts_b = df_b[group_col].value_counts()

    included, excluded = [], []
    for bucket in canonical_order:
        total = int(counts_a.get(bucket, 0) + counts_b.get(bucket, 0))
        if total >= min_bucket_support:
            included.append(bucket)
        else:
            excluded.append((bucket, total))

    if excluded:
        warnings.warn(
            f"Bucket(s) excluded for insufficient support "
            f"(min_bucket_support={min_bucket_support}): {excluded}. "
            f"This is a REPORTED exclusion, not a silent one — include this "
            f"in thesis diagnostics tables.",
            stacklevel=2,
        )

    diagnostics = {
        "canonical_order": canonical_order,
        "included_buckets": included,
        "excluded_buckets": excluded,
        "min_bucket_support": min_bucket_support,
    }
    return included, diagnostics


def build_sequence(
    df: pd.DataFrame,
    group_col: str,
    measure_col: str,
    agg_func: str,
    index: list[str],
    weights: pd.Series | None = None,
) -> np.ndarray:
    """See docstring in module interface (§5)."""
    if agg_func != "mean":
        raise NotImplementedError(
            f"agg_func='{agg_func}' not supported. Only 'mean' is currently "
            f"implemented; see docs/01_METHODOLOGY.md §7 for the extension "
            f"this would require for 'sum'/'count'."
        )

    if weights is None:
        # Stage 3: unweighted aggregate.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            grouped = df.groupby(group_col)[measure_col].mean()
        return grouped.reindex(index).fillna(0).to_numpy(dtype=float)

    # Stage 8: weighted aggregate, excluding trimmed (NaN-weight) rows.
    valid_mask = weights.notna()
    df_valid = df.loc[valid_mask].copy()
    w_valid = weights.loc[valid_mask]

    df_valid = df_valid.assign(_weight=w_valid.to_numpy())
    df_valid["_weighted_value"] = df_valid[measure_col] * df_valid["_weight"]

    grouped = df_valid.groupby(group_col).agg(
        weighted_sum=("_weighted_value", "sum"),
        weight_sum=("_weight", "sum"),
    )
    seq = (grouped["weighted_sum"] / grouped["weight_sum"]).reindex(index)
    return seq.fillna(0).to_numpy(dtype=float)
```

## 7. Known Gaps / TODOs

- `min_bucket_support` default (currently 5, matching `min_cell_support`) is
  a judgment call, not derived — should be revisited once real data shows
  how many respondents fall in `"20+"` and whether 5 is too strict/lenient.
- Zero-fill-vs-exclude for empty buckets (§4.2) remains an open robustness
  question, not yet tested empirically.
- `agg_func` is hard-restricted to `"mean"` with an explicit
  `NotImplementedError` rather than silently mishandling `"sum"`/`"count"` —
  intentional fail-loud design, but extending to those aggregations is
  still a TODO if the thesis scope grows to need them (currently not
  planned, per `docs/00_PROJECT_CONCEPT.md` §6).

## 8. How Reviewer #2 Would Critique This Module

- *"The original implementation silently dropped a bucket — how do we know
  there aren't other silent exclusions elsewhere in the pipeline?"* — This
  is a fair and uncomfortable question. The honest answer: this fix
  addresses the one instance found during review; a full audit of every
  `dropna`/`reindex`/`fillna` call across all modules for similar silent
  exclusions is warranted before final results are reported, and should be
  logged as a pre-submission checklist item.
- *"Why zero-fill rather than exclude empty buckets from the distance sum?"*
  — Currently answered only by "matches the original convention," which is
  not a justification, only an explanation of provenance. Needs an
  empirical robustness check (§4.2) before the thesis can defend this
  choice on its merits.

## 9. Complexity

$O(n_A + n_B)$ for both `determine_bucket_index` (value_counts) and
`build_sequence` (groupby + reindex) — linear, no change from the original
implementation's complexity, only its correctness.

## 10. Tests to Write (`tests/test_sequence_builder.py`)

1. **Bucket inclusion regression test**: construct a synthetic dataset where
   `"20+"` has, say, 40 respondents combined across both groups — assert it
   is **included** in the returned bucket index (this test would have
   caught the original bug immediately).
2. **Bucket exclusion reporting**: construct a dataset where one bucket has
   fewer than `min_bucket_support` respondents — assert it appears in
   `diagnostics["excluded_buckets"]` and that a warning is raised.
3. **`build_sequence` unweighted correctness**: small DataFrame with known
   values, hand-computed expected means per bucket.
4. **`build_sequence` weighted correctness**: reuse the India/USA toy
   example from `docs/00_PROJECT_CONCEPT.md` §5 — this is the same
   hand-computable example referenced in the reweighting module's test
   plan, giving both modules a shared, cross-checkable fixture.
5. **Consistency test**: calling `build_sequence(df, weights=None)` and
   calling it with `weights` set to an all-ones `pd.Series` should return
   identical results — a cheap way to catch divergence between the
   weighted and unweighted code paths.

## 11. Thesis-Ready Description

> Aggregate sequences are computed over a bucket index determined
> programmatically as the set of canonical experience buckets with
> sufficient combined support across both comparison groups (minimum
> [`min_bucket_support`] respondents), rather than a fixed a priori subset.
> [N] respondents with 20+ years of experience were [included in / excluded
> from, with count] the analysis on this basis. The same aggregation
> function is used to compute both the original sequences (Section X) and
> the counterfactual, reweighted sequences (Section Y), ensuring the
> reported gap reduction reflects only the effect of reweighting and not an
> incidental implementation difference between the two computations.
