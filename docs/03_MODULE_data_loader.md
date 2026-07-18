# Module: `data_loader.py`

**Pipeline stage:** Stage 1 (load dataset) + Stage 2 (define two groups)

**Status:** Adapted from the existing `load_and_preprocess_year()` /
`load_all_years()` functions in the original notebook. Those were written
for a multi-year loop (now descoped, see `docs/00_PROJECT_CONCEPT.md` §6);
this module simplifies them to the single-dataset, single-comparison scope.
**Not yet split into its own file / tested in isolation** — this doc
specifies the target; migration from the notebook is the next implementation
step after this documentation pass.

---

## 1. Why This Module Is Needed

Every downstream stage (sequence construction, SDEcho, reweighting) assumes
a clean DataFrame with the specific derived columns the pipeline depends on:
a bucketed experience column, a cleaned age-group column, and a normalized
remote-work column. Centralizing this cleaning in one module — rather than
repeating `pd.cut`/regex/`fillna` calls inline in the notebook — means every
stage downstream operates on data with guaranteed structure, and any change
to the cleaning rules (e.g., different bucket boundaries) happens in exactly
one place.

## 2. Where It Fits in the Pipeline

```
raw CSV  →  data_loader.py  →  cleaned DataFrame  →  sequence_builder.py
```

It is the only module in `src/` that touches the raw survey file directly.
Every other module operates on already-cleaned DataFrames it receives as
arguments — no module downstream re-reads or re-cleans data. This is a
deliberate boundary: **cleaning logic lives in exactly one place.**

## 3. Assumptions Introduced

1. **Bucket boundaries are fixed and hand-chosen**, not learned or
   configurable per experiment: `YearsCodePro` is cut into
   `[0,2), [2,5), [5,10), [10,20), [20,100)` labeled `0-2, 3-5, 6-10, 10-20,
   20+`. This is inherited unmodified from the original notebook. **This is
   an arbitrary methodological choice that must be justified in the thesis**
   (why these boundaries, not quartiles or deciles?) — currently
   unjustified; flagged in `docs/11_ASSUMPTIONS_AND_LIMITATIONS.md`.
2. **`AgeGroup` is extracted via regex** (`\d+-\d+`) from a free-text-ish
   `Age` column, silently dropping any row whose age doesn't match a clean
   `NN-NN` pattern (e.g., "Under 18", "65 years or older" become `NaN` and
   are later dropped via `dropna`). This means certain age brackets are
   **structurally excluded from the study by construction**, not by
   deliberate scoping — needs to be stated explicitly as a limitation.
3. **`RemoteWork` values are collapsed**: `"Hybrid (some remote, some
   in-person)"` → `"Hybrid"`, missing values → `"Unknown"`. `"Unknown"` is
   then treated as a legitimate category value throughout the rest of the
   pipeline (including as a valid reweighting cell) — worth deciding
   explicitly whether `"Unknown"` should be excluded from `candidate_attrs`
   reweighting cells, since "unknown" is not a real covariate value and
   reweighting on it doesn't have a clean interpretation.
4. **Rows missing any pipeline-required column are dropped** via
   `dropna(subset=[group_col, subgroup_col, measure_col])`. This drop
   happens **before** the two groups are split, so both groups are cleaned
   under an identical rule — good for comparability, but the number of rows
   dropped and *why* (missing bucket vs. missing subgroup vs. missing
   measure) is currently not reported. Should be added as a diagnostic
   return value (see §6 below).

## 4. Public API

```python
def load_and_preprocess_data(path: str, config: dict) -> pd.DataFrame:
    """
    Load the raw survey CSV and derive the columns required by the
    pipeline: bucketed experience, cleaned age group, normalized
    remote-work category.

    Args:
        path: filesystem path to the raw CSV.
        config: pipeline CONFIG dict (see docs/02_ARCHITECTURE.md §5);
            only `group_col`, `subgroup_col`, `measure_col` are consulted
            here, to know which columns are required for dropna.

    Returns:
        Cleaned DataFrame with derived columns added, rows with missing
        pipeline-required fields removed.

    Raises:
        FileNotFoundError: if `path` does not exist.
        KeyError: if a column required for a derivation (e.g.,
            "YearsCodePro" for bucketing) is absent from the raw CSV —
            fails loudly rather than silently skipping the derivation.
    """


def split_groups(
    df: pd.DataFrame, subgroup_col: str, val1: str, val2: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a cleaned DataFrame into the two comparison groups.

    Args:
        df: output of load_and_preprocess_data().
        subgroup_col: the column defining the two groups (e.g. "AgeGroup").
        val1, val2: the two values of subgroup_col that define group A
            (source, reweighted) and group B (target, aligned to).

    Returns:
        (df_A, df_B) — two DataFrames, each a filtered view of df.

    Raises:
        ValueError: if val1 or val2 does not appear in df[subgroup_col]
            (fails loudly — silently returning an empty group is a common
            source of confusing downstream errors, e.g. division by zero
            in cell proportions).
    """
```

## 5. Implementation (adapted from the original notebook)

```python
"""
data_loader.py

Stage 1-2 of the pipeline: load and clean the raw survey data, and split
it into the two comparison groups. This is the ONLY module that touches
raw survey files directly; every module downstream receives already-clean
DataFrames.
"""

import warnings
import pandas as pd


# Bucket boundaries for years-of-experience, fixed by design.
# See docs/11_ASSUMPTIONS_AND_LIMITATIONS.md for the justification gap
# this introduces (boundaries are hand-chosen, not empirically derived).
_EXPERIENCE_BINS = [0, 2, 5, 10, 20, 100]
_EXPERIENCE_LABELS = ["0-2", "3-5", "6-10", "10-20", "20+"]


def load_and_preprocess_data(path: str, config: dict) -> pd.DataFrame:
    """See docstring in module interface above."""
    df = pd.read_csv(path)

    if "YearsCodePro" in df.columns:
        df["YearsExpBucket"] = pd.cut(
            pd.to_numeric(df["YearsCodePro"], errors="coerce"),
            bins=_EXPERIENCE_BINS,
            labels=_EXPERIENCE_LABELS,
        )

    if "Age" in df.columns:
        df["AgeGroup"] = df["Age"].str.extract(r"(\d+-\d+)")

    if "RemoteWork" in df.columns:
        df["RemoteWork"] = (
            df["RemoteWork"]
            .fillna("Unknown")
            .replace({"Hybrid (some remote, some in-person)": "Hybrid"})
        )

    required_cols = [config["group_col"], config["subgroup_col"], config["measure_col"]]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Required column(s) {missing_cols} not found after preprocessing. "
            f"Check that the raw CSV contains the expected source columns."
        )

    n_before = len(df)
    df = df.dropna(subset=required_cols)
    n_after = len(df)
    n_dropped = n_before - n_after

    # Diagnostic print rather than silent drop; a future iteration should
    # return this as structured metadata rather than a print statement
    # (tracked as a TODO, not blocking for the current implementation pass).
    print(f"Loaded {n_before} rows; dropped {n_dropped} "
          f"({100 * n_dropped / n_before:.1f}%) with missing required fields; "
          f"{n_after} rows remain.")

    return df


def split_groups(
    df: pd.DataFrame, subgroup_col: str, val1: str, val2: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """See docstring in module interface above."""
    if val1 not in df[subgroup_col].unique():
        raise ValueError(f"'{val1}' not found in column '{subgroup_col}'.")
    if val2 not in df[subgroup_col].unique():
        raise ValueError(f"'{val2}' not found in column '{subgroup_col}'.")

    df_a = df[df[subgroup_col] == val1].copy()
    df_b = df[df[subgroup_col] == val2].copy()
    return df_a, df_b
```

## 6. Known Gaps / TODOs for This Module

- Row-drop diagnostics are currently a `print()`, not a structured return
  value. Once `evaluation.py`'s reporting tables are built, this should
  become part of a `LoadDiagnostics` dataclass (mirrors the pattern used for
  `ReweightingDiagnostics` in `docs/02_ARCHITECTURE.md` §3) so it can be
  included in thesis tables rather than only console output.
- `"Unknown"` as a `RemoteWork` category needs an explicit decision (keep,
  exclude from `candidate_attrs`, or exclude from reweighting cells
  specifically) — currently unresolved, tracked in
  `docs/11_ASSUMPTIONS_AND_LIMITATIONS.md`.
- Experience bucket boundaries are hard-coded; if evaluation later shows
  sensitivity to bucket choice, this should become a `config` parameter
  rather than a module constant — deferred until there's evidence it
  matters (avoid premature configurability).

## 7. How Reviewer #2 Would Critique This Module

- *"Why these five experience buckets and not deciles, or data-driven
  quantile bins?"* — Answer needed in the thesis: likely "matches SDEcho
  paper's own bucketing convention" if true, otherwise this is an
  unjustified modeling choice and should be tested as a robustness check
  (does the explained fraction change materially under different bucket
  granularity?).
- *"Regex-based age extraction silently discards non-standard age
  responses — how many rows, and does this bias the sample?"* — Needs a
  reported count and a sentence acknowledging the exclusion, at minimum.
- *"`dropna` before splitting groups seems safe, but does it drop
  differentially by group?"* — Worth a one-line check: report drop rate
  separately for group A and group B, not just overall, in case cleaning
  itself introduces an imbalance that later gets misattributed to a "real"
  compositional difference.

## 8. Complexity

$O(n)$ in the number of raw rows for all operations (`pd.cut`, regex
extraction, `fillna`, `dropna`, boolean filtering) — no joins, no
quadratic operations. Not a performance concern relative to SDEcho's
brute-force search cost.

## 9. Tests to Write (`tests/test_data_loader.py`)

1. **Bucketing correctness**: a tiny synthetic DataFrame with known
   `YearsCodePro` values (e.g., `[1, 4, 7, 15, 25]`) should map to exactly
   `["0-2", "3-5", "6-10", "10-20", "20+"]`.
2. **Age regex correctness**: `"25-34 years old"` → `"25-34"`; `"Under 18
   years old"` → `NaN` (and confirm the row is dropped downstream, not
   silently retained with a bad value).
3. **RemoteWork collapsing**: confirm `"Hybrid (some remote, some
   in-person)"` → `"Hybrid"` and `NaN` → `"Unknown"`.
4. **`split_groups` raises `ValueError`** on a nonexistent group value
   (e.g., typo'd `"25-35"` instead of `"25-34"`) rather than silently
   returning an empty DataFrame.
5. **Row-drop count is correct** on a small DataFrame with a known number
   of intentionally-missing required fields.

## 10. Thesis-Ready Description (for a Data / Preprocessing subsection)

> Raw survey responses are cleaned into the representation required by the
> pipeline: years of professional coding experience are discretized into
> five fixed buckets (`0-2, 3-5, 6-10, 10-20, 20+` years); free-text age
> ranges are normalized via pattern extraction into a canonical `NN-NN`
> format; and remote-work status is collapsed into three categories
> (`Remote, Hybrid, Onsite/Unknown`). Rows missing any field required for
> the aggregate sequence construction (bucketing attribute, comparison-group
> attribute, or outcome measure) are excluded; the exclusion rate is
> reported in [Table X] and discussed as a scope limitation in [Section Y].
