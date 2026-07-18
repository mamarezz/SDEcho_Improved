# Architecture

This document specifies the software architecture: how the pipeline in
`docs/00_PROJECT_CONCEPT.md` §8 and the formulas in `docs/01_METHODOLOGY.md`
map onto the `src/` module structure, including function signatures,
intermediate data representations, and data flow.

---

## 1. Module Map

```
project/
├── src/
│   ├── data_loader.py       Stage 1   — load + clean raw survey CSV
│   ├── sequence_builder.py  Stage 2-3 — group selection, aggregate sequences
│   ├── predicates.py        (shared)  — predicate enumeration, masking utilities
│   ├── sdecho.py            Stage 4-5 — sequence distance, SDEcho search
│   ├── reweighting.py       Stage 6-10— predicate selection, cell weighting,
│   │                                    counterfactual sequence, gap decomposition
│   ├── evaluation.py        —         — baselines, bootstrap CI, synthetic
│   │                                    ground-truth harness, ablations
│   └── visualization.py     —         — sequence plots, gap-decomposition
│                                        figures, diagnostic tables
├── notebooks/
│   └── thesis.ipynb         — orchestration only; imports from src/, no
│                               business logic lives in the notebook
├── data/                    — raw + processed survey CSVs (not versioned)
├── papers/                  — SDEcho, Scorpion, XQA survey PDFs
└── docs/                    — this documentation set
```

**Design principle:** the notebook is a *thin orchestration layer*. Every
function that does real work lives in `src/`, is independently importable,
independently testable, and independently documented. This directly serves
`IMPLEMENTATION_RULES.md`'s requirement to avoid large notebook cells and
duplicated code, and makes `evaluation.py` able to call the exact same
functions used in the main pipeline (no re-implementation for experiments).

---

## 2. Data Flow Diagram

```
data/stackoverflow2022.csv
        │
        ▼
┌─────────────────────┐
│ data_loader.py       │  load_and_preprocess_data()
│                      │  → cleaned pd.DataFrame
└─────────┬────────────┘
          │
          ▼
┌─────────────────────┐
│ sequence_builder.py  │  split_groups(df, subgroup_col, val1, val2)
│                      │  → (D_A, D_B)
│                      │
│                      │  build_sequence(D, group_col, measure_col, index)
│                      │  → np.ndarray  (calls _aggregate_sequence)
└─────────┬────────────┘
          │  s_A, s_B
          ▼
┌─────────────────────┐
│ sdecho.py            │  sequence_distance(s1, s2) → float   (Stage 4)
│                      │
│                      │  run_sdecho(D_A, D_B, candidate_attrs,     (Stage 5)
│                      │             max_order, k, min_support)
│                      │  → List[SDEchoResult]   (uses predicates.py)
└─────────┬────────────┘
          │  ranked predicates
          ▼
┌─────────────────────┐
│ reweighting.py        │  select_predicate(results, rank=0)          (Stage 6)
│                       │  → predicate dict
│                       │
│                       │  compute_cell_weights(D_A, D_B, attrs, τ)   (Stage 7)
│                       │  → (weights: pd.Series, diagnostics: dict)
│                       │
│                       │  weighted_aggregate_sequence(D_A, weights,  (Stage 8)
│                       │        group_col, measure_col, index)
│                       │  → np.ndarray  (s_A_cf)
│                       │
│                       │  compute_gap_decomposition(...)        (Stage 9-10)
│                       │  → GapDecompositionResult
└─────────┬─────────────┘
          │  GapDecompositionResult
          ▼
┌─────────────────────┐        ┌─────────────────────┐
│ evaluation.py         │  ←──→ │ visualization.py     │
│  baseline comparisons │       │  sequence plots       │
│  bootstrap CIs        │       │  gap-decomposition     │
│  synthetic validation │       │  bar charts            │
│  ablations             │      │  diagnostic tables     │
└───────────┬────────────┘      └───────────┬────────────┘
            │                                │
            ▼                                ▼
      results tables                  thesis-ready figures
      (CSV / DataFrame)                (PNG / PDF)
```

---

## 3. Intermediate Representations

All cross-module data is passed as **typed dataclasses**, not raw dicts —
this gives IDE autocompletion, catches key-name typos at development time,
and gives every result object a natural `__repr__`/serialization point for
`evaluation.py` result tables.

```python
# src/predicates.py

from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass(frozen=True)
class Predicate:
    """A conjunctive predicate over one or more attributes."""
    conditions: Dict[str, Any]  # {"Country": "United States of America"}

    @property
    def attrs(self) -> list:
        return list(self.conditions.keys())

    def __repr__(self) -> str:
        return " & ".join(f"{k}={v}" for k, v in self.conditions.items())
```

```python
# src/sdecho.py

from dataclasses import dataclass
from src.predicates import Predicate

@dataclass(frozen=True)
class SDEchoResult:
    """One ranked candidate explanation from SDEcho's predicate search."""
    predicate: Predicate
    gamma: float
    dist_before: float
    dist_after: float
    n1: int          # matching tuple count in group A
    n2: int          # matching tuple count in group B
```

```python
# src/reweighting.py

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ReweightingDiagnostics:
    """Diagnostic summary of a cell-based reweighting run."""
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
    """Final Stage 7-10 output: everything needed for reporting/plotting."""
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
```

**Why dataclasses over dicts:** the earlier `reweighting.py` draft used plain
dicts (fine for a first pass / quick validation). As the codebase grows to
include `evaluation.py`'s baseline comparisons and bootstrap CIs, plain dicts
become error-prone (typo'd keys fail silently). Migrating to
`@dataclass(frozen=True)` is a small, mechanical refactor — flagged here as
the next cleanup step once Stage 7 is validated on real data, not before.

---

## 4. Module Interfaces (public API surface)

### `data_loader.py`
```python
def load_and_preprocess_data(path: str, config: dict) -> pd.DataFrame:
    """Load one CSV and derive columns needed by the pipeline
    (bucket columns, cleaned categorical columns)."""

def split_groups(
    df: pd.DataFrame, subgroup_col: str, val1: str, val2: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into the two comparison groups."""
```

### `sequence_builder.py`
```python
def build_sequence(
    df: pd.DataFrame, group_col: str, measure_col: str,
    agg_func: str, index: list[str]
) -> np.ndarray:
    """Wraps _aggregate_sequence; the single source of truth for
    turning a DataFrame into an aggregate sequence, used identically
    in the original (Stage 3) and counterfactual (Stage 8) paths."""
```

### `predicates.py`
```python
def enumerate_predicates(
    df1: pd.DataFrame, df2: pd.DataFrame,
    candidate_attrs: list[str], max_order: int,
    max_values_per_attr: int | None = None,
) -> list[Predicate]:
    """Generates candidate predicates up to max_order attributes."""

def predicate_mask(df: pd.DataFrame, predicate: Predicate) -> pd.Series:
    """Boolean mask of tuples matching a predicate."""
```

### `sdecho.py`
```python
def sequence_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """Euclidean distance between two aggregate sequences."""

def run_sdecho(
    df1: pd.DataFrame, df2: pd.DataFrame,
    group_col: str, measure_col: str, agg_func: str, index: list[str],
    candidate_attrs: list[str], max_order: int, k: int,
    max_values_per_attr: int, min_support: int,
) -> list[SDEchoResult]:
    """Brute-force SDEcho predicate search, ranked ascending by gamma."""
```

### `reweighting.py`
```python
def select_predicate(results: list[SDEchoResult], rank: int = 0) -> Predicate:
    """Selects the rank-th predicate (0 = top-1) from SDEcho's ranked output."""

def compute_cell_weights(
    df_source: pd.DataFrame, df_target: pd.DataFrame,
    attrs: list[str], min_cell_support: int = 5,
) -> tuple[pd.Series, ReweightingDiagnostics]:
    """Exact cell-based (joint-distribution) reweighting factors."""

def weighted_aggregate_sequence(
    df: pd.DataFrame, weights: pd.Series,
    group_col: str, measure_col: str, index: list[str],
) -> np.ndarray:
    """Weighted-mean aggregate sequence under a reweighting."""

def compute_gap_decomposition(
    df_source: pd.DataFrame, df_target: pd.DataFrame,
    predicate: Predicate, group_col: str, measure_col: str,
    index: list[str], min_cell_support: int = 5,
) -> GapDecompositionResult:
    """Full Stage 7-10 orchestration."""
```

### `evaluation.py`
```python
def removal_baseline(
    df_source: pd.DataFrame, df_target: pd.DataFrame,
    predicate: Predicate, group_col: str, measure_col: str,
    index: list[str],
) -> float:
    """SDEcho's own dist_after-based reduction, for direct comparison
    against compute_gap_decomposition's explained fraction."""

def bootstrap_explained_fraction_ci(
    df_source: pd.DataFrame, df_target: pd.DataFrame,
    predicate: Predicate, group_col: str, measure_col: str,
    index: list[str], n_bootstrap: int = 1000, ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval on the explained fraction."""

def generate_synthetic_dataset(
    effect_size: float, n_per_group: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Synthetic two-group dataset with a KNOWN ground-truth explained
    fraction, for validating estimator correctness."""

def run_ablation(
    df: pd.DataFrame, config: dict, param_name: str, param_values: list,
) -> pd.DataFrame:
    """Sweeps one config parameter (e.g., min_cell_support), returns a
    results table for plotting sensitivity."""
```

### `visualization.py`
```python
def plot_sequence_comparison(
    result: GapDecompositionResult, index: list[str], title: str,
) -> matplotlib.figure.Figure:
    """Original vs. counterfactual vs. target sequence, side by side."""

def plot_gap_decomposition_bar(
    result: GapDecompositionResult,
) -> matplotlib.figure.Figure:
    """Bar chart: d_orig, d_cf, explained fraction annotation."""

def plot_weight_distribution(
    weights: pd.Series, diagnostics: ReweightingDiagnostics,
) -> matplotlib.figure.Figure:
    """Histogram of cell weights, for spotting extreme-weight cells."""

def render_diagnostics_table(
    diagnostics: ReweightingDiagnostics,
) -> pd.DataFrame:
    """Thesis-ready table: rows dropped, cells trimmed, weight range."""
```

---

## 5. Configuration

A single `config.py` (or `config.yaml`, either is fine — recommend a plain
Python dict for now to avoid adding a YAML dependency at this stage) holds
all pipeline parameters in one place, following `IMPLEMENTATION_RULES.md`'s
"no hard-coded paths, no magic numbers" rule:

```python
CONFIG = {
    "data_path": "data/stackoverflow2022.csv",
    "subgroup_col": "AgeGroup",
    "subgroup_val1": "25-34",
    "subgroup_val2": "35-44",
    "group_col": "YearsExpBucket",
    "measure_col": "ConvertedCompYearly",
    "agg_func": "mean",
    "candidate_attrs": ["EdLevel", "RemoteWork", "Country"],
    "max_order": 2,
    "sdecho_k": 10,
    "max_values_per_attr": 10,
    "sdecho_min_support": 20,
    "predicate_rank": 0,          # top-1
    "min_cell_support": 5,        # reweighting trimming threshold
    "reweight_direction": "A_to_B",  # "A_to_B" or "B_to_A"
    "n_bootstrap": 1000,
    "bootstrap_ci": 0.95,
}
```

Every function signature above takes its parameters explicitly (no hidden
global state) — `CONFIG` is unpacked once at the orchestration layer
(`thesis.ipynb` or a `run_pipeline.py` script) and passed down explicitly.
This keeps every `src/` function independently unit-testable without needing
to construct a full `CONFIG` object first.

---

## 6. Testing Strategy (maps to `docs/10_EVALUATION_PLAN.md`)

```
tests/
├── test_data_loader.py
├── test_sequence_builder.py
├── test_predicates.py
├── test_sdecho.py
├── test_reweighting.py       ← includes the hand-computable toy example
│                                 from docs/00_PROJECT_CONCEPT.md §5
└── test_evaluation.py        ← synthetic ground-truth recovery test
```

Each `src/` module gets a corresponding `tests/test_*.py` using `pytest`,
with at least one hand-computable example per function (small enough to
verify by hand, as in the India/USA toy example already worked through).
This is the "unit tests, sanity checks, synthetic examples" requirement from
`IMPLEMENTATION_RULES.md`, made concrete.

---

## 7. Why This Architecture (design rationale)

- **Modules mirror pipeline stages 1:1** — anyone reading the thesis
  methodology chapter can find the corresponding code in seconds; anyone
  reading the code can find the corresponding formula in `01_METHODOLOGY.md`
  in seconds. This traceability is valuable both for your own reproducibility
  and for a thesis committee auditing correctness.
- **`evaluation.py` never reimplements pipeline logic** — it only calls
  `reweighting.py`/`sdecho.py` functions with varied inputs (different
  configs, resampled data, synthetic data). This guarantees the numbers
  reported in the evaluation chapter are produced by the *exact same code*
  as the main pipeline, eliminating an entire class of "the eval script and
  the real pipeline silently diverged" bugs.
- **`visualization.py` is pure — takes result objects, returns figures, no
  side effects, no recomputation.** Keeps plotting logic swappable
  (matplotlib now, could move to plotly later) without touching any
  computational code.
