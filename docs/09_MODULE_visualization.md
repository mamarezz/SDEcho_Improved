# Module: `visualization.py`

**Pipeline stage:** Not a pipeline stage — consumes result objects from
`reweighting.py` and `evaluation.py` to produce thesis-ready figures and
tables. Touches no raw data and performs no computation of its own.

**Status:** Extends the interface originally sketched in
`docs/02_ARCHITECTURE.md` §4 with two additional functions
(`plot_bootstrap_distribution`, `plot_ablation_sensitivity`) that became
necessary once `evaluation.py` (file 9) was specified — that module
produces bootstrap resample distributions and ablation sweep tables that
the original architecture doc's visualization interface didn't yet account
for. Noted as an extension, not a contradiction, of the original plan.

---

## 1. Why This Module Is Needed

Every number this pipeline produces (a sequence, a distance, an explained
fraction, a confidence interval) needs a corresponding visual for the
thesis — a reader will not absorb "explained fraction = 0.83, 95% CI
[0.71, 0.91]" as easily as a bar with an error whisker. Centralizing all
plotting in one module, strictly separated from computation, means:

- Every result object (`GapDecompositionResult`, bootstrap diagnostics,
  ablation tables) has exactly one place responsible for rendering it —
  consistent styling across the whole thesis, no copy-pasted matplotlib
  boilerplate scattered across notebook cells (directly serving
  `IMPLEMENTATION_RULES.md`'s "avoid duplicated code" / "avoid large
  notebook cells" rules).
- Figures can be regenerated from saved result objects without re-running
  the (expensive) SDEcho search — useful during thesis writing, when
  figures often need cosmetic revision (fonts, labels, colors) long after
  the underlying computation is done and shouldn't need to be redone.

## 2. Where It Fits in the Pipeline

```
reweighting.py                    evaluation.py
GapDecompositionResult      bootstrap CI diagnostics, ablation DataFrame
        │                                  │
        └────────────────┬─────────────────┘
                          ▼
                  visualization.py
   plot_sequence_comparison()
   plot_gap_decomposition_bar()
   plot_weight_distribution()
   plot_bootstrap_distribution()      ← new, see status note above
   plot_ablation_sensitivity()        ← new, see status note above
   render_diagnostics_table()
                          │
                          ▼
              figures/ (PNG/PDF) + tables/ (CSV/LaTeX)
              for direct inclusion in the thesis document
```

## 3. Design Principles

1. **Pure functions, no side effects.** Every function takes a result
   object (or plain data) and returns a `matplotlib.figure.Figure` (or a
   `pd.DataFrame` for tables) — it does not save files, does not call
   `plt.show()`, and does not mutate its inputs. Saving to disk is the
   caller's responsibility (typically one line in `thesis.ipynb`:
   `fig.savefig(...)`), keeping this module trivially testable (can assert
   on figure structure/data without touching the filesystem) and reusable
   in contexts other than the notebook (e.g., a future batch figure-export
   script).
2. **Matplotlib only** — matches the library already used throughout the
   original SDEcho reimplementation notebook; no new plotting dependency is
   introduced, consistent with `IMPLEMENTATION_RULES.md`'s preference for
   simplicity over unnecessary tooling.
3. **Every figure must be interpretable without the code that generated
   it** — axis labels, titles, and legends are mandatory arguments or are
   derived directly from the result object's own fields (e.g., predicate
   repr, explained fraction value annotated on the bar chart itself), not
   left as a caller responsibility that could be forgotten.

## 4. Public API

```python
def plot_sequence_comparison(
    result: GapDecompositionResult,
    index: list[str],
    title: str,
) -> matplotlib.figure.Figure:
    """
    Three-line plot: original source sequence, counterfactual (reweighted)
    source sequence, and target sequence, all over the same bucket index —
    the single most important figure in the results chapter, since it
    shows directly how much of the visual gap closes under reweighting.
    """


def plot_gap_decomposition_bar(
    result: GapDecompositionResult,
    removal_reduction: float | None = None,
) -> matplotlib.figure.Figure:
    """
    Bar chart: d_orig, d_cf (and, if provided, the removal-baseline
    equivalent distance) side by side, annotated with the explained
    fraction. Including removal_reduction here directly visualizes the
    central removal-vs-reweighting comparison (docs/08_MODULE_evaluation.md
    §3.1) in a single figure.
    """


def plot_weight_distribution(
    weights: pd.Series,
    diagnostics: ReweightingDiagnostics,
) -> matplotlib.figure.Figure:
    """
    Histogram of (non-NaN) cell weights, with the trimmed-row percentage
    annotated — the figure a reader needs to judge whether extreme weights
    (docs/01_METHODOLOGY.md §10, point 4) are a real concern for this
    particular result.
    """


def plot_bootstrap_distribution(
    bootstrap_diagnostics: dict,
    point_estimate: float,
    ci: tuple[float, float],
) -> matplotlib.figure.Figure:
    """
    Histogram of the bootstrap-resampled explained-fraction distribution
    (from evaluation.bootstrap_explained_fraction_ci's diagnostics dict),
    with the point estimate and CI bounds marked as vertical lines.
    """


def plot_ablation_sensitivity(
    ablation_df: pd.DataFrame,
    param_name: str,
    metric: str = "explained_fraction",
) -> matplotlib.figure.Figure:
    """
    Line plot of `metric` against the swept parameter values, from
    evaluation.run_ablation's output — the figure used to argue (or
    concede) that a given config choice (e.g., min_cell_support) doesn't
    materially change the headline result.
    """


def render_diagnostics_table(
    diagnostics: ReweightingDiagnostics,
) -> pd.DataFrame:
    """
    Thesis-ready single-row table: rows dropped, cells trimmed/valid,
    weight range — suitable for direct LaTeX export via
    DataFrame.to_latex() or inclusion as a markdown table.
    """
```

## 5. Implementation

```python
"""
visualization.py

Pure plotting/table-rendering functions. Takes result objects produced by
reweighting.py and evaluation.py; performs no computation, no I/O side
effects (no plt.show(), no file writes) — see docs/09_MODULE_visualization.md
§3 for why this separation matters.
"""

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import pandas as pd


def plot_sequence_comparison(
    result: "GapDecompositionResult",
    index: list[str],
    title: str,
) -> matplotlib.figure.Figure:
    """See docstring in module interface (§4)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(index, result.s_source_orig, marker="o", linewidth=2,
            label="Source (original)")
    ax.plot(index, result.s_source_cf, marker="^", linewidth=2,
            linestyle="--", label="Source (counterfactual, reweighted)")
    ax.plot(index, result.s_target, marker="s", linewidth=2,
            label="Target")
    ax.set_xlabel("Bucket")
    ax.set_ylabel("Aggregate value")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_gap_decomposition_bar(
    result: "GapDecompositionResult",
    removal_reduction: float | None = None,
) -> matplotlib.figure.Figure:
    """See docstring in module interface (§4)."""
    labels = ["Original\ndistance", "Counterfactual\ndistance\n(reweighting)"]
    values = [result.d_orig, result.d_cf]

    if removal_reduction is not None:
        d_removal_equiv = result.d_orig * (1 - removal_reduction)
        labels.append("Distance after\nremoval (SDEcho)")
        values.append(d_removal_equiv)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["tab:blue", "tab:green", "tab:orange"][:len(values)])
    ax.set_ylabel("Sequence distance")
    ax.set_title(
        f"Gap decomposition — {result.predicate}\n"
        f"Explained fraction: {result.explained_fraction:.1%}"
    )
    for bar, val in zip(bars, values):
        ax.annotate(f"{val:.0f}", (bar.get_x() + bar.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 4), ha="center")
    fig.tight_layout()
    return fig


def plot_weight_distribution(
    weights: pd.Series,
    diagnostics: "ReweightingDiagnostics",
) -> matplotlib.figure.Figure:
    """See docstring in module interface (§4)."""
    valid_weights = weights.dropna()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(valid_weights, bins=30, color="tab:blue", edgecolor="white")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1,
               label="weight = 1 (no adjustment)")
    ax.set_xlabel("Cell weight")
    ax.set_ylabel("Tuple count")
    ax.set_title(
        f"Weight distribution — {diagnostics.pct_dropped_rows:.1f}% of rows "
        f"trimmed (insufficient common support)"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def plot_bootstrap_distribution(
    bootstrap_diagnostics: dict,
    point_estimate: float,
    ci: tuple[float, float],
) -> matplotlib.figure.Figure:
    """See docstring in module interface (§4)."""
    fractions = bootstrap_diagnostics["resampled_fractions"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(fractions, bins=40, color="tab:purple", alpha=0.7)
    ax.axvline(point_estimate, color="black", linewidth=2, label="Point estimate")
    ax.axvline(ci[0], color="black", linestyle="--", linewidth=1, label="95% CI")
    ax.axvline(ci[1], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Bootstrapped explained fraction")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Bootstrap distribution (n={bootstrap_diagnostics['n_valid']} valid / "
        f"{bootstrap_diagnostics['n_bootstrap']} resamples, "
        f"{bootstrap_diagnostics['pct_failed']:.1f}% failed)"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def plot_ablation_sensitivity(
    ablation_df: pd.DataFrame,
    param_name: str,
    metric: str = "explained_fraction",
) -> matplotlib.figure.Figure:
    """See docstring in module interface (§4)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ablation_df[param_name], ablation_df[metric], marker="o")
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric)
    ax.set_title(f"Sensitivity of {metric} to {param_name}")
    ax.grid(True)
    fig.tight_layout()
    return fig


def render_diagnostics_table(diagnostics: "ReweightingDiagnostics") -> pd.DataFrame:
    """See docstring in module interface (§4)."""
    return pd.DataFrame([{
        "Attributes": ", ".join(diagnostics.attrs),
        "Rows (source)": diagnostics.n_source_rows,
        "Rows dropped": diagnostics.n_dropped_rows,
        "% dropped": diagnostics.pct_dropped_rows,
        "Cells (valid)": diagnostics.n_cells_valid,
        "Cells (no target overlap)": diagnostics.n_cells_no_target_overlap,
        "Cells (below min support)": diagnostics.n_cells_below_min_support,
        "Min cell support (τ)": diagnostics.min_cell_support,
        "Weight range": f"[{diagnostics.min_weight:.3f}, {diagnostics.max_weight:.3f}]",
    }])
```

## 6. Known Gaps / TODOs

- **No saving/export convenience function yet** — every figure is returned
  to the caller, who is responsible for `fig.savefig(path, dpi=..., 
  bbox_inches="tight")`. Consider adding a thin
  `save_figure(fig, path, **kwargs)` wrapper purely to standardize DPI/
  format choices across the thesis's figures — a cosmetic convenience, not
  a correctness issue, low priority.
- **`plot_gap_decomposition_bar`'s removal-equivalent bar is a derived
  quantity** (`d_orig * (1 - removal_reduction)`), not stored directly on
  any result object — this is a display-only computation (not silently
  duplicating pipeline logic, since `removal_reduction` itself still comes
  from `evaluation.removal_baseline_reduction()`), but worth a comment in
  the code (already present) so it's not mistaken for a hidden
  recomputation of the distance itself.
- **No color/style consistency constants yet** (colors are hardcoded per
  function: `tab:blue`, `tab:green`, etc.) — fine for a first pass; if the
  thesis ends up with many figures, extracting a shared style module
  (`plt.style.use(...)` + a small palette dict) would improve visual
  consistency. Deferred as a polish step, not blocking.

## 7. How Reviewer #2 Would Critique This Module

- *"Your gap-decomposition bar chart puts SDEcho's removal-based distance
  next to your reweighting-based distance as if they're directly
  comparable bars — are they actually on the same footing?"* — This is a
  legitimate framing question: `d_removal_equiv` is a distance computed
  over a group with *fewer tuples* (post-removal), while `d_cf` is computed
  over the *same* tuple count (reweighted, not removed). The bar chart
  should carry a caption note making this distinction explicit, or a
  reader could misread the comparison as more apples-to-apples than it
  actually is. Add this caveat to the figure's caption text in the thesis,
  not just this doc.
- *"Bootstrap histogram shows resampled fractions but doesn't show the
  failure cases — could the true uncertainty be understated because
  failed resamples were silently excluded?"* — Yes, and this should be
  stated explicitly in the thesis text near this figure: the CI is
  conditional on cells retaining common support under resampling, and the
  `pct_failed` annotation in the title is there specifically so this
  caveat isn't hidden.

## 8. Complexity

All functions are $O(n)$ or better in their input size (a sequence of
length $|\mathcal{B}|$, a weight series of length $n$, or an ablation table
of length $|\text{param\_values}|$) — plotting cost is negligible relative
to any upstream computation stage.

## 9. Tests to Write (`tests/test_visualization.py`)

Since these are plotting functions, tests focus on **structural
correctness**, not pixel-level rendering:

1. **`plot_sequence_comparison` line count**: assert the returned figure
   has exactly 3 line series (source, counterfactual, target) with the
   expected labels.
2. **`plot_gap_decomposition_bar` bar count**: assert 2 bars when
   `removal_reduction=None`, 3 when provided.
3. **`render_diagnostics_table` schema**: assert the returned DataFrame has
   exactly the expected columns and one row, given a known
   `ReweightingDiagnostics` instance.
4. **No side effects**: assert none of these functions call `plt.show()`
   (can check via monkeypatching `plt.show` and asserting it's never
   invoked) — a regression test for the "pure function" design principle
   in §3.1.

## 10. Thesis-Ready Description

> Results are visualized via three complementary figure types: (1) a
> sequence-comparison plot showing the original source, counterfactual
> (reweighted) source, and target sequences over the shared bucket index,
> directly illustrating the visual effect of reweighting; (2) a gap-
> decomposition bar chart contrasting the original distance, the
> reweighting-based counterfactual distance, and — where computed — the
> equivalent distance under SDEcho's own removal-based mechanism, annotated
> with the explained fraction; and (3) diagnostic figures (weight
> distribution, bootstrap resample distribution) supporting the validity
> checks discussed in [Section: Limitations]. All figures are generated
> from stored result objects independent of re-running the underlying
> computation, ensuring reproducibility of the thesis's figures from a
> fixed set of saved experimental results.