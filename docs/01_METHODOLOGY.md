# Methodology

This document is the formal mathematical companion to `docs/00_PROJECT_CONCEPT.md`.
Every symbol and formula here maps directly onto a function in `src/`, so this
file can be lifted near-verbatim into the thesis's Methodology chapter, with
each subsection cross-referenced to its implementation.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $D_A, D_B$ | Tuple sets (DataFrames) for the two comparison groups (e.g., AgeGroup=25-34, AgeGroup=35-44) |
| $n_A, n_B$ | $\lvert D_A \rvert, \lvert D_B \rvert$ |
| $\mathcal{B}$ | Ordered set of buckets on the x-axis of the aggregate sequence (e.g., experience buckets `{0-2, 3-5, 6-10, 10-20}`) |
| $b$ | A single bucket, $b \in \mathcal{B}$ |
| $\text{group\_col}(t)$ | The bucketing attribute value of tuple $t$ |
| $\text{measure\_col}(t)$ | The numeric outcome value of tuple $t$ (e.g., `ConvertedCompYearly`) |
| $s_G[b]$ | Aggregate value (mean) of group $G$ within bucket $b$ |
| $s_G$ | The full aggregate sequence for group $G$, i.e., $(s_G[b])_{b \in \mathcal{B}}$ |
| $P$ | A predicate: a set of (attribute, value) pairs, e.g. $\{(\text{Country}, \text{USA})\}$ |
| $X$ | The attribute set of $P$, i.e., $X = \{\text{attr} : (\text{attr}, \cdot) \in P\}$ |
| $x(t)$ | The joint-cell value of tuple $t$ over attributes $X$ |
| $w(t)$ | Reweighting factor assigned to tuple $t$ |
| $d(\cdot, \cdot)$ | Sequence distance function |

---

## 2. Aggregate Sequence (Stage 3 — unchanged from SDEcho)

$$
s_G[b] \;=\; \frac{1}{|D_G^{(b)}|} \sum_{t \in D_G^{(b)}} \text{measure\_col}(t),
\qquad D_G^{(b)} = \{t \in D_G : \text{group\_col}(t) = b\}
$$

If $D_G^{(b)} = \emptyset$, we define $s_G[b] = 0$ by convention (matching the
existing `_aggregate_sequence` implementation via `fillna(0)`).

> **Implementation:** `_aggregate_sequence()` in `sequence_builder.py`
> (reused unmodified from the original SDEcho reimplementation).

---

## 3. Sequence Distance (Stage 4 — unchanged from SDEcho)

$$
d(s_1, s_2) \;=\; \sqrt{\sum_{b \in \mathcal{B}} \left(s_1[b] - s_2[b]\right)^2}
$$

Ordinary Euclidean distance over the bucket-indexed vectors. Kept identical
to SDEcho's own distance so that SDEcho's `dist_before`/`dist_after` and this
thesis's $d_{\text{orig}}$/$d_{\text{cf}}$ are directly comparable.

> **Implementation:** `_dist()` in `sdecho.py` (reused unmodified).

---

## 4. SDEcho Predicate Search (Stage 5 — unchanged, brute-force)

For each candidate predicate $P$ built from `candidate_attrs` up to
`max_order` attributes, with masks $M_1(P) = \{t \in D_A : t \models P\}$ and
$M_2(P) = \{t \in D_B : t \models P\}$:

$$
\gamma(P) \;=\; \frac{d\big(s_{D_A \setminus M_1(P)},\; s_{D_B \setminus M_2(P)}\big)}{d(s_{D_A}, s_{D_B})} \cdot \left(1 + \frac{|M_1(P)|}{n_A} + \frac{|M_2(P)|}{n_B}\right)
$$

Lower $\gamma$ = stronger explanation (removing $P$-matching tuples reduces
the distance more, penalized by how large a fraction of each group $P$
matches). Predicates are ranked ascending by $\gamma$, filtered by
`min_support` on $|M_1(P)| + |M_2(P)|$.

> **Implementation:** `compute_gamma()` in `sdecho.py` (reused unmodified;
> this is a *reimplementation*, not the original authors' code — see
> `docs/11_ASSUMPTIONS_AND_LIMITATIONS.md` for the fidelity discussion this
> implies).

---

## 5. Predicate Selection (Stage 6)

$$
P^{*} = \arg\min_{P \in \text{top-}k} \gamma(P)
$$

By default $k=1$ (top-1 predicate only). The set of covariate attributes
carried forward into reweighting is:

$$
X = \{\text{attr} : (\text{attr}, \cdot) \in P^{*}\}
$$

Note: the specific *value* of each attribute in $P^{*}$ (e.g., `USA`) is
**not** used beyond identifying which attribute(s) matter — reweighting
operates on the full joint domain of $X$, not only the value(s) named in
$P^{*}$. This is a deliberate design choice: SDEcho's predicate tells us
*which attributes* drive the divergence; reweighting then aligns the *entire*
distribution over those attributes, not just the one flagged value.

> **Implementation:** `get_covariate_attrs()` in `reweighting.py`.

---

## 6. Cell-Based Reweighting (Stage 7 — this thesis's core contribution)

### 6.1 Joint cell definition

$$
x(t) = \big(t.\text{attr}_1, \dots, t.\text{attr}_k\big), \quad \{\text{attr}_1,\dots,\text{attr}_k\} = X
$$

For $|X|=1$, $x(t)$ is simply the single attribute's value; for $|X|>1$,
$x(t)$ is a tuple, so reweighting matches the **joint** distribution over
$X$, not independent marginals (see `docs/00_PROJECT_CONCEPT.md` §6 and
`docs/11_ASSUMPTIONS_AND_LIMITATIONS.md` for the marginal-vs-joint tradeoff).

### 6.2 Empirical cell proportions

$$
\hat p_G(x) = \frac{\big|\{t \in D_G : x(t) = x\}\big|}{n_G}, \qquad G \in \{A, B\}
$$

### 6.3 Weight formula

For source group $D_A$ (reweighted) and target group $D_B$ (aligned to):

$$
w(t) =
\begin{cases}
\dfrac{\hat p_B\big(x(t)\big)}{\hat p_A\big(x(t)\big)} & \text{if } x(t) \in \mathcal{C}_{\text{valid}} \\[2mm]
\text{undefined (tuple trimmed)} & \text{otherwise}
\end{cases}
$$

where the **valid cell set** is:

$$
\mathcal{C}_{\text{valid}} = \Big\{\, x \;:\; x \in \text{dom}_A \cap \text{dom}_B,\;\; \text{count}_A(x) \geq \tau,\;\; \text{count}_B(x) \geq \tau \,\Big\}
$$

with $\tau = $ `min_cell_support` (default $\tau = 5$), and
$\text{dom}_G = \{x(t) : t \in D_G\}$.

By construction, after reweighting:

$$
\sum_{t \in D_A,\, x(t)=x} w(t) \Big/ \sum_{t \in D_A} w(t) \;=\; \hat p_B(x) \qquad \forall\, x \in \mathcal{C}_{\text{valid}}
$$

i.e., group $A$'s weighted distribution over $X$ exactly reproduces group
$B$'s empirical distribution, restricted to the valid (common-support) cells.

> **Implementation:** `compute_cell_weights()` in `reweighting.py`.

### 6.4 Direction of reweighting

Reweighting is directional: $D_A \to D_B$ means "reweight $A$'s covariate
distribution to match $B$'s." The reverse direction ($D_B \to D_A$) is a
distinct, generally non-symmetric operation — a known property of
Oaxaca–Blinder-style decompositions. The default direction is
`subgroup_val1 → subgroup_val2`, configurable per experiment.

---

## 7. Counterfactual Aggregate Sequence (Stage 8)

$$
s_A^{\text{cf}}[b] \;=\; \frac{\displaystyle\sum_{t \in D_A^{(b)},\; w(t)\text{ defined}} w(t)\cdot \text{measure\_col}(t)}{\displaystyle\sum_{t \in D_A^{(b)},\; w(t)\text{ defined}} w(t)}
$$

Tuples with undefined (trimmed) weight are excluded from both numerator and
denominator — equivalent to treating them as unobserved under this
counterfactual, not as zero-valued. If a bucket $b$ has no tuples with
defined weight, $s_A^{\text{cf}}[b] = 0$ by the same convention as §2.

> **Note:** this formula assumes `agg_func = "mean"`. Extending to `"sum"` or
> `"count"` requires a different weighted-aggregation formula and is
> explicitly **not implemented** (see `docs/11_ASSUMPTIONS_AND_LIMITATIONS.md`).

> **Implementation:** `weighted_aggregate_sequence()` in `reweighting.py`.

---

## 8. Gap Decomposition (Stages 9–10)

$$
d_{\text{orig}} = d(s_A, s_B), \qquad d_{\text{cf}} = d(s_A^{\text{cf}}, s_B)
$$

$$
\boxed{\;\text{ExplainedFraction}(P^{*}) = \dfrac{d_{\text{orig}} - d_{\text{cf}}}{d_{\text{orig}}}\;} \qquad \text{ResidualGap} = d_{\text{cf}}
$$

**Range and interpretation:**
- $\text{ExplainedFraction} = 1$: reweighting on $X$ fully closes the gap.
- $\text{ExplainedFraction} = 0$: reweighting on $X$ has no effect on the gap.
- $\text{ExplainedFraction} < 0$: reweighting **increases** the distance.
  This is a legitimate, reportable outcome (not an error) — it can occur
  because per-bucket weighted means interact non-monotonically with the
  Euclidean aggregation across buckets. Any occurrence must be flagged and
  discussed, not discarded.
- $\text{ExplainedFraction}$ is undefined when $d_{\text{orig}} = 0$
  (identical original sequences — nothing to explain).

> **Implementation:** `compute_gap_decomposition()` in `reweighting.py`.

---

## 9. Complexity

| Step | Complexity | Notes |
|---|---|---|
| Aggregate sequence construction | $O(n_A + n_B)$ | Single groupby |
| SDEcho predicate search | $O\big(n_A \cdot n_B \cdot \lvert \mathcal{P}\rvert\big)$ | $\lvert\mathcal{P}\rvert$ = number of enumerated predicates, exponential in `max_order` and `max_values_per_attr` — this is SDEcho's brute-force cost, unrelated to reweighting |
| Cell weight computation | $O(n_A + n_B)$ | `value_counts` + hash join, linear |
| Weighted aggregate sequence | $O(n_A)$ | Single groupby |
| **Total reweighting stage (7–10)** | $O(n_A + n_B)$ | Linear; no search, no optimization — consistent with the project's "no new optimization algorithm" constraint |

The reweighting stage is asymptotically cheap; nearly all computational cost
in the full pipeline is attributable to SDEcho's brute-force predicate
search (Stage 5), unchanged from the original implementation.

---

## 10. Assumptions Introduced by This Stage (summary — full discussion in doc 11)

1. **Categorical, discrete covariates.** Cell-based reweighting requires
   finite, enumerable joint cells; not applicable to continuous covariates
   without discretization (not needed here, since SDEcho's `candidate_attrs`
   are already categorical).
2. **Common support.** Cells absent from either group, or below
   `min_cell_support`, are trimmed; trimming is data loss and must be
   reported (`pct_dropped_rows` in diagnostics).
3. **Ignorability with respect to $X$ only.** The decomposition attributes
   gap reduction only to differences in the distribution over $X$ — the
   attribute(s) SDEcho found. It does **not** account for any other
   unmeasured or unmodeled covariate that might also differ between groups.
4. **Static, unstabilized weights.** No capping/winsorization is applied by
   default; extreme weights from rare-but-valid cells can inflate the
   variance of $s_A^{\text{cf}}$. This is deferred to an empirical robustness
   check (see `docs/10_EVALUATION_PLAN.md`) rather than pre-emptively solved.
5. **No causal interpretation.** Per `docs/00_PROJECT_CONCEPT.md` §7,
   ExplainedFraction is a descriptive counterfactual estimate under an
   observational reweighting, not a causal effect estimate.
