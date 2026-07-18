# Project Concept

## Thesis Title (working)

**Counterfactual Reweighting for Aggregate Sequence Explanation: Extending SDEcho with Compositional Gap Decomposition**

---

## 1. One-Paragraph Summary

SDEcho explains *why* two aggregate sequences differ by discovering a predicate
(e.g., `Country = USA`) whose associated tuples, when removed, most reduce the
distance between the sequences. This thesis adds a second stage: instead of
removing those tuples, we **reweight** one group's tuples so that its
distribution over the predicate's attribute(s) statistically matches the other
group's distribution — without deleting any data. We then re-measure the
sequence distance under this counterfactual reweighting and report what
fraction of the original gap is **explained** by the compositional
(distributional) difference on that attribute, versus what fraction remains
as an **unexplained residual**. This is a statistical decomposition exercise
(in the spirit of Oaxaca–Blinder / DiNardo–Fortin–Lemieux), not a causal
claim.

---

## 2. Research Question

> Given a predicate discovered by SDEcho that explains the divergence between
> two aggregate sequences, how much of that divergence can be attributed to
> differing distributions over the predicate's attribute(s) — as opposed to
> other, residual differences — when quantified via **reweighting** rather
> than **removal**?

A secondary, comparison-oriented question:

> Do SDEcho's removal-based reduction (`dist_after` when matching tuples are
> deleted) and this thesis's reweighting-based explained fraction agree, and
> if not, by how much and why?

---

## 3. Background — Why This Question Is Open

SDEcho computes, for a candidate predicate `P`:

- `dist_before`: the original sequence distance.
- `dist_after`: the sequence distance recomputed with `P`-matching tuples
  **removed**.
- `gamma = (dist_after / dist_before) * penalty`, used to rank predicates.

`dist_after` answers: *"if `P`-matching tuples didn't exist at all, how
different would the sequences be?"* This is a destructive counterfactual — it
changes sample size and discards the removed tuples' contribution entirely.

It does **not** answer a different, arguably more natural question: *"if the
two groups simply had the same **proportions** of `P`-matching tuples (rather
than none), how different would the sequences be?"* This compositional
question is exactly what classical decomposition methods (Oaxaca–Blinder,
DFL) answer for scalar or distributional outcomes — but it has not, to our
knowledge, been connected to XQA-style sequence-difference explanation
(SDEcho, Scorpion, COMPARE, Reptile), nor applied to structured, per-bucket
aggregate **sequences** rather than a single scalar gap.

This thesis fills that specific, narrow gap: **automatically discovering the
reweighting covariate via SDEcho, and applying it to sequence-valued
outcomes rather than scalar means.**

---

## 4. Contribution Statement (final wording, use in abstract/intro)

> We propose a two-stage pipeline that (1) reuses SDEcho to automatically
> discover an explanatory predicate for the divergence between two aggregate
> sequences, and (2) applies exact cell-based (joint-distribution) reweighting,
> seeded by that predicate, to estimate a counterfactual "aligned-covariate"
> sequence and quantify what fraction of the sequence-level gap is
> attributable to compositional (covariate distribution) differences versus
> residual differences. Unlike classical Oaxaca–Blinder/DFL decomposition,
> the reweighting covariate is not manually chosen by the analyst but is
> automatically discovered via an existing XQA explanation method, and the
> outcome being decomposed is a structured aggregate sequence rather than a
> scalar or single distribution. We further show empirically that this
> reweighting-based notion of "explained gap" is systematically different
> from SDEcho's own removal-based reduction metric.

---

## 5. Intuition (plain-language, for thesis introduction)

Two groups (A: ages 25–34, B: ages 35–44) have different average salaries by
experience bucket. SDEcho finds that `Country` explains much of this: Group A
has more India-based developers, Group B has more USA-based developers.

Instead of asking *"what if India-based developers didn't exist in Group A?"*
(removal — SDEcho's own mechanism), we ask *"what if Group A had the same
India/USA mix as Group B?"* (reweighting — this thesis's mechanism). We
answer this by counting each of Group A's tuples more or less heavily
(a weight), so that Group A's weighted country distribution exactly matches
Group B's, without deleting anyone. Recomputing Group A's aggregate sequence
under these weights gives a **counterfactual sequence**; comparing its
distance to Group B's sequence against the *original* distance yields the
**explained fraction**.

---

## 6. Scope

### In scope
- One dataset: Stack Overflow Developer Survey 2022.
- One pairwise group comparison at a time (configurable, not fixed to one pair).
- SDEcho, reimplemented brute-force (not the original paper's optimized
  search — search efficiency is explicitly not our contribution).
- Top-1 (extendable to top-k) SDEcho predicate as the reweighting covariate.
- Exact cell-based (joint-distribution) categorical reweighting — no
  propensity-score models, no logistic regression, no optimization search.
- Comparison against SDEcho's own removal-based reduction as a baseline.
- One synthetic dataset with a known, controlled ground-truth compositional
  effect, for validating correctness.
- Bootstrap confidence intervals on the explained fraction.
- Explicit common-support / cell-trimming policy and reporting.

### Out of scope (explicitly, with reason)
- **Temporal / multi-year analysis** — descoped to keep the thesis
  achievable; noted as future work.
- **Scorpion / point-level tuple-influence explanation** — a different
  granularity of explanation, not needed for the reweighting question;
  dropped to keep the pipeline focused and coherent.
- **Causal discovery / causal graphs / admissible-set reasoning
  (Salimi-style database repair)** — explicitly disclaimed; this thesis
  performs statistical reweighting, not causal intervention.
- **Propensity-score-model-based IPW (logistic regression)** — unnecessary
  given categorical, low-cardinality covariates from SDEcho predicates.
- **Search over reweighting covariates** — the covariate set is always taken
  from SDEcho's output, never independently searched or optimized.
- **Continuous-covariate / kernel-density (full DFL) reweighting** — not
  needed; SDEcho predicates are categorical attribute-value conjunctions.

---

## 7. Non-Goals (explicit disclaimers to repeat throughout the thesis)

- We do **not** claim that intervening on the real-world population (e.g.,
  changing the country composition of Group A) would reproduce the estimated
  gap reduction. This is a **descriptive counterfactual estimate under an
  ignorability-style assumption** (no unmeasured confounding beyond the
  attributes in the discovered predicate), not a validated causal effect.
- We do **not** propose a new predicate-discovery algorithm — SDEcho is used
  as-is (reimplemented, brute-force).
- We do **not** propose a new reweighting algorithm — exact cell-based
  reweighting is a well-established special case of inverse-probability
  weighting for categorical covariates.

---

## 8. Pipeline (final, approved version)

```
Stage 1  Load dataset
Stage 2  Define two comparison groups
Stage 3  Construct aggregate sequences (identical to SDEcho)
Stage 4  Compute sequence distance (Euclidean, identical to SDEcho)
Stage 5  Run SDEcho -> ranked explanatory predicates
Stage 6  Select predicate (top-1, extendable to top-k)
Stage 7  Exact cell-based reweighting on predicate's attribute(s)
Stage 8  Recompute weighted (counterfactual) aggregate sequence
Stage 9  Recompute sequence distance under reweighting
Stage 10 Report: original distance, counterfactual distance,
         explained fraction, residual gap, reweighting diagnostics
```

Stages 1–5 already exist (reused, not duplicated) from the original SDEcho
reimplementation. Stages 6–10 are this thesis's contribution.

---

## 9. Glossary

| Term | Meaning in this thesis |
|---|---|
| **Aggregate sequence** | A vector of aggregate values (e.g., mean salary) over an ordered bucketing attribute (e.g., years of experience), for one group. |
| **Predicate** | A conjunction of attribute=value conditions (e.g., `Country=USA & EdLevel=PhD`) that SDEcho identifies as explanatory. |
| **Cell** | A distinct joint combination of values of the predicate's attribute(s); the unit of reweighting. |
| **Reweighting** | Assigning a multiplicative weight to each tuple in the source group so that its weighted covariate distribution matches the target group's empirical distribution. |
| **Explained fraction** | `(d_original - d_counterfactual) / d_original` — the proportion of the sequence gap attributable to compositional (distributional) differences on the predicate's attributes. |
| **Residual gap** | `d_counterfactual` — what remains of the sequence distance after compositional alignment. |
| **Common support** | The requirement that a covariate cell have sufficient observations in *both* groups to be reliably reweighted; violations are trimmed and reported. |
| **Counterfactual (as used here)** | A statistical "what-if" under a reweighting operation on observed data — explicitly **not** a causal intervention claim. |

---

## 10. Relationship to Prior Work (short form — expand in related work chapter)

| Method | Grain | Mechanism | Relation to this thesis |
|---|---|---|---|
| **SDEcho** | Sequence-level | Predicate discovery via removal-based distance reduction | Reused unmodified as the front-end predicate discoverer |
| **Scorpion** | Point-level | Tuple influence within a single aggregate outlier | Not used (descoped) |
| **Oaxaca–Blinder** | Scalar mean gap | Manual covariate choice, explained/unexplained split | Mechanism adapted to sequence-valued outcomes; covariate is automatically discovered, not manually chosen |
| **DiNardo–Fortin–Lemieux** | Full distribution | Kernel reweighting | Conceptual ancestor; we use the simpler, exact discrete special case (no kernel density estimation needed for categorical data) |
| **Salimi et al. (causal DB repair)** | Tuple/database level | Causal admissible-set reasoning, minimal repair | Explicitly weaker causal ambition (no causal graph assumed); explicitly stronger automatic discoverability (covariate found by SDEcho, not specified by analyst) |
