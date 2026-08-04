# Project Context

Project title

Extending SDEcho with Counterfactual Distributional Reweighting for Aggregate Sequence Explanation

---

Main Goal

The project extends SDEcho with a diagnostic second stage.

SDEcho identifies predicates that are influential under a removal-based
counterfactual.

This thesis investigates what those predicates mean under statistical
distributional alignment. The goal is not necessarily to reduce the gap.
The goal is to distinguish whether a SDEcho-discovered predicate behaves as:

- a compositional explainer,
- a weak or non-compositional explainer,
- a gap-amplifying predicate,
- a bucket-specific explainer,
- or a proxy for deeper residual/within-cell differences.

The work does NOT perform causal intervention or causal discovery.

The work performs statistical counterfactual reweighting and uses the results
to generate disciplined hypotheses about possible underlying mechanisms.

---

Main Papers

1. Efficient Explanation of Aggregated Sequence Difference (SDEcho)

This is the primary paper.

2. XQA Survey

This survey summarizes previous XQA methods.

---

Dataset

Primary dataset:

Stack Overflow Developer Survey 2022

---

Current Scope

One dataset

One comparison

One aggregate sequence

No temporal analysis

No Scorpion

No causal discovery

No optimization

No deep learning

---

Current Pipeline

Dataset

↓

Aggregate Query

↓

Two Aggregate Sequences

↓

Sequence Distance

↓

SDEcho

↓

Top Predicate

↓

Counterfactual Reweighting

↓

Weighted Sequence

↓

Counterfactual Gap Change

↓

Explained Fraction / Gap Amplification

↓

Residual Gap and Bucket-Level Diagnostics

---

Main Research Question

Given a predicate discovered by SDEcho,

does statistically aligning the distribution of that predicate reduce,
preserve, or amplify the aggregate sequence difference?

What does the disagreement between SDEcho's removal-based explanation and
the reweighting-based counterfactual reveal about composition, residual
within-cell differences, and bucket-level heterogeneity?

---

Non-goals

We are NOT developing

- a new XQA algorithm

- a new search algorithm

- a causal inference method

- a fairness algorithm

The contribution is a second-stage analysis after SDEcho.

The contribution should be framed as explanation auditing, not simply as
improving SDEcho's gap reduction.
