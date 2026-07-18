# Project Context

Project title

Extending SDEcho with Counterfactual Distributional Reweighting for Aggregate Sequence Explanation

---

Main Goal

The project extends SDEcho.

SDEcho explains why two aggregate sequences differ.

This thesis investigates how much of the sequence difference can be reduced by statistically aligning the distribution of explanatory attributes discovered by SDEcho.

The work does NOT perform causal intervention.

The work performs statistical counterfactual reweighting.

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

Gap Reduction

↓

Explained Fraction

↓

Residual Gap

---

Main Research Question

Given the predicate discovered by SDEcho,

how much of the sequence difference can be reduced by statistically aligning the distribution of that predicate?

---

Non-goals

We are NOT developing

- a new XQA algorithm

- a new search algorithm

- a causal inference method

- a fairness algorithm

The contribution is a second-stage analysis after SDEcho.