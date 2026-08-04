# SDEcho Predicate Reweighting Audit

This master's thesis project extends SDEcho with a statistical reweighting
stage for aggregate sequence explanation.

SDEcho discovers predicates that are influential under removal. This project
asks a different question: if the distribution of a SDEcho-discovered predicate
were aligned between two groups, would the aggregate sequence gap shrink,
remain, or grow?

The goal is not only gap reduction. The goal is to audit what kind of
explanation a SDEcho predicate provides.

## Main Interpretation Categories

- **Compositional explainer**: reweighting reduces the aggregate sequence gap.
- **Weak/non-compositional explainer**: SDEcho ranks the predicate highly, but
  reweighting explains little of the gap.
- **Gap-amplifying predicate**: reweighting increases the sequence distance.
- **Bucket-specific explainer**: reweighting helps in some buckets and hurts in
  others.
- **Proxy/residual indicator**: the predicate suggests hidden structure or
  within-cell differences, but does not identify a cause.

## Important Assumption

This project performs statistical counterfactual reweighting. It does not make
causal claims.

Acceptable interpretation:

> Aligning `Country` changes the counterfactual salary sequence and suggests
> residual heterogeneity across experience buckets.

Avoid:

> Country causes the salary gap.

## Pipeline

1. Load Stack Overflow Developer Survey data.
2. Define two comparison groups.
3. Build aggregate sequences.
4. Compute original sequence distance.
5. Run SDEcho to rank predicates.
6. Reweight the source group to match the target distribution over predicate
   attributes.
7. Build the counterfactual sequence.
8. Report distance change, explained fraction, residual gap, gap amplification,
   bucket-level diagnostics, and common-support diagnostics.

## Key Files

- `PROJECT_CONTEXT.md`: top-level project framing.
- `IMPLEMENTATION_REFRAMING.md`: current implementation assumptions and
  reporting rules.
- `IMPLEMENTATION_RULES.md`: coding and thesis-quality rules.
- `run_pipeline.py`: end-to-end pipeline.
- `src/reweighting.py`: core reweighting and decomposition logic.
- `src/evaluation.py`: baselines, bootstrap, and synthetic validation.
- `docs/00_PROJECT_CONCEPT.md`: thesis concept and research framing.
