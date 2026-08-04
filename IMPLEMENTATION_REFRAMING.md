# Implementation Reframing

## Purpose

The implementation should support the thesis as an explanation-auditing
pipeline, not only as a gap-reduction pipeline.

SDEcho discovers predicates that are influential under removal. The
reweighting stage audits whether those predicates also behave as compositional
explanations when the source group is statistically aligned to the target
group.

## Core Rule

Do not treat a small or negative explained fraction as an implementation
failure by default.

A negative explained fraction means the counterfactual sequence is farther from
the target than the original sequence. This should be reported as **gap
amplification**, not hidden or clipped to zero.

## Predicate Interpretation Categories

Each SDEcho predicate analyzed through reweighting should be classified as one
of the following:

- **Compositional explainer**: reweighting reduces the aggregate sequence
  distance meaningfully.
- **Weak/non-compositional explainer**: SDEcho ranks the predicate highly, but
  reweighting explains little of the distance.
- **Gap-amplifying predicate**: reweighting increases the distance.
- **Bucket-specific explainer**: reweighting reduces the gap in some buckets
  and increases it in others.
- **Proxy/residual indicator**: the predicate suggests possible hidden
  structure, but the result should not be interpreted causally.

## Required Diagnostics

Every main result should report:

- original source sequence,
- target sequence,
- counterfactual source sequence,
- original distance,
- counterfactual distance,
- explained fraction,
- residual gap,
- whether the result is gap-reducing or gap-amplifying,
- per-bucket original gap,
- per-bucket counterfactual gap,
- per-bucket change in absolute gap,
- common-support diagnostics,
- weight range and dropped-row percentage,
- bootstrap uncertainty when feasible.

## Causal Language Policy

The implementation and generated result text must not claim that a predicate
causes the observed salary difference.

Acceptable language:

- "is associated with"
- "is consistent with"
- "suggests residual heterogeneity"
- "may act as a proxy for"
- "generates a hypothesis"

Avoid:

- "causes"
- "the market prefers"
- "the effect of country is"
- "changing this attribute would produce"

## First Implementation Priority

The first implementation step is to create a predicate-audit result layer that
computes and stores per-bucket diagnostics and assigns an interpretation
category to each reweighted predicate.

This should be implemented before adding new plots or more sequential
experiments, because the interpretation category becomes the stable reporting
unit for the thesis.
