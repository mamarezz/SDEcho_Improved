# Evaluation plan

## Positive control

The deterministic generator plants three independent prevalence shifts with
identical conditional premiums in both groups. The method must recover the
exact order and exact distance path recorded in
`synthetic_data/ground_truth.json`.

## Calibration regression

A correlated two-predicate fixture verifies that cyclic calibration requires
multiple passes and preserves both target margins. This prevents a one-pass
implementation from silently undoing the first constraint.

## Required diagnostics

Every accepted iteration reports:

- raw source and target support;
- weighted predicate prevalence;
- search-bucket distance after removal and full-sequence distance after
  balancing;
- original-scale and cumulative distance reduction;
- effective sample size and its source-sample ratio;
- minimum and maximum positive weight;
- maximum balance error.

The final accepted constraint set additionally exports per-intervention-bucket
raw/active event and complement counts, event/complement effective sample
sizes, and a balance table that marks constrained versus untouched buckets.
Candidate failures inside the configured top-k pool are exported separately.

## Stress tests to add before final empirical claims

- correlated and nested planted predicates;
- equal prevalence with conditional outcome differences;
- a gap-amplifying candidate;
- sparse or infeasible support;
- heavy-tailed outcomes and a robust aggregation sensitivity;
- discovery on one split and evaluation on held-out data;
- reverse target direction and alternative first-step paths.

The deterministic benchmark is a correctness test, not evidence that removal
and prevalence balancing are generally equivalent.
