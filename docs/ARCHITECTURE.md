# Architecture

The active data flow is:

```text
raw groups
   -> weighted mean sequences
   -> weighted SDEcho removal ranking
   -> exact predicate-event constraint
   -> cumulative calibration in a frozen intervention-bucket set
   -> residual weighted sequences
   -> repeat
```

## Module responsibilities

- `sequence_builder.py` validates positional weights, computes Kish effective
  sample size, and constructs strict weighted-mean sequences.
- `sdecho.py` evaluates fixed-weight tuple removal. It never recalibrates while
  scoring a candidate.
- `balancing.py` performs cyclic calibration over every selected
  predicate-by-bucket margin.
- `iterative_sdecho.py` owns bounded candidate-pool acceptance, rejection
  auditing, quality thresholds, ordered steps, cumulative effects, and stopping
  reasons.
- `visualization.py` converts typed results to transparent tables and plots.

## State carried between iterations

The raw source and target frames never change. The target weights remain fixed.
The source has a positional weight vector. After a predicate is accepted, that
vector is recomputed from mean-one base weights under all selected balance
constraints in the frozen intervention buckets. Rows in other buckets remain
at base weight. This prevents later correlated predicates from undoing earlier
balance while preserving one coherent weighted dataset.

## Result contract

Each `IterationStep` stores the selected predicate, weighted SDEcho score,
full-sequence distance before and after balancing, search-bucket distance after
removal, source sequence before and after, cumulative reduction, and
weight-quality diagnostics. Search and intervention use the same frozen bucket
set; the full residual still includes untouched buckets.

`IterativeSDEchoResult` stores the complete path, final weights, final residual,
and an explicit stopping reason. `CandidateRejection` records why a candidate
returned in the configured top-k pool failed a subsequent iterative acceptance
gate; it is not a log of predicates filtered inside SDEcho or outside the pool.
