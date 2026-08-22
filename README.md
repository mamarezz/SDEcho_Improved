# Sequential Counterfactual SDEcho

This project extends SDEcho into an ordered residual-explanation procedure for
aggregate sequences.

At every iteration it:

1. runs weighted SDEcho removal on the current pseudo-populations;
2. considers the configured top weighted-SDEcho pool and selects its
   highest-ranked predicate that passes support, weight, and positive-gain
   gates;
3. balances the exact binary predicate event in the source group to the fixed
   target group within a bucket set frozen from the original sequences;
4. recalibrates all previously selected predicate constraints together;
5. reruns SDEcho on the remaining weighted difference.

The output is an ordered path such as:

```text
Initial gap
  -> Country=USA
  -> Education=Doctoral, conditional on Country balance
  -> RemoteWork=Remote, conditional on both earlier balances
  -> residual gap
```

This is a descriptive counterfactual analysis, not a causal estimator.

## Project structure

```text
foundation.txt                 Complete theory, formulas, and assumptions
run_pipeline.py                Real Stack Overflow 2022 pipeline
src/
  predicates.py                Predicate representation and enumeration
  sequence_builder.py          Weighted sequence construction and ESS
  sdecho.py                    Weighted removal search
  balancing.py                 Cumulative predicate-event calibration
  iterative_sdecho.py          Iterative orchestration and result types
  data_loader.py               Survey loading and group preparation
  visualization.py             Path tables and figures
synthetic_data/
  generator.py                 Deterministic ordered ground truth
  benchmark.py                 End-to-end recovery runner
  visualization.py             Oracle-versus-recovery figure
  ground_truth.json            Human-readable expected order and distances
  results/                     Current tables and PNG/SVG figures
  archive/                     Clearly labeled retired benchmark outputs
tests/                         Unit and end-to-end validation
docs/                          Compact implementation and evaluation guides
```

## Run the deterministic validation

```powershell
python -m synthetic_data.benchmark
```

This command also writes the consolidated current results to
`synthetic_data/results/`, including distance-path, sequence-path, and
ground-truth recovery figures.

Expected predicate order:

```text
Country=USA -> Education=Doctoral -> RemoteWork=Remote
```

Expected distance path:

```text
50,000 -> 26,000 -> 12,000 -> 4,000
```

## Run the real-data pipeline

Place the Stack Overflow 2022 survey at
`data/stackoverflow2022.csv`, then run:

```powershell
python run_pipeline.py
```

Artifacts are written to `iterative_results/`:

- `iterations.csv`
- `candidate_rejections.csv`
- `sequence_path.csv`
- `final_balance.csv`
- `final_support.csv`
- `summary.json`
- `distance_path.png`
- `sequence_path.png`

## Run tests

```powershell
python -m pytest -q
```

The tests cover weighted sequence construction, hand-checkable SDEcho scores,
raw/active/effective support guards, correlated cumulative calibration, exact
ordered predicate and distance recovery, preservation of previous balance
constraints, and deterministic repeatability.

## Interpretation

Each step reports two different diagnostics:

- search-bucket distance after deleting the selected predicate from both
  weighted groups;
- full-sequence distance after balancing that predicate's prevalence in the
  source group inside the same frozen buckets.

They answer different questions and need not be equal. See `foundation.txt`
for the mathematical distinction, real-data evidence, support assumptions, and
claim boundaries.
