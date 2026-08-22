# Ordered synthetic ground truth

This folder validates the iterative method rather than one-shot top-k
attribution.  The generated population contains three independent binary
composition shifts with decreasing contributions:

1. `Country=USA`
2. `Education=Doctoral`
3. `RemoteWork=Remote`

The conditional salary premium for each predicate is identical in both
groups. Only prevalence differs for the three planted composition mechanisms.
A deliberately separate 2,000-unit target outcome shift remains after all
three predicate events are balanced within every sequence bucket.

Expected Euclidean distance path over four buckets:

```text
50,000 -> 26,000 -> 12,000 -> 4,000
```

Run the benchmark and regenerate the current tables and figures:

```powershell
python -m synthetic_data.benchmark
```

The tests verify the exact predicate order, exact distance path, preservation
of earlier balance constraints, and the final residual.

## Folder layout

```text
synthetic_data/
  generator.py            Active deterministic data-generating process
  ground_truth.json       Expected order and exact distance path
  benchmark.py            Active iterative validation runner
  visualization.py        Synthetic-specific recovery figure
  results/                Current iterative tables and PNG/SVG figures
  archive/
    legacy_one_shot/      Historical outputs from the retired benchmark
```

The archive is retained only for traceability. Its results use the former
two-predicate, one-shot attribute-alignment methodology and must not be mixed
with the active three-predicate iterative results.
