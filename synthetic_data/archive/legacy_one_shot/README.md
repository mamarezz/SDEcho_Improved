# Legacy one-shot synthetic benchmark

These files are preserved only for traceability. They belong to the retired
two-conjunctive-predicate benchmark and its one-shot attribute-alignment
methodology. They are not outputs of the active iterative SDEcho implementation.

The `data/` and `figures/` directories contain the later legacy run. Its main
summary reports 30.2% removal for each individual planted predicate, 80.0% for
their removal union, 46.6% and 41.9% for the individual attribute-alignment
interventions, and 79.9% for top-k alignment with dynamic bucket selection.

The `superseded_first_results/` directory is an earlier benchmark calibration
with different salary/residual numbers, not merely another rendering of the
same data-generating process. It reports 80.0% rather than 79.9% for top-k
alignment and must not be mixed with the later legacy run.

For the current three-predicate sequential benchmark, use
`synthetic_data/results/` and regenerate it with:

```powershell
python -m synthetic_data.benchmark
```
