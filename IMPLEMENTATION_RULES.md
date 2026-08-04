# General Instructions

Always assume this project is a master's thesis.

The implementation must prioritize

- correctness
- reproducibility
- readability
- simplicity

over optimization.

---

Never redesign the research direction unless explicitly asked.

Instead,

improve the existing implementation.

Current Research Framing

The project is now framed as explanation auditing.

Do not assume the goal is always to reduce the sequence gap.

The implementation should determine whether a SDEcho-discovered predicate is:

- compositional,
- weak/non-compositional,
- gap-amplifying,
- bucket-specific,
- or a proxy/residual indicator.

Small or negative explained fractions are valid findings and should be
reported, not hidden.

---

When suggesting code,

always explain

1. Why the module is needed.

2. Where it fits in the pipeline.

3. Which assumptions it introduces.

4. Possible alternatives.

Then write the code.

---

Every implementation should answer

Is it faithful to SDEcho?

What assumptions are made?

How will it be evaluated?

How can it fail?

How would Reviewer #2 criticize it?

Does it distinguish removal-based influence from reweighting-based
composition?

---

Coding Rules

Always use

- modular functions

- descriptive variable names

- comments

- type hints where useful

- docstrings

Avoid

large notebook cells

duplicated code

magic numbers

hard-coded paths

global variables

---

Research Rules

Never confuse

statistical reweighting

with

causal intervention.

Never write

"This intervention causes..."

Instead write

"This counterfactual reweighting estimates..."

---

Evaluation Rules

Every new module should eventually be testable.

Whenever possible,

suggest

unit tests,

sanity checks,

synthetic examples,

and expected outputs.

Every main reweighting result should include bucket-level diagnostics:

- original bucket gap
- counterfactual bucket gap
- change in absolute bucket gap
- whether the bucket improved or worsened

This is required because a predicate can reduce the global distance while
worsening individual buckets, or increase the global distance while revealing
important heterogeneity.

---

Documentation Rules

For every important function,

provide

- explanation

- pseudocode

- complexity

- thesis-ready description.

---

Communication Style

If multiple solutions exist,

recommend the simplest solution suitable for a master's thesis.

Do not recommend unnecessary complexity.

Always tell me when an idea is likely to create unnecessary implementation work.



CONFIG = {
    "data_path": "data/stackoverflow2022.csv",
    "subgroup_col": "AgeGroup",
    "subgroup_val1": "25-34",
    "subgroup_val2": "35-44",
    "group_col": "YearsExpBucket",
    "measure_col": "ConvertedCompYearly",
    "agg_func": "mean",
    "candidate_attrs": ["EdLevel", "RemoteWork", "Country"],
    "max_order": 2,
    "sdecho_k": 10,
    "max_values_per_attr": 10,
    "sdecho_min_support": 20,
    "predicate_rank": 0,          # top-1
    "min_cell_support": 5,        # reweighting trimming threshold
    "reweight_direction": "A_to_B",  # "A_to_B" or "B_to_A"
    "n_bootstrap": 1000,
    "bootstrap_ci": 0.95,
}
