# Module: `predicates.py`

**Pipeline stage:** Shared utility, used by Stage 5 (SDEcho predicate search)
and referenced by Stage 6–7 (predicate selection, reweighting covariate
extraction).

**Status:** Extracted and formalized from `_enumerate_predicates()` and
`_predicate_mask()` in the original notebook, plus the `Predicate` dataclass
introduced in `docs/02_ARCHITECTURE.md` §3. Logic is unchanged from the
original brute-force implementation; this module only relocates and
type-defines it.

---

## 1. Why This Module Is Needed

Both SDEcho's predicate search (Stage 5) and this thesis's reweighting stage
(Stage 7) need a shared, unambiguous representation of "a conjunctive
predicate over one or more categorical attributes." Rather than passing
raw `dict[str, Any]` objects between `sdecho.py` and `reweighting.py` (which
risks key-typo bugs and gives no single place to define predicate-related
behavior like matching/masking), this module centralizes:

1. The `Predicate` data type itself (see `docs/02_ARCHITECTURE.md` §3).
2. Enumeration of candidate predicates from two groups' data.
3. Boolean masking — which rows of a DataFrame match a given predicate.

This is a **pure utility module** — it has no notion of "gamma," "distance,"
or "reweighting"; it only knows how to build and apply predicates. This
keeps it independently testable and reusable by any future module that
needs predicate logic (e.g., if `evaluation.py`'s ablations need to enumerate
predicates under different `max_order` settings).

## 2. Where It Fits in the Pipeline

```
                    predicates.py
                   ┌───────────────┐
                   │ Predicate      │
                   │ enumerate_...  │
                   │ predicate_mask │
                   └───────┬────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼                             ▼
       sdecho.py                    reweighting.py
  (Stage 5: search over          (Stage 6: predicate.attrs
   enumerated predicates,          extracts covariate set;
   ranks by gamma using            Stage 7 uses predicate_mask
   predicate_mask)                 conceptually via cell keys)
```

Note: reweighting (Stage 7) does **not** call `predicate_mask()` directly —
it operates on the *joint cell* defined by `predicate.attrs` (see
`docs/01_METHODOLOGY.md` §6.1), which is a coarser operation than matching
the specific predicate values. `predicate_mask()` is used by SDEcho's search
(which needs exact value matches to compute `dist_after`), while reweighting
only needs the *attribute names*, not the matched values. This distinction
is worth restating clearly in the thesis text, since it's easy to conflate
"the predicate SDEcho found" with "the values used for reweighting" — they
are related but not identical (see `docs/01_METHODOLOGY.md` §5, final
paragraph).

## 3. Assumptions Introduced

1. **`max_values_per_attr` truncates each attribute to its top-N most
   frequent values** (by combined count across both groups) before building
   predicates. This means a rare-but-highly-explanatory value (e.g., a
   country with few respondents but an unusually large pay gap) can **never
   be discovered** by SDEcho, regardless of how well it would explain the
   divergence — the search space excludes it before search even begins.
   This is already an implicit assumption in the original code
   (`max_values_per_attr=10` in `CONFIG`), but was not previously stated as
   a limitation. It should be explicitly flagged in the thesis and, ideally,
   tested as an ablation (does increasing `max_values_per_attr` change the
   top predicate found?).
2. **Predicate enumeration is fully combinatorial up to `max_order`**: for
   `max_order=2` and two attributes each truncated to 10 values, this is
   `10 + 10 + 100 = 120` predicates (single-attribute predicates for each of
   2 attributes, plus their pairwise combinations); for three attributes,
   the order-2 term alone is $\binom{3}{2} \times 10 \times 10 = 300$. This
   grows combinatorially in both the number of `candidate_attrs` and
   `max_order` — a real, previously under-stated scalability constraint
   (see `docs/01_METHODOLOGY.md` §9).
3. **A predicate's `attrs` are unordered** (implemented as a dict / frozen
   dataclass field) — `{Country: USA, EdLevel: PhD}` and
   `{EdLevel: PhD, Country: USA}` are the same predicate. This needs
   consistent hashing/equality if predicates are ever deduplicated or used
   as dictionary keys (e.g., in ablation result tables) — Python dicts are
   unordered by equality but the `Predicate` dataclass as currently
   specified is **not hashable** (dict fields aren't hashable). If
   deduplication or set-membership is needed later, `conditions` should be
   stored as a `frozenset` of `(attr, value)` tuples instead of a `dict` —
   flagged as a design decision to make now rather than retrofit later.

## 4. Public API

```python
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Predicate:
    """
    A conjunctive predicate over one or more categorical attributes.

    NOTE: `conditions` is currently a plain dict, which makes Predicate
    unhashable (cannot be used as a dict key or in a set). If ablations
    need to deduplicate or index predicates, migrate `conditions` to a
    frozenset of (attr, value) tuples instead — see §3.3 above.
    """
    conditions: dict[str, Any]

    @property
    def attrs(self) -> list[str]:
        """Attribute names involved in this predicate, in insertion order."""
        ...

    def __repr__(self) -> str:
        """Human-readable form, e.g. 'Country=USA & EdLevel=PhD'."""
        ...


def enumerate_predicates(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    candidate_attrs: list[str],
    max_order: int,
    max_values_per_attr: int | None = None,
) -> list[Predicate]:
    """
    Enumerate candidate predicates over combinations of candidate_attrs,
    up to max_order attributes per predicate.

    Args:
        df1, df2: the two comparison groups (used to determine each
            attribute's most frequent values, if max_values_per_attr is set).
        candidate_attrs: attribute names eligible for predicate search.
        max_order: maximum number of attributes conjoined in one predicate
            (1 = single-attribute predicates only, 2 = also pairwise, etc.).
        max_values_per_attr: if set, restrict each attribute to its
            top-N most frequent values (combined across df1, df2) before
            enumerating — see §3.1 for the limitation this introduces.

    Returns:
        List of Predicate objects, one per (attribute-combination,
        value-combination) pair up to max_order.
    """


def predicate_mask(df: pd.DataFrame, predicate: Predicate) -> pd.Series:
    """
    Boolean mask of rows in `df` matching all conditions in `predicate`.

    Args:
        df: DataFrame to mask.
        predicate: the Predicate to match against.

    Returns:
        pd.Series[bool], aligned to df.index.
    """
```

## 5. Implementation

```python
"""
predicates.py

Shared predicate representation, enumeration, and masking logic used by
sdecho.py's predicate search (Stage 5) and referenced by reweighting.py's
covariate-set extraction (Stage 6). This module has no knowledge of
distance metrics, gamma scores, or reweighting — it only builds and
matches predicates.
"""

import itertools
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Predicate:
    """A conjunctive predicate over one or more categorical attributes."""
    conditions: dict[str, Any]

    @property
    def attrs(self) -> list[str]:
        return list(self.conditions.keys())

    def __repr__(self) -> str:
        return " & ".join(f"{k}={v}" for k, v in self.conditions.items())


def enumerate_predicates(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    candidate_attrs: list[str],
    max_order: int,
    max_values_per_attr: int | None = None,
) -> list[Predicate]:
    """See docstring in module interface (§4)."""
    value_map: dict[str, list] = {}
    for attr in candidate_attrs:
        combined = pd.concat([df1[attr], df2[attr]]).dropna()
        if max_values_per_attr is not None:
            top_values = combined.value_counts().nlargest(max_values_per_attr).index
            value_map[attr] = list(top_values)
        else:
            value_map[attr] = list(combined.unique())

    predicates: list[Predicate] = []
    for order in range(1, max_order + 1):
        for attr_combo in itertools.combinations(candidate_attrs, order):
            value_lists = [value_map[a] for a in attr_combo]
            for value_combo in itertools.product(*value_lists):
                conditions = dict(zip(attr_combo, value_combo))
                predicates.append(Predicate(conditions=conditions))

    return predicates


def predicate_mask(df: pd.DataFrame, predicate: Predicate) -> pd.Series:
    """See docstring in module interface (§4)."""
    mask = pd.Series(True, index=df.index)
    for attr, value in predicate.conditions.items():
        mask &= (df[attr] == value)
    return mask
```

## 6. Known Gaps / TODOs

- `Predicate.conditions` as a plain `dict` is unhashable — migrate to
  `frozenset[tuple[str, Any]]` before `evaluation.py`'s ablations need to
  deduplicate or index predicates (§3.3). Not urgent for the current
  single-run pipeline, but should be done before the ablation study is
  implemented (`docs/10_EVALUATION_PLAN.md`).
- `max_values_per_attr` truncation (§3.1) needs to be reported as a
  limitation with an accompanying sensitivity check (does the top predicate
  change under `max_values_per_attr = 10` vs. `20` vs. unrestricted?) —
  currently only flagged, not tested.
- No caching: `enumerate_predicates` recomputes `value_counts()` for every
  call, including repeated calls during ablations. Not a correctness issue,
  only a performance one — not worth optimizing prematurely given the
  scale of this thesis's data (see `docs/01_METHODOLOGY.md` §9: SDEcho's
  search dominates total runtime regardless).

## 7. How Reviewer #2 Would Critique This Module

- *"You truncate to the top-10 most frequent values per attribute before
  searching — doesn't this bias you toward finding predicates about common
  categories (e.g., large countries) and structurally prevent discovery of
  explanations driven by smaller, specific subgroups?"* — Yes, and this is
  a genuine, not-yet-tested limitation. It should be explicitly named in
  the thesis's limitations section and ideally tested once with an enlarged
  `max_values_per_attr` on a subset of runs to check whether conclusions
  are sensitive to this cutoff.
- *"Is `max_order=2` an arbitrary choice or a principled one?"* — Currently
  arbitrary, chosen for combinatorial tractability (§3.2). This is a
  defensible practical constraint but should be stated as such, not implied
  to be theoretically motivated.

## 8. Complexity

- `enumerate_predicates`: $O\left(\sum_{o=1}^{\text{max\_order}} \binom{|A|}{o} \cdot V^o\right)$
  where $|A|$ = `len(candidate_attrs)`, $V$ = `max_values_per_attr` (or
  average domain size if unrestricted) — combinatorial in both `max_order`
  and $V$, as noted in §3.2.
- `predicate_mask`: $O(n \cdot |X|)$ per call, where $|X|$ is the predicate's
  attribute count (typically 1–2) — cheap per call, but called once per
  enumerated predicate inside `sdecho.py`'s search loop, so the *total* cost
  across a full SDEcho run is $O(|\mathcal{P}| \cdot n)$, consistent with
  the complexity already stated in `docs/01_METHODOLOGY.md` §9.

## 9. Tests to Write (`tests/test_predicates.py`)

1. **Enumeration count**: for a known small `candidate_attrs` list and
   `max_order`, assert the exact number of predicates generated matches the
   combinatorial formula in §8 (a good regression test against accidental
   off-by-one errors in the `itertools.combinations` range).
2. **`max_values_per_attr` truncation correctness**: synthetic data with a
   known frequency ranking — assert only the top-N values appear in
   generated predicates.
3. **`predicate_mask` correctness**: single-attribute and multi-attribute
   predicates against a small hand-constructed DataFrame with known
   matching/non-matching rows.
4. **`Predicate.__repr__` formatting**: snapshot test to catch accidental
   formatting regressions (useful since predicate reprs will appear
   directly in thesis tables/console output).
5. **Order-independence**: `Predicate({"A": 1, "B": 2})` and
   `Predicate({"B": 2, "A": 1})` should be treated as equivalent once the
   hashability fix (§6) is made — write this test now as a specification,
   even before the fix lands, so it's ready to run once `conditions`
   becomes a `frozenset`.

## 10. Thesis-Ready Description

> Candidate explanatory predicates are enumerated as conjunctions of up to
> [`max_order`] attributes drawn from a fixed candidate set
> ([`candidate_attrs`]), with each categorical attribute restricted to its
> [`max_values_per_attr`] most frequent values (combined across both
> comparison groups) to bound the search space. This restriction means
> predicates involving rare attribute values are not eligible for
> discovery, regardless of their explanatory strength — a limitation
> discussed in [Section Z] and partially assessed via a sensitivity
> analysis over [`max_values_per_attr`] in [Section W].
