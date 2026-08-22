"""Synthetic population with a known sequential explanation path."""

from itertools import product

import pandas as pd

from src.predicates import Predicate


BUCKETS = ["0-2", "3-5", "6-10", "10-20"]
GROUP_COL = "YearsExpBucket"
SUBGROUP_COL = "ComparisonGroup"
MEASURE_COL = "Salary"
SOURCE_GROUP = "Group A"
TARGET_GROUP = "Group B"
CANDIDATE_ATTRIBUTES = ["Country", "Education", "RemoteWork"]

EXPECTED_PREDICATES = (
    Predicate({"Country": "USA"}),
    Predicate({"Education": "Doctoral"}),
    Predicate({"RemoteWork": "Remote"}),
)

BASE_SALARY = {
    "0-2": 60_000.0,
    "3-5": 80_000.0,
    "6-10": 100_000.0,
    "10-20": 120_000.0,
}
PREMIUMS = {
    "Country": 60_000.0,
    "Education": 35_000.0,
    "RemoteWork": 16_000.0,
}
SOURCE_PREVALENCE = {
    "Country": 0.10,
    "Education": 0.20,
    "RemoteWork": 0.25,
}
TARGET_PREVALENCE = {
    "Country": 0.30,
    "Education": 0.40,
    "RemoteWork": 0.50,
}
IRREDUCIBLE_TARGET_SHIFT = 2_000.0
ROWS_PER_GROUP_BUCKET = 200

# Four equal bucket gaps: 25k initially, then 13k, 6k and the 2k residual.
EXPECTED_DISTANCE_PATH = (50_000.0, 26_000.0, 12_000.0, 4_000.0)
EXPECTED_BUCKET_GAP_PATH = (25_000.0, 13_000.0, 6_000.0, 2_000.0)


def _cell_count(group_prevalence: dict[str, float], flags: tuple[int, int, int]) -> int:
    probability = 1.0
    for attribute, flag in zip(CANDIDATE_ATTRIBUTES, flags):
        prevalence = group_prevalence[attribute]
        probability *= prevalence if flag else 1.0 - prevalence
    count = probability * ROWS_PER_GROUP_BUCKET
    rounded = round(count)
    if abs(count - rounded) > 1e-10:
        raise AssertionError("synthetic probabilities must produce integer cell counts")
    return int(rounded)


def generate_iterative_ground_truth() -> pd.DataFrame:
    """Return a deterministic, fully crossed two-group synthetic dataset.

    The three binary attributes are independent within each group and bucket.
    Their conditional salary premiums are identical between groups; only their
    prevalence differs.  A 2,000 target-only shift remains after all three
    predicate distributions are balanced.
    """
    rows: list[dict[str, object]] = []
    row_id = 1
    for bucket in BUCKETS:
        for group, prevalence in (
            (SOURCE_GROUP, SOURCE_PREVALENCE),
            (TARGET_GROUP, TARGET_PREVALENCE),
        ):
            for flags in product((0, 1), repeat=3):
                count = _cell_count(prevalence, flags)
                country = "USA" if flags[0] else "Other"
                education = "Doctoral" if flags[1] else "Standard"
                remote_work = "Remote" if flags[2] else "On-site"
                salary = (
                    BASE_SALARY[bucket]
                    + flags[0] * PREMIUMS["Country"]
                    + flags[1] * PREMIUMS["Education"]
                    + flags[2] * PREMIUMS["RemoteWork"]
                    + (IRREDUCIBLE_TARGET_SHIFT if group == TARGET_GROUP else 0.0)
                )
                active = []
                if flags[0]:
                    active.append("Country=USA")
                if flags[1]:
                    active.append("Education=Doctoral")
                if flags[2]:
                    active.append("RemoteWork=Remote")
                for _ in range(count):
                    rows.append(
                        {
                            "RowID": row_id,
                            SUBGROUP_COL: group,
                            GROUP_COL: bucket,
                            "Country": country,
                            "Education": education,
                            "RemoteWork": remote_work,
                            MEASURE_COL: salary,
                            "ActiveGroundTruthPredicates": " | ".join(active),
                        }
                    )
                    row_id += 1

    dataset = pd.DataFrame(rows)
    expected_rows = len(BUCKETS) * 2 * ROWS_PER_GROUP_BUCKET
    if len(dataset) != expected_rows:
        raise AssertionError(f"expected {expected_rows} rows, found {len(dataset)}")
    return dataset


def split_synthetic_groups(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the generated dataset into source and fixed target groups."""
    source = dataset[dataset[SUBGROUP_COL] == SOURCE_GROUP].copy()
    target = dataset[dataset[SUBGROUP_COL] == TARGET_GROUP].copy()
    return source, target
