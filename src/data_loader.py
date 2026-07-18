# src/data_loader.py

import pandas as pd
import numpy as np
from typing import Tuple


def load_and_preprocess_data(path: str, config: dict) -> pd.DataFrame:
    """
    Load one CSV and derive columns needed by the pipeline
    (bucket columns, cleaned categorical columns).
    
    Args:
        path: Path to CSV file
        config: Configuration dictionary with column specifications
    
    Returns:
        Preprocessed DataFrame ready for analysis
    
    Notes:
        - Derives YearsExpBucket from YearsCodePro if needed
        - Derives AgeGroup from Age if needed
        - Cleans RemoteWork values
        - Drops rows with missing critical values
    """
    df = pd.read_csv(path)
    
    # Derive YearsExpBucket from YearsCodePro if needed
    if "YearsCodePro" in df.columns and "YearsExpBucket" not in df.columns:
        df["YearsExpBucket"] = pd.cut(
            pd.to_numeric(df["YearsCodePro"], errors="coerce"),
            bins=[0, 2, 5, 10, 20],
            labels=["0-2", "3-5", "6-10", "10-20"],
        )
    
    # Derive AgeGroup from Age if needed
    if "Age" in df.columns and "AgeGroup" not in df.columns:
        df["AgeGroup"] = df["Age"].str.extract(r"(\d+-\d+)")
    
    # Clean RemoteWork values
    if "RemoteWork" in df.columns:
        df["RemoteWork"] = (
            df["RemoteWork"]
            .fillna("Unknown")
            .replace({"Hybrid (some remote, some in-person)": "Hybrid"})
        )
    
    # Drop rows with missing critical values
    group_col = config.get("group_col")
    subgroup_col = config.get("subgroup_col")
    measure_col = config.get("measure_col")
    
    required_cols = [c for c in [group_col, subgroup_col, measure_col] if c]
    if required_cols:
        df = df.dropna(subset=[c for c in required_cols if c in df.columns])
    
    return df


def split_groups(
    df: pd.DataFrame, subgroup_col: str, val1: str, val2: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into the two comparison groups.
    
    Args:
        df: Preprocessed DataFrame
        subgroup_col: Column name defining the groups
        val1: Value for group A
        val2: Value for group B
    
    Returns:
        Tuple of (df_group1, df_group2) as copies
    
    Notes:
        - Returns copies to avoid modifying original
        - Groups may have different sizes
    """
    df_A = df[df[subgroup_col] == val1].copy()
    df_B = df[df[subgroup_col] == val2].copy()
    
    return df_A, df_B


def get_bucket_index(df: pd.DataFrame, group_col: str) -> list[str]:
    """
    Get the ordered list of bucket labels present in the data.
    
    Args:
        df: DataFrame
        group_col: Bucket column name
    
    Returns:
        Ordered list of unique bucket labels (as strings)
    
    Notes:
        - Uses a predefined order for experience buckets
        - Only returns buckets present in the data
    """
    desired_order = ["0-2", "3-5", "6-10", "10-20"]
    present = set(df[group_col].dropna().astype(str).unique())
    return [x for x in desired_order if x in present]