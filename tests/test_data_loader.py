import pandas as pd

from src.data_loader import get_bucket_index, load_and_preprocess_data, split_groups


def test_loader_derives_buckets_age_groups_and_remote_label(tmp_path):
    path = tmp_path / "survey.csv"
    pd.DataFrame(
        {
            "YearsCodePro": [1, 4, 8],
            "Age": ["25-34 years old", "35-44 years old", "25-34 years old"],
            "RemoteWork": ["Hybrid (some remote, some in-person)", "Remote", None],
            "Salary": [10.0, 20.0, 30.0],
        }
    ).to_csv(path, index=False)

    result = load_and_preprocess_data(
        str(path),
        {
            "group_col": "YearsExpBucket",
            "subgroup_col": "AgeGroup",
            "measure_col": "Salary",
        },
    )

    assert result["AgeGroup"].tolist() == ["25-34", "35-44", "25-34"]
    assert result["YearsExpBucket"].astype(str).tolist() == ["0-2", "3-5", "6-10"]
    assert result["RemoteWork"].tolist() == ["Hybrid", "Remote", "Unknown"]
    assert get_bucket_index(result, "YearsExpBucket") == ["0-2", "3-5", "6-10"]


def test_split_groups_returns_independent_copies():
    frame = pd.DataFrame({"Group": ["A", "B"], "Value": [1, 2]})
    source, target = split_groups(frame, "Group", "A", "B")
    source.loc[source.index[0], "Value"] = 99

    assert frame.loc[0, "Value"] == 1
    assert target["Value"].tolist() == [2]
