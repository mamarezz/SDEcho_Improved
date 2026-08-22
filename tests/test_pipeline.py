from pathlib import Path

from synthetic_data.generator import (
    BUCKETS,
    CANDIDATE_ATTRIBUTES,
    GROUP_COL,
    MEASURE_COL,
    SOURCE_GROUP,
    SUBGROUP_COL,
    TARGET_GROUP,
    generate_iterative_ground_truth,
)


def test_zero_step_run_overwrites_diagnostic_artifacts(monkeypatch, tmp_path: Path):
    import run_pipeline

    dataset = generate_iterative_ground_truth()
    monkeypatch.setattr(
        run_pipeline, "load_and_preprocess_data", lambda *_args, **_kwargs: dataset
    )
    monkeypatch.setattr(run_pipeline, "get_bucket_index", lambda *_args: BUCKETS)
    (tmp_path / "final_balance.csv").write_text("stale", encoding="utf-8")
    (tmp_path / "final_support.csv").write_text("stale", encoding="utf-8")

    result = run_pipeline.main(
        {
            "output_dir": str(tmp_path),
            "subgroup_col": SUBGROUP_COL,
            "source_group": SOURCE_GROUP,
            "target_group": TARGET_GROUP,
            "group_col": GROUP_COL,
            "measure_col": MEASURE_COL,
            "candidate_attrs": CANDIDATE_ATTRIBUTES,
            "max_order": 1,
            "max_iterations": 0,
            "balance_bucket_policy": "all",
        }
    )

    assert not result.steps
    assert (tmp_path / "final_balance.csv").read_text(encoding="utf-8").startswith(
        "predicate,bucket,"
    )
    assert (tmp_path / "final_support.csv").read_text(encoding="utf-8").startswith(
        "predicate,bucket,"
    )
