import numpy as np

from src.predicates import Predicate
from src.reweighting import (
    GapDecompositionResult,
    ReweightingDiagnostics,
    audit_predicate_reweighting,
)
from src.visualization import (
    render_bucket_audit_table,
    render_predicate_audit_summary,
)


def _audit_result():
    diagnostics = ReweightingDiagnostics(
        attrs=["country"],
        n_source_rows=10,
        n_dropped_rows=0,
        pct_dropped_rows=0.0,
        n_cells_source_total=2,
        n_cells_no_target_overlap=0,
        n_cells_below_min_support=0,
        n_cells_valid=2,
        min_cell_support=1,
        max_weight=1.0,
        min_weight=1.0,
    )
    result = GapDecompositionResult(
        predicate=Predicate({"country": "USA"}),
        attrs=["country"],
        s_source_orig=np.array([0.0, 10.0]),
        s_source_cf=np.array([4.0, 20.0]),
        s_target=np.array([10.0, 10.0]),
        d_orig=10.0,
        d_cf=9.6,
        explained_fraction=0.04,
        residual_gap=9.6,
        diagnostics=diagnostics,
    )
    return audit_predicate_reweighting(result, ["0-2", "3-5"])


def test_render_predicate_audit_summary_contains_category():
    table = render_predicate_audit_summary(_audit_result())

    assert "Audit Category" in table["Metric"].to_list()
    assert "bucket-specific" in table["Value"].to_list()


def test_render_bucket_audit_table_contains_bucket_statuses():
    table = render_bucket_audit_table(_audit_result())

    assert table["Bucket"].to_list() == ["0-2", "3-5"]
    assert table["Status"].to_list() == ["improved", "worsened"]
