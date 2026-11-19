"""Tests for EDXGB's feature pipeline reproduction."""

from __future__ import annotations

import numpy as np
import pytest

from ebd.realdata.edxgb import (
    DERIVED,
    build_matrix,
    derive_rates,
    feature_columns,
    first_alert_time,
)


def test_ptr_btr_match_the_published_formulas():
    row = {
        "bidirectional_duration_ms": "1000",
        "dst2src_packets": "10", "src2dst_packets": "40",
        "dst2src_bytes": "500", "src2dst_bytes": "8000",
    }
    d = derive_rates(row)
    assert d["PTR_I"] == pytest.approx(0.010)
    assert d["PTR_O"] == pytest.approx(0.040)
    assert d["BTR_I"] == pytest.approx(0.500)
    assert d["BTR_O"] == pytest.approx(8.000)


def test_zero_duration_does_not_divide_by_zero():
    d = derive_rates({"bidirectional_duration_ms": "0", "src2dst_packets": "5"})
    assert all(v == 0.0 for v in d.values())


def test_identifier_columns_excluded():
    hdr = ["src_ip", "dst_ip", "src_mac", "dst_port", "protocol",
           "bidirectional_bytes", "Stage", "Activity",
           "bidirectional_first_seen_ms"]
    cols = feature_columns(hdr)
    # who -> excluded; how -> kept
    assert "src_ip" not in cols and "dst_ip" not in cols and "src_mac" not in cols
    assert "Stage" not in cols and "Activity" not in cols
    assert "bidirectional_first_seen_ms" not in cols     # absolute time leaks
    assert "dst_port" in cols and "protocol" in cols
    assert "bidirectional_bytes" in cols


def test_matrix_shape_and_derived_appended():
    cols = ["bidirectional_bytes"]
    rows = [{"bidirectional_bytes": "100", "bidirectional_duration_ms": "10",
             "src2dst_bytes": "50", "dst2src_bytes": "5",
             "src2dst_packets": "2", "dst2src_packets": "1"}]
    X = build_matrix(rows, cols)
    assert X.shape == (1, 1 + len(DERIVED))
    assert X[0, 0] == pytest.approx(100.0)
    assert X[0, 1 + DERIVED.index("BTR_O")] == pytest.approx(5.0)


def test_non_numeric_and_missing_values_are_safe():
    X = build_matrix([{"a": "", "b": "junk"}], ["a", "b", "missing"])
    assert np.all(np.isfinite(X))


def test_first_alert_uses_flow_end_not_start():
    rows = [
        {"bidirectional_last_seen_ms": "5000"},
        {"bidirectional_last_seen_ms": "2000"},   # earlier record
        {"bidirectional_last_seen_ms": "9000"},
    ]
    scores = np.array([0.9, 0.9, 0.9])
    assert first_alert_time(rows, scores, 0.5) == pytest.approx(2.0)


def test_first_alert_ignores_unflagged_rows():
    rows = [{"bidirectional_last_seen_ms": "1000"},
            {"bidirectional_last_seen_ms": "8000"}]
    scores = np.array([0.1, 0.9])
    assert first_alert_time(rows, scores, 0.5) == pytest.approx(8.0)


def test_first_alert_none_when_never_flagged():
    rows = [{"bidirectional_last_seen_ms": "1000"}]
    assert first_alert_time(rows, np.array([0.01]), 0.5) is None
