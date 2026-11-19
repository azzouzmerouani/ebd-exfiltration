"""Unit tests for the EBD metric core and the H1 ranking comparison."""

from __future__ import annotations

import numpy as np

from ebd.episode import Episode
from ebd.metrics import (
    compare_rankings,
    compute_ebd,
    ebd_at_fpb,
    normalized_ebd,
)


def _episode(window_bytes, label=True):
    wb = np.asarray(window_bytes, dtype=float)
    return Episode(episode_id="t", label=label, window_bytes=wb, sender_total=float(wb.sum()))


def test_ebd_early_alert_is_small():
    ep = _episode([10, 10, 10, 10, 10])  # 50 MB total
    # alert during window 0 -> only the first window's bytes are gone
    assert compute_ebd(ep, 0) == 10.0
    assert compute_ebd(ep, 2) == 30.0


def test_ebd_never_fires_is_full_asset():
    ep = _episode([10, 10, 10])
    assert compute_ebd(ep, None) == 30.0
    assert normalized_ebd(ep, None) == 1.0


def test_ebd_before_any_data_is_zero():
    ep = _episode([5, 5, 5])
    assert compute_ebd(ep, -1) == 0.0


def test_normalized_ebd_is_a_fraction():
    ep = _episode([25, 25, 25, 25])  # 100 MB
    assert normalized_ebd(ep, 0) == 0.25
    assert normalized_ebd(ep, 1) == 0.50


def test_ebd_at_fpb_respects_budget():
    from ebd.metrics import EBDOperatingPoint

    pts = [
        EBDOperatingPoint(threshold=0.1, mean_ebd=5.0, mean_nebd=0.05,
                          detection_rate=1.0, fpb=0.40, fp_per_hour=0.0),   # too many FPs
        EBDOperatingPoint(threshold=0.5, mean_ebd=30.0, mean_nebd=0.30,
                          detection_rate=1.0, fpb=0.04, fp_per_hour=0.0),   # feasible
        EBDOperatingPoint(threshold=0.9, mean_ebd=80.0, mean_nebd=0.80,
                          detection_rate=0.9, fpb=0.01, fp_per_hour=0.0),   # feasible, worse
    ]
    op = ebd_at_fpb(pts, max_fpb=0.05)
    assert op.fpb <= 0.05
    assert op.mean_ebd == 30.0  # best feasible EBD, not the infeasible 5.0


def test_h1_diverges_when_f1_and_ebd_disagree():
    # deliberately inverted: the F1 winner is the EBD loser
    f1 = {"late_but_sure": 0.95, "early": 0.80, "mid": 0.85}
    ebd = {"late_but_sure": 200.0, "early": 10.0, "mid": 100.0}
    cmp = compare_rankings(f1, ebd)
    assert cmp.diverges
    assert cmp.kendall_tau < 0.5
    assert cmp.f1_rank["late_but_sure"] == 1
    assert cmp.ebd_rank["early"] == 1


def test_h1_falsified_when_rankings_agree():
    f1 = {"a": 0.9, "b": 0.8, "c": 0.7}
    ebd = {"a": 10.0, "b": 20.0, "c": 30.0}  # same order as F1
    cmp = compare_rankings(f1, ebd)
    assert not cmp.diverges
    assert cmp.kendall_tau >= 0.5
