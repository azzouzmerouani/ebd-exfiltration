"""Tests for the trained base detector and the D5 control harness."""

from __future__ import annotations

import numpy as np

from ebd.d5 import d5_compare, match_fpb_threshold
from ebd.detectors import WindowClassifier
from ebd.synth import DatasetConfig, make_dataset
from ebd.triggers import ECTSTrigger


def _small_dataset():
    cfg = DatasetConfig(replicates_per_cell=2, n_benign_light=60, n_benign_bulk=30)
    return make_dataset(cfg).split()


def test_window_classifier_separates_attack_from_benign():
    tr_a, tr_b, te_a, te_b = _small_dataset()
    clf = WindowClassifier().fit(tr_b, tr_a)

    def peak(ep):
        clf.reset()
        return max(clf.update(ep.window_bytes[k], ep) for k in range(ep.n_windows))

    # a trained persistence-aware classifier should score sustained attacks
    # higher than short benign bursts, on average
    atk = np.mean([peak(e) for e in te_a])
    ben = np.mean([peak(e) for e in te_b])
    assert atk > ben


def test_match_fpb_threshold_respects_budget():
    tr_a, tr_b, te_a, te_b = _small_dataset()
    base = WindowClassifier().fit(tr_b, tr_a)
    trig = ECTSTrigger(base=base, cost_model="byte")
    trig.calibrate(tr_a, tr_b)

    from ebd.d5 import benign_fpb
    from ebd.harness import auto_thresholds

    grid = auto_thresholds(trig, list(tr_b) + list(te_a))
    thr = match_fpb_threshold(trig, te_b, target_fpb=0.05, grid=grid)
    assert benign_fpb(trig, te_b, thr) <= 0.05 + 1e-9


def test_byte_floored_never_less_aggressive_than_time():
    # The decoupled fix: its cutoff is never above the time cutoff, so at the
    # same operating threshold it detects at least whatever time detects.
    tr_a, tr_b, te_a, te_b = _small_dataset()
    base = WindowClassifier().fit(tr_b, tr_a)
    floored = ECTSTrigger(base=base, cost_model="byte-floored")
    time = ECTSTrigger(base=base, cost_model="time")
    floored.calibrate(tr_a, tr_b)
    time.calibrate(tr_a, tr_b)

    # Score each trigger in a full, separate pass (they share the base object, so
    # interleaving their updates on one window would corrupt the shared state).
    def run(det, ep):
        det.reset()
        return np.array([det.update(ep.window_bytes[k], ep) for k in range(ep.n_windows)])

    for ep in te_a[:20]:
        mf = run(floored, ep)
        mt = run(time, ep)
        # cutoff_floored <= cutoff_time => margin_floored >= margin_time, every window
        assert np.all(mf >= mt - 1e-9)


def test_d5_compare_runs_and_is_well_formed():
    tr_a, tr_b, te_a, te_b = _small_dataset()
    base = WindowClassifier().fit(tr_b, tr_a)      # shared base -> identical p_t
    byte = ECTSTrigger(base=base, cost_model="byte")
    time = ECTSTrigger(base=base, cost_model="time")

    res = d5_compare(byte, time, tr_a, tr_b, te_a, te_b, target_fpb=0.1, n_boot=200, seed=1)

    assert res.base_name == "classifier"
    assert res.det_byte > 0.0                       # the base actually detects
    assert res.ci_lo <= res.paired_nebd_saving <= res.ci_hi
    assert {c.rate for c in res.per_rate} <= {"burst", "fast", "throttled", "drip"}
    # both triggers respect (roughly) the same FP budget they were matched to
    assert res.fpb_byte <= 0.25 and res.fpb_time <= 0.25
