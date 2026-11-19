"""Unit tests for the detector suite and the ECTS byte/time contrast."""

from __future__ import annotations

import numpy as np

from ebd.detectors import CumulativeBytes, CusumRate, EWMARate, VolumeThreshold
from ebd.episode import Episode
from ebd.triggers import ECTSTrigger


def _ep(window_bytes, label=True, dt=1.0):
    wb = np.asarray(window_bytes, dtype=float)
    return Episode(episode_id="t", label=label, window_bytes=wb, dt=dt,
                   sender_total=float(wb.sum()))


def _run(det, ep):
    det.reset()
    return np.array([det.update(ep.window_bytes[k], ep) for k in range(ep.n_windows)])


def test_all_detectors_return_finite_scores():
    ep = _ep([1, 5, 0, 3, 10])
    for det in (VolumeThreshold(), EWMARate(), CumulativeBytes(), CusumRate(k=1.0)):
        s = _run(det, ep)
        assert s.shape == (ep.n_windows,)
        assert np.all(np.isfinite(s))


def test_cumulative_is_monotonic_nondecreasing():
    ep = _ep([2, 0, 3, 1])
    s = _run(CumulativeBytes(), ep)
    assert np.all(np.diff(s) >= 0)
    assert s[-1] == ep.total_bytes


def test_volume_tracks_current_window():
    ep = _ep([1, 9, 2])
    s = _run(VolumeThreshold(), ep)
    assert list(s) == [1.0, 9.0, 2.0]


def test_cusum_accumulates_on_sustained_rate_then_decays():
    det = CusumRate(k=1.0)
    ep = _ep([5, 5, 5, 0, 0, 0])  # above slack, then idle
    s = _run(det, ep)
    assert s[2] > s[0]                       # accumulates while rate stays high
    assert np.all(np.diff(s[2:]) <= 0)       # one-sided CUSUM only decays when idle
    assert s[-1] < s[2]                      # drains down (by the slack each window)


def test_ects_byte_fires_earlier_than_time_on_a_fast_transfer():
    # Same base scorer, same calibration; only the delay cost differs.
    fast = [_ep([20, 20, 20, 20, 20]) for _ in range(6)]      # 100 MB, fast
    benign = [_ep([0.1, 0.1, 0.1], label=False) for _ in range(6)]

    byte = ECTSTrigger(base=EWMARate(0.3), cost_model="byte")
    time = ECTSTrigger(base=EWMARate(0.3), cost_model="time")
    byte.calibrate(fast, benign)
    time.calibrate(fast, benign)

    probe = _ep([20, 20, 20, 20, 20])

    def first_positive_margin(det):
        s = _run(det, probe)
        hits = np.nonzero(s >= 0.0)[0]
        return int(hits[0]) if hits.size else probe.n_windows

    # byte cost raises the urgency on a fast flow, so it should not fire later
    assert first_positive_margin(byte) <= first_positive_margin(time)


def test_ects_requires_calibration_before_use():
    det = ECTSTrigger(base=EWMARate(), cost_model="byte")
    ep = _ep([5, 5])
    det.reset()
    try:
        det.update(ep.window_bytes[0], ep)
    except RuntimeError as e:
        assert "calibrate" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError before calibration")
