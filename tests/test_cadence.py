"""Tests for the export-cadence emission model and the EBD floor."""

from __future__ import annotations

import numpy as np
import pytest

from ebd.cadence import (
    OracleFlowDetector,
    cadence_floor,
    emission_windows,
    theoretical_floor,
)
from ebd.episode import Episode


def _ep(window_bytes, dt=1.0):
    wb = np.asarray(window_bytes, dtype=float)
    return Episode(episode_id="t", label=True, window_bytes=wb, dt=dt,
                   sender_total=float(wb.sum()))


def test_short_transfer_emits_only_at_teardown():
    # 5 windows = 5 s, far inside a 120 s flow timeout -> one record, at the end
    ep = _ep([10] * 5)
    assert emission_windows(ep, flow_timeout=120.0) == [4]


def test_transfer_shorter_than_cadence_is_total_loss():
    ep = _ep([10] * 5)                       # D = 5 s
    f = cadence_floor(ep, cadence_s=120.0)   # T = 120 s
    assert f.total_loss
    assert f.nebd == pytest.approx(1.0)


def test_long_transfer_emits_at_each_active_timeout():
    ep = _ep([1] * 25)                        # 25 s
    # flushes at 10 s and 20 s, then final teardown record at 25 s
    assert emission_windows(ep, flow_timeout=10.0) == [9, 19, 24]


def test_floor_matches_closed_form_for_constant_rate():
    ep = _ep([2] * 100)                       # D = 100 s, constant rate
    for T in (10.0, 25.0, 50.0, 200.0):
        f = cadence_floor(ep, cadence_s=T)
        assert f.nebd == pytest.approx(theoretical_floor(100.0, T), abs=0.02)


def test_floor_is_rate_inverted():
    # same asset, different speeds: the FASTER transfer loses more, not less
    fast = _ep([50] * 10)      # 500 MB in 10 s
    slow = _ep([1] * 500)      # 500 MB in 500 s
    T = 120.0
    assert cadence_floor(fast, T).nebd == pytest.approx(1.0)
    assert cadence_floor(slow, T).nebd < 0.3


def test_idle_teardown_emits_a_record_early():
    # 3 s of traffic, then a 10 s gap: an idle timeout of 5 s tears the flow down
    ep = _ep([5, 5, 5] + [0] * 10 + [5, 5])
    without = emission_windows(ep, flow_timeout=120.0, idle_timeout=None)
    with_idle = emission_windows(ep, flow_timeout=120.0, idle_timeout=5.0)
    assert with_idle[0] < without[0]      # idle teardown helps the incumbent


def test_oracle_fires_exactly_at_first_record():
    ep = _ep([1] * 25)
    det = OracleFlowDetector(flow_timeout=10.0)
    det.reset()
    scores = [det.update(ep.window_bytes[k], ep) for k in range(ep.n_windows)]
    first = emission_windows(ep, 10.0)[0]
    assert scores[first - 1] == 0.0        # silent before the record exists
    assert scores[first] == 1.0            # fires the moment it does


def test_oracle_ebd_equals_the_floor():
    from ebd.metrics import first_alert_window, normalized_ebd

    ep = _ep([2] * 100)
    det = OracleFlowDetector(flow_timeout=25.0)
    det.reset()
    sc = np.array([det.update(ep.window_bytes[k], ep) for k in range(ep.n_windows)])
    aw = first_alert_window(sc, 0.5)
    assert normalized_ebd(ep, aw) == pytest.approx(cadence_floor(ep, 25.0).nebd)
