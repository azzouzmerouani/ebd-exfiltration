"""Tests for measuring the record-emission interval T from flow records."""

from __future__ import annotations

import pytest

from ebd.realdata import FlowRecord, group_into_episodes
from ebd.realdata.cadence_probe import (
    effective_T,
    floor_for,
    measure_active_timeout,
)


def _f(start, duration, src_port=1000):
    return FlowRecord(
        src_ip="10.0.0.1", src_port=src_port, dst_ip="10.0.0.2", dst_port=443,
        protocol=6, start=start, duration=duration,
        fwd_bytes=1000, bwd_bytes=10,
    )


def test_active_timeout_inferred_from_ceiling():
    # nothing exceeds 1800s -> that is the timeout
    flows = [_f(0, 60), _f(100, 1797.8), _f(5000, 30)]
    m = measure_active_timeout(flows)
    assert m.inferred_active_timeout == 1800.0
    assert m.n_over_ceiling == 0


def test_shorter_timeout_inferred_when_ceiling_is_lower():
    flows = [_f(0, 59.9), _f(100, 60.0), _f(200, 30.0)]
    assert measure_active_timeout(flows).inferred_active_timeout == 60.0


def test_effective_T_uses_connection_length_when_short():
    # many short beacon connections: first record arrives when the first closes
    flows = [_f(i * 60.0, 58.0, src_port=1000 + i) for i in range(20)]
    ep = group_into_episodes(flows)[0]
    assert effective_T(ep) == pytest.approx(58.0)


def test_effective_T_uses_earliest_ENDING_flow_not_earliest_starting():
    """A later, shorter connection can emit its record first."""
    flows = [
        _f(0.0, 1797.8, src_port=1000),   # starts first, ends last
        _f(200.0, 54.0, src_port=1001),   # starts later, ends at t=254
    ]
    ep = group_into_episodes(flows)[0]
    assert effective_T(ep) == pytest.approx(254.0)


def test_effective_T_clamped_by_active_timeout():
    # one long-lived connection: the exporter forces a record at the timeout
    flows = [_f(0.0, 5000.0)]
    ep = group_into_episodes(flows)[0]
    assert effective_T(ep, active_timeout=1800.0) == pytest.approx(1800.0)


def test_floor_is_T_over_D():
    flows = [_f(i * 60.0, 58.0, src_port=1000 + i) for i in range(100)]
    ep = group_into_episodes(flows)[0]
    expected = 58.0 / ep.duration
    assert floor_for(ep) == pytest.approx(expected, rel=1e-6)


def test_chunking_lowers_the_floor_versus_one_long_connection():
    """The counter-intuitive result: many short connections help the detector."""
    span = 3600.0
    chunked = [_f(i * 60.0, 58.0, src_port=1000 + i) for i in range(60)]
    single = [_f(0.0, span)]
    ep_chunked = group_into_episodes(chunked)[0]
    ep_single = group_into_episodes(single)[0]
    assert floor_for(ep_chunked) < floor_for(ep_single)


def test_floor_never_exceeds_one():
    flows = [_f(0.0, 10.0)]
    ep = group_into_episodes(flows)[0]
    assert floor_for(ep, active_timeout=1800.0) == 1.0
