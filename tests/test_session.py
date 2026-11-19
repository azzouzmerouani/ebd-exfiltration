"""Session granularity: T_session >= T_flow, and the chunking reversal."""

from __future__ import annotations

import pytest

from ebd.realdata.episodes import FlowRecord, group_into_episodes
from ebd.realdata.floorsweep import effective_T_at
from ebd.realdata.session import (
    byte_floor_session, effective_T_session, n_records, sessionize,
)


def _ep(spans, base=1000.0):
    flows = [
        FlowRecord(src_ip="10.0.0.1", src_port=40000 + i, dst_ip="10.0.0.2",
                   dst_port=443, protocol=6, start=base + s, duration=d,
                   fwd_bytes=1000, bwd_bytes=10)
        for i, (s, d) in enumerate(spans)
    ]
    eps = group_into_episodes(flows, gap_seconds=1e9)
    return eps[0]


def test_session_T_never_below_flow_T():
    """The core structural claim of Sec. 4.1, checked on many shapes."""
    shapes = [
        [(0.0, 5.0)],
        [(0.0, 5.0), (10.0, 5.0), (20.0, 5.0)],
        [(0.0, 900.0), (100.0, 5.0)],
        [(0.0, 1.0), (2.0, 1.0), (400.0, 1.0)],
        [(0.0, 60.0), (61.0, 60.0), (122.0, 60.0)],
    ]
    for spans in shapes:
        ep = _ep(spans)
        for gap in (60.0, 300.0, 600.0, 1800.0):
            assert effective_T_session(ep, gap) >= effective_T_at(ep, 1800.0) - 1e-9


def test_one_session_record_means_total_loss():
    """If the whole episode is a single aggregate, its floor is 1."""
    bins = [100] * 100
    ep = _ep([(0.0, 10.0), (30.0, 10.0), (60.0, 10.0), (90.0, 10.0)])
    assert n_records(ep, 300.0) == 1
    assert byte_floor_session(ep, bins, 1000.0, 1.0, 10_000,
                              gap_seconds=300.0) == pytest.approx(1.0)


def test_a_long_silence_splits_the_session_and_lowers_the_floor():
    bins = [100] * 1000
    ep = _ep([(0.0, 10.0), (500.0, 10.0)])
    assert n_records(ep, 60.0) == 2
    assert n_records(ep, 600.0) == 1
    assert (byte_floor_session(ep, bins, 1000.0, 1.0, 100_000, 60.0)
            < byte_floor_session(ep, bins, 1000.0, 1.0, 100_000, 600.0))


def test_chunking_reverses_between_granularities():
    """Sec. 4.4's corollary and Sec. 7.9's reversal, as an executable statement.

    The comparison must hold the episode's SPAN constant, otherwise a chunked
    episode that simply finishes earlier looks like an improvement at both
    granularities and the test proves nothing. Here one 80 s connection is
    compared against eight contiguous 10 s connections covering the same 80 s.
    """
    bins = [100] * 80
    span = 80.0
    single = _ep([(0.0, span)])
    chunked = _ep([(i * 10.0, 10.0) for i in range(8)])
    assert chunked.duration == pytest.approx(single.duration)

    # Flow granularity: the first chunk closes at t+10 instead of t+80.
    assert effective_T_at(chunked, 1800.0) < effective_T_at(single, 1800.0)

    # Session granularity: the chunks are summarised into one record that
    # cannot be emitted before the last of them ends, so nothing is gained.
    assert (effective_T_session(chunked, 300.0)
            == pytest.approx(effective_T_session(single, 300.0)))
    assert byte_floor_session(chunked, bins, 1000.0, 1.0, 8000,
                              300.0) == pytest.approx(1.0)


def test_extra_chunks_can_only_push_the_session_record_later():
    """The reversal's mechanism: additional chunks extend the aggregation window."""
    base = [(i * 10.0, 10.0) for i in range(8)]           # spans 80 s
    more = base + [(80.0, 10.0), (90.0, 10.0)]            # spans 100 s
    assert (effective_T_session(_ep(more), 300.0)
            > effective_T_session(_ep(base), 300.0))
    # ...while at flow granularity the first record arrives at the same instant.
    assert (effective_T_at(_ep(more), 1800.0)
            == pytest.approx(effective_T_at(_ep(base), 1800.0)))


def test_sessionize_keeps_every_flow_exactly_once():
    ep = _ep([(0.0, 5.0), (10.0, 5.0), (500.0, 5.0)])
    for gap in (1.0, 60.0, 600.0):
        groups = sessionize(ep.flows, gap)
        assert sum(g.n_flows for g in groups) == len(ep.flows)
