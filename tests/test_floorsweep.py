"""The counterfactual cadence formula, Eq. (4')."""

from __future__ import annotations

import pytest

from ebd.realdata.episodes import FlowRecord, group_into_episodes
from ebd.realdata.cadence_probe import effective_T as legacy_T
from ebd.realdata.floorsweep import (
    byte_floor_at, effective_T_at, sweep, time_floor_at,
)


def _ep(spans, base=1000.0):
    """Episode from (start_offset, duration) pairs."""
    flows = [
        FlowRecord(src_ip="10.0.0.1", src_port=40000 + i, dst_ip="10.0.0.2",
                   dst_port=443, protocol=6, start=base + s, duration=d,
                   fwd_bytes=1000, bwd_bytes=10)
        for i, (s, d) in enumerate(spans)
    ]
    eps = group_into_episodes(flows, gap_seconds=1e9)
    assert len(eps) == 1
    return eps[0]


def test_short_connection_beats_the_timeout():
    """A connection that closes early emits before the active timeout fires."""
    ep = _ep([(0.0, 5.0)])
    assert effective_T_at(ep, 1800.0) == pytest.approx(5.0)
    assert effective_T_at(ep, 2.0) == pytest.approx(2.0)


def test_T_eff_never_exceeds_the_timeout():
    ep = _ep([(0.0, 10_000.0)])
    for ta in (1.0, 15.0, 300.0, 1800.0):
        assert effective_T_at(ep, ta) <= ta + 1e-9


def test_monotone_in_the_timeout():
    """Lowering the active timeout can only make the first record earlier."""
    ep = _ep([(0.0, 900.0), (120.0, 30.0), (700.0, 5.0)])
    ts = [15.0, 60.0, 120.0, 300.0, 600.0, 1800.0]
    vals = [effective_T_at(ep, t) for t in ts]
    assert vals == sorted(vals)


def test_reduces_to_equation_4_at_the_exported_timeout():
    """Eq. (4') must agree with Eq. (4) where both are defined.

    This is the regression check the sweep driver asserts per episode; if it
    ever fails, the counterfactual is not trustworthy.
    """
    for spans in ([(0.0, 900.0), (120.0, 30.0)],
                  [(0.0, 5.0)],
                  [(0.0, 1700.0), (10.0, 1700.0)],
                  [(0.0, 3000.0), (2000.0, 5.0)]):
        ep = _ep(spans)
        assert effective_T_at(ep, 1800.0) == pytest.approx(legacy_T(ep, 1800.0))


def test_byte_floor_reads_the_timeline_not_the_duration():
    """A front-loaded episode has a byte floor far above its duration fraction."""
    # 100 one-second bins; the whole asset lands in the first two.
    bins = [500, 500] + [0] * 98
    ep = _ep([(0.0, 100.0)])
    b = byte_floor_at(ep, bins, 1000.0, 1.0, 1000, active_timeout=5.0)
    t = time_floor_at(ep, active_timeout=5.0)
    assert b == pytest.approx(1.0)      # everything already gone
    assert t == pytest.approx(0.05)     # but only 5% of the duration elapsed
    assert b > t


def test_total_loss_when_the_transfer_fits_in_one_interval():
    bins = [100] * 10
    ep = _ep([(0.0, 10.0)])
    assert byte_floor_at(ep, bins, 1000.0, 1.0, 1000,
                         active_timeout=60.0) == pytest.approx(1.0)


def test_sweep_is_monotone_non_increasing_as_the_timeout_tightens():
    bins = [10] * 1000
    ep = _ep([(0.0, 1000.0)])
    rows = sweep(ep, bins, 1000.0, 1.0, 10_000)
    floors = [r.byte_floor for r in sorted(rows, key=lambda r: r.active_timeout)]
    assert floors == sorted(floors)
