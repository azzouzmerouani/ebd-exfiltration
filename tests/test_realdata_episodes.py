"""Unit tests for grouping labeled flow records into exfiltration episodes."""

from __future__ import annotations

import pytest

from ebd.realdata import FlowRecord, group_into_episodes
from ebd.realdata.episodes import sensitivity


def _flow(start, duration, src_port=1000, fwd=1_000_000, bwd=1000,
          src="10.0.0.5", dst="203.0.113.9", dport=443):
    return FlowRecord(
        src_ip=src, src_port=src_port, dst_ip=dst, dst_port=dport, protocol=6,
        start=start, duration=duration, fwd_bytes=fwd, bwd_bytes=bwd,
        fwd_packets=fwd // 1500, bwd_packets=10,
        stage="Data Exfiltration", activity="upload",
    )


def test_single_flow_is_one_episode():
    eps = group_into_episodes([_flow(0.0, 30.0)])
    assert len(eps) == 1
    assert eps[0].n_flows == 1
    assert not eps[0].is_multi_flow


def test_exporter_split_stays_one_episode():
    # A 300 s transfer split by a 120 s flow timeout: three back-to-back records.
    flows = [_flow(0.0, 120.0), _flow(120.0, 120.0), _flow(240.0, 60.0)]
    eps = group_into_episodes(flows)
    assert len(eps) == 1
    assert eps[0].n_flows == 3
    assert eps[0].duration == pytest.approx(300.0)
    assert eps[0].fwd_bytes == 3_000_000


def test_parallel_connections_stay_one_episode():
    # Four simultaneous streams differing only in source port.
    flows = [_flow(0.0, 60.0, src_port=p) for p in (1001, 1002, 1003, 1004)]
    eps = group_into_episodes(flows)
    assert len(eps) == 1
    assert eps[0].n_flows == 4
    assert len(eps[0].five_tuples()) == 4      # distinct tuples, one episode
    assert eps[0].duration == pytest.approx(60.0)


def test_distant_transfers_are_separate_episodes():
    # Same peer, but an hour apart -> two incidents, not one.
    flows = [_flow(0.0, 30.0), _flow(3600.0, 30.0)]
    eps = group_into_episodes(flows)
    assert len(eps) == 2
    assert eps[0].t_start < eps[1].t_start


def test_gap_threshold_boundary_is_exclusive():
    # Gap exactly equal to the threshold stays together; just over splits.
    same = group_into_episodes([_flow(0.0, 10.0), _flow(130.0, 10.0)], gap_seconds=120.0)
    assert len(same) == 1
    split = group_into_episodes([_flow(0.0, 10.0), _flow(130.01, 10.0)], gap_seconds=120.0)
    assert len(split) == 2


def test_different_destinations_never_merge():
    flows = [
        _flow(0.0, 10.0, dst="203.0.113.9"),
        _flow(1.0, 10.0, dst="198.51.100.4"),
    ]
    eps = group_into_episodes(flows)
    assert len(eps) == 2


def test_different_destination_ports_never_merge():
    flows = [_flow(0.0, 10.0, dport=443), _flow(1.0, 10.0, dport=21)]
    assert len(group_into_episodes(flows)) == 2


def test_long_flow_keeps_window_open():
    # A 500 s flow overlapped by a later short one: the running end must be the
    # max, not the last-seen, or the short flow would wrongly start a new episode.
    flows = [_flow(0.0, 500.0), _flow(10.0, 5.0, src_port=1002), _flow(520.0, 5.0)]
    eps = group_into_episodes(flows, gap_seconds=120.0)
    assert len(eps) == 1
    assert eps[0].n_flows == 3


def test_episode_geometry_fields():
    flows = [_flow(100.0, 50.0), _flow(150.0, 25.0)]
    ep = group_into_episodes(flows)[0]
    assert ep.t_start == pytest.approx(100.0)
    assert ep.t_end == pytest.approx(175.0)
    assert ep.duration == pytest.approx(75.0)
    assert ep.src_ip == "10.0.0.5"
    assert ep.dst_port == 443


def test_episodes_returned_in_time_order():
    flows = [_flow(500.0, 5.0), _flow(0.0, 5.0, dst="198.51.100.4")]
    eps = group_into_episodes(flows)
    assert [e.t_start for e in eps] == sorted(e.t_start for e in eps)


def test_sensitivity_is_monotonic_non_increasing():
    # Widening the gap can only merge episodes, never create them.
    flows = [_flow(i * 200.0, 20.0) for i in range(6)]
    sens = sensitivity(flows, gaps=(30.0, 60.0, 120.0, 300.0, 600.0))
    counts = [sens[g] for g in sorted(sens)]
    assert counts == sorted(counts, reverse=True)
    assert sens[300.0] == 1        # 180 s silences all merge at a 300 s threshold


def test_negative_gap_rejected():
    with pytest.raises(ValueError):
        group_into_episodes([_flow(0.0, 1.0)], gap_seconds=-1.0)


def test_empty_input():
    assert group_into_episodes([]) == []
