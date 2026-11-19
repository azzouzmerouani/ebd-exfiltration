"""Unit tests for the streaming PCAP reader and byte-timeline builder.

Every fixture is a pcap constructed byte-by-byte in memory, so the expected
counts are known exactly rather than inferred from a capture.
"""

from __future__ import annotations

import io
import struct

import pytest

# Real captures carry real epoch timestamps; the reader uses a plausibility
# window to re-find record boundaries after corruption, so fixtures must too.
BASE = 1625182937.0

from ebd.realdata.pcapreader import (
    PcapFormatError,
    TimelineBuilder,
    iter_packets,
)


# --------------------------------------------------------------------------- #
# builders                                                                     #
# --------------------------------------------------------------------------- #

def _ipv4(src, dst, proto, l4: bytes) -> bytes:
    total = 20 + len(l4)
    hdr = struct.pack(
        "!BBHHHBBH4s4s", 0x45, 0, total, 0, 0, 64, proto, 0,
        bytes(int(x) for x in src.split(".")),
        bytes(int(x) for x in dst.split(".")),
    )
    return hdr + l4


def _tcp(sport, dport, seq, payload: bytes) -> bytes:
    # data offset 5 (20 bytes), no options
    hdr = struct.pack("!HHIIBBHHH", sport, dport, seq, 0, 5 << 4, 0x18, 0, 0, 0)
    return hdr + payload


def _udp(sport, dport, payload: bytes) -> bytes:
    return struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload


def _eth(l3: bytes, vlan: bool = False) -> bytes:
    if vlan:
        return (b"\x00" * 12 + struct.pack("!H", 0x8100)
                + struct.pack("!H", 0) + struct.pack("!H", 0x0800) + l3)
    return b"\x00" * 12 + struct.pack("!H", 0x0800) + l3


def _pcap(frames, linktype=1, nano=False, endian="<") -> bytes:
    # The canonical magic value, written in the chosen byte order — exactly what
    # a real writer does on that platform.
    magic = 0xA1B23C4D if nano else 0xA1B2C3D4
    out = struct.pack(endian + "IHHiIII", magic, 2, 4, 0, 0, 262144, linktype)
    for ts, frame in frames:
        sec = int(ts)
        frac = int(round((ts - sec) * (1e9 if nano else 1e6)))
        out += struct.pack(endian + "IIII", sec, frac, len(frame), len(frame)) + frame
    return out


# --------------------------------------------------------------------------- #
# parsing                                                                      #
# --------------------------------------------------------------------------- #

def test_parses_tcp_payload_length():
    f = _eth(_ipv4("10.0.0.1", "10.0.0.2", 6, _tcp(1234, 443, 1, b"x" * 100)))
    pkts = list(iter_packets(io.BytesIO(_pcap([(BASE + 0.5, f)]))))
    assert len(pkts) == 1
    p = pkts[0]
    assert (p.src_ip, p.src_port, p.dst_ip, p.dst_port) == ("10.0.0.1", 1234, "10.0.0.2", 443)
    assert p.protocol == 6
    assert p.payload_len == 100
    assert p.ts == pytest.approx(BASE + 0.5, abs=1e-3)


def test_parses_udp_payload_length():
    f = _eth(_ipv4("10.0.0.1", "10.0.0.2", 17, _udp(53, 5353, b"y" * 40)))
    p = list(iter_packets(io.BytesIO(_pcap([(BASE + 5.0, f)]))))[0]
    assert p.protocol == 17 and p.payload_len == 40 and p.seq is None


def test_headers_are_excluded_from_payload():
    # 60-byte payload inside eth+ip+tcp framing must report exactly 60
    f = _eth(_ipv4("1.1.1.1", "2.2.2.2", 6, _tcp(1, 2, 0, b"z" * 60)))
    assert list(iter_packets(io.BytesIO(_pcap([(BASE + 1.0, f)]))))[0].payload_len == 60


def test_vlan_tagged_frames_are_parsed():
    f = _eth(_ipv4("10.0.0.1", "10.0.0.2", 6, _tcp(1, 2, 0, b"a" * 10)), vlan=True)
    pkts = list(iter_packets(io.BytesIO(_pcap([(BASE + 1.0, f)]))))
    assert len(pkts) == 1 and pkts[0].payload_len == 10


def test_nanosecond_and_big_endian_variants():
    f = _eth(_ipv4("10.0.0.1", "10.0.0.2", 6, _tcp(1, 2, 0, b"a" * 5)))
    for nano in (False, True):
        for endian in ("<", ">"):
            pk = list(iter_packets(io.BytesIO(_pcap([(BASE + 7.25, f)], nano=nano, endian=endian))))
            assert len(pk) == 1
            assert pk[0].ts == pytest.approx(BASE + 7.25, abs=1e-3)


def test_non_ipv4_frames_are_skipped():
    arp = b"\x00" * 12 + struct.pack("!H", 0x0806) + b"\x00" * 28
    f = _eth(_ipv4("10.0.0.1", "10.0.0.2", 6, _tcp(1, 2, 0, b"a" * 8)))
    pkts = list(iter_packets(io.BytesIO(_pcap([(BASE + 1.0, arp), (BASE + 2.0, f)]))))
    assert len(pkts) == 1


def test_keep_filter_applied():
    a = _eth(_ipv4("10.0.0.1", "10.0.0.2", 6, _tcp(1, 443, 0, b"a" * 10)))
    b = _eth(_ipv4("10.0.0.9", "10.0.0.2", 6, _tcp(1, 443, 0, b"b" * 10)))
    keep = lambda s, sp, d, dp, pr: s == "10.0.0.1"
    pkts = list(iter_packets(io.BytesIO(_pcap([(BASE + 1.0, a), (BASE + 2.0, b)])), keep=keep))
    assert len(pkts) == 1 and pkts[0].src_ip == "10.0.0.1"


def test_bad_magic_rejected():
    with pytest.raises(PcapFormatError):
        list(iter_packets(io.BytesIO(b"\x00" * 40)))


def test_truncated_stream_stops_cleanly():
    f = _eth(_ipv4("10.0.0.1", "10.0.0.2", 6, _tcp(1, 2, 0, b"a" * 10)))
    raw = _pcap([(BASE + 1.0, f)])[:-4]          # cut mid-packet
    assert list(iter_packets(io.BytesIO(raw))) == []


# --------------------------------------------------------------------------- #
# damage recovery (the real captures are not clean)                            #
# --------------------------------------------------------------------------- #

def test_resync_recovers_packets_after_a_corrupt_header():
    from ebd.realdata.pcapreader import ParseStats

    good = [(BASE + i, _eth(_ipv4("10.0.0.1", "10.0.0.2", 6,
                                  _tcp(1, 443, i * 100, b"p" * 100))))
            for i in range(6)]
    raw = bytearray(_pcap(good))
    # Corrupt the 3rd record header in place (offset: 24 global + 2 full records)
    rec = 16 + len(good[0][1])
    off = 24 + 2 * rec
    raw[off:off + 8] = b"\xff\xff\xff\xff\xff\xff\xff\xff"

    st = ParseStats()
    pkts = list(iter_packets(io.BytesIO(bytes(raw)), stats=st))
    # the damaged record is lost, the rest are recovered
    assert len(pkts) >= 4
    assert st.resyncs >= 1
    assert st.resync_bytes_skipped > 0


def test_record_whose_length_swallows_next_header_is_repaired():
    from ebd.realdata.pcapreader import ParseStats

    # Exactly the real-file defect: declared length overruns into later records.
    good = [(BASE + i, _eth(_ipv4("10.0.0.1", "10.0.0.2", 6,
                                  _tcp(1, 443, i * 100, b"q" * 60))))
            for i in range(5)]
    raw = bytearray(_pcap(good))
    frame_len = len(good[0][1])
    off = 24                                   # first record header
    # claim the first packet is far longer than it is
    struct.pack_into("<I", raw, off + 8, frame_len + 500)

    st = ParseStats()
    pkts = list(iter_packets(io.BytesIO(bytes(raw)), stats=st))
    # later packets must still be found rather than swallowed
    assert len(pkts) >= 4
    assert st.resyncs >= 1


def test_clean_file_needs_no_resync():
    from ebd.realdata.pcapreader import ParseStats

    good = [(BASE + i, _eth(_ipv4("10.0.0.1", "10.0.0.2", 6,
                                  _tcp(1, 443, i * 100, b"r" * 40))))
            for i in range(8)]
    st = ParseStats()
    pkts = list(iter_packets(io.BytesIO(_pcap(good)), stats=st))
    assert len(pkts) == 8
    assert st.resyncs == 0
    assert st.resync_bytes_skipped == 0
    assert st.loss_fraction == 0.0


# --------------------------------------------------------------------------- #
# timeline                                                                     #
# --------------------------------------------------------------------------- #

def _mk(ts, seq, n, src="10.0.0.1"):
    from ebd.realdata.pcapreader import Packet
    return Packet(ts, src, 1234, "10.0.0.2", 443, 6, n, seq)


def test_timeline_bins_and_totals():
    tb = TimelineBuilder(bin_seconds=1.0)
    for i, ts in enumerate([100.0, 100.4, 101.2, 103.9]):
        tb.add(_mk(ts, seq=i * 100, n=50))
    times, vals = tb.timeline()
    assert tb.total_bytes == 200
    assert vals == [100, 50, 0, 50]          # bin 102 is an explicit zero
    assert times[0] == pytest.approx(100.0)


def test_retransmission_is_not_double_counted():
    tb = TimelineBuilder()
    tb.add(_mk(1.0, seq=1000, n=500))
    tb.add(_mk(2.0, seq=1000, n=500))        # same seq, retransmit
    assert tb.total_bytes == 500
    assert tb.duplicate_bytes == 500


def test_same_seq_on_different_connection_still_counts():
    tb = TimelineBuilder()
    tb.add(_mk(1.0, seq=1, n=100, src="10.0.0.1"))
    tb.add(_mk(1.0, seq=1, n=100, src="10.0.0.7"))
    assert tb.total_bytes == 200


def test_zero_payload_packets_ignored():
    tb = TimelineBuilder()
    tb.add(_mk(1.0, seq=1, n=0))             # pure ACK
    assert tb.total_bytes == 0 and tb.n_packets == 0


def test_bytes_by_is_cumulative():
    tb = TimelineBuilder(bin_seconds=1.0)
    tb.add(_mk(10.0, seq=1, n=10))
    tb.add(_mk(11.0, seq=2, n=20))
    tb.add(_mk(12.0, seq=3, n=30))
    assert tb.bytes_by(10.5) == 10
    assert tb.bytes_by(11.5) == 30
    assert tb.bytes_by(99.0) == 60
