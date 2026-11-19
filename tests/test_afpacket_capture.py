"""The tcpdump stand-in used for the netem capture writes standard pcap."""

from __future__ import annotations

import struct

from ebd.testbed.afpacket_capture import (
    pcap_global_header, pcap_record, tcp_port_match,
)


def _frame(sport: int, dport: int, payload: int = 100, proto: int = 6) -> bytes:
    eth = b"\x00" * 12 + b"\x08\x00"
    total = 20 + 20 + payload
    ip = bytes([0x45, 0]) + struct.pack(">H", total) + b"\x00" * 5 + bytes([proto]) \
        + b"\x00" * 10
    tcp = struct.pack(">HHII", sport, dport, 1, 0) + bytes([0x50, 0x18]) + b"\x00" * 6
    return eth + ip + tcp + b"\x5a" * payload


def test_port_filter_matches_either_direction_and_only_tcp_ipv4():
    assert tcp_port_match(_frame(40000, 9000), 9000)
    assert tcp_port_match(_frame(9000, 40000), 9000)
    assert not tcp_port_match(_frame(40000, 9001), 9000)
    assert not tcp_port_match(_frame(40000, 9000, proto=17), 9000)
    assert not tcp_port_match(b"\x00" * 12 + b"\x86\xdd" + b"\x00" * 60, 9000)


def test_pcap_header_and_record_layout():
    h = pcap_global_header(96)
    magic, vmaj, vmin, tz, sig, snap, link = struct.unpack("<IHHiIII", h)
    assert (magic, vmaj, vmin, snap, link) == (0xA1B2C3D4, 2, 4, 96, 1)
    fr = _frame(40000, 9000, payload=1400)
    rec = pcap_record(1_700_000_000, 123456, fr, 96)
    s, us, incl, orig = struct.unpack("<IIII", rec[:16])
    assert (s, us, incl, orig) == (1_700_000_000, 123456, 96, len(fr))
    assert rec[16:] == fr[:96]
