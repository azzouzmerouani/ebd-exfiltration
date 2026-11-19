"""A minimal packet capture for the testbed, for when tcpdump cannot run.

    python -m ebd.testbed.afpacket_capture --iface lo --port 40000 --out cap.pcap

Why this exists
---------------
The network-emulation run of Sec. 9 (M5: 20 ms RTT, 0.5% loss, 1500-byte MTU)
is executed inside an unprivileged user + network namespace, which is what lets
``tc netem`` shape that namespace's own loopback without root on the host. Inside
such a namespace tcpdump starts as uid 0 and then tries to drop privileges to its
own service account, which the namespace forbids, so it aborts after writing an
empty file. This module records the same packets the same way tcpdump would:

* an ``AF_PACKET`` socket bound to the interface, with **kernel** receive
  timestamps (``SO_TIMESTAMPNS``), so packet times are not Python's clock;
* on the loopback interface every frame is seen twice by a packet socket, once
  outgoing and once incoming; libpcap drops the outgoing copy on loopback and so
  do we, which keeps one record per frame as in a tcpdump trace;
* classic pcap format, ``DLT_EN10MB``, the same 96-byte snaplen the original
  capture used. Payload length is later read from ``ip.len``, so truncating the
  payload costs nothing.

Only IPv4 TCP frames to or from ``--port`` are written, the equivalent of
tcpdump's ``tcp port N`` filter. The capture runs until SIGINT or SIGTERM.
"""

from __future__ import annotations

import argparse
import signal
import socket
import struct
import time
from pathlib import Path

ETH_P_ALL = 0x0003
PACKET_OUTGOING = 4
SO_TIMESTAMPNS = 35          # linux/asm-generic/socket.h
DLT_EN10MB = 1
PCAP_MAGIC = 0xA1B2C3D4


def tcp_port_match(frame: bytes, port: int) -> bool:
    """True for an IPv4/TCP Ethernet frame whose source or destination is ``port``."""
    if len(frame) < 14 + 20 or frame[12:14] != b"\x08\x00":
        return False
    ip = frame[14:]
    if ip[0] >> 4 != 4 or ip[9] != 6:
        return False
    ihl = (ip[0] & 0x0F) * 4
    if len(ip) < ihl + 4:
        return False
    sport, dport = struct.unpack(">HH", ip[ihl:ihl + 4])
    return sport == port or dport == port


def pcap_global_header(snaplen: int) -> bytes:
    return struct.pack("<IHHiIII", PCAP_MAGIC, 2, 4, 0, 0, snaplen, DLT_EN10MB)


def pcap_record(ts_sec: int, ts_usec: int, frame: bytes, snaplen: int) -> bytes:
    incl = frame[:snaplen]
    return struct.pack("<IIII", ts_sec, ts_usec, len(incl), len(frame)) + incl


def _kernel_timestamp(ancdata) -> tuple[int, int] | None:
    for level, ctype, data in ancdata:
        if level == socket.SOL_SOCKET and ctype == SO_TIMESTAMPNS and len(data) >= 16:
            sec, nsec = struct.unpack("qq", data[:16])
            return int(sec), int(nsec // 1000)
    return None


def capture(iface: str, port: int, out: Path, snaplen: int = 96,
            ready_file: Path | None = None, announce: bool = False) -> int:
    """Capture until SIGINT/SIGTERM. Returns the number of frames written.

    The caller learns that the socket is live either from ``ready_file`` or,
    with ``announce``, from a ``READY`` line on stdout (which needs no file to
    be deleted afterwards).
    """
    stop = {"flag": False}

    def _stop(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_ALL))
    s.bind((iface, 0))
    s.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 * 1024 * 1024)
    s.settimeout(0.25)
    is_loopback = iface == "lo"

    n = 0
    last_flush = time.monotonic()
    with open(out, "wb") as fh:
        fh.write(pcap_global_header(snaplen))
        fh.flush()
        if ready_file is not None:
            ready_file.write_text("ready\n")
        if announce:
            print("READY", flush=True)
        while not stop["flag"]:
            try:
                frame, anc, _flags, addr = s.recvmsg(65535, socket.CMSG_SPACE(16))
            except socket.timeout:
                continue
            except InterruptedError:
                continue
            if is_loopback and len(addr) > 2 and addr[2] == PACKET_OUTGOING:
                continue
            if not tcp_port_match(frame, port):
                continue
            ts = _kernel_timestamp(anc)
            if ts is None:                       # should not happen; fall back
                now = time.time()
                ts = (int(now), int((now % 1) * 1e6))
            fh.write(pcap_record(ts[0], ts[1], frame, snaplen))
            n += 1
            if time.monotonic() - last_flush > 0.5:
                fh.flush()
                last_flush = time.monotonic()
        fh.flush()
    s.close()
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AF_PACKET capture (tcpdump stand-in)")
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--snaplen", type=int, default=96)
    ap.add_argument("--ready-file", type=Path, default=None)
    ap.add_argument("--announce", action="store_true",
                    help="print READY on stdout once the socket is live")
    args = ap.parse_args(argv)
    n = capture(args.iface, args.port, args.out, args.snaplen, args.ready_file,
                args.announce)
    print(f"afpacket_capture: wrote {n} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
