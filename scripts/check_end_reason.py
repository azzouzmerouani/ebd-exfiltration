"""How did the first connections of DAPT2021 episode 443 #3 end?

    uv run python scripts/check_end_reason.py

The emission-latency result of the paper (floor of 443 #3 rising from 0.42% to
46.0% at L = 15 s) assumes the record of the episode's first connection waits
for an inactive timeout. On a FIN/RST-aware exporter (e.g. Cisco Flexible
NetFlow) a connection closed by FIN or RST is exported without that wait. The
released flow records cannot tell: nfstream marks them with expiration_id 0
(idle timeout) whatever the TCP state. This script reads the TCP flags of the
episode's earliest connections from the public PCAP (streamed by HTTP range
requests from the DAPT2021 bucket, as in the ground-truth builder) and reports,
for each connection, the time of its last packet, of its first FIN and of its
first RST, relative to the episode start.

Writes results/end_reason_443_3.csv.
"""
from __future__ import annotations

import csv
import struct
from pathlib import Path

from ebd.realdata import groundtruth as gtmod
from ebd.realdata import pcapreader as pr

PCAP_KEY = "raw/network-captures/Week5_Day5-6_06252021-06262021/net1013x.pcap"
SRC, DST, DPORT = "10.1.3.17", "10.8.10.84", 443
T_START = 1624665799.796946          # episode start (groundtruth.jsonl)
SPORTS = (60793, 60794, 60796, 60797, 60798)   # the connections whose records come first
WINDOW = 400.0                       # seconds after T_START to scan

FIN, RST = 0x01, 0x04


def main() -> int:
    flags: dict[tuple, list[tuple[float, int]]] = {}
    orig = pr._parse_at

    def parse_with_flags(buf, off, end, ts):
        pkt = orig(buf, off, end, ts)
        if pkt is not None and pkt.protocol == pr.PROTO_TCP:
            ihl = (buf[off] & 0x0F) * 4
            f = buf[off + ihl + 13]
            a = (pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port)
            if (a[0] == SRC and a[2] == DST and a[3] == DPORT and a[1] in SPORTS) or \
               (a[0] == DST and a[1] == DPORT and a[2] == SRC and a[3] in SPORTS):
                sport = a[1] if a[0] == SRC else a[3]
                flags.setdefault(sport, []).append((ts, f))
        return pkt

    pr._parse_at = parse_with_flags
    size = gtmod.object_size(PCAP_KEY)
    lo = pr.find_record_offset(lambda o, n: gtmod.fetch_range(PCAP_KEY, o, n), size, T_START - 15)
    keep = lambda s, sp, d, dp, p: {s, d} == {SRC, DST}
    with gtmod._PrefixedStream(PCAP_KEY, lo, size - 1) as stream:
        for _ in pr.iter_packets(stream, keep=keep, stop_after_ts=T_START + WINDOW):
            pass
    pr._parse_at = orig

    rows = []
    for sport in SPORTS:
        pk = sorted(flags.get(sport, []))
        if not pk:
            rows.append(dict(src_port=sport, packets=0))
            continue
        fin = next((t for t, f in pk if f & FIN), None)
        rst = next((t for t, f in pk if f & RST), None)
        rows.append(dict(src_port=sport, packets=len(pk),
                         first_s=round(pk[0][0] - T_START, 3),
                         last_s=round(pk[-1][0] - T_START, 3),
                         first_fin_s="" if fin is None else round(fin - T_START, 3),
                         first_rst_s="" if rst is None else round(rst - T_START, 3)))
    out = Path(__file__).resolve().parents[1] / "results/end_reason_443_3.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["src_port", "packets", "first_s", "last_s",
                                           "first_fin_s", "first_rst_s"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
