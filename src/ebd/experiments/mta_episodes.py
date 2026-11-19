"""Floors of the real-malware episodes annotated from Malware-Traffic-Analysis.net.

    python -m ebd.experiments.mta_episodes

Input: one packet table per infection in ``data/mta/``, exported by the authors
from the published capture in Wireshark (columns No., Time, Source,
Destination, Protocol, Length, Stream index, TCP Segment Len, Flags; for AMOS
also the HTTP method and path). Each table already holds only the packets
between the infected host and the exfiltration peer named in the post's
indicators; no payload and no credentials are kept.

Episode: the TCP streams that carry the theft (for AMOS only the
``POST /contact`` uploads; the later backdoor polling is excluded). V_e(t) is
the cumulative TCP payload sent by the infected host; t = 0 at its first
payload byte; D is the time of its last payload byte. A connection ends at
its last packet; it is FIN/RST-closed if any packet carries FIN or RST.
T_eff is the end of the earliest-ending connection; the floor at L = 0 is
V_e(T_eff)/V_e.

Composed pipelines use the release delays measured in the lab. Columns
``Cisco-like NetFlow``, ``Zeek`` and ``softflowd defaults`` keep the rev17
configuration (``pipeline_profiles_measured``: 300 s collector files for NetFlow,
25 s overhead, 5-min rule). Columns ``<release>_p<period>`` sweep the rule period
(0 = a stream processor reading each record 2 s after release; 10 to 900 s)
with records forwarded as they arrive and the directly observed overhead
(``pipeline_sweep.OVH_DIRECT``), averaged over the phase (24 grid points);
``*_p300_min``/``_max`` give the best and worst phase. ``margin_s`` = D - T_eff:
any release-plus-read delay at least this long loses the transfer in full.
``first_conn_bytes``: upload of the earliest-ending connection;
``floor_L0_excl_first``: floor at L = 0 with that connection removed;
``largest_conn_share``: share of the upload in the largest connection (an
episode reduced to that connection is a single-connection total loss). No model
is involved. Writes ``data/mta/mta_episodes.csv``.
"""
from __future__ import annotations

import csv
import ipaddress
from bisect import bisect_right
from pathlib import Path

from .pipeline_profiles import _phases
from .pipeline_profiles_measured import PROFILES_MEASURED
from .pipeline_sweep import OVH_DIRECT, PERIODS, RELEASE, STREAM

INFECTIONS = [  # file, family, channel, streams (None = all with upload payload)
    ("2023-10-11-lumma-packets.csv", "Lumma", "HTTP", None),
    ("2024-09-17-snake-ftp-packets.csv", "Snake KeyLogger", "FTP", None),
    ("2024-09-16-snake-smtp-packets.csv", "Snake KeyLogger", "SMTP", None),
    ("2023-11-22-agenttesla-ftp-packets.csv", "AgentTesla", "FTP", None),
    ("2024-10-23-redline-packets.csv", "RedLine", "HTTP", None),
    ("2024-01-12-stealc-packets.csv", "StealC", "HTTP", None),
    ("2026-09-10-amos-packets.csv", "AMOS (macOS)", "HTTP", "POST /contact"),
]
PIPES = {"Cisco-like NetFlow": "netflow_onprem", "Zeek": "zeek_conn",
         "softflowd defaults": "softflowd_default"}


def _private(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def _rows(path: Path, select, drop=None):
    rows = list(csv.DictReader(path.open()))
    if select is not None:  # streams whose request line matches
        keep = {r["Stream index"] for r in rows if (r.get("http_request") or "").startswith(select)}
        rows = [r for r in rows if r["Stream index"] in keep]
    if drop is not None:
        rows = [r for r in rows if r["Stream index"] != drop]
    return rows


def _stream_uploads(path: Path, select):
    out = {}
    for r in _rows(path, select):
        if _private(r["Source"]) and not _private(r["Destination"]):
            out[r["Stream index"]] = out.get(r["Stream index"], 0) + int(r["TCP Segment Len"] or 0)
    return {k: v for k, v in out.items() if v > 0}


def _upload_of(path: Path, select, stream):
    return sum(int(r["TCP Segment Len"] or 0) for r in _rows(path, select)
               if r["Stream index"] == stream and _private(r["Source"]) and not _private(r["Destination"]))


def episode(path: Path, select, drop=None):
    rows = _rows(path, select, drop)
    up = [(float(r["Time"]), int(r["TCP Segment Len"] or 0), r["Stream index"]) for r in rows
          if _private(r["Source"]) and not _private(r["Destination"]) and int(r["TCP Segment Len"] or 0) > 0]
    streams = {s for _, _, s in up}
    up.sort()
    t0 = up[0][0]
    times = [t - t0 for t, _, _ in up]
    cum, acc = [], 0
    for _, b, _ in up:
        acc += b
        cum.append(acc)
    conns = {}
    for r in rows:
        s = r["Stream index"]
        if s not in streams:
            continue
        t = float(r["Time"]) - t0
        f = int(r["Flags"], 16) if r["Flags"] else 0
        end, fin = conns.get(s, (-1e18, False))
        conns[s] = (max(end, t), fin or bool(f & 0x5))
    return times, cum, conns


def frac(times, cum, t):
    i = bisect_right(times, t)
    return cum[i - 1] / cum[-1] if i else 0.0


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    out = []
    extra = {}
    for fname, fam, ch, sel in INFECTIONS:
        times, cum, conns = episode(root / "data/mta" / fname, sel)
        first_end, first_fin = min(conns.values())
        first_id = min(conns, key=conns.get)
        row = dict(file=fname, family=fam, channel=ch, conns=len(conns), V_bytes=cum[-1],
                   D_s=round(times[-1], 2), T_eff_s=round(first_end, 2),
                   floor_L0=round(100 * frac(times, cum, first_end), 2))
        row["margin_s"] = round(times[-1] - first_end, 2)
        # rev17 configuration (25 s overhead, NetFlow with 300 s collector files, 5-min rule)
        for col, key in PIPES.items():
            p = PROFILES_MEASURED[key]
            rel = first_end + (p["fin"] if first_fin else p["nofin"])
            offs, _ = _phases(p["window"], p["period"])
            v = [frac(times, cum, rel + p["fixed"] + o) for o in offs]
            row[col] = round(100 * sum(v) / len(v), 1)
        # rule-period sweep with the measured release, records forwarded as they
        # arrive, and the directly observed overhead (pipeline_sweep)
        for tag, r in (("netflow", RELEASE["netflow"]), ("zeek", RELEASE["zeek"]),
                       ("softflowd", RELEASE["softflowd_default"])):
            rel = first_end + (r["fin"] if first_fin else r["nofin"])
            for period in PERIODS:
                ovh = STREAM if period == 0 else OVH_DIRECT
                offs, _ = _phases(0.0, period)
                v = [frac(times, cum, rel + ovh + o) for o in offs]
                row[f"{tag}_p{int(period)}"] = round(100 * sum(v) / len(v), 1)
                if period == 300:
                    row[f"{tag}_p300_min"] = round(100 * min(v), 1)
                    row[f"{tag}_p300_max"] = round(100 * max(v), 1)
        up_first = _upload_of(root / "data/mta" / fname, sel, first_id)
        row["first_conn_bytes"] = up_first
        if len(conns) > 1:
            t2, c2, k2 = episode(root / "data/mta" / fname, sel, drop=first_id)
            row["floor_L0_excl_first"] = round(100 * frac(t2, c2, min(k2.values())[0]), 2)
        else:
            row["floor_L0_excl_first"] = ""
        sizes = _stream_uploads(root / "data/mta" / fname, sel)
        row["largest_conn_share"] = round(100 * max(sizes.values()) / sum(sizes.values()), 1)
        out.append(row)
    dst = root / "data/mta/mta_episodes.csv"
    with dst.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
