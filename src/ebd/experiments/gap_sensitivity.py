"""Sensitivity of the DAPT2021 floors to the episode gap G (rev12 review, DA-1).

    python -m ebd.experiments.gap_sensitivity

The seven reference episodes are cut from the exfiltration flow records with a
gap of G = 120 s. A larger gap can merge an episode with a nearby transfer on
the same (src, dst, dst port) key; the earlier transfer then supplies the first
record. This driver regroups the flows at G in {120, 180, 300, 600} s and, for
every merged episode that contains a reference episode, recomputes the floor
under flow records available L seconds after their flow ends.

Byte timeline of a merged episode: the packet-level 1 s bins of every reference
episode it contains (``groundtruth.jsonl``), plus the outbound bytes of the
other flows accrued uniformly over each flow's lifetime (these flows were not
reconstructed from packets). Denominator: the sum of both. At G = 120 s this
reproduces Table ``semantics`` exactly. No model is involved.

Writes ``results/gap_sensitivity.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from ..realdata.episodes import group_into_episodes
from .cadence_sweep import load_flows

GAPS = (120.0, 180.0, 300.0, 600.0)
LATENCIES = (0.0, 15.0, 120.0, 300.0, 600.0, 900.0, 1800.0)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _pcap_delivered(g: dict, t: float) -> float:
    idx = int(math.floor((t - g["bin_start"]) / g["bin_seconds"]))
    if idx < 0:
        return 0.0
    return float(sum(g["bins"][: idx + 1]))


def _flow_delivered(flows, t: float) -> float:
    acc = 0.0
    for f in flows:
        if f.duration <= 0:
            acc += f.fwd_bytes if f.start <= t else 0.0
        else:
            acc += f.fwd_bytes * min(1.0, max(0.0, (t - f.start) / f.duration))
    return acc


def floors(flows, gt: dict, ref_ids: list[str], other_flows, latencies=LATENCIES) -> dict:
    total = sum(gt[r]["pcap_bytes"] for r in ref_ids) + sum(f.fwd_bytes for f in other_flows)
    first_end = min(f.start + f.duration for f in flows)
    out = {}
    for L in latencies:
        t = first_end + L
        got = sum(_pcap_delivered(gt[r], t) for r in ref_ids) + _flow_delivered(other_flows, t)
        out[L] = min(1.0, got / max(total, 1.0))
    return out


def main(argv: list[str] | None = None) -> int:
    root = _root()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flows", type=Path, default=root / "data/realdata/exfil_flows.csv")
    ap.add_argument("--groundtruth", type=Path, default=root / "data/realdata/groundtruth.jsonl")
    ap.add_argument("--out", type=Path, default=root / "results/gap_sensitivity.csv")
    args = ap.parse_args(argv)

    gt = {}
    for line in args.groundtruth.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            gt[d["episode_id"]] = d
    all_flows = load_flows(args.flows)
    ref_eps = {e.episode_id: e for e in group_into_episodes(all_flows, 120.0) if e.episode_id in gt}
    ref_of_flow = {}
    for rid, e in ref_eps.items():
        for f in e.flows:
            ref_of_flow[id(f)] = rid

    rows = []
    for G in GAPS:
        for e in group_into_episodes(all_flows, G):
            refs = sorted({ref_of_flow[id(f)] for f in e.flows if id(f) in ref_of_flow})
            if not refs:
                continue
            other = [f for f in e.flows if id(f) not in ref_of_flow]
            fl = floors(e.flows, gt, refs, other)
            first = min(e.flows, key=lambda f: f.start + f.duration)
            for rid in refs:
                rows.append(dict(
                    gap_s=G, reference_episode=rid, merged_episode=e.episode_id,
                    n_references=len(refs), merged_start=e.t_start,
                    precursor_s=round(min(gt[r]["t_start"] for r in refs) - e.t_start, 3),
                    other_flows=len(other), other_bytes=sum(f.fwd_bytes for f in other),
                    first_record_from_reference=id(first) in ref_of_flow,
                    **{f"floor_L{int(L)}": round(v, 6) for L, v in fl.items()}))

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    for G in GAPS:
        sub = [r for r in rows if r["gap_s"] == G]
        means = {L: 100 * sum(r[f"floor_L{int(L)}"] for r in sub) / len(sub) for L in LATENCIES}
        print(f"G={int(G):>4}s  " + "  ".join(f"L{int(L)}={m:6.2f}%" for L, m in means.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
