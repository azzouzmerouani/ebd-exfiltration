"""Composed-pipeline floors on DAPT2021 at a larger episode gap (rev14 review, DA-1).

    python -m ebd.experiments.pipeline_profiles_gap

Combines ``pipeline_profiles`` (the DAPT2021 profiles: release, collection or
aggregation, ingestion and the rule phase, averaged over the phase) with the
episode regrouping of ``gap_sensitivity``: at G = 300 s an episode may merge with
a nearby transfer on the same key, whose flows then supply the first record.
Byte timeline of a merged episode as in ``gap_sensitivity``: packet-level bins
of the reference episodes plus the outbound bytes of the other flows accrued
uniformly over each flow's lifetime. Release at teardown follows the
destination port as in ``pipeline_profiles`` (port-443 first records carry FIN,
port-3306 ones do not); for a merged first record from a non-reference flow this
is an assumption. At G = 120 s the output reproduces the DAPT2021 rows of
``results/pipeline_profiles.csv`` (asserted). No model is involved.

Writes ``results/pipeline_profiles_gap.csv``.
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

from ..realdata.episodes import group_into_episodes
from .cadence_sweep import load_flows
from .gap_sensitivity import _flow_delivered, _pcap_delivered
from .pipeline_profiles import PROFILES, _phases

GAPS = (120.0, 300.0)


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    gt = {}
    for line in (root / "data/realdata/groundtruth.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            gt[d["episode_id"]] = d
    all_flows = load_flows(root / "data/realdata/exfil_flows.csv")
    ref_eps = {e.episode_id: e for e in group_into_episodes(all_flows, 120.0) if e.episode_id in gt}
    ref_of_flow = {id(f): rid for rid, e in ref_eps.items() for f in e.flows}
    rows = []
    for G in GAPS:
        for e in group_into_episodes(all_flows, G):
            refs = sorted({ref_of_flow[id(f)] for f in e.flows if id(f) in ref_of_flow})
            if not refs:
                continue
            other = [f for f in e.flows if id(f) not in ref_of_flow]
            total = sum(gt[r]["pcap_bytes"] for r in refs) + sum(f.fwd_bytes for f in other)
            t_first_packet = min(min(gt[r]["t_start"] for r in refs),
                                 min((f.start for f in other), default=float("inf")))

            def frac(t):
                got = sum(_pcap_delivered(gt[r], t) for r in refs) + _flow_delivered(other, t)
                return min(1.0, got / max(total, 1.0))

            for prof, p in PROFILES.items():
                if prof == "dns_query_log":
                    continue
                if p["anchor"] == "flow_end":
                    rel = p["fin"] if e.dst_port == 443 else p["nofin"]
                    t0 = min(f.start + f.duration for f in e.flows) + rel
                else:
                    t0 = t_first_packet
                offs, worst = _phases(p["window"], p["period"])
                v = [frac(t0 + p["fixed"] + o) for o in offs]
                for rid in refs:
                    rows.append(dict(gap_s=G, profile=prof, reference_episode=rid,
                                     merged_episode=e.episode_id, other_flows=len(other),
                                     floor_mean_phase=sum(v) / len(v),
                                     floor_worst_phase=frac(t0 + p["fixed"] + worst)))
    # regression check against pipeline_profiles at G = 120 s
    ref = {(r["profile"], r["unit"]): float(r["floor_mean_phase"])
           for r in csv.DictReader((root / "results/pipeline_profiles.csv").open())
           if r["corpus"] == "dapt2021"}
    for r in rows:
        if r["gap_s"] == 120.0:
            assert abs(r["floor_mean_phase"] - ref[(r["profile"], r["reference_episode"])]) < 1e-9, r
    out = root / "results/pipeline_profiles_gap.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    for G in GAPS:
        for prof in PROFILES:
            sub = [r["floor_mean_phase"] for r in rows if r["gap_s"] == G and r["profile"] == prof]
            if sub:
                no1 = [r["floor_mean_phase"] for r in rows if r["gap_s"] == G and r["profile"] == prof
                       and "1625264949" not in r["reference_episode"]]
                print(f"G={int(G):>4} {prof:<16} n={len(sub)} mean {100*st.mean(sub):6.2f}% "
                      f"median {100*st.median(sub):6.2f}%  mean without 443#1 {100*st.mean(no1):6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
