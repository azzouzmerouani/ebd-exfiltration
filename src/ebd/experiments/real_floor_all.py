"""Floor over all nominal DAPT2021 episodes, from flow records alone.

    python -m ebd.experiments.real_floor_all

The packet-level floor in the paper is computed for the seven episodes that
carry at least 0.1 MB, because only those were reconstructed from the PCAPs.
This module reports the floor for all 465 nominal episodes using a flow-level
approximation: bytes of each flow are assumed to accrue uniformly over the
flow's lifetime, and the floor is the fraction of the episode's outbound
(src->dst) bytes accrued when the first record covering the episode is
emitted.  For a single-flow episode the first record is the whole transfer
unless the active timeout cuts it, so its floor is 1 by construction.

The approximation is validated against the packet-level floors on the seven
reconstructed episodes (column ``pcap_floor`` in the per-episode output).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ebd.realdata.cadence_probe import effective_T
from ebd.realdata.episodes import FlowRecord, group_into_episodes

ACTIVE_TIMEOUT = 1800.0


def load_flows(path: Path) -> list[FlowRecord]:
    out = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            out.append(FlowRecord(
                r["src_ip"], int(r["src_port"]), r["dst_ip"], int(r["dst_port"]),
                int(r["protocol"]), int(r["bidirectional_first_seen_ms"]) / 1e3,
                int(r["bidirectional_duration_ms"]) / 1e3,
                int(r["src2dst_bytes"]), int(r["dst2src_bytes"]),
                activity=r.get("Activity", "")))
    return out


def flow_floor(ep, active_timeout: float = ACTIVE_TIMEOUT) -> float:
    total = sum(f.fwd_bytes for f in ep.flows)
    if total <= 0:
        return 1.0
    t_first = ep.t_start + effective_T(ep, active_timeout)
    acc = 0.0
    for f in ep.flows:
        if f.duration <= 0:
            acc += f.fwd_bytes if f.start <= t_first else 0.0
        else:
            frac = min(1.0, max(0.0, (t_first - f.start) / f.duration))
            acc += f.fwd_bytes * frac
    return min(1.0, acc / total)


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser()
    p.add_argument("--flows", type=Path, default=root / "data/realdata/exfil_flows.csv")
    p.add_argument("--floors", type=Path, default=root / "results/real_floor.csv")
    p.add_argument("--groundtruth", type=Path, default=root / "data/realdata/groundtruth.jsonl")
    p.add_argument("--out", type=Path, default=root / "results/real_floor_all.csv")
    a = p.parse_args(argv)

    eps = group_into_episodes(load_flows(a.flows))
    # packet-level floors of the reconstructed episodes, at the paper's T
    pcap = {}
    from ebd.experiments.real_floor import load_groundtruth  # noqa
    from ebd.realdata.cadence_probe import byte_floor
    gt = {r["episode_id"]: r for r in load_groundtruth(a.groundtruth)}
    by_id = {e.episode_id: e for e in eps}
    for eid, r in gt.items():
        if eid in by_id:
            pcap[eid] = byte_floor(by_id[eid], r["bin_start"], r["bin_seconds"],
                                   r["bins"], r["pcap_bytes"])

    rows = []
    for e in eps:
        rows.append(dict(
            episode_id=e.episode_id, dst_port=e.dst_port, n_flows=e.n_flows,
            fwd_bytes=e.fwd_bytes, duration_s=round(e.duration, 3),
            t_eff_s=round(effective_T(e, ACTIVE_TIMEOUT), 3),
            flow_floor=round(flow_floor(e), 6),
            pcap_floor="" if e.episode_id not in pcap else round(pcap[e.episode_id], 6),
            reconstructed=e.episode_id in gt))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    def summ(sel, label):
        n = len(sel)
        if not n:
            return
        b = sum(r["fwd_bytes"] for r in sel)
        unw = sum(r["flow_floor"] for r in sel) / n
        wtd = sum(r["flow_floor"] * r["fwd_bytes"] for r in sel) / b if b else float("nan")
        one = sum(1 for r in sel if r["flow_floor"] >= 0.999)
        print(f"{label:<28} n={n:4d} bytes={b/1e6:8.3f} MB  unweighted={unw*100:6.2f}%"
              f"  byte-weighted={wtd*100:6.2f}%  floor=1: {one}")
    summ(rows, "all nominal episodes")
    summ([r for r in rows if r["n_flows"] == 1], "single-flow")
    summ([r for r in rows if r["n_flows"] > 1], "multi-flow")
    summ([r for r in rows if r["fwd_bytes"] >= 1e5], ">= 0.1 MB (flow bytes)")
    summ([r for r in rows if r["fwd_bytes"] < 1e5], "< 0.1 MB (flow bytes)")
    print("\nvalidation on reconstructed episodes (flow approx vs packet floor):")
    for r in rows:
        if r["reconstructed"]:
            print(f"  {r['episode_id'][:44]:<45} flow={r['flow_floor']*100:7.3f}%  pcap={float(r['pcap_floor'])*100:7.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
