"""Time-denominated latency against byte-denominated loss on the same episodes.

    python -m ebd.experiments.time_vs_bytes

For the reproduced detector under the strict protocol, reports per episode the
detection latency in seconds after the episode starts (the denomination of
Puccetti and Ceccarelli, 2026), the latency as a fraction of the episode's
duration, and nEBD.  Kendall's tau between the time latency and nEBD measures
how far the two denominations agree in ordering the episodes.
"""
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path


def tau(x, y):
    c = d = 0
    for i, j in combinations(range(len(x)), 2):
        s = (x[i] - x[j]) * (y[i] - y[j])
        c += s > 0
        d += s < 0
    n = len(x) * (len(x) - 1) / 2
    return (c - d) / n, c, d


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--protocol", choices=("strict", "loo"), default="strict")
    proto = ap.parse_args(argv).protocol
    root = Path(__file__).resolve().parents[3]
    det = json.load(open(root / f"results/real_edxgb_detail_{proto}_wilson.json"))["episodes"]
    gap = {r["episode_id"]: r for r in csv.DictReader(open(root / f"results/real_edxgb_gap_{proto}_wilson.csv"))}
    rows = []
    for eid, d in det.items():
        dur = d["t_end"] - d["t_start"]
        fired = d["alert_s"] is not None
        rows.append(dict(episode_id=eid, duration_s=round(dur, 1),
                         latency_s="" if not fired else round(d["alert_s"], 1),
                         latency_frac_D="" if not fired else round(d["alert_s"] / dur, 6),
                         nebd=round(float(gap[eid]["actual"]), 6), fired=fired))
    with (root / f"results/time_vs_bytes_{proto}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    f = [r for r in rows if r["fired"]]
    t, c, dd = tau([r["latency_s"] for r in f], [r["nebd"] for r in f])
    t2, c2, d2 = tau([r["latency_frac_D"] for r in f], [r["nebd"] for r in f])
    for r in sorted(rows, key=lambda r: r["nebd"]):
        print(r)
    print(f"fired episodes {len(f)}; tau(latency s, nEBD) = {t:+.3f} ({c} concordant, {dd} discordant pairs)")
    print(f"tau(latency/D, nEBD) = {t2:+.3f} ({c2} concordant, {d2} discordant pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
