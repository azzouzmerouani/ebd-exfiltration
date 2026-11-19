"""Flow granularity vs session granularity, on the seven real episodes.

    uv run ebd-granularity

The paper states that the floor holds at both granularities and that
``T_session >= T_flow``, but measures only the flow case and flags the session
case as an open question. This closes it.

Produces ``results/granularity.csv`` and ``figures/granularity.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..realdata.episodes import group_into_episodes
from ..realdata.floorsweep import byte_floor_at, effective_T_at
from ..realdata.session import (
    DEFAULT_SESSION_GAPS, byte_floor_session, effective_T_session, n_records,
)
from .cadence_sweep import load_flows


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    root = _project_root()
    p = argparse.ArgumentParser(description="Floor at two record granularities")
    p.add_argument("--flows", type=Path,
                   default=root / "data/realdata/exfil_flows.csv")
    p.add_argument("--groundtruth", type=Path,
                   default=root / "data/realdata/groundtruth.jsonl")
    p.add_argument("--active-timeout", type=float, default=1800.0)
    p.add_argument("--out", type=Path, default=root / "results/granularity.csv")
    p.add_argument("--figure", type=Path, default=root / "figures/granularity.png")
    args = p.parse_args(argv)

    flows = load_flows(args.flows)
    episodes = {e.episode_id: e for e in group_into_episodes(flows)}
    gt = [json.loads(l) for l in args.groundtruth.read_text().splitlines() if l.strip()]

    print("=" * 84)
    print("  THE FLOOR AT TWO RECORD GRANULARITIES")
    print("  Session records are emitted at last_flow_end -- the EARLIEST they")
    print("  could exist -- so every session figure here is a lower bound.")
    print("=" * 84)

    gaps = DEFAULT_SESSION_GAPS
    hdr = "".join(f"{int(g):>9}s" for g in gaps)
    print(f"\n{'episode':<22}{'D (h)':>7}{'flow':>8}   session floor at idle gap S{hdr[27:]}")
    print(f"{'':<22}{'':>7}{'floor':>8}   " + "".join(f"{int(g):>9}s" for g in gaps))
    print("-" * 84)

    rows = []
    n_total_loss = {g: 0 for g in gaps}
    for rec in gt:
        ep = episodes.get(rec["episode_id"])
        if ep is None:
            continue
        bins, bs, b0 = rec["bins"], rec["bin_seconds"], rec["bin_start"]
        total = rec["pcap_bytes"]

        f_floor = byte_floor_at(ep, bins, b0, bs, total, args.active_timeout)
        f_T = effective_T_at(ep, args.active_timeout)

        cells = []
        for g in gaps:
            s_floor = byte_floor_session(ep, bins, b0, bs, total, g)
            s_T = effective_T_session(ep, g)
            nrec = n_records(ep, g)
            if s_floor >= 0.999:
                n_total_loss[g] += 1
            cells.append(f"{s_floor * 100:>9.2f}")
            rows.append({
                "episode_id": ep.episode_id,
                "duration_s": f"{ep.duration:.3f}",
                "asset_bytes": total,
                "n_flows": ep.n_flows,
                "flow_T_eff_s": f"{f_T:.3f}",
                "flow_byte_floor": f"{f_floor:.8f}",
                "session_gap_s": g,
                "session_T_eff_s": f"{s_T:.3f}",
                "session_byte_floor": f"{s_floor:.8f}",
                "session_records": nrec,
            })
        short = rec["episode_id"].split("@")[0].split("->")[1]
        print(f"{short:<22}{ep.duration / 3600:>7.1f}{f_floor * 100:>8.2f}   "
              + "".join(cells))

    n_ep = len({r["episode_id"] for r in rows})
    print("-" * 84)
    print(f"\n  episodes that are a TOTAL LOSS (floor = 100%) at session granularity:")
    for g in gaps:
        print(f"    S = {int(g):>5}s   {n_total_loss[g]}/{n_ep}")

    mean_flow = sum(float(r["flow_byte_floor"]) for r in rows
                    if r["session_gap_s"] == gaps[0]) / max(n_ep, 1)
    print(f"\n  mean flow-granularity floor:    {mean_flow * 100:6.2f}%")
    for g in gaps:
        m = sum(float(r["session_byte_floor"]) for r in rows
                if r["session_gap_s"] == g) / max(n_ep, 1)
        print(f"  mean session floor at S={int(g):>5}s: {m * 100:6.2f}%  "
              f"({m / mean_flow if mean_flow else float('inf'):.0f}x the flow floor)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {args.out}")

    _plot(rows, gaps, args.figure)
    print(f"  wrote {args.figure}")
    return 0


def _plot(rows, gaps, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    eids = sorted({r["episode_id"] for r in rows},
                  key=lambda e: float(next(x["duration_s"] for x in rows
                                           if x["episode_id"] == e)))
    labels = [f"{e.split('@')[0].split('->')[1]}\n"
              f"{float(next(x['duration_s'] for x in rows if x['episode_id'] == e)) / 3600:.1f}h"
              for e in eids]

    flow = [float(next(r["flow_byte_floor"] for r in rows if r["episode_id"] == e))
            * 100 for e in eids]
    series = {g: [float(next(r["session_byte_floor"] for r in rows
                            if r["episode_id"] == e and r["session_gap_s"] == g))
                  * 100 for e in eids] for g in gaps}

    x = np.arange(len(eids))
    w = 0.8 / (len(gaps) + 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ax.bar(x - 0.4 + w * 0.5, flow, w, label="flow granularity", color="#1b3a4b")
    cmap = plt.get_cmap("Reds")
    for i, g in enumerate(gaps):
        ax.bar(x - 0.4 + w * (i + 1.5), series[g], w,
               label=f"session, $S$={int(g)}s", color=cmap(0.35 + 0.16 * i))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("% of asset gone before the\nfirst record exists")
    ax.set_ylim(0, 108)
    ax.axhline(100, color="k", lw=0.7, ls=":")
    ax.text(len(eids) - 0.5, 101.5, "total loss", fontsize=7, ha="right")
    ax.set_title("Session-granularity consumers are blind to the whole episode\n"
                 "architectural floor at two record granularities, seven real episodes",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc="center left")
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
