"""The architectural floor across the telemetry cadence, on real episodes.

    uv run ebd-sweep

Answers the question the Discussion section poses but the paper did not
previously measure: *if the exporter were configured differently, how much
earlier could any flow-record consumer possibly act?*

Produces ``results/cadence_sweep.csv`` and ``figures/cadence_sweep.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..realdata.episodes import FlowRecord, group_into_episodes
from ..realdata.cadence_probe import effective_T as legacy_effective_T
from ..realdata.floorsweep import DEFAULT_TIMEOUTS, sweep, effective_T_at


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_flows(path: Path) -> list[FlowRecord]:
    """Load the exfiltration flow records exactly as the other drivers do."""
    out: list[FlowRecord] = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                start_ms = float(row["bidirectional_first_seen_ms"])
                dur_ms = float(row["bidirectional_duration_ms"])
            except (KeyError, ValueError):
                continue
            out.append(
                FlowRecord(
                    src_ip=row["src_ip"],
                    src_port=int(float(row["src_port"])),
                    dst_ip=row["dst_ip"],
                    dst_port=int(float(row["dst_port"])),
                    protocol=int(float(row["protocol"])),
                    start=start_ms / 1000.0,
                    duration=dur_ms / 1000.0,
                    fwd_bytes=int(float(row["src2dst_bytes"])),
                    bwd_bytes=int(float(row["dst2src_bytes"])),
                    fwd_packets=int(float(row["src2dst_packets"])),
                    bwd_packets=int(float(row["dst2src_packets"])),
                    source_file=row.get("source_file", ""),
                )
            )
    return out


def main(argv: list[str] | None = None) -> int:
    root = _project_root()
    p = argparse.ArgumentParser(description="Floor vs telemetry cadence")
    p.add_argument("--flows", type=Path,
                   default=root / "data/realdata/exfil_flows.csv")
    p.add_argument("--groundtruth", type=Path,
                   default=root / "data/realdata/groundtruth.jsonl")
    p.add_argument("--out", type=Path, default=root / "results/cadence_sweep.csv")
    p.add_argument("--figure", type=Path,
                   default=root / "figures/cadence_sweep.png")
    args = p.parse_args(argv)

    flows = load_flows(args.flows)
    episodes = {e.episode_id: e for e in group_into_episodes(flows)}
    gt = [json.loads(l) for l in args.groundtruth.read_text().splitlines() if l.strip()]

    print("=" * 78)
    print("  THE ARCHITECTURAL FLOOR ACROSS THE TELEMETRY CADENCE")
    print("  Eq. (4'): T_eff(e; Ta) = min_f min(s_f + Ta, e_f) - t_start(e)")
    print("=" * 78)

    rows = []
    for rec in gt:
        ep = episodes.get(rec["episode_id"])
        if ep is None:
            print(f"  ! no flow group for {rec['episode_id']}")
            continue

        # Regression check: Eq. (4') must reproduce Eq. (4) at the timeout the
        # capture was actually exported with. If this ever fails, one of the two
        # is wrong and the counterfactual is not trustworthy.
        legacy = legacy_effective_T(ep, 1800.0)
        general = effective_T_at(ep, 1800.0)
        assert abs(legacy - general) < 1e-6, (
            f"Eq. (4') disagrees with Eq. (4) at Ta=1800 on {ep.episode_id}: "
            f"{general:.6f} vs {legacy:.6f}"
        )

        rows.extend(
            sweep(
                ep,
                bins=rec["bins"],
                bin_start=rec["bin_start"],
                bin_seconds=rec["bin_seconds"],
                total_bytes=rec["pcap_bytes"],
                timeouts=DEFAULT_TIMEOUTS,
            )
        )

    # --- report -----------------------------------------------------------
    by_ep: dict[str, list] = {}
    for r in rows:
        by_ep.setdefault(r.episode_id, []).append(r)

    header = "  ".join(f"{int(t):>7}s" for t in DEFAULT_TIMEOUTS)
    print(f"\n{'episode':<40}{'D (h)':>8}   {header}")
    print("-" * (48 + len(header)))
    for eid, rs in by_ep.items():
        short = eid.split("@")[0].split("->")[1]
        d_h = rs[0].duration / 3600.0
        cells = "  ".join(f"{r.byte_floor * 100:>7.2f}" for r in rs)
        print(f"{short:<40}{d_h:>8.1f}   {cells}")

    n_ep = len(by_ep)
    print("\n  mean byte floor (%) by cadence:")
    for i, ta in enumerate(DEFAULT_TIMEOUTS):
        m = sum(rs[i].byte_floor for rs in by_ep.values()) / max(n_ep, 1)
        worst = max(rs[i].byte_floor for rs in by_ep.values())
        print(f"    Ta={int(ta):>5}s   mean {m * 100:6.3f}%   worst {worst * 100:6.3f}%")

    # Headline: reduction from the deployed default to the most aggressive
    i_default = DEFAULT_TIMEOUTS.index(1800.0)
    i_tight = DEFAULT_TIMEOUTS.index(15.0)
    mean_def = sum(rs[i_default].byte_floor for rs in by_ep.values()) / n_ep
    mean_tight = sum(rs[i_tight].byte_floor for rs in by_ep.values()) / n_ep
    print(f"\n  1800s -> 15s reduces the mean floor "
          f"{mean_def * 100:.3f}% -> {mean_tight * 100:.3f}% "
          f"({mean_def / mean_tight if mean_tight else float('inf'):.1f}x)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode_id", "active_timeout_s", "t_eff_s", "duration_s",
                    "byte_floor", "time_floor", "asset_bytes"])
        for r in rows:
            w.writerow([r.episode_id, r.active_timeout, f"{r.t_eff:.4f}",
                        f"{r.duration:.4f}", f"{r.byte_floor:.8f}",
                        f"{r.time_floor:.8f}", r.asset_bytes])
    print(f"\n  wrote {args.out}")

    _plot(by_ep, args.figure)
    print(f"  wrote {args.figure}")
    return 0


def _plot(by_ep: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for eid, rs in sorted(by_ep.items(), key=lambda kv: kv[1][0].duration):
        xs = [r.active_timeout for r in rs]
        ys = [r.byte_floor * 100 for r in rs]
        label = f"{eid.split('@')[0].split('->')[1]}  D={rs[0].duration / 3600:.1f}h"
        ax.plot(xs, ys, marker="o", ms=4, lw=1.4, label=label)

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("exporter active timeout $T_a$ (s, log scale)")
    ax.set_ylabel("architectural floor: % of asset gone\nbefore any record exists")
    ax.set_title("The floor is a deployment choice, not a constant\n"
                 "oracle floor vs telemetry cadence, seven real episodes",
                 fontsize=10)
    ax.set_xticks(list(DEFAULT_TIMEOUTS))
    ax.set_xticklabels([f"{int(t)}" for t in DEFAULT_TIMEOUTS])
    ax.grid(alpha=0.3, which="both", lw=0.4)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
