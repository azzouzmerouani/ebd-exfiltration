"""Figure for Sec. 8: the floor on the DNS corpus, at three pipeline positions.

    uv run python -m ebd.experiments.dns_figure

Reads the committed outputs of the v4.2 DNS run (``results/dns_v42/``) and the
DAPT2021 cadence sweep (``results/cadence_sweep.csv``) and writes
``figures/dns_floors.png``. Left: each tool family's floor at the three pipeline
positions, on identical packets. Right: the mean floor against the record
cadence: the DNS flow floor against the exporter's active timeout, the DNS
session floor against the aggregation window, and, for contrast, the DAPT2021
flow floor against the active timeout.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

# Categorical slots 1-3 of the reference palette, validated all-pairs (light).
COLORS = {"per-query": "#2a78d6", "flow": "#eb6834", "session": "#1baf7a"}
MARKERS = {"per-query": "o", "flow": "s", "session": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAMILY_LABEL = {"cobaltstrike": "Cobalt Strike", "dns2tcp": "dns2tcp", "dnscat2": "dnscat2",
                "dnsexfiltrator": "DNSExfiltrator", "dnspot": "dnspot", "iodine": "iodine",
                "ozymandns": "OzymanDNS", "tcp_over_dns": "tcp-over-dns"}


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = _root()
    ap = argparse.ArgumentParser(description="DNS floor figure")
    ap.add_argument("--dns", type=Path, default=root / "results/dns_v42")
    ap.add_argument("--dapt-sweep", type=Path, default=root / "results/cadence_sweep.csv")
    ap.add_argument("--out", type=Path, default=root / "figures/dns_floors.png")
    args = ap.parse_args(argv)

    floors = list(csv.DictReader((args.dns / "dns_floors.csv").open()))
    fam = defaultdict(lambda: defaultdict(list))
    ports = defaultdict(list)
    for r in floors:
        fam[r["family"]][r["consumer"]].append(float(r["floor_measured"]) * 100)
        if r["consumer"] == "flow":
            ports[r["family"]].append(int(r["n_ports"]))
    # order: single-port families first (the flow-blind ones), then by flow floor
    order = sorted(fam, key=lambda f: (-np.mean(fam[f]["flow"]), f))

    sweep = list(csv.DictReader((args.dns / "dns_cadence_sweep.csv").open()))
    by_axis = defaultdict(lambda: defaultdict(list))
    for r in sweep:
        by_axis[r["axis"]][float(r["param"])].append(float(r["floor_measured"]) * 100)
    dapt = defaultdict(list)
    for r in csv.DictReader(args.dapt_sweep.open()):
        dapt[float(r["active_timeout_s"])].append(float(r["byte_floor"]) * 100)

    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})

    y = np.arange(len(order))[::-1]                       # first family at the top
    for cons, off in (("per-query", 0.2), ("flow", 0.0), ("session", -0.2)):
        xs_ = [np.mean(fam[f][cons]) for f in order]
        a.scatter(xs_, y + off, s=24, marker=MARKERS[cons], color=COLORS[cons],
                  edgecolor="white", linewidth=0.7, zorder=3, label=cons)
    a.set_xscale("log")
    a.set_xlim(1e-4, 300)
    a.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    a.set_xticklabels(["0.0001", "0.001", "0.01", "0.1", "1", "10", "100"])
    a.set_yticks(y)
    a.set_yticklabels([f"{FAMILY_LABEL[f]} ({int(np.median(ports[f])):,} "
                       f"port{'s' if np.median(ports[f]) > 1 else ''})" for f in order],
                      fontsize=7)
    a.set_xlabel("floor (% of asset, log scale)")
    a.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    a.set_axisbelow(True)
    for s_ in ("top", "right"):
        a.spines[s_].set_visible(False)
    a.legend(frameon=False, loc="upper center", fontsize=7, handletextpad=0.2,
             ncol=1, bbox_to_anchor=(0.47, 1.0), labelspacing=0.35)
    a.set_title("(a) floor by tool family, same packets", fontsize=8.5, loc="left", color=INK)

    def series(d):
        xs = sorted(d)
        return xs, [np.mean(d[v]) for v in xs]

    xs, ys = series(by_axis["flow active timeout"])
    b.plot(xs, ys, color=COLORS["flow"], marker=MARKERS["flow"], markersize=4.5,
           linewidth=1.6, markeredgecolor="white", markeredgewidth=0.6)
    b.annotate("DNS, flow record:\nactive timeout", (xs[-2], ys[-2]), textcoords="offset points",
               xytext=(-40, -30), fontsize=7, color=INK)
    xs2, ys2 = series(by_axis["session window"])
    b.plot(xs2, ys2, color=COLORS["session"], marker=MARKERS["session"], markersize=4.5,
           linewidth=1.6, markeredgecolor="white", markeredgewidth=0.6)
    b.annotate("DNS, session:\naggregation window", (xs2[0], ys2[0]), textcoords="offset points",
               xytext=(-30, -26), fontsize=7, color=INK)
    xs3, ys3 = series(dapt)
    b.plot(xs3, ys3, color=INK2, marker="o", markersize=3.5, linewidth=1.2,
           linestyle=(0, (3, 2)))
    b.annotate("DAPT2021, flow record: active timeout", (xs3[-1], ys3[-1]),
               textcoords="offset points", xytext=(-138, 7), fontsize=7, color=INK2)
    b.set_xscale("log")
    b.set_xticks([15, 30, 60, 120, 300, 600, 1800])
    b.set_xticklabels(["15", "30", "60", "120", "300", "600", "1800"])
    b.set_ylim(-3, 105)
    b.set_xlabel("record cadence (s)")
    b.set_ylabel("mean floor (% of asset)")
    b.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    b.set_axisbelow(True)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    b.set_title("(b) mean floor against the cadence", fontsize=8.5, loc="left", color=INK)

    fig.tight_layout(w_pad=1.6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
