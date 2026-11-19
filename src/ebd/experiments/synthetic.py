"""Pilot experiment: end-to-end EBD benchmark on synthetic episodes.

Run it::

    uv run ebd-synth                 # or: uv run python -m ebd.experiments.synthetic

It generates the synthetic dataset, trains and evaluates the six detectors,
prints the results table and the H1 verdict, and writes a CSV plus two figures
(the EBD-FPB trade-off curves and the F1-vs-EBD ranking map) under ``results/``
and ``figures/``.

Read the banner it prints. This validates the *pipeline and the metrics*, not
the paper's empirical claim: the detectors are stand-ins and the data is
generated to a known shape. The claim is settled only on lab captures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..detectors import (
    CumulativeBytes,
    CusumRate,
    EWMARate,
    VolumeThreshold,
    WindowClassifier,
)
from ..harness import BenchmarkReport, run_benchmark
from ..metrics import ebd_fpb_curve
from ..synth import DatasetConfig, make_dataset
from ..triggers import ECTSTrigger


def build_detectors() -> list:
    """The pilot detector suite.

    Three plain baselines, one change detector, one trained classifier, and the
    two ECTS triggers that share a trained base scorer and differ *only* in the
    delay-cost denomination — the byte/time pair whose separation the whole paper
    rests on (experiment D5, isolated in ``ebd.experiments.d5_control``).
    """
    return [
        VolumeThreshold(),
        EWMARate(alpha=0.3),
        CumulativeBytes(),
        CusumRate(),
        WindowClassifier(),
        ECTSTrigger(base=WindowClassifier(), cost_model="time"),  # D5 control
        ECTSTrigger(base=WindowClassifier(), cost_model="byte"),  # contribution
    ]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #

def _plot_curves(detectors, test_attack, test_benign, asset_mb, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..harness import auto_thresholds

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for det in detectors:
        thr = auto_thresholds(det, list(test_attack) + list(test_benign))
        curve = ebd_fpb_curve(det, test_attack, test_benign, thr)
        pts = sorted(curve, key=lambda p: p.fpb)
        fpb = [p.fpb for p in pts]
        nebd = [p.mean_nebd for p in pts]
        ax.plot(fpb, nebd, marker="o", ms=3, lw=1.4, label=det.name)
    ax.set_xlabel("False-positive burden (fraction of benign flows alerted)")
    ax.set_ylabel("mean normalized EBD  (fraction of asset lost)")
    ax.set_title("EBD–FPB trade-off  (lower-left is better)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_ranking(report: BenchmarkReport, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = report.ranking
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    n = len(r.detectors)
    ax.plot([1, n], [1, n], ls="--", color="0.6", lw=1, label="rankings agree")
    for d in r.detectors:
        x, y = r.f1_rank[d], r.ebd_rank[d]
        ax.scatter(x, y, s=60, zorder=3)
        ax.annotate(d, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("F1 rank  (1 = best by F1)")
    ax.set_ylabel("EBD rank  (1 = least data leaked)")
    verdict = "DIVERGE — H1 supported" if r.diverges else "agree — H1 falsified"
    ax.set_title(f"Ranking map   τ = {r.kendall_tau:+.3f}   → {verdict}")
    ax.set_xticks(range(1, n + 1))
    ax.set_yticks(range(1, n + 1))
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EBD synthetic pilot benchmark")
    parser.add_argument("--seed", type=int, default=DatasetConfig.seed)
    parser.add_argument("--max-fpb", type=float, default=0.05,
                        help="false-positive budget for the EBD operating point")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    print("=" * 74)
    print("  EBD PILOT — SYNTHETIC DATA")
    print("  Validates the pipeline and the metrics, NOT the paper's claim.")
    print("  Detectors are stand-ins; data is generated to a known shape.")
    print("  The empirical result is settled only on lab captures.")
    print("=" * 74)

    cfg = DatasetConfig(seed=args.seed)
    ds = make_dataset(cfg)
    tr_atk, tr_ben, te_atk, te_ben = ds.split()
    asset = ds.median_asset_mb
    print(
        f"\ndataset: {len(ds.attack)} attack + {len(ds.benign)} benign episodes"
        f"  |  median asset = {asset:.0f} MB  |  matrix = 36 cells"
        f"\nsplit:   train {len(tr_atk)}A/{len(tr_ben)}B   test {len(te_atk)}A/{len(te_ben)}B"
        f"  |  FPB budget = {args.max_fpb:.0%}\n"
    )

    detectors = build_detectors()
    report = run_benchmark(
        detectors, tr_atk, tr_ben, te_atk, te_ben,
        max_fpb=args.max_fpb, asset_mb=asset,
    )

    df = report.to_frame()
    print("results (sorted by EBD, best first):\n")
    print(df.to_string(index=False))
    print("\n" + report.ranking.as_table())

    # locate the byte/time ECTS pair and report the controlled contrast (D5)
    byte = next((r for r in report.results if r.name.startswith("ects-byte")), None)
    time = next((r for r in report.results if r.name.startswith("ects-time")), None)
    if byte and time:
        delta = time.mean_ebd - byte.mean_ebd
        print(
            f"\nD5 control  (byte vs time cost, same base scorer & FPB):"
            f"\n  byte-cost mean EBD = {byte.mean_ebd:7.2f} MB"
            f"\n  time-cost mean EBD = {time.mean_ebd:7.2f} MB"
            f"\n  byte saves {delta:+.2f} MB "
            f"({'contribution holds' if delta > 0 else 'contribution NOT shown'} on this run)"
        )

    root = _project_root()
    (root / "results").mkdir(exist_ok=True)
    csv_path = root / "results" / "synthetic_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path.relative_to(root)}")

    if not args.no_figures:
        (root / "figures").mkdir(exist_ok=True)
        c = root / "figures" / "ebd_fpb_curves.png"
        m = root / "figures" / "ranking_map.png"
        _plot_curves(detectors, te_atk, te_ben, asset, c)
        _plot_ranking(report, m)
        print(f"wrote {c.relative_to(root)}")
        print(f"wrote {m.relative_to(root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
