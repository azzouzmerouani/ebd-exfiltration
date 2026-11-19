"""Runnable experiment D5: does byte-cost beat time-cost?

    uv run ebd-d5      # or: uv run python -m ebd.experiments.d5_control

Builds one **shared, trained** base scorer (:class:`WindowClassifier`), wraps it
in the two ECTS triggers that differ only in delay-cost denomination, and runs
the controlled comparison — matched false-positive budget, paired per-episode
EBD on the same test attacks, a per-rate breakdown and a paired bootstrap.

Read the banner. This is a synthetic *instrument test*: it shows whether the
harness can detect a byte-vs-time effect when the base scorer discriminates, and
where along the rate axis the effect lives. It is not evidence about real traffic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..d5 import d5_compare
from ..detectors import WindowClassifier
from ..synth import DatasetConfig, make_dataset
from ..triggers import ECTSTrigger


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _plot_per_rate(result, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cells = result.per_rate
    rates = [c.rate for c in cells]
    byte = [c.ebd_mb_byte for c in cells]
    time = [c.ebd_mb_time for c in cells]
    x = np.arange(len(rates))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - w / 2, time, w, label="time-cost (D5 control)", color="#c44")
    ax.bar(x + w / 2, byte, w, label="byte-cost (contribution)", color="#3a7")
    ax.set_xticks(x)
    ax.set_xticklabels(rates)
    ax.set_xlabel("exfiltration rate (adversary-matrix axis)")
    ax.set_ylabel("mean EBD (MB) — lower is better")
    verdict = "byte beats time" if result.supported else "byte does not beat time"
    ax.set_title(f"D5: EBD by rate   (base={result.base_name})   → {verdict}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D5 byte-vs-time control")
    parser.add_argument("--seed", type=int, default=DatasetConfig.seed)
    parser.add_argument("--max-fpb", type=float, default=0.05)
    parser.add_argument("--variant", choices=["byte", "byte-floored"], default="byte",
                        help="'byte' = naive myopic rule; 'byte-floored' = decoupled fix")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    print("=" * 74)
    print("  EXPERIMENT D5 — byte-vs-time control (SYNTHETIC INSTRUMENT TEST)")
    print("  Shows whether the harness detects a byte-vs-time effect when the")
    print("  base scorer discriminates. Not evidence about real traffic.")
    print("=" * 74)

    cfg = DatasetConfig(seed=args.seed)
    ds = make_dataset(cfg)
    tr_a, tr_b, te_a, te_b = ds.split()
    print(
        f"\ndataset: {len(ds.attack)} attack + {len(ds.benign)} benign"
        f"  |  median asset {ds.median_asset_mb:.0f} MB  |  36 cells"
        f"\nsplit:   train {len(tr_a)}A/{len(tr_b)}B   test {len(te_a)}A/{len(te_b)}B\n"
    )

    # one shared, trained base -> identical p_t for both triggers
    base = WindowClassifier().fit(tr_b, tr_a)
    byte = ECTSTrigger(base=base, cost_model=args.variant)
    time = ECTSTrigger(base=base, cost_model="time")
    print(f"variant under test: {args.variant}   (vs time-cost control)\n")

    result = d5_compare(byte, time, tr_a, tr_b, te_a, te_b,
                        target_fpb=args.max_fpb, seed=args.seed)
    print(result.report())

    root = _project_root()
    (root / "results").mkdir(exist_ok=True)
    import csv

    csv_path = root / "results" / f"d5_per_rate_{args.variant}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rate", "n", "ebd_mb_byte", "ebd_mb_time", "saving_mb",
                    "nebd_byte", "nebd_time", "det_byte", "det_time"])
        for c in result.per_rate:
            w.writerow([c.rate, c.n, f"{c.ebd_mb_byte:.3f}", f"{c.ebd_mb_time:.3f}",
                        f"{c.saving_mb:.3f}", f"{c.nebd_byte:.4f}", f"{c.nebd_time:.4f}",
                        f"{c.det_byte:.4f}", f"{c.det_time:.4f}"])
    print(f"\nwrote {csv_path.relative_to(root)}")

    if not args.no_figures:
        (root / "figures").mkdir(exist_ok=True)
        fig_path = root / "figures" / f"d5_per_rate_{args.variant}.png"
        _plot_per_rate(result, fig_path)
        print(f"wrote {fig_path.relative_to(root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
