"""Seed-stability of the H1 ranking-divergence result.

    uv run ebd-stability --seeds 20

The headline H1 number — Kendall's tau between the F1 ranking and the EBD
ranking — is load-bearing for the paper *whatever* happens with the byte-cost
trigger. A single-seed tau is a point estimate; before it goes in a figure we
need its spread across regenerated datasets. This runs the full benchmark on
many seeds and reports the tau distribution and how often it clears the
pre-registered divergence bar (tau < 0.5).

Synthetic, like everything in the pilot: it characterises the *estimator's*
stability, not a fact about real traffic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..detectors import CumulativeBytes, CusumRate, EWMARate, VolumeThreshold
from ..harness import run_benchmark
from ..synth import DatasetConfig, make_dataset
from ..triggers import ECTSTrigger
from .synthetic import build_detectors


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_clustered_suite() -> list:
    """The naive-only suite (no strong trained detector).

    This is the regime H1 is *about*: several detectors with comparable F1, where
    EBD would have to be the thing that separates them. Contrast with the ``full``
    suite, whose trained classifier dominates on F1 and so trivially aligns the
    two rankings.
    """
    return [
        VolumeThreshold(), EWMARate(0.3), CumulativeBytes(), CusumRate(),
        ECTSTrigger(base=EWMARate(0.3), cost_model="time"),
        ECTSTrigger(base=EWMARate(0.3), cost_model="byte"),
    ]


def run_seed(seed: int, max_fpb: float, suite: str) -> tuple[float, bool]:
    cfg = DatasetConfig(seed=seed)
    ds = make_dataset(cfg)
    tr_a, tr_b, te_a, te_b = ds.split()
    detectors = build_clustered_suite() if suite == "clustered" else build_detectors()
    report = run_benchmark(detectors, tr_a, tr_b, te_a, te_b,
                           max_fpb=max_fpb, asset_mb=ds.median_asset_mb)
    return report.ranking.kendall_tau, report.ranking.diverges


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed-stability of H1 (Kendall tau)")
    parser.add_argument("--seeds", type=int, default=20, help="number of seeds")
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--max-fpb", type=float, default=0.05)
    parser.add_argument("--suite", choices=["full", "clustered"], default="full",
                        help="'full' includes the trained classifier; "
                             "'clustered' is the naive-only regime H1 is about")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    print("=" * 70)
    print("  H1 SEED-STABILITY — Kendall tau (F1 ranking vs EBD ranking)")
    print("  Synthetic: characterises estimator spread, not real traffic.")
    print("=" * 70)

    print(f"  suite under test: {args.suite}")
    seeds = [args.base_seed + i for i in range(args.seeds)]
    taus, diverge = [], []
    for i, s in enumerate(seeds, 1):
        t, d = run_seed(s, args.max_fpb, args.suite)
        taus.append(t)
        diverge.append(d)
        print(f"  seed {s:>5}  ({i:>2}/{len(seeds)})   tau = {t:+.3f}   "
              f"{'diverge' if d else 'agree'}")

    taus = np.array(taus)
    frac = float(np.mean(diverge))
    print("\nsummary")
    print("-" * 40)
    print(f"  seeds            : {len(seeds)}")
    print(f"  tau  mean±sd     : {taus.mean():+.3f} ± {taus.std():.3f}")
    print(f"  tau  min / max   : {taus.min():+.3f} / {taus.max():+.3f}")
    print(f"  tau  median      : {np.median(taus):+.3f}")
    print(f"  P(tau < 0.5)     : {frac:.0%}  (H1 divergence bar)")
    verdict = "STABLE divergence" if frac >= 0.9 else (
        "MOSTLY diverges" if frac >= 0.6 else "UNSTABLE — reconsider")
    print(f"  verdict          : {verdict}")

    root = _project_root()
    (root / "results").mkdir(exist_ok=True)
    import csv

    csv_path = root / "results" / f"h1_stability_{args.suite}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "kendall_tau", "diverges"])
        for s, t, d in zip(seeds, taus, diverge):
            w.writerow([s, f"{t:.4f}", int(d)])
    print(f"\nwrote {csv_path.relative_to(root)}")

    if not args.no_figures:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        ax.hist(taus, bins=min(12, len(seeds)), color="#47a", edgecolor="white")
        ax.axvline(0.5, color="#c33", ls="--", lw=1.5, label="H1 bar (tau=0.5)")
        ax.axvline(taus.mean(), color="k", ls="-", lw=1.2,
                   label=f"mean {taus.mean():+.3f}")
        ax.set_xlabel("Kendall tau (F1 ranking vs EBD ranking)")
        ax.set_ylabel("seeds")
        ax.set_title(f"H1 stability over {len(seeds)} seeds — P(tau<0.5) = {frac:.0%}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = root / "figures" / f"h1_stability_{args.suite}.png"
        (root / "figures").mkdir(exist_ok=True)
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out.relative_to(root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
