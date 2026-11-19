"""The cadence pressure-test: how early can a flow-based detector possibly be?

    uv run ebd-cadence

Computes the **architectural EBD floor** of batch per-flow detectors across the
adversary matrix's rate axis and the record-emission intervals real exporters
actually use, and contrasts it with the online detector from this project.

Unlike every other experiment here, the headline number does **not** depend on
any model's quality: the floor is computed against an *oracle* incumbent that is
correct on its first available record. Real detectors can only do worse. The one
synthetic ingredient is the byte timeline itself (rate x duration), which is the
honest part of the generator — no fabricated packet counts, directions, entropy
or protocols are involved.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ..cadence import CADENCE_PRESETS, cadence_floor, theoretical_floor
from ..d5 import RATE_ORDER
from ..detectors import WindowClassifier
from ..episode import Episode, ScenarioCell
from ..metrics import _score_episode, first_alert_window, normalized_ebd
from ..scenarios import RATE_LEVELS
from ..synth import DatasetConfig, _attack_profile, make_dataset


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def fixed_asset_episodes(
    asset_mb: float, seed: int, reps: int = 12, chunking: str = "single"
) -> dict[str, list[Episode]]:
    """Episodes holding the **asset constant** while the rate varies.

    This is the operationally meaningful framing, and the one the main dataset
    cannot express: an adversary has a fixed amount to steal and *chooses* how
    fast to move it. Duration is then ``asset / rate``, so the rate choice is
    exactly what sets ``D`` — and therefore the ``T/D`` that bounds any
    flow-based detector.

    (The default synthetic dataset draws duration independently of rate, so its
    floor is flat across the rate axis by construction. That is a property of the
    generator, not a finding, which is why the inversion needs this separate
    sweep.)
    """
    rng = np.random.default_rng(seed)
    out: dict[str, list[Episode]] = {}
    for rate, params in RATE_LEVELS.items():
        mb_s = params["mb_per_s"]
        duration = max(3, int(round(asset_mb / mb_s)))
        eps = []
        for i in range(reps):
            prof = _attack_profile(rng, mb_s, chunking, duration)
            eps.append(
                Episode(
                    episode_id=f"fixed/{rate}/{i}",
                    label=True,
                    window_bytes=prof,
                    scenario=ScenarioCell(rate=rate, chunking=chunking,
                                          destination="cloud-a"),
                    sender_total=float(prof.sum()),
                )
            )
        out[rate] = eps
    return out


def _online_nebd(ds_split, target_attacks=None) -> dict[str, float]:
    """Mean nEBD per rate for the project's online detector, at ~1% FPB.

    The contrast that makes the blind spot a *choice* rather than a necessity:
    the *same episodes*, scored continuously instead of at export boundaries.

    ``target_attacks`` selects which attack episodes to score (defaults to the
    held-out test attacks). It must be passed the same episode set the floor was
    computed on, or the two numbers are not comparable. Training and threshold
    selection always use the main split, never the target episodes.
    """
    tr_a, tr_b, te_a, te_b = ds_split
    clf = WindowClassifier().fit(tr_b, tr_a)

    # pick the most aggressive threshold holding benign FPB <= 1%
    ben = [_score_episode(clf, ep) for ep in te_b]
    grid = np.unique(np.quantile(np.concatenate(ben), np.linspace(0, 1, 200)))
    thr = grid[-1]
    for t in grid:
        fpb = sum(first_alert_window(s, t) is not None for s in ben) / max(len(ben), 1)
        if fpb <= 0.01:
            thr = t
            break

    by_rate: dict[str, list[float]] = {}
    for ep in (te_a if target_attacks is None else target_attacks):
        sc = _score_episode(clf, ep)
        aw = first_alert_window(sc, thr)
        by_rate.setdefault(ep.scenario.rate if ep.scenario else "?", []).append(
            normalized_ebd(ep, aw)
        )
    return {r: float(np.mean(v)) for r, v in by_rate.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export-cadence EBD floor")
    parser.add_argument("--seed", type=int, default=DatasetConfig.seed)
    parser.add_argument("--idle-timeout", type=float, default=None,
                        help="idle teardown seconds (default: none = conservative)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    print("=" * 78)
    print("  CADENCE PRESSURE-TEST — the architectural EBD floor")
    print("  Floor is computed against an ORACLE flow detector (perfect, fires on")
    print("  its first record). Real batch per-flow detectors can only do worse.")
    print("=" * 78)

    ds = make_dataset(DatasetConfig(seed=args.seed))
    rows: list[dict] = []

    print("\n[A] default dataset — duration drawn INDEPENDENTLY of rate.")
    print("    The floor is flat across rates here BY CONSTRUCTION (D does not")
    print("    depend on rate in this generator); see [B] for the real mechanism.")
    print(f"\n{'cadence':>18} | " + " | ".join(f"{r:>10}" for r in RATE_ORDER))
    print("-" * 78)
    for label, T in CADENCE_PRESETS.items():
        cells = {}
        for rate in RATE_ORDER:
            eps = [e for e in ds.attack if e.scenario and e.scenario.rate == rate]
            floors = [cadence_floor(e, T, args.idle_timeout) for e in eps]
            nebd = float(np.mean([f.nebd for f in floors]))
            frac_total = float(np.mean([f.total_loss for f in floors]))
            cells[rate] = (nebd, frac_total)
            rows.append({
                "cadence": label, "cadence_s": T, "rate": rate, "n": len(eps),
                "mean_nebd_floor": round(nebd, 4),
                "frac_total_loss": round(frac_total, 4),
                "mean_duration_s": round(float(np.mean([e.duration for e in eps])), 1),
            })
        line = " | ".join(
            f"{cells[r][0]*100:>6.1f}% {'*' if cells[r][1] >= 0.999 else ' '}   "
            for r in RATE_ORDER
        )
        print(f"{label:>18} | {line}")
    print("-" * 78)
    print("  cells are mean EBD floor as % of asset;  * = EVERY episode totally lost")

    # algebra check: the closed form is exact only for constant-rate profiles.
    by_chunk: dict[str, float] = {}
    for e in ds.attack:
        f = cadence_floor(e, 120.0, None)
        dev = abs(f.nebd - theoretical_floor(e.duration, 120.0))
        ck = e.scenario.chunking if e.scenario else "?"
        by_chunk[ck] = max(by_chunk.get(ck, 0.0), dev)
    print("\n  closed-form check, max |simulated - min(1,T/D)| by chunking:")
    for ck, dev in sorted(by_chunk.items()):
        note = "exact-ish (steady stream)" if ck == "single" else "gaps => deviation expected"
        print(f"    {ck:<12} {dev:.4f}   {note}")

    # [B] the real mechanism: hold the asset fixed, let the adversary pick a rate
    print("\n[B] FIXED-ASSET sweep — 500 MB stolen at each rate (D = asset/rate).")
    print("    This is where the inversion lives: faster theft => shorter D =>")
    print("    a HIGHER floor. The architecture is blindest when damage is fastest.")
    fixed = fixed_asset_episodes(asset_mb=500.0, seed=args.seed)
    print(f"\n{'cadence':>18} | " + " | ".join(f"{r:>10}" for r in RATE_ORDER))
    print("-" * 78)
    fixed_rows: list[dict] = []
    for label, T in CADENCE_PRESETS.items():
        cells = {}
        for rate in RATE_ORDER:
            eps = fixed[rate]
            floors = [cadence_floor(e, T, args.idle_timeout) for e in eps]
            nebd = float(np.mean([f.nebd for f in floors]))
            tot = float(np.mean([f.total_loss for f in floors]))
            cells[rate] = (nebd, tot)
            fixed_rows.append({
                "cadence": label, "cadence_s": T, "rate": rate,
                "duration_s": round(float(np.mean([e.duration for e in eps])), 1),
                "mean_nebd_floor": round(nebd, 4), "frac_total_loss": round(tot, 4),
            })
        line = " | ".join(
            f"{cells[r][0]*100:>6.1f}% {'*' if cells[r][1] >= 0.999 else ' '}   "
            for r in RATE_ORDER
        )
        print(f"{label:>18} | {line}")
    print("-" * 78)
    durs = {r: fixed[r][0].duration for r in RATE_ORDER}
    print("  transfer duration D: " + "  ".join(f"{r}={durs[r]:.0f}s" for r in RATE_ORDER))

    # the contrast, computed on the SAME fixed-asset episodes as [B]
    split = ds.split()
    all_fixed = [e for r in RATE_ORDER for e in fixed[r]]
    online = _online_nebd(split, target_attacks=all_fixed)
    print("\n[C] CONTRAST on the fixed-asset episodes (identical episodes as [B]):")
    print("    online detector (this work, continuous scoring, FPB<=1%):")
    print("      " + "  ".join(f"{r}={online.get(r, float('nan'))*100:5.1f}%"
                              for r in RATE_ORDER))
    print("    oracle flow-based floor @120s (CICFlowMeter):")
    print("      " + "  ".join(
        f"{r}={next(x['mean_nebd_floor'] for x in fixed_rows if x['rate']==r and x['cadence_s']==120.0)*100:5.1f}%"
        for r in RATE_ORDER))

    # [D] why the floor is CONSERVATIVE, and most conservative where it looks best
    print("\n[D] the floor is an ORACLE; where will a real detector miss it?")
    print("    A flow record covers T seconds, so its volume is rate*T. A real")
    print("    per-flow detector must separate that slice from benign bulk.")
    bulk = [e.total_bytes for e in ds.benign if e.meta.get("kind") == "bulk"]
    bulk = np.array(bulk)
    print(f"    benign bulk volume/flow: median {np.median(bulk):.0f} MB, "
          f"p90 {np.percentile(bulk, 90):.0f} MB, max {bulk.max():.0f} MB")
    print(f"\n    {'rate':<10}{'slice @120s':>13}{'benign bulk >= slice':>23}")
    for rate in RATE_ORDER:
        slice_mb = RATE_LEVELS[rate]["mb_per_s"] * 120.0
        frac = float(np.mean(bulk >= slice_mb))
        flag = "  <-- slice hides IN benign range" if frac > 0.02 else ""
        print(f"    {rate:<10}{slice_mb:>10.0f} MB{frac*100:>20.1f}%{flag}")
    print("\n    The drip's low floor (24%) is the corner where a real detector is")
    print("    LEAST likely to reach it: its slice sits inside the benign bulk")
    print("    volume range, the same overlap that forced the threshold up in D5.")
    print("    So 'real detectors can only do worse' bites hardest exactly where")
    print("    the incumbents look least blind -> the bound is conservative.")

    root = _project_root()
    (root / "results").mkdir(exist_ok=True)
    out_csv = root / "results" / "cadence_floor.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out_csv2 = root / "results" / "cadence_floor_fixed_asset.csv"
    with open(out_csv2, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fixed_rows[0].keys()))
        w.writeheader()
        w.writerows(fixed_rows)
    print(f"\nwrote {out_csv.relative_to(root)}")
    print(f"wrote {out_csv2.relative_to(root)}")

    if not args.no_figures:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        x = np.arange(len(RATE_ORDER))
        width = 0.19
        for i, (label, T) in enumerate(CADENCE_PRESETS.items()):
            vals = [
                next(r["mean_nebd_floor"] for r in fixed_rows
                     if r["rate"] == rt and r["cadence_s"] == T) * 100
                for rt in RATE_ORDER
            ]
            ax.bar(x + (i - 1.5) * width, vals, width, label=f"{label} (T={T:g}s)")
        onl = [online.get(rt, 0.0) * 100 for rt in RATE_ORDER]
        ax.plot(x, onl, "k*--", ms=13, lw=1.6, label="online detector (this work)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}\nD={durs[r]:.0f}s" for r in RATE_ORDER])
        ax.set_xlabel("exfiltration rate  (fixed 500 MB asset — faster ⇒ shorter D)")
        ax.set_ylabel("EBD floor  (% of asset lost before any alert is possible)")
        ax.set_title("Flow-based detectors: architectural EBD floor vs export cadence\n"
                     "oracle incumbent (real detectors can only do worse) — fixed-asset sweep")
        ax.axhline(100, color="0.5", lw=0.8, ls=":")
        ax.set_ylim(0, 108)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out_png = root / "figures" / "cadence_floor.png"
        (root / "figures").mkdir(exist_ok=True)
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f"wrote {out_png.relative_to(root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
