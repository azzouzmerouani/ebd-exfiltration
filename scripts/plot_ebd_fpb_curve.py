"""EBD--FPB trade-off of the four detectors over the threshold grid (Fig. ebd_fpb).

    python3 scripts/plot_ebd_fpb_curve.py

Reads results/real_multimodel_sweep_wilson.csv (written by ebd-revision
multimodel) and writes paper/figs/ebd_fpb_curve.png.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rows = defaultdict(list)
for r in csv.DictReader((ROOT / "results/real_multimodel_sweep_wilson.csv").open()):
    rows[r["model"]].append(r)
STYLE = {"EDXGB (reproduced)": ("#1b6ca8", "o"), "XGBoost (default)": ("#e07b39", "s"),
         "Random Forest": ("#3a9a5b", "^"), "Logistic Regression": ("#8e5ea2", "D")}
FLOOR_X = 0.02  # where a held-out burden of exactly 0 is drawn on the log axis
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for m, rs in rows.items():
    rs = sorted(rs, key=lambda r: float(r["threshold"]))
    c, mk = STYLE[m]
    xs = [max(float(r["heldout_fpb"]) * 100, FLOOR_X) for r in rs]
    ys = [float(r["mean_ebd_pct"]) for r in rs]
    ax.plot(xs, ys, "-", color=c, marker=mk, ms=3.5, lw=1.1, label=m)
    for r, x, y in zip(rs, xs, ys):
        if r["is_operating_point"] == "True":
            lo = max(float(r["heldout_lo"]) * 100, FLOOR_X)
            hi = float(r["heldout_hi"]) * 100
            ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt=mk, color=c, ms=9,
                        mfc="white", mew=2, capsize=3, zorder=5)
ax.axvline(1.0, color="grey", ls="--", lw=0.8)
ax.text(1.06, 95, "budget 1%", color="grey", fontsize=8)
ax.set_xscale("log")
ax.set_xlim(0.015, 130)
ax.set_ylim(0, 105)
ax.set_xlabel("held-out FPB (% of 3,334 benign host-service keys; 0 drawn at 0.02%)")
ax.set_ylabel("mean nEBD over the 7 episodes (%)")
ax.grid(alpha=0.3, which="both", lw=0.4)
ax.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.9)
fig.tight_layout()
fig.savefig(ROOT / "paper/figs/ebd_fpb_curve.png", dpi=200)
print("wrote paper/figs/ebd_fpb_curve.png")
