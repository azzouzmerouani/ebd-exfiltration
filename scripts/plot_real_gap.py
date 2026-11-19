"""Fig. 1: EDXGB against its floor on the seven DAPT2021 episodes, log scale.

    python3 scripts/plot_real_gap.py

Reads results/real_edxgb_gap_{loo,strict}_wilson.csv; writes
paper/figs/real_edxgb_gap.png.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"1625264949": "443 #1", "1625199408": "3306 #1", "1625185759": "3306 #2",
         "1624318576": "443 #2", "1624665799": "443 #3", "1624784014": "443 #4",
         "1624863977": "443 #5"}
ORDER = list(NAMES)


def load(proto):
    out = {}
    for r in csv.DictReader((ROOT / f"results/real_edxgb_gap_{proto}_wilson.csv").open()):
        k = r["episode_id"].split("@")[1].split(".")[0]
        out[k] = (100 * float(r["floor"]), 100 * float(r["actual"]))
    return out


loo, strict = load("loo"), load("strict")
x = np.arange(len(ORDER))
fig, ax = plt.subplots(figsize=(6.6, 3.6))
w = 0.26
floor = [loo[k][0] for k in ORDER]
ax.bar(x - w, floor, w, color="#444444", label="floor (any record consumer)")
for off, data, lab, col in ((0, loo, "EDXGB, leave-one-out", "#1b6ca8"),
                            (w, strict, "EDXGB, strict", "#e07b39")):
    vals = [data[k][1] for k in ORDER]
    ax.bar(x + off, vals, w, color=col, label=lab)
ax.set_yscale("log")
ax.set_ylim(0.02, 150)
ax.set_xticks(x)
ax.set_xticklabels([NAMES[k] for k in ORDER])
ax.set_ylabel("nEBD (% of asset, log scale)")
ax.axhline(100, color="grey", lw=0.6, ls=":")
ax.legend(fontsize=8, loc="upper left", frameon=True, framealpha=0.9)
ax.grid(axis="y", alpha=0.3, which="both", lw=0.4)
fig.tight_layout()
fig.savefig(ROOT / "paper/figs/real_edxgb_gap.png", dpi=200)
print("wrote paper/figs/real_edxgb_gap.png")
