"""Figures for Sec. 9 and Sec. 10 from the committed CSVs.

    python3 scripts/plot_section9.py

Reads results/fast_regime_wilson.csv and results/escape_wilson.csv; writes
paper/figs/fast_regime.png and paper/figs/escape.png.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COL = {5.0: "#1b6ca8", 15.0: "#e07b39", 60.0: "#3a9a5b"}

rows = [r for r in csv.DictReader((ROOT / "results/fast_regime_wilson.csv").open())
        if float(r["asset_bytes"]) < 5e6]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
for ax, k, title in ((axes[0], "1", "one connection"),
                     (axes[1], "8", "eight sequential connections (T1030)")):
    for ta, c in COL.items():
        sub = sorted((r for r in rows if r["n_connections"] == k
                      and float(r["active_timeout_s"]) == ta),
                     key=lambda r: float(r["T_over_D"]))
        x = [float(r["T_over_D"]) for r in sub]
        ax.plot(x, [100 * float(r["floor"]) for r in sub], "-o", color=c, ms=3.5,
                lw=1.4, label=f"floor, $T_a$={int(ta)} s")
        ax.plot(x, [100 * float(r["ebd"]) for r in sub], "--x", color=c, ms=4,
                lw=1.0, alpha=0.8, label=f"detector, $T_a$={int(ta)} s")
    ax.axvline(1.0, color="grey", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("$T/D$ (active timeout / transfer duration)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, lw=0.4)
axes[0].set_ylabel("nEBD (% of asset)")
axes[0].set_ylim(-3, 105)
axes[1].legend(fontsize=7, loc="center right", ncol=1, framealpha=0.9)
fig.tight_layout()
fig.savefig(ROOT / "paper/figs/fast_regime.png", dpi=200)

esc = sorted(csv.DictReader((ROOT / "results/escape_wilson.csv").open()),
             key=lambda r: float(r["duration_s"]))
fig, ax = plt.subplots(figsize=(6.4, 3.4))
w = 0.2
for j, (key, lab, c) in enumerate((("floor_Ta5", "floor, $T_a$=5 s", COL[5.0]),
                                   ("floor_Ta15", "floor, $T_a$=15 s", COL[15.0]),
                                   ("floor_Ta60", "floor, $T_a$=60 s", COL[60.0]),
                                   ("incremental_ebd", "incremental trigger", "#c0392b"))):
    ax.bar([i + (j - 1.5) * w for i in range(len(esc))],
           [100 * float(r[key]) for r in esc], w, color=c, label=lab)
ax.set_xticks(range(len(esc)))
ax.set_xticklabels([f"{float(r['duration_s']):.1f} s" for r in esc])
ax.set_xlabel("transfer duration $D$")
ax.set_ylabel("nEBD (% of asset)")
ax.set_ylim(0, 105)
ax.legend(fontsize=7, loc="upper center", ncol=2, framealpha=0.9)
ax.grid(axis="y", alpha=0.3, lw=0.4)
fig.tight_layout()
fig.savefig(ROOT / "paper/figs/escape.png", dpi=200)
print("wrote paper/figs/fast_regime.png and paper/figs/escape.png")
