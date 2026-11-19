"""Fig. front_loaded_episode: the byte curve of DAPT2021 episode 443 #3 (G = 120 s)
with the first record (the floor) and the first flagged record of the
leave-one-out model at the in-sample threshold (results/time_vs_bytes_loo.csv).

    PYTHONPATH=src python scripts/plot_front_loaded.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ebd.experiments.cadence_sweep import load_flows
from ebd.experiments.pipeline_profiles import _frac
from ebd.realdata.episodes import group_into_episodes

ROOT = Path(__file__).resolve().parents[1]
EID = "10.1.3.17->10.8.10.84:443@1624665799.794"
g = [json.loads(l) for l in (ROOT / "data/realdata/groundtruth.jsonl").open() if EID in l][0]
e = {x.episode_id: x for x in group_into_episodes(load_flows(ROOT / "data/realdata/exfil_flows.csv"))}[EID]
t_floor = min(f.start + f.duration for f in e.flows) - g["t_start"]
lat = [float(r["latency_s"]) for r in csv.DictReader((ROOT / "results/time_vs_bytes_loo.csv").open())
       if r["episode_id"] == EID][0]
ts = [k * 0.5 for k in range(0, 2 * 240)]
v = [100 * _frac(g, g["t_start"] + t) for t in ts]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(ts, v, color="black", lw=1.4, label="asset delivered, $V_e(t)/V_e$")
for x, lab, c, ls in ((t_floor, f"first record available ({t_floor:.0f} s): floor {100*_frac(g, g['t_start']+t_floor):.2f}%", "#1f77b4", "--"),
                      (lat, f"first flagged record ({lat:.0f} s): nEBD {100*_frac(g, g['t_start']+lat):.1f}%", "#d62728", "-")):
    ax.axvline(x, color=c, ls=ls, lw=1.1, label=lab)
ax.set_xlabel("time since the first packet of the episode (s)")
ax.set_ylabel("% of the asset")
ax.set_xlim(0, 240)
ax.set_ylim(0, 100)
ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.tight_layout()
out = ROOT / "paper/figs/front_loaded_episode.png"
fig.savefig(out, dpi=200)
print("wrote", out, round(t_floor, 1), lat)
