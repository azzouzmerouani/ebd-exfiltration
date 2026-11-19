"""Composed floors against the rule period, with the delays measured in the lab.

    python -m ebd.experiments.pipeline_sweep

Same composition as ``pipeline_profiles`` (release after the flow end, an
optional collector file, ingestion and rule overhead, a scheduled rule, averaged
over the phase of every clock-driven stage), on DAPT2021 and on the second DNS
corpus (CIC-Bell-DNS-EXF-2021), for:

* release (lab-measured medians, lab/README.md): ``netflow`` (softflowd
  configured like Cisco Flexible NetFlow), ``zeek`` (Zeek 9.0 defaults),
  ``softflowd_default`` (softflowd as packaged), and its two parts:
  ``softflowd_fin_wait`` (the measured 315 s wait after a FIN, other flows as
  ``netflow``) and ``softflowd_idle`` (the configured 3600 s idle timeout for
  flows without FIN, FIN-closed flows as ``netflow``);
* rule period 0 (a stream processor reading each record ``STREAM`` s after its
  release), 10, 20, 30, 60, 300 and 900 s (Kibana recommends at least 60 s but
  does not enforce it by default);
* release per flow: the first record is the earliest over all flows of
  end + release, with the release chosen from each flow's own FIN/RST counts
  (FIN-closed and RST-closed flows as measured or configured; RST under
  softflowd defaults: its 120 s ``tcp.rst`` timeout plus the measured 15 s
  expiry lag, configured, not measured);
* overhead from release to alert, excluding the wait for the next scheduled
  run: ``OVH_DIRECT`` (6.4 s, mean over the 52 observed alerts that matched
  their derivation; median 4.5 s, range 1.5-16.6 s) and ``OVH_HIGH`` (25 s,
  the earlier indirect estimate: mean wait minus mean phase);
* collector: records forwarded as they arrive (the lab's alert path) or kept
  in 300 s nfdump files (``window`` = 300, NetFlow only).

Writes ``results/pipeline_sweep.csv`` (one row per corpus, profile, period,
overhead and collector; mean and median over the units). No model is involved.
"""
from __future__ import annotations

import csv
import statistics as st
import tempfile
from pathlib import Path

from .pipeline_profiles import main as _compose

STREAM = 2.0       # ingestion 1 s + index refresh 1 s, measured
OVH_DIRECT = 6.4   # release -> alert minus the wait for the next scheduled run, observed
OVH_HIGH = 25.0    # indirect estimate used in rev15-rev17
PERIODS = (0.0, 10.0, 20.0, 30.0, 60.0, 300.0, 900.0)

RELEASE = {
    "netflow": dict(fin=1.6, rst=1.6, nofin=15.7, udp=15.7, maxlife=1800.0),
    "zeek": dict(fin=7.8, rst=7.8, nofin=300.0, udp=64.9, maxlife=float("inf")),
    "softflowd_default": dict(fin=314.7, rst=134.7, nofin=3600.0, udp=330.2, maxlife=604800.0),
    "softflowd_fin_wait": dict(fin=314.7, rst=134.7, nofin=15.7, udp=330.2, maxlife=604800.0),
    "softflowd_idle": dict(fin=1.6, rst=1.6, nofin=3600.0, udp=15.7, maxlife=604800.0),
}


def configs():
    for rel, r in RELEASE.items():
        for period in PERIODS:
            windows = (0.0, 300.0) if rel == "netflow" and period else (0.0,)
            for window in windows:
                for ovh in ((STREAM,) if period == 0 else (OVH_DIRECT, OVH_HIGH)):
                    name = f"{rel}|p{int(period)}|w{int(window)}|o{ovh:g}"
                    yield name, dict(r, window=window, fixed=ovh, period=period, anchor="flow_end",
                                     dns_only=False, per_flow=True)


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    profs = dict(configs())
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sweep_units.csv"
        _compose(profs, str(out))
        units = list(csv.DictReader(out.open()))
    rows = []
    for name in profs:
        rel, p, w, o = name.split("|")
        for corpus in ("dapt2021", "cic"):
            v = [float(u["floor_mean_phase"]) for u in units if u["profile"] == name and u["corpus"] == corpus]
            rows.append(dict(corpus=corpus, release=rel, period_s=int(p[1:]), window_s=int(w[1:]),
                             overhead_s=float(o[1:]), n=len(v), mean=round(100 * st.mean(v), 4),
                             median=round(100 * st.median(v), 4),
                             total_loss=sum(x >= 0.999 for x in v)))
    dst = root / "results/pipeline_sweep.csv"
    with dst.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {dst}")
    for r in rows:
        if r["window_s"] == 0 and r["overhead_s"] in (STREAM, OVH_DIRECT):
            print(f"{r['corpus']:<9} {r['release']:<20} period {r['period_s']:>4} "
                  f"mean {r['mean']:6.2f}% median {r['median']:6.2f}% total {r['total_loss']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
