"""When does the first record a detector can score actually exist?

    uv run ebd-semantics

The floor of Proposition 1 depends on one quantity, the instant the first
record covering an episode becomes available to the detector. Sec. 4.4 and the
DAPT2021 floors take that instant as the end of the first flow to finish -- a
lower bound. This driver recomputes the floor of the seven DAPT2021 episodes
under the emission semantics deployed pipelines actually have:

* ``flow+L``  -- flow records, available L seconds after the flow ends
  (L = 0; 15 s, Cisco's default inactive timeout; 120 s, nfstream's default
  idle timeout when a flow is not expired on FIN; 300 s, a collector that
  rotates its files every five minutes, as nfcapd does by default);
* ``idle-gap S`` -- a session aggregator that closes a session after S seconds
  of silence: available at the last flow of the first session plus S;
* ``tumbling W`` -- aggregates over fixed windows of W seconds, emitted at each
  window boundary over the flows completed by then (averaged over the
  window's phase, and at its worst phase);
* ``polled P`` -- a NEDS-like consumer: a per-host-pair aggregate updated as
  each flow completes and scored every P seconds (NEDS polls every 2 min).

It also reports each episode's longest internal silence, which decides when an
idle-gap aggregator can close, and the reproduced detector's EBD under the
same latencies (from the per-record detail of ``real_edxgb``).

Floor only for the aggregators: no model is retrained, so these are bounds on
any detector with that emission semantics, read off the same byte timelines.
Writes ``results/emission_semantics.csv`` and
``results/emission_semantics_ebd.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from ..realdata.episodes import group_into_episodes
from ..realdata.session import sessionize
from .cadence_sweep import load_flows

LATENCIES = (0.0, 15.0, 120.0, 300.0, 600.0, 900.0, 1800.0)
IDLE_GAPS = (60.0, 120.0, 300.0, 600.0, 1800.0)
WINDOWS = (60.0, 300.0, 600.0, 900.0)
POLLS = (120.0, 300.0, 600.0)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def frac(g: dict, t: float) -> float:
    """Fraction of the asset delivered by absolute time ``t`` (1 s bins)."""
    idx = int(math.floor((t - g["bin_start"]) / g["bin_seconds"]))
    if idx < 0:
        return 0.0
    return min(1.0, sum(g["bins"][: idx + 1]) / max(g["pcap_bytes"], 1))


def _phase_avg(g: dict, t_first: float, period: float, steps: int = 120) -> tuple[float, float]:
    """Floor for emission at the next multiple of ``period`` after ``t_first``,
    averaged over a uniformly random phase, and at the worst phase."""
    vals = [frac(g, t_first + (period - (k + 0.5) * period / steps)) for k in range(steps)]
    return sum(vals) / len(vals), frac(g, t_first + period)


def main(argv: list[str] | None = None) -> int:
    root = _root()
    ap = argparse.ArgumentParser(description="Floors under deployed emission semantics")
    ap.add_argument("--flows", type=Path, default=root / "data/realdata/exfil_flows.csv")
    ap.add_argument("--groundtruth", type=Path, default=root / "data/realdata/groundtruth.jsonl")
    ap.add_argument("--details", nargs="*", type=Path, default=[
        root / "results/real_edxgb_detail_loo_wilson.json",
        root / "results/real_edxgb_detail_strict_wilson.json"])
    ap.add_argument("--out", type=Path, default=root / "results/emission_semantics.csv")
    ap.add_argument("--ebd-out", type=Path, default=root / "results/emission_semantics_ebd.csv")
    args = ap.parse_args(argv)

    gt = {}
    for line in args.groundtruth.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            gt[d["episode_id"]] = d
    episodes = {e.episode_id: e for e in group_into_episodes(load_flows(args.flows))}

    rows = []
    for eid, g in gt.items():
        e = episodes[eid]
        flows = sorted(e.flows, key=lambda f: f.start)
        first_end = min(f.start + f.duration for f in flows)
        run_end, silences = flows[0].start + flows[0].duration, []
        for f in flows[1:]:
            silences.append(max(0.0, f.start - run_end))
            run_end = max(run_end, f.start + f.duration)
        base = dict(episode_id=eid, duration_s=g["t_end"] - g["t_start"],
                    n_flows=len(flows), longest_silence_s=max(silences, default=0.0))
        for L in LATENCIES:
            rows.append({**base, "semantics": "flow", "param_s": L,
                         "floor": frac(g, first_end + L), "floor_worst": frac(g, first_end + L)})
        for S in IDLE_GAPS:
            first_sess_end = min(s.emission_time for s in sessionize(flows, S))
            rows.append({**base, "semantics": "idle-gap", "param_s": S,
                         "floor": frac(g, first_sess_end + S),
                         "floor_worst": frac(g, first_sess_end + S),
                         "floor_no_wait": frac(g, first_sess_end)})
        for W in WINDOWS:
            avg, worst = _phase_avg(g, first_end, W)
            rows.append({**base, "semantics": "tumbling", "param_s": W,
                         "floor": avg, "floor_worst": worst})
        for P in POLLS:
            avg, worst = _phase_avg(g, first_end, P)
            rows.append({**base, "semantics": "polled", "param_s": P,
                         "floor": avg, "floor_worst": worst})

    ebd_rows = []
    for path in args.details:
        if not path.exists():
            continue
        det = json.loads(path.read_text())
        protocol = "strict" if "strict" in path.name else "loo"
        for eid, d in det["episodes"].items():
            g = gt[eid]
            for L in LATENCIES:
                if d.get("alert_s") is None:
                    ebd = 1.0
                else:
                    ebd = frac(g, d["t_start"] + d["alert_s"] + L)
                floor = frac(g, d["t_start"] + d["first_record_s"] + L)
                ebd_rows.append(dict(protocol=protocol, episode_id=eid, latency_s=L,
                                     floor=floor, ebd=ebd, gap=max(0.0, ebd - floor),
                                     at_floor=(ebd - floor) <= 0.01))

    for path, data in ((args.out, rows), (args.ebd_out, ebd_rows)):
        if not data:
            continue
        keys = sorted({k for r in data for k in r}, key=lambda k: list(data[0]).index(k)
                      if k in data[0] else 99)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
        print(f"wrote {path}")

    # console summary
    names = {e: e.split("@")[0][-9:] for e in gt}
    print(f"\n{'semantics':<12}{'param':>7}  mean floor  worst-phase   total losses")
    for sem, params in (("flow", LATENCIES), ("idle-gap", IDLE_GAPS),
                        ("tumbling", WINDOWS), ("polled", POLLS)):
        for p in params:
            sub = [r for r in rows if r["semantics"] == sem and r["param_s"] == p]
            m = sum(r["floor"] for r in sub) / len(sub)
            mw = sum(r["floor_worst"] for r in sub) / len(sub)
            tl = sum(r["floor"] >= 0.999 for r in sub)
            print(f"{sem:<12}{int(p):>7}  {100*m:9.2f}%  {100*mw:9.2f}%   {tl}/7")
    for proto in ("loo", "strict"):
        for L in LATENCIES:
            sub = [r for r in ebd_rows if r["protocol"] == proto and r["latency_s"] == L]
            if sub:
                print(f"{proto:<7} L={int(L):>4}s  at floor {sum(r['at_floor'] for r in sub)}/7  "
                      f"mean floor {100*sum(r['floor'] for r in sub)/7:.2f}%  "
                      f"mean EBD {100*sum(r['ebd'] for r in sub)/7:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
