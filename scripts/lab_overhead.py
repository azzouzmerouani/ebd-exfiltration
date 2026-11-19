"""Direct estimate of the ingestion + rule overhead on the lab pipeline.

    python scripts/lab_overhead.py

For each first record whose observed alert came within 25 s of its derived
time (not delayed by the per-run alert cap), the overhead is the time from
release to the observed alert minus the wait from ingestion to the next
scheduled run of the ingest-time rule (whose start is at least 1 s, one index
refresh, after ingestion). Prints mean, median, range, and a 95% interval from
a bootstrap that resamples rule runs (alerts in the same run are not
independent). Reads results/lab_pipeline/{delays.csv,rule_runs.jsonl}.
"""
import csv
import json
import random
import statistics as st
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def estimates():
    runs = sorted((ts(r["sched"]), ts(r["start"])) for r in map(json.loads, (ROOT / "results/lab_pipeline/rule_runs.jsonl").open())
                  if r["rule"] == "lab-exfil-peer-ingested")
    out = []
    for r in csv.DictReader((ROOT / "results/lab_pipeline/delays.csv").open()):
        if r["found"] != "True" or not r["alert_ing_observed_minus_derived"]:
            continue
        if abs(float(r["alert_ing_observed_minus_derived"])) > 25:
            continue
        t0 = float(r["t_start"])
        ing = t0 + float(r["ingest"])
        sched = [s for s, st_ in runs if st_ >= ing + 1.0][0]
        out.append((sched, t0 + float(r["alert_ingested"]) - (t0 + float(r["release"])) - (sched - ing)))
    return out


def main():
    est = estimates()
    v = [x for _, x in est]
    by_run = defaultdict(list)
    for s, x in est:
        by_run[s].append(x)
    keys = list(by_run)
    rng = random.Random(20261101)
    boot = []
    for _ in range(10000):
        sample = [x for k in (rng.choice(keys) for _ in keys) for x in by_run[k]]
        boot.append(st.mean(sample))
    boot.sort()
    print(f"n={len(v)} alerts in {len(keys)} runs; mean {st.mean(v):.1f} s, median {st.median(v):.1f} s, "
          f"range {min(v):.1f}-{max(v):.1f} s; run-bootstrap 95% CI of the mean "
          f"{boot[250]:.1f}-{boot[9749]:.1f} s")


if __name__ == "__main__":
    main()
