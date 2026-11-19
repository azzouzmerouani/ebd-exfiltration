#!/usr/bin/env bash
# Round-3 runs that need XGBoost (not available in the cloud sandbox).
#
#   bash scripts/run_round3.sh            # roughly 1-2 h single-threaded
#
# 1. Out-of-fold calibration (R1-M2): the threshold is chosen on 5-fold
#    out-of-fold scores of the training benign keys instead of in-sample ones.
# 2. Variance over the benign split (R1-M6, DA-3). EDXGB's hyperparameters use
#    no row or column subsampling, so XGBoost's seed does not change the model;
#    the variance that matters is over which benign keys are held out. Ten
#    split seeds, full features and no ports.
# 3. Timing-only baselines and the random-score null (no XGBoost; already run
#    in the sandbox, re-run here for the manifest).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/round3

uv run ebd-baseline --variants full,no_ports --permutations 0 --oof 5 \
  --out results/round3/real_baseline_oof5.csv \
  --episodes-out results/round3/real_baseline_oof5_episodes.csv

for s in $(seq 20260815 20260824); do
  uv run ebd-baseline --seed "$s" --variants full,no_ports --permutations 0 \
    --out "results/round3/real_baseline_split_${s}.csv" \
    --episodes-out "results/round3/real_baseline_split_${s}_episodes.csv"
done

uv run ebd-timing-baseline

python3 - <<'PY'
import csv, glob, statistics as st
rows = []
for p in sorted(glob.glob("results/round3/real_baseline_split_*[0-9].csv")):
    rows += list(csv.DictReader(open(p)))
for v in ("full", "no_ports"):
    r = [x for x in rows if x["variant"] == v]
    af = [int(x["at_floor"]) for x in r]
    fp = [float(x["heldout_fpb"]) for x in r]
    adm = sum(x["admissible"] == "True" for x in r)
    print(f"{v:9s} splits {len(r)}  at-floor {sorted(af)}  median {st.median(af)}  "
          f"held-out FPB {min(fp):.2%}-{max(fp):.2%}  admissible {adm}/{len(r)}")
PY
echo "done: results/round3/"
