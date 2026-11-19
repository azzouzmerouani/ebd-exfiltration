#!/usr/bin/env bash
# Revision-14 runs on the Mac (rev13 review: R1-1 leave-one-out calibration; R1-5/EIC-6
# logistic-regression reproducibility).
#
#   mkdir -p results/round7 && bash scripts/run_round7.sh 2>&1 | tee results/round7/log.txt
#
# Do not unpack any archive into this repository while it runs. Several hours.
#  * every leave-one-out model gets its own threshold (quantile grid, five-fold
#    out-of-fold calibration on its own training-benign keys) and its own
#    held-out burden (--loo-calibrate); before, every LOO model was scored at the
#    strict model's threshold;
#  * logistic regression is now solved to tol 1e-10 (it stopped after ~60 lbfgs
#    iterations at the default tol); the four-family table is run twice and hashed.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/round7
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
TR=data/realdata/train_flows.csv
GT=data/realdata/groundtruth.jsonl

echo "== 1. Four-family table, twice, hashed =="
for k in 1 2; do
  uv run ebd-multimodel --train $TR --groundtruth $GT --rule wilson --suffix _wilson_r7
  shasum -a 256 results/real_multimodel_*_wilson_r7.csv results/real_multimodel_h1_wilson_r7.txt \
    > results/round7/multimodel_sha256_run$k.txt
done
diff -q results/round7/multimodel_sha256_run1.txt results/round7/multimodel_sha256_run2.txt \
  && echo "four-family table: identical in two runs" \
  || echo "WARNING: four-family table differs between two runs"
for neg in all benign; do
  uv run ebd-multimodel --train $TR --groundtruth $GT --rule wilson --grid quantile --oof 5 \
    --negatives $neg --suffix _wilson_q_oof5_${neg}_r7
done

echo "== 2. Leave-one-out with each model's own operating point (single runs) =="
for neg in all benign; do
  uv run ebd-baseline --variants full,no_ports,ports_only --permutations 0 \
    --grid quantile --oof 5 --negatives $neg --loo --loo-calibrate \
    --out results/round7/loo_cal_${neg}.csv \
    --episodes-out results/round7/loo_cal_${neg}_episodes.csv
done

echo "== 3. Leave-one-out with own operating points over ten resamplings (EDXGB features) =="
for b in $(seq 1 10); do
  s=$((20260814 + b))
  uv run ebd-baseline --seed "$s" --exfil-bootstrap 1 --variants full --permutations 0 \
    --grid quantile --oof 5 --negatives all --loo --loo-calibrate \
    --out "results/round7/var_loo_cal_all_${s}.csv" \
    --episodes-out "results/round7/var_loo_cal_all_${s}_episodes.csv"
done

echo "== 4. Summary =="
python3 - <<'PY'
import csv, glob, statistics as st
for pat, label in (("results/round7/loo_cal_all.csv", "single/all"),
                   ("results/round7/loo_cal_benign.csv", "single/benign"),
                   ("results/round7/var_loo_cal_all_*[0-9].csv", "ten resamplings/all")):
    rows = []
    for p in sorted(glob.glob(pat)):
        rows += list(csv.DictReader(open(p)))
    for v in sorted({r["variant"] for r in rows}):
        r = [x for x in rows if x["variant"] == v]
        af = sorted(int(x["at_floor"]) for x in r)
        print(f"{label:22s} {v:10s} n={len(r):2d} at-floor {af} "
              f"own thr {min(float(x['loo_thr_min']) for x in r):.4g}-{max(float(x['loo_thr_max']) for x in r):.4g} "
              f"own FPB {min(float(x['loo_fpb_min']) for x in r):.2%}-{max(float(x['loo_fpb_max']) for x in r):.2%} "
              f"admissible models {sum(int(x['loo_admissible']) for x in r)}/{7*len(r)}")
PY
echo "done: send back results/round7/ and results/real_multimodel_*_r7*"
