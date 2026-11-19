#!/usr/bin/env bash
# Revision-13 runs on the Mac (rev12 review: R1-1, R1-4, R1-6, DA-3, R2-3).
#
#   mkdir -p results/round6 && bash scripts/run_round6.sh 2>&1 | tee results/round6/log.txt
#
# Do not unpack any archive into this repository while it runs.
# Roughly 4-6 h single-threaded. What changed from round 5:
#  * BLAS/OpenMP pinned to one thread, and the four-family table is run twice
#    and hashed both times, so its reproducibility is tested, not assumed (R1-6);
#  * the ten resamplings use out-of-fold calibration as well as the quantile
#    grid, as the robustness table states (R1-1);
#  * the destination-disjoint split uses the quantile grid and out-of-fold
#    calibration, for both full and no_ports, and records the training
#    positives of each group (R1-4, DA-3);
#  * the CIC-Bell-DNS-EXF-2021 captures are re-measured with retried chunks
#    removed (R2-3), if the dataset folder is where it was in round 5.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/round6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
TR=data/realdata/train_flows.csv
GT=data/realdata/groundtruth.jsonl
CIC=${CIC:-/Users/snow/Desktop/PhD/EBD/cic-dataset-2026/extracted}

echo "== 1. Four-family table, twice, hashed (default grid, as published) =="
for k in 1 2; do
  uv run ebd-multimodel --train $TR --groundtruth $GT --rule wilson
  shasum -a 256 results/real_multimodel_*_wilson.csv results/real_multimodel_h1_wilson.txt \
    > results/round6/multimodel_sha256_run$k.txt
done
if diff -q results/round6/multimodel_sha256_run1.txt results/round6/multimodel_sha256_run2.txt; then
  echo "four-family table: identical in two runs"
else
  echo "WARNING: four-family table differs between two runs"
fi
for neg in all benign; do
  uv run ebd-multimodel --train $TR --groundtruth $GT --rule wilson --grid quantile --oof 5 \
    --negatives $neg --suffix _wilson_q_oof5_${neg}
done
shasum -a 256 results/real_multimodel_*_wilson*.csv results/real_multimodel_h1_wilson*.txt \
  > results/round6/multimodel_sha256.txt

echo "== 2. Variance with out-of-fold calibration: 10 resamplings (full, no_ports; strict and LOO) =="
for neg in all benign; do
  for b in $(seq 1 10); do
    s=$((20260814 + b))
    for proto in strict loo; do
      flag=""; [ "$proto" = loo ] && flag="--loo"
      uv run ebd-baseline --seed "$s" --exfil-bootstrap 1 --variants full,no_ports \
        --permutations 0 --grid quantile --oof 5 --negatives $neg $flag \
        --out "results/round6/var_${proto}_${neg}_${s}.csv" \
        --episodes-out "results/round6/var_${proto}_${neg}_${s}_episodes.csv"
    done
  done
done

echo "== 3. Destination-disjoint split, quantile grid, out-of-fold calibration =="
for neg in all benign; do
  uv run ebd-baseline --variants full,no_ports --permutations 0 --peer-disjoint \
    --grid quantile --oof 5 --negatives $neg \
    --out results/round6/peer_disjoint_${neg}.csv \
    --episodes-out results/round6/peer_disjoint_${neg}_episodes.csv
done

echo "== 4. CIC-Bell-DNS-EXF-2021 with retried chunks removed =="
if [ -d "$CIC" ]; then
  PYTHONPATH=src uv run python scripts/cic_dns_floor.py --root "$CIC" --dedup --out results/cic_dns_dedup
else
  echo "skipped: $CIC not found (set CIC=/path/to/extracted)"
fi

echo "== 5. Summary =="
python3 - <<'PY'
import csv, glob, statistics as st
def summ(pat, label):
    rows = []
    for p in sorted(glob.glob(pat)):
        rows += list(csv.DictReader(open(p)))
    for v in sorted({(r["variant"], r["group"]) for r in rows}):
        r = [x for x in rows if (x["variant"], x["group"]) == v]
        af = [int(x["at_floor"]) for x in r]; a0 = [int(x["at_floor_0pp"]) for x in r]
        fp = [float(x["heldout_fpb"]) for x in r]
        adm = sum(x["admissible"] == "True" for x in r)
        pos = sorted({x.get("n_train_pos", "") for x in r})
        print(f"{label:26s} {v[0]:9s} {v[1]:18s} n={len(r):2d} at-floor {sorted(af)} (0pp med {st.median(a0)}) "
              f"FPB {min(fp):.2%}-{max(fp):.2%} admissible {adm}/{len(r)} train_pos {pos}")
for neg in ("all", "benign"):
    for proto in ("strict", "loo"):
        summ(f"results/round6/var_{proto}_{neg}_*[0-9].csv", f"variance {proto}/{neg}")
    summ(f"results/round6/peer_disjoint_{neg}.csv", f"peer-disjoint {neg}")
PY
echo "done: send back results/round6/, results/cic_dns_dedup/ and the refreshed results/real_multimodel_* files"
