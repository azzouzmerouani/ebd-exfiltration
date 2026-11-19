#!/usr/bin/env bash
# Revision-11 runs that need XGBoost (supersede run_round4.sh and run_round3.sh).
#
#   mkdir -p results/round5 && bash scripts/run_round5.sh 2>&1 | tee results/round5/log.txt
#
# Roughly 3-5 h single-threaded. What changed from round 4 (rev10 review):
#  * the benchmark audit found that attack flows of other stages, including
#    gateway-vantage records of the scored exfiltration connections, train as
#    negatives: every table is run with --negatives all (as published) and
#    --negatives benign (attack rows of other stages removed);
#  * variance is taken over the EXFILTRATION side of the training set
#    (--exfil-bootstrap, whole training episodes resampled), not only over the
#    benign split, and under both strict and leave-one-out (--loo);
#  * the threshold grid is the training-score quantile grid (--grid quantile),
#    which resolves the 0.35-0.40 gap, and the four-family table is calibrated
#    out of fold (--oof 5);
#  * the destination-disjoint split is read for no_ports only, because without
#    the scored peer the remaining labels say "this peer is benign".
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/round5
TR=data/realdata/train_flows.csv
GT=data/realdata/groundtruth.jsonl

echo "== 1. Refresh the published four-family table (hashes, admissible flag, neutral H1 wording) =="
uv run ebd-multimodel --train $TR --groundtruth $GT --rule wilson
shasum -a 256 results/real_multimodel_*_wilson.csv results/real_multimodel_h1_wilson.txt \
  > results/round5/multimodel_sha256.txt

echo "== 2. Four families: quantile grid, out-of-fold calibration, both negative policies =="
for neg in all benign; do
  uv run ebd-multimodel --train $TR --groundtruth $GT --rule wilson --grid quantile --oof 5 \
    --negatives $neg --suffix _wilson_q_oof5_${neg}
done

echo "== 3. EDXGB features: strict and LOO x negatives x {full,no_ports,ports_only,peer_rule} =="
for neg in all benign; do
  for proto in strict loo; do
    flag=""; [ "$proto" = loo ] && flag="--loo"
    uv run ebd-baseline --variants full,no_ports,ports_only,peer_rule --permutations 0 \
      --grid quantile --oof 5 --negatives $neg $flag \
      --out results/round5/baseline_${proto}_${neg}.csv \
      --episodes-out results/round5/baseline_${proto}_${neg}_episodes.csv
  done
done

echo "== 4. Variance: 10 exfiltration bootstraps x 10 benign splits (full, no_ports; strict and LOO) =="
for neg in all benign; do
  for b in $(seq 1 10); do
    s=$((20260814 + b))
    for proto in strict loo; do
      flag=""; [ "$proto" = loo ] && flag="--loo"
      uv run ebd-baseline --seed "$s" --exfil-bootstrap 1 --variants full,no_ports \
        --permutations 0 --grid quantile --negatives $neg $flag \
        --out "results/round5/var_${proto}_${neg}_${s}.csv" \
        --episodes-out "results/round5/var_${proto}_${neg}_${s}_episodes.csv"
    done
  done
done

echo "== 5. Destination-disjoint split (read no_ports only) =="
for neg in all benign; do
  uv run ebd-baseline --variants no_ports --permutations 0 --peer-disjoint --negatives $neg \
    --out results/round5/peer_disjoint_${neg}.csv \
    --episodes-out results/round5/peer_disjoint_${neg}_episodes.csv
done

echo "== 6. Sandbox analyses, re-run for the manifest =="
uv run ebd-timing-baseline
uv run ebd-floor-all
uv run ebd-response
uv run ebd-time-vs-bytes --protocol strict
uv run ebd-time-vs-bytes --protocol loo
uv run ebd-baseline --variants peer_rule --permutations 0 \
  --out results/real_peer_rule.csv --episodes-out results/real_peer_rule_episodes.csv
uv run python -m ebd.experiments.rev11_audit > /dev/null

python3 - <<'PY'
import csv, glob, statistics as st
def summ(pat, label):
    rows = []
    for p in sorted(glob.glob(pat)):
        rows += list(csv.DictReader(open(p)))
    for v in sorted({r["variant"] for r in rows}):
        r = [x for x in rows if x["variant"] == v]
        af = [int(x["at_floor"]) for x in r]; a0 = [int(x["at_floor_0pp"]) for x in r]
        fp = [float(x["heldout_fpb"]) for x in r]
        adm = sum(x["admissible"] == "True" for x in r)
        print(f"{label:26s} {v:10s} n={len(r):2d} at-floor(1pp) {sorted(af)} (0pp median {st.median(a0)}) "
              f"FPB {min(fp):.2%}-{max(fp):.2%} admissible {adm}/{len(r)}")
for neg in ("all", "benign"):
    for proto in ("strict", "loo"):
        summ(f"results/round5/baseline_{proto}_{neg}.csv", f"single {proto}/{neg}")
        summ(f"results/round5/var_{proto}_{neg}_*[0-9].csv", f"variance {proto}/{neg}")
    summ(f"results/round5/peer_disjoint_{neg}.csv", f"peer-disjoint {neg}")
PY
echo "done: send back results/round5/ and the refreshed results/real_multimodel_* files"
