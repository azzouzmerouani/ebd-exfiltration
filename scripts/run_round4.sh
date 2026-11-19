#!/usr/bin/env bash
# Runs that need XGBoost or the DAPT2021 PCAPs (neither available in the cloud
# sandbox). Supersedes run_round3.sh: it contains all of its steps.
#
#   bash scripts/run_round4.sh 2>&1 | tee results/round4/log.txt
#
# Roughly 2-4 h single-threaded. Needs network access to the public DAPT2021
# bucket for step 6 only.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/round4
TR=data/realdata/train_flows.csv
GT=data/realdata/groundtruth.jsonl

echo "== 1. Refresh Table multimodel (paper grid), so hashes and the admissible flag match =="
uv run ebd-multimodel --train $TR --groundtruth $GT --rule wilson
shasum -a 256 results/real_multimodel_*_wilson.csv results/real_multimodel_h1_wilson.txt \
  > results/round4/multimodel_sha256.txt

echo "== 2. Same, threshold grid extended below 0.05 (grid floor no longer binds) =="
uv run ebd-multimodel --train $TR --groundtruth $GT --rule wilson --grid extended \
  --suffix _wilson_ext

echo "== 3. Out-of-fold calibration (5 folds), paper grid and extended grid =="
for g in default extended; do
  uv run ebd-baseline --variants full,no_ports,ports_only,peer_rule --permutations 0 \
    --oof 5 --grid $g \
    --out results/round4/baseline_oof5_${g}.csv \
    --episodes-out results/round4/baseline_oof5_${g}_episodes.csv
done

echo "== 4. Variance over ten benign splits (in-sample and out-of-fold calibration) =="
for s in $(seq 20260815 20260824); do
  uv run ebd-baseline --seed "$s" --variants full,no_ports,ports_only,peer_rule \
    --permutations 0 \
    --out "results/round4/baseline_split_${s}.csv" \
    --episodes-out "results/round4/baseline_split_${s}_episodes.csv"
  uv run ebd-baseline --seed "$s" --variants full,no_ports --permutations 0 --oof 5 \
    --out "results/round4/baseline_split_${s}_oof5.csv" \
    --episodes-out "results/round4/baseline_split_${s}_oof5_episodes.csv"
done

echo "== 5. Destination-disjoint split: no training exfiltration to the scored peer =="
uv run ebd-baseline --variants full,no_ports,ports_only,peer_rule --permutations 0 \
  --peer-disjoint \
  --out results/round4/baseline_peer_disjoint.csv \
  --episodes-out results/round4/baseline_peer_disjoint_episodes.csv

echo "== 6. TCP end of the first connections of 443 #3 (needs the public PCAP) =="
uv run python scripts/check_end_reason.py || echo "PCAP step failed; continue"

echo "== 7. Sandbox analyses, re-run for the manifest =="
uv run ebd-timing-baseline
uv run ebd-floor-all
uv run ebd-response
uv run ebd-time-vs-bytes --protocol strict
uv run ebd-time-vs-bytes --protocol loo

python3 - <<'PY'
import csv, glob, statistics as st
def load(pat):
    rows = []
    for p in sorted(glob.glob(pat)):
        rows += list(csv.DictReader(open(p)))
    return rows
for label, pat in (("in-sample", "results/round4/baseline_split_*[0-9].csv"),
                   ("oof5", "results/round4/baseline_split_*_oof5.csv")):
    rows = load(pat)
    for v in ("full", "no_ports", "ports_only", "peer_rule"):
        r = [x for x in rows if x["variant"] == v]
        if not r:
            continue
        af = [int(x["at_floor"]) for x in r]
        fp = [float(x["heldout_fpb"]) for x in r]
        adm = sum(x["admissible"] == "True" for x in r)
        gb = sum(x.get("grid_bottom_binds") == "True" for x in r)
        print(f"{label:9s} {v:10s} splits {len(r)}  at-floor {sorted(af)}  median {st.median(af)}  "
              f"held-out FPB {min(fp):.2%}-{max(fp):.2%}  admissible {adm}/{len(r)}  grid floor binds {gb}")
for p in ("results/round4/baseline_peer_disjoint.csv",):
    for x in csv.DictReader(open(p)):
        print(f"peer-disjoint {x['variant']:10s} {x['group']:16s} fired {x['fired']}/{x['episodes']} "
              f"at-floor {x['at_floor']} FPB {float(x['heldout_fpb']):.2%} admissible {x['admissible']}")
PY
echo "done: results/round4/ — send this folder and the refreshed results/real_multimodel_* files back"
