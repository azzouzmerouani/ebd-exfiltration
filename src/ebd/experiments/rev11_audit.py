"""Revision-11 audit of the DAPT2021 detector benchmark (no XGBoost needed).

    python -m ebd.experiments.rev11_audit

Writes results/rev11_audit.json and prints a summary:
* benign flows to the two exfiltration peers (why a lookup rule works);
* the first record of each reconstructed episode (bytes, packets, payload by then);
* other-stage attack flows that train as negatives, and how many of them are
  feature-identical copies of scored exfiltration flows (e.g. gateway vantage);
* exact and relative at-floor counts from the committed per-episode files;
* the record-count cost of shorter active timeouts on the benign records;
* the cadence-reporting count by stratum of the coded sample.
"""
from __future__ import annotations

import collections
import csv
import json
import math
from pathlib import Path

from ebd.realdata.cadence_probe import effective_T
from ebd.realdata.episodes import FlowRecord, group_into_episodes

ROOT = Path(__file__).resolve().parents[3]
FEAT = ("bidirectional_duration_ms", "src2dst_bytes", "dst2src_bytes",
        "src2dst_packets", "dst2src_packets")
NAMES = {"1625264949.712": "443 #1", "1625199408.918": "3306 #1", "1625185759.453": "3306 #2",
         "1624318576.594": "443 #2", "1624665799.794": "443 #3", "1624784014.250": "443 #4",
         "1624863977.550": "443 #5"}


def name(eid):
    return NAMES[eid.split("@")[1]]


def main() -> int:
    csv.field_size_limit(10_000_000)
    rows = list(csv.DictReader(open(ROOT / "data/realdata/train_flows.csv")))
    gt = {json.loads(l)["episode_id"]: json.loads(l)
          for l in open(ROOT / "data/realdata/groundtruth.jsonl") if l.strip()}
    out = {}

    benign = [r for r in rows if r.get("Stage", "").strip() in ("Benign", "Normal", "")]
    peers = {("10.8.10.84", "443"), ("10.1.5.21", "3306")}
    out["benign_flows_to_exfil_peers"] = {f"{a}:{p}": sum(1 for r in benign if (r["dst_ip"], r["dst_port"]) == (a, p)) for a, p in peers}
    out["benign_flows_dst_port_3306"] = sum(1 for r in benign if r["dst_port"] == "3306")
    out["benign_flows_dst_ip_peers"] = {a: sum(1 for r in benign if r["dst_ip"] == a) for a, _ in peers}

    exf = [r for r in rows if r.get("Stage", "").strip() == "Data Exfiltration"]
    frecs, owner = [], {}
    for r in exf:
        f = FlowRecord(r["src_ip"], int(r["src_port"]), r["dst_ip"], int(r["dst_port"]), int(r["protocol"]),
                       int(r["bidirectional_first_seen_ms"]) / 1e3, int(r["bidirectional_duration_ms"]) / 1e3,
                       int(r["src2dst_bytes"]), int(r["dst2src_bytes"]))
        frecs.append(f)
        owner[id(f)] = r
    eps = {e.episode_id: e for e in group_into_episodes(frecs) if e.episode_id in gt}
    held_ids = {id(owner[id(f)]) for e in eps.values() for f in e.flows}

    first = {}
    for eid, e in eps.items():
        f = min(e.flows, key=lambda x: x.end)
        r = owner[id(f)]
        g = gt[eid]
        t = e.t_start + effective_T(e)
        idx = int((t - g["bin_start"]) // g["bin_seconds"])
        payload = sum(g["bins"][: max(idx + 1, 0)])
        first[name(eid)] = dict(src2dst_bytes=int(r["src2dst_bytes"]), dst2src_bytes=int(r["dst2src_bytes"]),
                                packets=int(r["bidirectional_packets"]), duration_s=int(r["bidirectional_duration_ms"]) / 1e3,
                                fin=int(r["bidirectional_fin_packets"]), payload_delivered_by_first_record=int(payload),
                                activity=r["Activity"])
    out["first_record"] = first

    other = [r for r in rows if r.get("Stage", "").strip() not in ("Benign", "Normal", "", "Data Exfiltration")]
    out["other_stage_attack_rows_trained_as_negative"] = len(other)
    out["other_stage_by_stage"] = dict(collections.Counter(r["Stage"] for r in other))
    sig = collections.defaultdict(set)
    for eid, e in eps.items():
        for f in e.flows:
            sig[tuple(owner[id(f)][k] for k in FEAT)].add(name(eid))
    dup = collections.Counter()
    dup_src = collections.Counter()
    for r in other:
        s = tuple(r[k] for k in FEAT)
        if s in sig:
            for n in sig[s]:
                dup[n] += 1
            dup_src[(r["src_ip"], r["source_file"].split("/")[-1], r["Stage"])] += 1
    out["feature_identical_negatives_per_episode"] = dict(dup)
    out["feature_identical_negatives_total"] = sum(1 for r in other if tuple(r[k] for k in FEAT) in sig)
    out["feature_identical_negatives_by_source"] = {f"{a}|{b}|{c}": n for (a, b, c), n in dup_src.most_common(6)}
    out["scored_flows_per_episode"] = {name(eid): len(e.flows) for eid, e in eps.items()}

    for proto in ("strict", "loo"):
        g = list(csv.DictReader(open(ROOT / f"results/real_edxgb_gap_{proto}_wilson.csv")))
        gaps = [float(r["gap"]) for r in g]
        ratio = [float(r["actual"]) / float(r["floor"]) for r in g]
        out[f"at_floor_{proto}"] = {"0pp": sum(x <= 1e-9 for x in gaps), "0.3pp": sum(x <= 0.003 + 1e-9 for x in gaps),
                                     "1pp": sum(x <= 0.01 for x in gaps), "ratio<=2": sum(x <= 2 for x in ratio),
                                     "ratios": {name(r["episode_id"]): round(q, 2) for r, q in zip(g, ratio)}}
    sem = list(csv.DictReader(open(ROOT / "results/emission_semantics_ebd.csv")))
    out["at_floor_under_latency"] = {f"{p}@{L}": sum(r["at_floor"] == "True" for r in sem if r["protocol"] == p and float(r["latency_s"]) == L)
                                     for p in ("strict", "loo") for L in sorted({float(r["latency_s"]) for r in sem})}

    def n_records(r, ta):
        d = int(r["bidirectional_duration_ms"]) / 1e3
        return max(1, math.ceil(d / ta)) if d > 0 else 1
    base = len(benign)
    out["benign_record_inflation"] = {str(ta): round(sum(n_records(r, ta) for r in benign) / base, 3)
                                      for ta in (15, 60, 120, 300, 600, 1800)}

    frame = {r["key"]: r for r in csv.DictReader(open(ROOT / "data/survey/frame.csv"))}
    first_c = {r["key"]: r for r in csv.DictReader(open(ROOT / "data/survey/coded.csv"))}
    second = {r["key"]: r for r in csv.DictReader(open(ROOT / "data/survey/second_coding.csv"))}
    rc_units = {"flow_record", "session_aggregate", "host_timeseries"}
    strata = collections.defaultdict(lambda: collections.Counter())
    for k, f in frame.items():
        a, b = first_c.get(k), second.get(k)
        if not a or not b:
            continue
        if a["record_consumer"] == "yes" and b["unit_of_inference"] in rc_units:
            s = f["scope"] + "/" + f["channel"]
            strata[s]["both_rc"] += 1
            strata[s]["second_cadence_" + b["cadence_stated"]] += 1
    out["cadence_by_stratum"] = {k: dict(v) for k, v in strata.items()}

    resp = list(csv.DictReader(open(ROOT / "results/response_sensitivity.csv")))
    med = {}
    for R in sorted({int(r["R_s"]) for r in resp}):
        f = sorted(float(r["floor_plus_R"]) for r in resp if int(r["R_s"]) == R)
        d = sorted(float(r["detector_plus_R"]) for r in resp if int(r["R_s"]) == R)
        med[str(R)] = dict(floor_median=round(f[3], 4), floor_max=round(f[-1], 4), det_median=round(d[3], 4), det_max=round(d[-1], 4))
    out["response_median_max"] = med

    (ROOT / "results/rev11_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
