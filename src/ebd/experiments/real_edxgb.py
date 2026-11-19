"""Phase D/E: measure a real detector's EBD against the architectural floor.

    uv run ebd-edxgb --train <flows.csv> --groundtruth <groundtruth.jsonl>

Trains EDXGB's feature pipeline + XGBoost predictor (see
:mod:`ebd.realdata.edxgb` for exactly what is and is not reproduced), then for
each reconstructed episode measures:

* **floor**  — ``min(1, T_eff/D)``, what no flow-record consumer can beat;
* **actual** — the fraction of the asset already gone when the detector's first
  positive flow record became available;
* **gap**    — actual minus floor: how much further than the architecture
  requires this detector let the data go.

Splitting
---------
The held-out set is **whole episodes**, never individual flows. Flows from one
episode are strongly correlated — they share a peer, a rate and a technique — so
a flow-level split would train and test on the same incident and report a
detector far better than it is. This is also why our figures will not reproduce
EDXGB's published numbers, and should not be expected to.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    from ebd.realdata import FlowRecord, group_into_episodes
    from ebd.realdata.cadence_probe import byte_floor, effective_T
    from ebd.realdata.edxgb import (
        build_matrix, feature_columns, first_alert_time, train,
    )

    ap = argparse.ArgumentParser(description="Real detector vs architectural floor")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--groundtruth", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--loo", action="store_true",
                    help="leave-one-episode-out: retrain per episode on the other six")
    ap.add_argument("--at-floor-pp", type=float, default=1.0,
                    help="gap (percentage points of asset) counted as reaching the floor")
    ap.add_argument("--fpb-budget", type=float, default=0.01,
                    help="false-positive burden budget; the operating threshold is "
                         "chosen as the most aggressive one meeting it on TRAIN benign")
    ap.add_argument("--benign-holdout", type=float, default=0.5,
                    help="fraction of benign sessions held out to measure FPB")
    ap.add_argument("--rule", choices=("point", "wilson"), default="point",
                    help="operating-point rule (ebd.realdata.operating_point). "
                         "'point' reproduces Table 2's archived run; 'wilson' is "
                         "the a-priori rule used throughout the revised paper")
    ap.add_argument("--n-jobs", type=int, default=4,
                    help="xgboost threads; 1 makes reruns bit-identical on a machine")
    ap.add_argument("--suffix", default=None,
                    help="output-name suffix (default '_wilson' for the Wilson rule, "
                         "'' for the point rule, which overwrites the archived file)")
    args = ap.parse_args(argv)
    from ebd.realdata.operating_point import choose_threshold, measure_burden, unit_max_scores

    csv.field_size_limit(10_000_000)
    rows = list(csv.DictReader(open(args.train)))
    print("=" * 78)
    print("  PHASE D — EDXGB feature pipeline + XGBoost, on real episodes")
    print("  Deep DNN/CNN extractors omitted: a result AT the floor is")
    print("  conclusive; a poor result is not (see ebd.realdata.edxgb).")
    print("=" * 78)
    print(f"\nflow rows: {len(rows)}")

    gt = {json.loads(l)["episode_id"]: json.loads(l)
          for l in open(args.groundtruth) if l.strip()}
    print(f"reconstructed episodes: {len(gt)}")

    # rebuild episodes from the exfiltration rows so each flow knows its episode
    exf = [r for r in rows if r.get("Stage", "").strip() == "Data Exfiltration"]
    frecs, idx_of = [], {}
    for i, r in enumerate(exf):
        try:
            frecs.append(FlowRecord(
                src_ip=r["src_ip"], src_port=int(r["src_port"]), dst_ip=r["dst_ip"],
                dst_port=int(r["dst_port"]), protocol=int(r["protocol"]),
                start=float(r["bidirectional_first_seen_ms"]) / 1000,
                duration=float(r["bidirectional_duration_ms"]) / 1000,
                fwd_bytes=int(r["src2dst_bytes"]), bwd_bytes=int(r["dst2src_bytes"]),
                fwd_packets=int(r["src2dst_packets"]),
                bwd_packets=int(r["dst2src_packets"]),
                stage=r["Stage"], activity=r["Activity"],
                source_file=r.get("source_file", ""),
            ))
            idx_of[id(frecs[-1])] = i
        except (KeyError, ValueError):
            continue
    episodes = group_into_episodes(frecs)
    held = {e.episode_id: e for e in episodes if e.episode_id in gt}
    print(f"held-out episodes present in both: {len(held)}")

    held_rows = {eid: [exf[idx_of[id(f)]] for f in e.flows if id(f) in idx_of]
                 for eid, e in held.items()}
    held_row_ids = {id(r) for rs in held_rows.values() for r in rs}

    # Benign traffic must also be split. Previously every benign flow was in
    # training, so no false-positive rate could be measured at all -- and EBD
    # quoted without one is unfalsifiable: a detector that alerts on the first
    # record of everything reaches the floor trivially. Benign flows are grouped
    # into sessions with the same rule used for attack episodes, and the split is
    # by SESSION so correlated flows cannot straddle it.
    benign = [r for r in rows if r.get("Stage", "").strip() in ("Benign", "Normal", "")]
    bkey = {}
    for r in benign:
        k = (r.get("src_ip"), r.get("dst_ip"), r.get("dst_port"))
        bkey.setdefault(k, []).append(r)
    bsessions = list(bkey.values())
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(bsessions))
    cut = int(len(bsessions) * (1.0 - args.benign_holdout))
    train_benign = [r for i in order[:cut] for r in bsessions[i]]
    test_benign_sessions = [bsessions[i] for i in order[cut:]]
    test_benign = [r for sess in test_benign_sessions for r in sess]
    test_benign_ids = {id(r) for r in test_benign}
    print(f"benign: {len(benign)} flows in {len(bsessions)} sessions -> "
          f"train {len(train_benign)}, held out {len(test_benign)} "
          f"in {len(test_benign_sessions)} sessions")

    header = list(rows[0].keys())
    cols = feature_columns(header)
    tr = [r for r in rows
          if id(r) not in held_row_ids and id(r) not in test_benign_ids]
    y = np.array([1 if r.get("Stage", "").strip() == "Data Exfiltration" else 0
                  for r in tr])
    print(f"\ntrain rows {len(tr)}  (exfil {int(y.sum())}, other {len(y)-int(y.sum())})")
    print(f"features: {len(cols)} behavioural + 4 derived (PTR/BTR)")

    if args.loo:
        print("\nSPLIT: leave-one-episode-out — each episode scored by a model\n"
              "       retrained on the other six (plus all benign).")
    else:
        print("\nSPLIT: all seven episodes held out simultaneously — STRICTER than\n"
              "       leave-one-out, since no model sees any evaluated episode.")
    det = train(tr, y, cols, seed=args.seed, n_jobs=args.n_jobs)

    def detector_for(eid):
        """Model used to score ``eid``.

        Default: one model that saw none of the seven. With ``--loo``: a model
        retrained on the other six, which is the fairer test of the detector
        (more representative exfiltration in training) while still never
        showing it the episode being scored.
        """
        if not args.loo:
            return det
        keep_ids = {id(r) for e2, rs2 in held_rows.items() if e2 != eid for r in rs2}
        tr2 = [r for r in rows
               if id(r) not in held_row_ids and id(r) not in test_benign_ids] + \
              [r for r in rows if id(r) in keep_ids]
        y2 = np.array([1 if r.get("Stage", "").strip() == "Data Exfiltration" else 0
                       for r in tr2])
        return train(tr2, y2, cols, seed=args.seed, n_jobs=args.n_jobs)

    # ---- operating point ---------------------------------------------------
    # EBD is meaningless without a stated false-positive burden (Sec. 3.2), so
    # the threshold is CHOSEN on TRAIN benign to meet the budget, then the FPB
    # actually achieved is measured on HELD-OUT benign. Selecting on train and
    # reporting on held-out keeps the operating point honest.
    def session_fpb(sessions, model, thr):
        """Fraction of benign sessions raising at least one alert."""
        if not sessions:
            return 0.0
        hits = 0
        for sess in sessions:
            sc = model.score(sess)
            if len(sc) and float(sc.max()) >= thr:
                hits += 1
        return hits / len(sessions)

    train_benign_sessions = [v for v in bkey.values()
                             if v and id(v[0]) not in test_benign_ids]
    grid = np.unique(np.round(np.linspace(0.05, 0.95, 19), 3))
    # Each benign session is scored once; a session alerts iff its loudest
    # record clears the threshold (same quantity session_fpb computes).
    train_max = unit_max_scores([det.score(s) for s in train_benign_sessions])
    test_max = unit_max_scores([det.score(s) for s in test_benign_sessions])
    op = choose_threshold(train_max, args.fpb_budget, rule=args.rule, grid=grid)
    threshold = op["threshold"]
    burden = measure_burden(test_max, threshold)
    fpb_test = burden["heldout_fpb"]
    flow_fp = float((det.score(test_benign) >= threshold).mean()) if test_benign else 0.0
    print(f"\noperating point ({op['rule']} rule): threshold={threshold:.3f} for FPB "
          f"budget {args.fpb_budget:.1%}; train benign {op['train_fpb']:.2%} "
          f"(upper 95% {op['train_upper']:.2%}) on {op['train_n']} sessions"
          f"{'' if op['admissible'] else '  ** NO GRID VALUE MEETS THE BUDGET **'}")
    print(f"  measured on HELD-OUT benign: session FPB={fpb_test:.2%} "
          f"[{burden['heldout_lo']:.2%}, {burden['heldout_hi']:.2%}]  "
          f"flow-level FP rate={flow_fp:.3%}  "
          f"({len(test_benign_sessions)} sessions, {len(test_benign)} flows)")
    args.threshold = threshold

    print(f"\n{'episode':<30}{'V_e(MB)':>9}{'floor':>8}{'actual':>9}{'gap':>8}  outcome")
    print("-" * 78)
    out = []
    detail: dict[str, dict] = {}
    for eid, e in sorted(held.items(), key=lambda kv: gt[kv[0]]["t_end"] - gt[kv[0]]["t_start"]):
        g = gt[eid]
        D = g["t_end"] - g["t_start"]
        # Floor must be denominated in BYTES, like EBD itself: min(1,T/D) is a
        # fraction of duration and only equals the byte fraction at constant
        # rate, which a beacon is not.
        floor = byte_floor(e, g["bin_start"], g["bin_seconds"], g["bins"],
                           g["pcap_bytes"])
        time_floor = min(1.0, effective_T(e) / max(D, 1e-9))
        rws = held_rows[eid]
        model = detector_for(eid)
        scores = model.score(rws) if rws else np.array([])
        t_alert = first_alert_time(rws, scores, args.threshold) if len(scores) else None
        total = max(g["pcap_bytes"], 1)
        if t_alert is None:
            actual, tag = 1.0, "never fired"
        else:
            idx = int((t_alert - g["bin_start"]) // g["bin_seconds"])
            leaked = sum(g["bins"][: max(idx + 1, 0)]) if idx >= 0 else 0
            actual = min(1.0, leaked / total)
            tag = f"fired +{t_alert - g['t_start']:.0f}s"
        gap = actual - floor
        # Invariant: a detector cannot alert before its first record exists, so
        # the measured EBD is never below the floor. A negative gap means the
        # floor is being computed from the wrong quantity -- it is what exposed
        # both the duration-vs-bytes mix-up and the earliest-starting vs
        # earliest-ENDING flow bug. Fail loudly rather than publish nonsense.
        if gap < -1e-9:
            raise AssertionError(
                f"negative gap {gap:.4f} on {eid}: actual={actual:.4f} "
                f"floor={floor:.4f} -- floor computed from the wrong quantity"
            )
        det_rate = float((scores >= args.threshold).mean()) if len(scores) else 0.0
        print(f"{eid[:29]:<30}{g['pcap_bytes']/1e6:>9.3f}{floor*100:>7.2f}%"
              f"{actual*100:>8.2f}%{gap*100:>+7.2f}%  {tag} ({det_rate:.0%} of {len(rws)} flows)")
        out.append({"episode_id": eid, "asset_bytes": g["pcap_bytes"],
                    "duration_s": D, "floor": floor, "time_floor": time_floor,
                    "actual": actual, "gap": gap,
                    "flows": len(rws), "flow_detect_rate": det_rate,
                    "fired": t_alert is not None})
        # Per-record detail for the front-loaded analysis (Sec. 7): when each of
        # the earliest-ENDING records became available, and what it scored.
        ends = []
        for r, s in zip(rws, scores):
            try:
                ends.append((float(r["bidirectional_last_seen_ms"]) / 1000.0, float(s)))
            except (KeyError, TypeError, ValueError):
                continue
        ends.sort()
        detail[eid] = {
            "t_start": g["t_start"], "t_end": g["t_end"],
            "first_record_s": round(ends[0][0] - g["t_start"], 3) if ends else None,
            "alert_s": round(t_alert - g["t_start"], 3) if t_alert is not None else None,
            "n_records": len(ends), "n_flagged": int(sum(s >= args.threshold for _, s in ends)),
            "earliest_records": [{"end_s": round(t - g["t_start"], 3), "score": round(s, 6)}
                                 for t, s in ends[:12]],
        }
    print("-" * 78)

    fired = [o for o in out if o["fired"]]
    thr_pp = args.at_floor_pp / 100.0
    at_floor = [o for o in out if o["gap"] <= thr_pp]
    big = [o for o in out if o["gap"] > thr_pp]
    if out:
        print(f"episodes detected: {len(fired)}/{len(out)}")
        print(f"AT FLOOR (gap <= {args.at_floor_pp:.2f} pp of asset): {len(at_floor)}/{len(out)}")
        if at_floor and big:
            print(f"  largest at-floor gap {max(o['gap'] for o in at_floor)*100:.2f} pp"
                  f"   smallest large gap {min(o['gap'] for o in big)*100:.2f} pp"
                  f"   -> separation {min(o['gap'] for o in big)/max(max(o['gap'] for o in at_floor),1e-9):.0f}x")
            print("  (the bucketing is insensitive to the cutoff anywhere in that range)")
    if fired:
        print(f"mean floor  {np.mean([o['floor'] for o in fired])*100:.2f}%")
        print(f"mean actual {np.mean([o['actual'] for o in fired])*100:.2f}%")
        print(f"mean gap    {np.mean([o['gap'] for o in fired])*100:+.2f}%")

    root = _project_root()
    (root / "results").mkdir(exist_ok=True)
    for o in out:
        o["threshold"] = round(threshold, 4)
        o["session_fpb_heldout"] = round(fpb_test, 5)
        o["flow_fp_rate_heldout"] = round(flow_fp, 6)
        o["rule"] = op["rule"]
        o["admissible"] = op["admissible"]
        o["train_benign_sessions"] = op["train_n"]
        o["session_fpb_train"] = round(op["train_fpb"], 6)
        o["session_fpb_train_upper"] = round(op["train_upper"], 6)
        o["heldout_benign_sessions"] = burden["heldout_n"]
        o["session_fpb_heldout_lo"] = round(burden["heldout_lo"], 6)
        o["session_fpb_heldout_hi"] = round(burden["heldout_hi"], 6)
    tag = "loo" if args.loo else "strict"
    suffix = args.suffix if args.suffix is not None else \
        ("_wilson" if args.rule == "wilson" else "")
    p = root / "results" / f"real_edxgb_gap_{tag}{suffix}.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"\nwrote {p.relative_to(root)}")
    dp = root / "results" / f"real_edxgb_detail_{tag}{suffix}.json"
    dp.write_text(json.dumps({"threshold": threshold, "rule": op["rule"],
                              "episodes": detail}, indent=1))
    print(f"wrote {dp.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
