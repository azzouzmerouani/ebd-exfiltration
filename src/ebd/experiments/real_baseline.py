"""Separability, feature ablations and a label-permutation null for Sec. 7.

    uv run ebd-baseline            # about 5-10 min, single-threaded

Answers the second-round reviewers (R1-M6, DA-M4): the DAPT2021 attainment
result ("five of seven at the floor") had no separability measure, no null
baseline, and its features include port fields that could act as shortcuts on a
capture whose exfiltration goes to one destination. Under the strict protocol of
``real_edxgb`` (all seven episodes held out, benign split by host-service key,
upper-Wilson operating-point rule) this driver reports, for each variant:

* ``full``       -- EDXGB's features as in Table 2 (must reproduce its strict column);
* ``no_ports``   -- the same without ``src_port``, ``dst_port`` and ``protocol``;
* ``rates_only`` -- only EDXGB's four rate features (PTR/BTR in each direction);
* ``perm_k``     -- ``full`` features, training labels randomly permuted
  (``--permutations`` times): what the rule and the floor give a detector that
  has learnt nothing.

and for each: unit-level AUC (episode = max score over its records, against
held-out benign keys, with a Hanley-McNeil interval), flow-level AUC, the
threshold, held-out burden, episodes fired, episodes at the floor, mean and
median nEBD.

Writes ``results/real_baseline.csv`` (one row per variant) and
``results/real_baseline_episodes.csv`` (one row per variant x episode).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

PORT_COLUMNS = {"src_port", "dst_port", "protocol"}


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted as one half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def hanley_mcneil(a: float, n_pos: int, n_neg: int) -> tuple[float, float]:
    if not (0 < a < 1) or not n_pos or not n_neg:
        return (a, a)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = math.sqrt((a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a))
                   / (n_pos * n_neg))
    return max(0.0, a - 1.96 * se), min(1.0, a + 1.96 * se)


def main(argv: list[str] | None = None) -> int:
    from ebd.realdata import FlowRecord, group_into_episodes
    from ebd.realdata.cadence_probe import byte_floor
    from ebd.realdata.edxgb import build_matrix, feature_columns, first_alert_time, train
    from ebd.realdata.operating_point import (DEFAULT_GRID, choose_threshold, extended_grid,
                                              measure_burden, quantile_grid,
                                              training_negatives)

    root = _root()
    ap = argparse.ArgumentParser(description="Separability, ablations and a null for Sec. 7")
    ap.add_argument("--train", type=Path, default=root / "data/realdata/train_flows.csv")
    ap.add_argument("--groundtruth", type=Path, default=root / "data/realdata/groundtruth.jsonl")
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--permutations", type=int, default=20)
    ap.add_argument("--fpb-budget", type=float, default=0.01)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--at-floor-pp", type=float, default=1.0)
    ap.add_argument("--oof", type=int, default=0,
                    help="if K > 1, calibrate the threshold on out-of-fold scores of "
                         "the training benign keys (K folds by key) instead of "
                         "in-sample scores (round-3 R1-M2)")
    ap.add_argument("--variants", default="",
                    help="comma-separated subset of variants to run (default: all)")
    ap.add_argument("--negatives", choices=("all", "benign"), default="all",
                    help="'benign' drops attack rows of other stages from training, "
                         "so they no longer train as negatives (rev11 audit)")
    ap.add_argument("--exfil-bootstrap", type=int, default=0,
                    help="if B > 0, resample the training exfiltration episodes with "
                         "replacement (seeded by --seed) before training: variance "
                         "over the exfiltration side of the training set")
    ap.add_argument("--loo", action="store_true",
                    help="score each evaluated episode with a model that also trains on "
                         "the other six reconstructed episodes (threshold from the "
                         "strict model, as in real_edxgb)")
    ap.add_argument("--grid", choices=("default", "extended", "quantile"), default="default",
                    help="threshold grid: the paper's 0.05..0.95, or that grid extended "
                         "below 0.05 with log-spaced values and training scores")
    ap.add_argument("--loo-calibrate", action="store_true",
                    help="under --loo, choose each retrained model's own threshold (same "
                         "calibration and grid) and measure its own held-out burden "
                         "(rev13 review R1-1); without it every LOO model is scored at "
                         "the strict model's threshold")
    ap.add_argument("--peer-disjoint", action="store_true",
                    help="score each evaluated episode with a model trained without any "
                         "exfiltration flow to the same (dst_ip, dst_port): a "
                         "destination-disjoint split that removes the peer/port shortcut")
    ap.add_argument("--out", type=Path, default=root / "results/real_baseline.csv")
    ap.add_argument("--episodes-out", type=Path,
                    default=root / "results/real_baseline_episodes.csv")
    args = ap.parse_args(argv)

    csv.field_size_limit(10_000_000)
    rows = list(csv.DictReader(open(args.train)))
    gt = {json.loads(l)["episode_id"]: json.loads(l)
          for l in open(args.groundtruth) if l.strip()}

    # ---- the same split as real_edxgb (strict protocol) --------------------
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
    held = {e.episode_id: e for e in group_into_episodes(frecs) if e.episode_id in gt}
    held_rows = {eid: [exf[idx_of[id(f)]] for f in e.flows if id(f) in idx_of]
                 for eid, e in held.items()}
    held_ids = {id(r) for rs in held_rows.values() for r in rs}

    benign = [r for r in rows if r.get("Stage", "").strip() in ("Benign", "Normal", "")]
    bkey: dict = {}
    for r in benign:
        bkey.setdefault((r.get("src_ip"), r.get("dst_ip"), r.get("dst_port")), []).append(r)
    bsessions = list(bkey.values())
    order = np.random.default_rng(args.seed).permutation(len(bsessions))
    cut = int(len(bsessions) * 0.5)
    test_sessions = [bsessions[i] for i in order[cut:]]
    test_ids = {id(r) for s in test_sessions for r in s}
    train_sessions = [v for v in bkey.values() if v and id(v[0]) not in test_ids]
    tr = [r for r in rows if id(r) not in held_ids and id(r) not in test_ids]
    tr = training_negatives(tr, args.negatives)
    if args.exfil_bootstrap > 0:
        # resample the training exfiltration episodes (whole episodes) with replacement
        tr_ex = [r for r in tr if r.get("Stage", "").strip() == "Data Exfiltration"]
        tr_fr, tr_own = [], {}
        for r in tr_ex:
            f = FlowRecord(src_ip=r["src_ip"], src_port=int(r["src_port"]), dst_ip=r["dst_ip"],
                           dst_port=int(r["dst_port"]), protocol=int(r["protocol"]),
                           start=float(r["bidirectional_first_seen_ms"]) / 1000,
                           duration=float(r["bidirectional_duration_ms"]) / 1000,
                           fwd_bytes=int(r["src2dst_bytes"]), bwd_bytes=int(r["dst2src_bytes"]))
            tr_fr.append(f)
            tr_own[id(f)] = r
        tr_eps = group_into_episodes(tr_fr)
        pick = np.random.default_rng(args.seed + 7).integers(0, len(tr_eps), len(tr_eps))
        resampled = [tr_own[id(f)] for i in pick for f in tr_eps[i].flows]
        tr = [r for r in tr if r.get("Stage", "").strip() != "Data Exfiltration"] + resampled
    y = np.array([1 if r.get("Stage", "").strip() == "Data Exfiltration" else 0 for r in tr])
    print(f"train rows {len(tr)} (exfil {int(y.sum())}); held-out benign keys "
          f"{len(test_sessions)}; training benign keys {len(train_sessions)}")

    header = list(rows[0].keys())
    full_cols = feature_columns(header)
    # name -> (columns, permute labels, append rate features, kind)
    variants = {
        "full": (full_cols, False, True, "xgb"),
        "no_ports": ([c for c in full_cols if c not in PORT_COLUMNS], False, True, "xgb"),
        "rates_only": ([], False, True, "xgb"),
        # shortcut baselines: port fields alone, and a non-learnt peer rule
        "ports_only": ([c for c in full_cols if c in PORT_COLUMNS], False, False, "xgb"),
        "peer_rule": ([], False, False, "peer_rule"),
    }
    for k in range(args.permutations):
        variants[f"perm_{k:02d}"] = (full_cols, True, True, "xgb")
    if args.variants:
        keep = {v.strip() for v in args.variants.split(",") if v.strip()}
        variants = {k: v for k, v in variants.items() if k in keep}

    # out-of-fold calibration: K folds over the training benign keys
    oof_folds: list[list[int]] = []
    if args.oof > 1:
        fold_of = np.random.default_rng(args.seed + 2).permutation(len(train_sessions)) % args.oof
        oof_folds = [list(np.flatnonzero(fold_of == f)) for f in range(args.oof)]

    # index arrays so each benign key's max score is one vectorised pass
    def flat(sessions):
        rws, owner = [], []
        for j, s in enumerate(sessions):
            rws.extend(s)
            owner.extend([j] * len(s))
        return rws, np.asarray(owner)

    tr_b_rows, tr_b_owner = flat(train_sessions)
    te_b_rows, te_b_owner = flat(test_sessions)

    def key_max(scores, owner, n):
        m = np.full(n, -np.inf)
        np.maximum.at(m, owner, scores)
        return m

    summary, per_ep = [], []
    rng = np.random.default_rng(args.seed + 1)
    class PeerRule:
        """Alert on any record whose (dst_ip, dst_port) received exfiltration in training."""

        def __init__(self, rows, labels):
            self.peers = {(r.get("dst_ip"), r.get("dst_port"))
                          for r, l in zip(rows, labels) if l == 1}

        def score(self, rows):
            return np.array([1.0 if (r.get("dst_ip"), r.get("dst_port")) in self.peers else 0.0
                             for r in rows])

    def fit(kind, rows_, labels_, cols, rates):
        if kind == "peer_rule":
            return PeerRule(rows_, labels_)
        return train(rows_, labels_, cols, seed=args.seed, n_jobs=args.n_jobs, rates=rates)

    # evaluation groups: one group of all episodes, or one per destination peer
    if args.peer_disjoint:
        peers = sorted({(e.dst_ip, str(e.dst_port)) for e in held.values()})
        groups = [(f"{ip}:{port}", {eid for eid, e in held.items()
                                    if (e.dst_ip, str(e.dst_port)) == (ip, port)}, (ip, port))
                  for ip, port in peers]
    else:
        groups = [("all", set(held), None)]

    def peer_of(r):
        return (r.get("dst_ip"), r.get("dst_port"))

    for (name, (cols, permute, rates, kind)), (gname, gids, gpeer) in \
            [(v, g) for v in variants.items() for g in groups]:
        if gpeer is None:
            tr_g, y_g = tr, y
        else:
            keep_g = [i for i, r in enumerate(tr) if not (y[i] == 1 and peer_of(r) == gpeer)]
            tr_g, y_g = [tr[i] for i in keep_g], y[keep_g]
        labels = rng.permutation(y_g) if permute else y_g
        det = fit(kind, tr_g, labels, cols, rates)

        def calibrate(det_, rows_, labels_):
            """The operating point of one model: threshold on its own (OOF or
            in-sample) training-benign key maxima, burden on the held-out keys."""
            if oof_folds:
                tm = np.full(len(train_sessions), -np.inf)
                for idxs in oof_folds:
                    out_ids = {id(r) for j in idxs for r in train_sessions[j]}
                    keep = [i for i, r in enumerate(rows_) if id(r) not in out_ids]
                    det_k = fit(kind, [rows_[i] for i in keep], labels_[keep], cols, rates)
                    for j in idxs:
                        s_ = det_k.score(train_sessions[j])
                        tm[j] = float(s_.max()) if len(s_) else -np.inf
            else:
                tm = key_max(det_.score(tr_b_rows), tr_b_owner, len(train_sessions))
            ts = det_.score(te_b_rows)
            tmx = key_max(ts, te_b_owner, len(test_sessions))
            g = {"extended": extended_grid, "quantile": quantile_grid}.get(
                args.grid, lambda m: DEFAULT_GRID)(tm)
            if kind == "peer_rule":
                g = np.array([0.5])
            op_ = choose_threshold(tm, args.fpb_budget, rule="wilson", grid=g)
            return op_, measure_burden(tmx, op_["threshold"]), ts, tmx

        op, burden, te_scores, te_max = calibrate(det, tr_g, labels)
        thr = op["threshold"]

        ep_max, ep_flow_scores, outs, loo_ops = [], [], [], []
        for eid, e in sorted((k, v) for k, v in held.items() if k in gids):
            g = gt[eid]
            rws = held_rows[eid]
            if args.loo and kind != "peer_rule":
                extra = [r for e2, rs2 in held_rows.items() if e2 != eid and e2 in gids for r in rs2]
                lab_e = np.concatenate([labels, np.ones(len(extra), int)])
                det_e = fit(kind, tr_g + extra, lab_e, cols, rates)
                sc = det_e.score(rws)
                if args.loo_calibrate:
                    op_e, burden_e, _, _ = calibrate(det_e, tr_g + extra, lab_e)
                    thr_e = op_e["threshold"]
                    loo_ops.append((thr_e, burden_e["heldout_fpb"],
                                    bool(op_e["admissible"]) and burden_e["heldout_fpb"] <= args.fpb_budget))
                else:
                    thr_e = thr
            else:
                sc = det.score(rws)
                thr_e = thr
            ep_max.append(float(sc.max()) if len(sc) else -np.inf)
            ep_flow_scores.append(sc)
            floor = byte_floor(e, g["bin_start"], g["bin_seconds"], g["bins"], g["pcap_bytes"])
            t = first_alert_time(rws, sc, thr_e) if len(sc) else None
            if t is None:
                ebd = 1.0
            else:
                idx = int((t - g["bin_start"]) // g["bin_seconds"])
                ebd = min(1.0, sum(g["bins"][: max(idx + 1, 0)]) / max(g["pcap_bytes"], 1)) \
                    if idx >= 0 else 0.0
            assert ebd >= floor - 1e-9, (name, eid, ebd, floor)
            outs.append(dict(variant=name, group=gname, episode_id=eid, threshold=thr_e,
                             floor=floor, ebd=ebd,
                             gap=ebd - floor, fired=t is not None,
                             at_floor=(ebd - floor) <= args.at_floor_pp / 100,
                             at_floor_0pp=(ebd - floor) <= 1e-9,
                             at_floor_2pp=(ebd - floor) <= 0.02))
        per_ep.extend(outs)
        a_unit = auc(np.array(ep_max), te_max)
        lo, hi = hanley_mcneil(a_unit, len(ep_max), len(te_max))
        a_flow = auc(np.concatenate(ep_flow_scores), te_scores)
        ebds = [o["ebd"] for o in outs]
        flagged_flows = int((te_scores >= thr).sum())
        row = dict(variant=name, group=gname, kind=kind, negatives=args.negatives,
                   exfil_bootstrap=args.exfil_bootstrap, protocol="loo" if args.loo else "strict",
                   n_train_pos=int(np.sum(y_g)),
                   n_train_pos_3306=int(sum(1 for i, r in enumerate(tr_g)
                                            if y_g[i] == 1 and str(r.get("dst_port")) == "3306")),
                   n_features=len(cols) + (4 if rates else 0), permuted=permute,
                   split_seed=args.seed, calibration=f"oof{args.oof}" if oof_folds else "in_sample",
                   grid=args.grid if kind != "peer_rule" else "rule",
                   threshold=thr,
                   grid_bottom_binds=bool(op.get("grid_bottom_binds", False)) and kind != "peer_rule",
                   admissible=bool(op["admissible"]) and burden["heldout_fpb"] <= args.fpb_budget,
                   rule_met_in_training=op["admissible"],
                   train_upper=round(op["train_upper"], 6),
                   heldout_fpb=round(burden["heldout_fpb"], 6),
                   heldout_lo=round(burden["heldout_lo"], 6),
                   heldout_hi=round(burden["heldout_hi"], 6),
                   heldout_keys=burden["heldout_n"],
                   heldout_flagged_flows=flagged_flows,
                   heldout_benign_flows=int(len(te_scores)),
                   unit_auc=round(a_unit, 4), unit_auc_lo=round(lo, 4), unit_auc_hi=round(hi, 4),
                   flow_auc=round(a_flow, 4),
                   fired=sum(o["fired"] for o in outs),
                   episodes=len(outs),
                   at_floor=sum(o["at_floor"] for o in outs),
                   at_floor_0pp=sum(o["at_floor_0pp"] for o in outs),
                   at_floor_2pp=sum(o["at_floor_2pp"] for o in outs),
                   mean_ebd_pct=round(100 * float(np.mean(ebds)), 4),
                   median_ebd_pct=round(100 * float(np.median(ebds)), 4))
        if loo_ops:
            row.update(loo_calibrated=True,
                       loo_thr_min=round(min(o[0] for o in loo_ops), 6),
                       loo_thr_max=round(max(o[0] for o in loo_ops), 6),
                       loo_fpb_min=round(min(o[1] for o in loo_ops), 6),
                       loo_fpb_max=round(max(o[1] for o in loo_ops), 6),
                       loo_admissible=sum(o[2] for o in loo_ops))
        summary.append(row)
        print(f"{name:<11} [{gname}] thr {thr:.4g}{' (grid floor)' if row['grid_bottom_binds'] else ''} FPB {row['heldout_fpb']:.2%} "
              f"unitAUC {a_unit:.3f} [{lo:.2f},{hi:.2f}] flowAUC {a_flow:.3f} "
              f"fired {row['fired']}/{len(outs)} at-floor {row['at_floor']}/{len(outs)} "
              f"mean nEBD {row['mean_ebd_pct']:.2f}% median {row['median_ebd_pct']:.2f}%")

    perm = [r for r in summary if r["permuted"]]
    full_rows = [r for r in summary if r["variant"] == "full"]
    if perm and full_rows:
        af = sorted(r["at_floor"] for r in perm)
        print(f"\npermutation null ({len(perm)}): at-floor counts {af}; "
              f"unit AUC median {np.median([r['unit_auc'] for r in perm]):.3f}; "
              f"P(at_floor >= full) = "
              f"{np.mean([r['at_floor'] >= full_rows[0]['at_floor'] for r in perm]):.2f}")

    for path, data in ((args.out, summary), (args.episodes_out, per_ep)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            w.writeheader()
            w.writerows(data)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
