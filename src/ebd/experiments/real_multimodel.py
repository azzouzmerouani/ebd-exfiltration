"""Sec. 7.8: does the F1/EBD divergence of Sec. 6.2 survive across detector
architectures, on the same seven real episodes real_edxgb.py evaluates?

    uv run ebd-multimodel --train <flows.csv> --groundtruth <groundtruth.jsonl>

Four classifiers share EDXGB's feature pipeline (ebd.realdata.edxgb) and the
strict split of real_edxgb.py --- all seven episodes held out simultaneously,
benign held out by session --- and differ only in the classifier on top:

* EDXGB       --- the reproduction of real_edxgb.py (lr=0.01, depth=10, 30
                   estimators), retrained here single-threaded;
* XGBoost     --- the same library at default hyperparameters;
* Random Forest, Logistic Regression --- scikit-learn, class-balanced.

For each, this measures the same two things two ways: an episode-level F1
over a threshold sweep, and mean EBD at a threshold chosen to meet an FPB
budget on training benign (Sec. 3.3's operating-point rule; the training
sessions are in the model's training set, so the training burden is an
in-sample estimate and the held-out burden is the one reported). It also
writes the whole EBD--FPB trade-off over the threshold grid
(real_multimodel_sweep*.csv), so detectors can be compared at matched held-out
burden as well as at their own operating points. The two
thresholds are selected independently and need not coincide -- see Sec. 7.8's
Logistic Regression / Random Forest case study, which needs the max score
per episode at both thresholds to explain; that diagnostic is written
separately (--maxscore-out) rather than mixed into the per-episode table.

xgboost's histogram tree method is run single-threaded (n_jobs=1): under
n_jobs>1 its per-thread histogram reduction is platform-dependent, which
changes two of Table 2's seven EDXGB episodes when retrained on different
hardware with an identical seed (Sec. 7.8 reports this as a finding, not
just a caveat).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass
class GenericDetector:
    """A trained classifier over EDXGB's feature matrix (build_matrix)."""

    model: object
    columns: list
    scaler: object = None

    def score(self, rows):
        from ebd.realdata.edxgb import build_matrix
        if not rows:
            return np.array([])
        X = build_matrix(rows, self.columns)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]


def train_edxgb(tr_rows, y, cols, seed):
    """EDXGB's reproduction (ebd.realdata.edxgb.train), single-threaded.

    Same hyperparameters as the module (lr=0.01, depth=10, 30 estimators);
    the only change is n_jobs=1 in place of the module's hardcoded n_jobs=4,
    for bit-reproducible reruns across this comparison (see module docstring
    and Sec. 7.8's discussion of the two episodes this moves).
    """
    from xgboost import XGBClassifier
    from ebd.realdata.edxgb import build_matrix
    X = build_matrix(tr_rows, cols)
    yy = np.asarray(y, dtype=int)
    pos, neg = max(int(yy.sum()), 1), max(len(yy) - int(yy.sum()), 1)
    model = XGBClassifier(
        learning_rate=0.01, max_depth=10, n_estimators=30,
        scale_pos_weight=neg / pos, eval_metric="logloss",
        random_state=seed, n_jobs=1, tree_method="hist",
    )
    model.fit(X, yy)
    return GenericDetector(model=model, columns=cols)


def train_xgb_default(tr_rows, y, cols, seed):
    """Off-the-shelf XGBoost: default hyperparameters, same imbalance handling."""
    from xgboost import XGBClassifier
    from ebd.realdata.edxgb import build_matrix
    X = build_matrix(tr_rows, cols)
    yy = np.asarray(y, dtype=int)
    pos, neg = max(int(yy.sum()), 1), max(len(yy) - int(yy.sum()), 1)
    model = XGBClassifier(
        scale_pos_weight=neg / pos, eval_metric="logloss",
        random_state=seed, n_jobs=1, tree_method="hist",
    )
    model.fit(X, yy)
    return GenericDetector(model=model, columns=cols)


def train_rf(tr_rows, y, cols, seed):
    """Random Forest -- a standard flow-based NIDS baseline."""
    from sklearn.ensemble import RandomForestClassifier
    from ebd.realdata.edxgb import build_matrix
    X = build_matrix(tr_rows, cols)
    yy = np.asarray(y, dtype=int)
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=1,
    )
    model.fit(X, yy)
    return GenericDetector(model=model, columns=cols)


def train_lr(tr_rows, y, cols, seed):
    """Logistic Regression -- the linear baseline, standardised features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from ebd.realdata.edxgb import build_matrix
    X = build_matrix(tr_rows, cols)
    yy = np.asarray(y, dtype=int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # rev13 review (EIC-6, R1-5): at sklearn's default tol=1e-4 lbfgs stopped after
    # ~60 iterations, short of the optimum, and the scores near the grid thresholds
    # differed between platforms and runs; solve to a tight tolerance instead.
    model = LogisticRegression(max_iter=20000, tol=1e-10, class_weight="balanced",
                               random_state=seed)
    # rev12 review (R1-6): lbfgs over a multithreaded BLAS gave different F1* in
    # two runs on the same data; pin BLAS/OpenMP to one thread for the fit.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover
        model.fit(Xs, yy)
    else:
        with threadpool_limits(limits=1):
            model.fit(Xs, yy)
    return GenericDetector(model=model, columns=cols, scaler=scaler)


MODELS = {
    "EDXGB (reproduced)": train_edxgb,
    "XGBoost (default)": train_xgb_default,
    "Random Forest": train_rf,
    "Logistic Regression": train_lr,
}


def _session_fpb(session_scores, thr):
    if not session_scores:
        return 0.0
    hits = sum(1 for sc in session_scores if len(sc) and float(sc.max()) >= thr)
    return hits / len(session_scores)


def _outcome(g, rws, scores, thr, first_alert_time):
    """(fraction of the asset gone, bytes gone, fired) for one episode at ``thr``.

    The byte timeline ``g`` is the episode's ground truth (groundtruth.jsonl);
    the alert instant is the end of the first record scoring >= thr.
    """
    total = max(g["pcap_bytes"], 1)
    t_alert = first_alert_time(rws, scores, thr) if len(scores) else None
    if t_alert is None:
        return 1.0, total, False
    idx = int((t_alert - g["bin_start"]) // g["bin_seconds"])
    leaked = sum(g["bins"][: max(idx + 1, 0)]) if idx >= 0 else 0
    return min(1.0, leaked / total), leaked, True


def main(argv: list[str] | None = None) -> int:
    from ebd.realdata import FlowRecord, group_into_episodes
    from ebd.realdata.cadence_probe import byte_floor
    from ebd.realdata.edxgb import feature_columns, first_alert_time
    from ebd.metrics import compare_rankings

    ap = argparse.ArgumentParser(description="F1 vs EBD across four detector families")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--groundtruth", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--fpb-budget", type=float, default=0.01)
    ap.add_argument("--benign-holdout", type=float, default=0.5)
    ap.add_argument("--rule", choices=("wilson", "point"), default="wilson",
                    help="operating-point rule (ebd.realdata.operating_point); "
                         "'point' reproduces the archived 19 Aug run")
    ap.add_argument("--suffix", default=None,
                    help="output-name suffix (default: '_wilson' for the Wilson "
                         "rule, '' for the point rule, which overwrites the archive)")
    ap.add_argument("--negatives", choices=("all", "benign"), default="all",
                    help="'benign' drops attack rows of other stages from training")
    ap.add_argument("--oof", type=int, default=0,
                    help="if K > 1, calibrate each model's threshold on K-fold "
                         "out-of-fold scores of the training benign keys")
    ap.add_argument("--grid", choices=("default", "extended", "quantile"), default="default",
                    help="'extended' adds thresholds below 0.05 (log-spaced and the "
                         "training scores) so the bottom of the grid does not bind")
    ap.add_argument("--sweep-out", type=Path, default=None,
                    help="per-threshold trade-off table (default: "
                         "results/real_multimodel_sweep<suffix>.csv)")
    args = ap.parse_args(argv)
    from ebd.realdata.operating_point import (choose_threshold, extended_grid, measure_burden,
                                             quantile_grid, training_negatives,
                                             unit_max_scores, wilson_interval,
                                             wilson_upper)
    suffix = args.suffix if args.suffix is not None else ("_wilson" if args.rule == "wilson" else "")

    csv.field_size_limit(10_000_000)
    grid = np.unique(np.round(np.linspace(0.05, 0.95, 19), 3))

    rows = list(csv.DictReader(open(args.train)))
    gt = {json.loads(l)["episode_id"]: json.loads(l)
          for l in open(args.groundtruth) if l.strip()}
    print(f"flow rows: {len(rows)}   reconstructed episodes: {len(gt)}")

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
                fwd_packets=int(r["src2dst_packets"]), bwd_packets=int(r["dst2src_packets"]),
                stage=r["Stage"], activity=r["Activity"], source_file=r.get("source_file", ""),
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

    benign = [r for r in rows if r.get("Stage", "").strip() in ("Benign", "Normal", "")]
    bkey: dict = {}
    for r in benign:
        k = (r.get("src_ip"), r.get("dst_ip"), r.get("dst_port"))
        bkey.setdefault(k, []).append(r)
    bsessions = list(bkey.values())
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(bsessions))
    cut = int(len(bsessions) * (1.0 - args.benign_holdout))
    test_benign_sessions = [bsessions[i] for i in order[cut:]]
    test_benign = [r for sess in test_benign_sessions for r in sess]
    test_benign_ids = {id(r) for r in test_benign}
    train_benign_sessions = [v for v in bkey.values() if v and id(v[0]) not in test_benign_ids]
    print(f"benign: {len(benign)} flows in {len(bsessions)} sessions -> "
          f"held out {len(test_benign)} in {len(test_benign_sessions)} sessions")

    header = list(rows[0].keys())
    cols = feature_columns(header)
    tr = [r for r in rows if id(r) not in held_row_ids and id(r) not in test_benign_ids]
    tr = training_negatives(tr, args.negatives)
    y = np.array([1 if r.get("Stage", "").strip() == "Data Exfiltration" else 0 for r in tr])
    print(f"train rows {len(tr)} (exfil {int(y.sum())}, other {len(y) - int(y.sum())})\n")

    summary, per_episode, maxscores, sweep = [], [], [], []

    for name, train_fn in MODELS.items():
        print(f"=== {name} ===")
        det = train_fn(tr, y, cols, args.seed)

        train_benign_scores = [det.score(sess) for sess in train_benign_sessions]
        test_benign_scores = [det.score(sess) for sess in test_benign_sessions]
        held_scores = {eid: det.score(held_rows[eid]) for eid in held}

        tb_max = unit_max_scores(train_benign_scores)
        if args.oof > 1:
            fold_of = np.random.default_rng(args.seed + 2).permutation(len(train_benign_sessions)) % args.oof
            tb_max = np.full(len(train_benign_sessions), -np.inf)
            for f in range(args.oof):
                idxs = list(np.flatnonzero(fold_of == f))
                out_ids = {id(r) for j in idxs for r in train_benign_sessions[j]}
                keep = [i for i, r in enumerate(tr) if id(r) not in out_ids]
                det_k = train_fn([tr[i] for i in keep], y[keep], cols, args.seed)
                for j in idxs:
                    sc = det_k.score(train_benign_sessions[j])
                    tb_max[j] = float(sc.max()) if len(sc) else -np.inf
        grid_m = {"extended": extended_grid, "quantile": quantile_grid}.get(
            args.grid, lambda m: grid)(tb_max)
        op = choose_threshold(tb_max, args.fpb_budget, rule=args.rule, grid=grid_m)
        ebd_threshold = op["threshold"]
        burden = measure_burden(unit_max_scores(test_benign_scores), ebd_threshold)
        fpb_heldout = burden["heldout_fpb"]
        assert abs(fpb_heldout - _session_fpb(test_benign_scores, ebd_threshold)) < 1e-12
        all_tb = np.concatenate(test_benign_scores) if test_benign_scores else np.array([])
        flow_fp = float((all_tb >= ebd_threshold).mean()) if len(all_tb) else 0.0

        floors = {eid: byte_floor(e, gt[eid]["bin_start"], gt[eid]["bin_seconds"],
                                  gt[eid]["bins"], gt[eid]["pcap_bytes"])
                  for eid, e in held.items()}
        ep_out = []
        for eid, e in held.items():
            g = gt[eid]
            floor = floors[eid]
            actual, leaked_bytes, fired = _outcome(g, held_rows[eid], held_scores[eid],
                                                   ebd_threshold, first_alert_time)
            gap = max(actual - floor, 0.0)
            row = dict(model=name, episode_id=eid, asset_bytes=g["pcap_bytes"],
                       floor_pct=floor * 100, actual_pct=actual * 100, gap_pct=gap * 100,
                       leaked_bytes=leaked_bytes, fired=fired)
            ep_out.append(row)
            per_episode.append(row)

        # The whole EBD--FPB trade-off over the grid, so detectors can also be
        # compared at matched held-out burden (Sec. 7.8). Descriptive only: the
        # operating point above never sees these held-out columns.
        tr_max = unit_max_scores(train_benign_scores)
        te_max = unit_max_scores(test_benign_scores)
        for t in sorted({float(x) for x in grid} | {float(ebd_threshold)}):
            tr_k, te_k = int((tr_max >= t).sum()), int((te_max >= t).sum())
            lo, hi = wilson_interval(te_k, len(te_max))
            outs = {eid: _outcome(gt[eid], held_rows[eid], held_scores[eid], t,
                                  first_alert_time) for eid in held}
            gaps = [max(outs[eid][0] - floors[eid], 0.0) for eid in held]
            sweep.append(dict(
                model=name, threshold=float(t),
                train_benign_sessions=len(tr_max), train_alerts=tr_k,
                train_fpb=tr_k / max(len(tr_max), 1),
                train_upper=wilson_upper(tr_k, len(tr_max)),
                heldout_benign_sessions=len(te_max), heldout_alerts=te_k,
                heldout_fpb=te_k / max(len(te_max), 1), heldout_lo=lo, heldout_hi=hi,
                mean_ebd_pct=100.0 * float(np.mean([outs[eid][0] for eid in held])),
                mean_gap_pct=100.0 * float(np.mean(gaps)),
                fired=sum(1 for eid in held if outs[eid][2]),
                at_floor=sum(1 for gp in gaps if gp <= 0.01),
                is_operating_point=bool(abs(t - ebd_threshold) < 1e-9),
            ))

        mean_nebd = float(np.mean([o["actual_pct"] for o in ep_out]))
        op_row = [r for r in sweep if r["model"] == name and r["is_operating_point"]]
        assert len(op_row) == 1 and abs(op_row[0]["mean_ebd_pct"] - mean_nebd) < 1e-9
        mean_gap = float(np.mean([o["gap_pct"] for o in ep_out]))
        mean_leaked = float(np.mean([o["leaked_bytes"] for o in ep_out]))
        median_leaked = float(np.median([o["leaked_bytes"] for o in ep_out]))
        detect_rate = sum(o["fired"] for o in ep_out) / len(ep_out)

        best = dict(threshold=float("nan"), precision=0.0, recall=0.0, f1=-1.0)
        for t in grid:
            tp = sum(1 for eid in held if len(held_scores[eid]) and held_scores[eid].max() >= t)
            fn = len(held) - tp
            fp = sum(1 for sc in test_benign_scores if len(sc) and sc.max() >= t)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            if f1 > best["f1"]:
                best = dict(threshold=float(t), precision=precision, recall=recall, f1=f1)

        for eid in held:
            sc = held_scores[eid]
            maxscores.append(dict(
                model=name, episode_id=eid,
                max_score=float(sc.max()) if len(sc) else float("nan"),
                f1_threshold=best["threshold"], ebd_threshold=ebd_threshold,
                fires_at_f1_threshold=bool(len(sc) and sc.max() >= best["threshold"]),
                fires_at_ebd_threshold=bool(len(sc) and sc.max() >= ebd_threshold),
            ))

        summary.append(dict(
            model=name, f1=best["f1"], f1_threshold=best["threshold"],
            precision=best["precision"], recall=best["recall"],
            ebd_score_pct=mean_nebd, mean_gap_pct=mean_gap,
            mean_bytes_lost=mean_leaked, median_bytes_lost=median_leaked,
            ebd_threshold=ebd_threshold, fpb_heldout=fpb_heldout,
            flow_fp_rate=flow_fp, detect_rate=detect_rate,
            rule=op["rule"],
            # round 3: admissible requires the held-out burden within budget too
            admissible=bool(op["admissible"]) and burden["heldout_fpb"] <= op["budget"],
            rule_met_in_training=op["admissible"],
            grid=args.grid, negatives=args.negatives,
            calibration=f"oof{args.oof}" if args.oof > 1 else "in_sample",
            grid_bottom_binds=bool(op.get("grid_bottom_binds", False)),
            train_benign_sessions=op["train_n"], train_fpb=op["train_fpb"],
            train_upper=op["train_upper"],
            heldout_benign_sessions=burden["heldout_n"],
            fpb_heldout_lo=burden["heldout_lo"], fpb_heldout_hi=burden["heldout_hi"],
        ))
        print(f"  F1={best['f1']:.3f} (thr={best['threshold']:.2f}, "
              f"P={best['precision']:.2f}, R={best['recall']:.2f})   "
              f"EBD={mean_nebd:.2f}%  mean bytes lost={mean_leaked/1e6:.3f} MB  "
              f"detect_rate={detect_rate:.0%}  FPB(heldout)={fpb_heldout:.2%} "
              f"[{burden['heldout_lo']:.2%}, {burden['heldout_hi']:.2%}]  "
              f"(EBD thr {ebd_threshold:.2f}; train {op['train_fpb']:.2%}, "
              f"upper {op['train_upper']:.2%}, rule {op['rule']})")

    f1_by = {s["model"]: s["f1"] for s in summary}
    ebd_by = {s["model"]: s["ebd_score_pct"] for s in summary}
    cmp = compare_rankings(f1_by, ebd_by)
    print("\n" + cmp.as_table())

    # Sensitivity: is the divergence one episode's leverage over the ranking,
    # or does it survive leaving any single episode out? (Sec. 7.8)
    sensitivity = []
    for excl in held:
        ebd_excl = {}
        for m in MODELS:
            vals = [o["actual_pct"] for o in per_episode
                    if o["model"] == m and o["episode_id"] != excl]
            ebd_excl[m] = sum(vals) / len(vals)
        c = compare_rankings(f1_by, ebd_excl)
        sensitivity.append(dict(excluded_episode=excl, tau=c.kendall_tau,
                                 p_value=c.p_value, diverges=c.diverges))
    print("\nleave-one-episode-out tau sensitivity:")
    for s in sensitivity:
        print(f"  exclude {s['excluded_episode'][:30]:<32} "
              f"tau={s['tau']:+.3f}  diverges={s['diverges']}")

    root = _project_root()
    (root / "results").mkdir(exist_ok=True)
    res = root / "results"
    with open(res / f"real_multimodel_summary{suffix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    with open(res / f"real_multimodel_per_episode{suffix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_episode[0].keys()))
        w.writeheader(); w.writerows(per_episode)
    with open(res / f"real_multimodel_maxscores{suffix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(maxscores[0].keys()))
        w.writeheader(); w.writerows(maxscores)
    with open(res / f"real_multimodel_h1{suffix}.txt", "w") as fh:
        fh.write(cmp.as_table() + "\n")
    with open(res / f"real_multimodel_sensitivity{suffix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sensitivity[0].keys()))
        w.writeheader(); w.writerows(sensitivity)
    sweep_out = args.sweep_out or (res / f"real_multimodel_sweep{suffix}.csv")
    with open(sweep_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
        w.writeheader(); w.writerows(sweep)
    print(f"\nwrote results/real_multimodel_{{summary,per_episode,maxscores,sensitivity,"
          f"sweep}}{suffix}.csv, results/real_multimodel_h1{suffix}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
