"""Sec. 7.9: a session-granularity detector, built and run on the seven episodes.

    uv run ebd-session --train data/realdata/train_flows.csv \
                       --groundtruth data/realdata/groundtruth.jsonl

The bound of Sec. 4 holds for every record consumer with ``T`` read as the
emission interval of the record it consumes. Sec. 7.9 measures the floor at
session granularity from the sessioniser's own emission times; this driver adds
the detector, so the section reports a classifier that was trained and run, not
only a bound computed from timestamps.

Construction
------------
* **Model held constant.** The predictor is EDXGB's XGBoost (learning rate
  0.01, depth 10, 30 estimators, ``scale_pos_weight`` for the imbalance), run
  single-threaded so reruns on one machine are bit-identical. Only the unit of
  inference changes, which is the variable of interest.
* **Session records.** Flows are grouped by the episode key
  (source IP, destination IP, destination port) and split on silences longer
  than the session idle timeout ``S``. Following NEDS's described inputs, a
  record carries the per-column mean of its flows' EDXGB features, plus the
  number of flows, the mean time between their starts, the session span, and the
  summed byte and packet counts in each direction.
* **Emission time.** A record is credited to the end of the last flow it
  summarises. A real idle-gap aggregator must also wait out ``S`` before it can
  know the session is over, so every number here is a lower bound on the
  family's exposure.
* **Split.** Strict protocol of Sec. 7.5: every session record containing a flow
  of the seven reconstructed episodes is held out; benign traffic is split by
  the same benign session key and seed as the flow-level detector, so the two
  are held to the same benign data.
* **Operating point.** :mod:`ebd.realdata.operating_point`: the most aggressive
  grid threshold whose upper 95% Wilson bound on training benign session records
  meets the 1% budget; the burden is then measured on held-out benign records.

Outputs ``results/session_detector.csv`` (one row per ``S``),
``results/session_detector_episodes.csv`` and ``figures/session_detector.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..realdata.operating_point import choose_threshold, measure_burden

DEFAULT_SESSION_GAPS = (60.0, 300.0, 600.0, 1800.0)
BENIGN_STAGES = ("Benign", "Normal", "")
EXFIL = "Data Exfiltration"
SUM_COLUMNS = ("src2dst_bytes", "dst2src_bytes", "bidirectional_bytes",
               "src2dst_packets", "dst2src_packets", "bidirectional_packets")
EXTRA_NAMES = ("n_flows", "mean_interflow_s", "span_s") + tuple(f"sum_{c}" for c in SUM_COLUMNS)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def sessionize_rows(keys: list, starts: np.ndarray, ends: np.ndarray,
                    gap_seconds: float) -> np.ndarray:
    """Session id per row: same key, split on silences longer than ``gap_seconds``.

    The silence is measured from the running end of the session, as in
    :func:`ebd.realdata.session.sessionize` and the episode grouping, so a long
    flow that is still open does not let a later flow start a new session.
    Rows with an invalid timestamp get id -1.
    """
    sid = np.full(len(keys), -1, dtype=np.int64)
    by_key: dict = defaultdict(list)
    for i, k in enumerate(keys):
        if np.isfinite(starts[i]) and np.isfinite(ends[i]):
            by_key[k].append(i)
    nxt = 0
    for k in sorted(by_key, key=lambda x: tuple(map(str, x))):
        idx = sorted(by_key[k], key=lambda i: (starts[i], ends[i]))
        running_end = None
        for i in idx:
            if running_end is None or starts[i] - running_end > gap_seconds:
                cur = nxt
                nxt += 1
                running_end = ends[i]
            else:
                running_end = max(running_end, ends[i])
            sid[i] = cur
    return sid


def session_features(X: np.ndarray, sid: np.ndarray, starts: np.ndarray,
                     ends: np.ndarray, sums: np.ndarray):
    """Aggregate per-flow features into one record per session id (>= 0).

    Returns (session ids in order, feature matrix, first start, last end,
    number of flows).
    """
    valid = sid >= 0
    order = np.argsort(sid[valid], kind="stable")
    rows = np.nonzero(valid)[0][order]
    s_sorted = sid[rows]
    uniq, first = np.unique(s_sorted, return_index=True)
    counts = np.diff(np.append(first, len(rows)))
    means = np.add.reduceat(X[rows], first, axis=0) / counts[:, None]
    tot = np.add.reduceat(sums[rows], first, axis=0)
    st, en = starts[rows], ends[rows]
    t_first = np.minimum.reduceat(st, first)
    t_last = np.maximum.reduceat(en, first)
    mean_iat = np.zeros(len(uniq))
    for j, (a, n) in enumerate(zip(first, counts)):
        if n > 1:
            s = np.sort(st[a:a + n])
            mean_iat[j] = float(np.mean(np.diff(s)))
    extra = np.column_stack([counts, mean_iat, t_last - t_first, tot])
    feats = np.nan_to_num(np.hstack([means, extra]).astype(np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0)
    return uniq, feats, t_first, t_last, counts


def _fraction_by(g: dict, t: float) -> float:
    total = max(int(g["pcap_bytes"]), 1)
    idx = int((t - g["bin_start"]) // g["bin_seconds"])
    if idx < 0:
        return 0.0
    return min(1.0, float(sum(g["bins"][: idx + 1])) / total)


def _xgb(seed: int, scale_pos_weight: float):
    from xgboost import XGBClassifier
    return XGBClassifier(
        learning_rate=0.01, max_depth=10, n_estimators=30,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=seed, n_jobs=1, tree_method="hist",
    )


def main(argv: list[str] | None = None) -> int:
    from sklearn.metrics import roc_auc_score

    from ..realdata import FlowRecord, group_into_episodes
    from ..realdata.edxgb import build_matrix, feature_columns
    from ..realdata.session import byte_floor_session

    root = _project_root()
    ap = argparse.ArgumentParser(description="Session-granularity detector vs its floor")
    ap.add_argument("--train", type=Path, default=root / "data/realdata/train_flows.csv")
    ap.add_argument("--groundtruth", type=Path, default=root / "data/realdata/groundtruth.jsonl")
    ap.add_argument("--gaps", type=float, nargs="+", default=list(DEFAULT_SESSION_GAPS))
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--fpb-budget", type=float, default=0.01)
    ap.add_argument("--benign-holdout", type=float, default=0.5)
    ap.add_argument("--rule", choices=("wilson", "point"), default="wilson")
    ap.add_argument("--out", type=Path, default=root / "results/session_detector.csv")
    ap.add_argument("--episodes-out", type=Path,
                    default=root / "results/session_detector_episodes.csv")
    ap.add_argument("--figure", type=Path, default=root / "figures/session_detector.png")
    args = ap.parse_args(argv)

    csv.field_size_limit(10_000_000)
    rows = list(csv.DictReader(open(args.train)))
    gt = {json.loads(l)["episode_id"]: json.loads(l)
          for l in open(args.groundtruth) if l.strip()}
    print(f"flow rows: {len(rows)}   reconstructed episodes: {len(gt)}")

    # --- the seven episodes, identified exactly as in Sec. 7.5 -------------
    exf_idx = [i for i, r in enumerate(rows) if r.get("Stage", "").strip() == EXFIL]
    frecs, row_of = [], {}
    for i in exf_idx:
        r = rows[i]
        try:
            f = FlowRecord(
                src_ip=r["src_ip"], src_port=int(r["src_port"]), dst_ip=r["dst_ip"],
                dst_port=int(r["dst_port"]), protocol=int(r["protocol"]),
                start=float(r["bidirectional_first_seen_ms"]) / 1000,
                duration=float(r["bidirectional_duration_ms"]) / 1000,
                fwd_bytes=int(r["src2dst_bytes"]), bwd_bytes=int(r["dst2src_bytes"]),
                fwd_packets=int(r["src2dst_packets"]), bwd_packets=int(r["dst2src_packets"]),
                stage=r["Stage"], activity=r["Activity"], source_file=r.get("source_file", ""),
            )
        except (KeyError, ValueError):
            continue
        frecs.append(f)
        row_of[id(f)] = i
    episodes = {e.episode_id: e for e in group_into_episodes(frecs) if e.episode_id in gt}
    print(f"held-out episodes present in both: {len(episodes)}")
    ep_rows = {eid: [row_of[id(f)] for f in e.flows if id(f) in row_of]
               for eid, e in episodes.items()}
    row_episode = np.full(len(rows), -1, dtype=np.int64)
    ep_ids = sorted(episodes)
    for j, eid in enumerate(ep_ids):
        row_episode[ep_rows[eid]] = j

    # --- per-flow arrays ---------------------------------------------------
    header = list(rows[0].keys())
    cols = feature_columns(header)
    X = build_matrix(rows, cols)
    starts = np.array([_float(r.get("bidirectional_first_seen_ms"), np.nan) for r in rows]) / 1000.0
    durs = np.array([_float(r.get("bidirectional_duration_ms"), np.nan) for r in rows]) / 1000.0
    ends = starts + np.maximum(durs, 0.0)
    sums = np.array([[_float(r.get(c), 0.0) or 0.0 for c in SUM_COLUMNS] for r in rows])
    stage = np.array([r.get("Stage", "").strip() for r in rows])
    is_exf = stage == EXFIL
    is_benign = np.isin(stage, BENIGN_STAGES)
    keys = [(r.get("src_ip"), r.get("dst_ip"), r.get("dst_port")) for r in rows]

    # benign split by key, exactly as real_edxgb.py / real_multimodel.py
    bkey: dict = {}
    for i in np.nonzero(is_benign)[0]:
        bkey.setdefault(keys[i], []).append(i)
    bkeys = list(bkey)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(bkeys))
    cut = int(len(bkeys) * (1.0 - args.benign_holdout))
    heldout_keys = {bkeys[i] for i in order[cut:]}
    row_heldout_benign_key = np.array([is_benign[i] and keys[i] in heldout_keys
                                       for i in range(len(rows))])

    summary, per_episode = [], []
    for S in args.gaps:
        sid = sessionize_rows(keys, starts, ends, S)
        uniq, F, t_first, t_last, nflows = session_features(X, sid, starts, ends, sums)
        pos_of = {int(s): j for j, s in enumerate(uniq)}
        n_sess = len(uniq)

        def per_session_any(mask):
            out = np.zeros(n_sess, dtype=bool)
            for i in np.nonzero(mask & (sid >= 0))[0]:
                out[pos_of[int(sid[i])]] = True
            return out

        def per_session_all(mask):
            bad = per_session_any(~mask)
            return ~bad

        s_exf = per_session_any(is_exf)
        s_benign = per_session_all(is_benign)
        s_heldout_episode = per_session_any(row_episode >= 0)
        s_heldout_benign = per_session_any(row_heldout_benign_key) & s_benign

        train_mask = ~s_heldout_episode & ~s_heldout_benign
        y = s_exf.astype(int)
        Xtr, ytr = F[train_mask], y[train_mask]
        pos = max(int(ytr.sum()), 1)
        neg = max(len(ytr) - int(ytr.sum()), 1)
        model = _xgb(args.seed, neg / pos)
        model.fit(Xtr, ytr)
        scores = model.predict_proba(F)[:, 1]

        train_benign = train_mask & s_benign
        op = choose_threshold(scores[train_benign], args.fpb_budget, rule=args.rule)
        thr = op["threshold"]
        burden = measure_burden(scores[s_heldout_benign], thr)
        auc_mask = s_heldout_episode | s_heldout_benign
        try:
            auc = float(roc_auc_score(s_heldout_episode[auc_mask].astype(int),
                                      scores[auc_mask]))
        except ValueError:
            auc = float("nan")

        ep_out = []
        for j, eid in enumerate(ep_ids):
            g = gt[eid]
            e = episodes[eid]
            sess = sorted({pos_of[int(sid[i])] for i in ep_rows[eid] if sid[i] >= 0})
            emit = [float(t_last[s]) for s in sess]
            t_emit = min(emit)
            floor = _fraction_by(g, t_emit)
            floor_ep_only = byte_floor_session(e, g["bins"], g["bin_start"],
                                               g["bin_seconds"], g["pcap_bytes"], S)
            if floor + 1e-12 < floor_ep_only:
                raise AssertionError(
                    f"{eid}: detector-session floor {floor:.4f} below the "
                    f"episode-only floor {floor_ep_only:.4f}")
            alerts = [float(t_last[s]) for s in sess if scores[s] >= thr]
            if alerts:
                t_alert = min(alerts)
                ebd = _fraction_by(g, t_alert)
            else:
                t_alert, ebd = None, 1.0
            if ebd + 1e-12 < floor:
                raise AssertionError(f"{eid}: EBD {ebd:.4f} below its floor {floor:.4f}")
            row = dict(
                gap_s=S, episode_id=eid, n_session_records=len(sess),
                first_emission_s=round(t_emit - g["t_start"], 3),
                floor=round(floor, 6), floor_episode_only=round(floor_ep_only, 6),
                fired=bool(alerts),
                alert_s=round(t_alert - g["t_start"], 3) if alerts else "",
                ebd=round(ebd, 6), gap=round(ebd - floor, 6),
                max_score=round(float(max(scores[s] for s in sess)), 6),
            )
            ep_out.append(row)
            per_episode.append(row)

        n_ep = len(ep_out)
        rec = dict(
            gap_s=S, rule=op["rule"], budget=args.fpb_budget,
            n_session_records=n_sess,
            n_train=int(train_mask.sum()), n_train_pos=int(ytr.sum()),
            threshold=thr, admissible=op["admissible"],
            train_benign_units=op["train_n"], train_alerts=op["train_k"],
            train_fpb=round(op["train_fpb"], 6), train_upper=round(op["train_upper"], 6),
            heldout_benign_units=burden["heldout_n"], heldout_alerts=burden["heldout_k"],
            heldout_fpb=round(burden["heldout_fpb"], 6),
            heldout_lo=round(burden["heldout_lo"], 6), heldout_hi=round(burden["heldout_hi"], 6),
            heldout_auc=round(auc, 4),
            heldout_episode_sessions=int(s_heldout_episode.sum()),
            episodes=n_ep,
            floor_total_loss=sum(o["floor"] >= 0.999 for o in ep_out),
            ebd_total_loss=sum(o["ebd"] >= 0.999 for o in ep_out),
            fired=sum(o["fired"] for o in ep_out),
            fired_at_floor=sum(o["fired"] and o["gap"] <= 0.01 for o in ep_out),
            mean_floor=round(float(np.mean([o["floor"] for o in ep_out])), 6),
            mean_ebd=round(float(np.mean([o["ebd"] for o in ep_out])), 6),
        )
        summary.append(rec)
        print(f"S={S:>6.0f}s  records={n_sess:>6}  thr={thr:.2f} "
              f"(train FPB {op['train_fpb']:.3%}, upper {op['train_upper']:.3%})  "
              f"held-out FPB {burden['heldout_fpb']:.3%} on {burden['heldout_n']}  "
              f"AUC {auc:.3f}  floor=1 {rec['floor_total_loss']}/{n_ep}  "
              f"EBD=1 {rec['ebd_total_loss']}/{n_ep}  fired {rec['fired']}/{n_ep}")

    for path, data in ((args.out, summary), (args.episodes_out, per_episode)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"wrote {path}")
    _plot(summary, args.figure)
    print(f"wrote {args.figure}")
    return 0


def _plot(summary, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gaps = [r["gap_s"] for r in summary]
    n = summary[0]["episodes"] if summary else 7
    x = np.arange(len(gaps))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x - w / 2, [r["floor_total_loss"] for r in summary], w,
           color="#3b4a6b", label="architectural: episode over before\nits first session record (floor = 100%)")
    ax.bar(x + w / 2, [r["fired"] for r in summary], w,
           color="#c8792a", label="detector quality: episodes on which\nthe classifier fires at all")
    for xi, r in zip(x, summary):
        ax.text(xi + w / 2, r["fired"] + 0.12, f"AUC {r['heldout_auc']:.2f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S = {g:.0f} s" for g in gaps])
    ax.set_ylim(0, n + 1.6)
    ax.set_ylabel(f"episodes (of {n})")
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
