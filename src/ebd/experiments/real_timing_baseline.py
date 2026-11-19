"""Timing-only baselines and a burden-matched random null for Sec. 7 (round 3).

    python -m ebd.experiments.real_timing_baseline     # no xgboost needed, ~2 min

Answers the round-3 reviewers (R1-M5, DA-3): "at the floor" could be reached by
any rule that fires on the first record it sees within the burden budget, and
the permuted-label null of ``real_baseline`` never fires, so it cannot show what
chance attainment looks like. Under exactly the split and operating-point rule
of ``real_baseline`` (all seven episodes held out, benign split by host-service
key, upper-Wilson rule at a 1% budget), this driver scores:

* ``bytes_out``  -- a model-free rule: a record's score is its outbound bytes
  (``src2dst_bytes``); the threshold is the smallest admissible byte count;
* ``bytes_total`` -- the same on bytes in both directions;
* ``random_k``   -- each record gets an independent U(0,1) score; the threshold
  is chosen by the same rule on a fine grid, so the null fires at (close to) the
  same burden as a real detector would. ``--draws`` repetitions.

Writes ``results/real_timing_baseline.csv`` and
``results/real_timing_baseline_episodes.csv`` in the layout of ``real_baseline``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ebd.experiments.real_baseline import auc, hanley_mcneil


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    from ebd.realdata import FlowRecord, group_into_episodes
    from ebd.realdata.cadence_probe import byte_floor
    from ebd.realdata.edxgb import first_alert_time
    from ebd.realdata.operating_point import choose_threshold, measure_burden

    root = _root()
    ap = argparse.ArgumentParser(description="Timing-only baselines and a random null")
    ap.add_argument("--train", type=Path, default=root / "data/realdata/train_flows.csv")
    ap.add_argument("--groundtruth", type=Path, default=root / "data/realdata/groundtruth.jsonl")
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--fpb-budget", type=float, default=0.01)
    ap.add_argument("--at-floor-pp", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=root / "results/real_timing_baseline.csv")
    ap.add_argument("--episodes-out", type=Path,
                    default=root / "results/real_timing_baseline_episodes.csv")
    args = ap.parse_args(argv)

    csv.field_size_limit(10_000_000)
    rows = list(csv.DictReader(open(args.train)))
    gt = {json.loads(l)["episode_id"]: json.loads(l)
          for l in open(args.groundtruth) if l.strip()}

    # ---- identical split to real_baseline / real_edxgb (strict protocol) -----
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
    print(f"held-out benign keys {len(test_sessions)}; training benign keys "
          f"{len(train_sessions)}; episodes {len(held)}")

    def flat(sessions):
        rws, owner = [], []
        for j, s in enumerate(sessions):
            rws.extend(s)
            owner.extend([j] * len(s))
        return rws, np.asarray(owner)

    tr_rows, tr_owner = flat(train_sessions)
    te_rows, te_owner = flat(test_sessions)

    def key_max(scores, owner, n):
        m = np.full(n, -np.inf)
        np.maximum.at(m, owner, scores)
        return m

    def col(rws, f):
        return np.array([float(f(r)) for r in rws])

    out_b = lambda r: int(r["src2dst_bytes"])
    tot_b = lambda r: int(r["src2dst_bytes"]) + int(r["dst2src_bytes"])
    byte_grid = np.unique(np.round(np.logspace(1, 10, 361)))
    unit_grid = np.round(np.linspace(0.0, 1.0, 10001), 4)

    variants = [("bytes_out", "rule", out_b, byte_grid),
                ("bytes_total", "rule", tot_b, byte_grid)]
    for k in range(args.draws):
        variants.append((f"random_{k:02d}", "random", None, unit_grid))

    rng = np.random.default_rng(args.seed + 7)
    summary, per_ep = [], []
    for name, kind, f, grid in variants:
        if kind == "rule":
            sc_tr, sc_te = col(tr_rows, f), col(te_rows, f)
            sc_ep = {eid: col(rws, f) for eid, rws in held_rows.items()}
        else:
            sc_tr, sc_te = rng.random(len(tr_rows)), rng.random(len(te_rows))
            sc_ep = {eid: rng.random(len(rws)) for eid, rws in held_rows.items()}
        tr_max = key_max(sc_tr, tr_owner, len(train_sessions))
        te_max = key_max(sc_te, te_owner, len(test_sessions))
        op = choose_threshold(tr_max, args.fpb_budget, rule="wilson", grid=grid)
        thr = op["threshold"]
        burden = measure_burden(te_max, thr)

        outs, ep_max, ep_all = [], [], []
        for eid, e in sorted(held.items()):
            g = gt[eid]
            rws, sc = held_rows[eid], sc_ep[eid]
            ep_max.append(float(sc.max()) if len(sc) else -np.inf)
            ep_all.append(sc)
            floor = byte_floor(e, g["bin_start"], g["bin_seconds"], g["bins"], g["pcap_bytes"])
            t = first_alert_time(rws, sc, thr) if len(sc) else None
            if t is None:
                ebd = 1.0
            else:
                idx = int((t - g["bin_start"]) // g["bin_seconds"])
                ebd = min(1.0, sum(g["bins"][: max(idx + 1, 0)]) / max(g["pcap_bytes"], 1)) \
                    if idx >= 0 else 0.0
            assert ebd >= floor - 1e-9, (name, eid, ebd, floor)
            outs.append(dict(variant=name, episode_id=eid, n_records=len(rws),
                             floor=round(floor, 6), ebd=round(ebd, 6),
                             gap=round(ebd - floor, 6), fired=t is not None,
                             at_floor=(ebd - floor) <= args.at_floor_pp / 100))
        per_ep.extend(outs)
        a_unit = auc(np.array(ep_max), te_max)
        lo, hi = hanley_mcneil(a_unit, len(ep_max), len(te_max))
        ebds = [o["ebd"] for o in outs]
        row = dict(variant=name, kind=kind, threshold=thr,
                   admissible=bool(op["admissible"]) and burden["heldout_fpb"] <= args.fpb_budget,
                   rule_met_in_training=op["admissible"],
                   train_upper=round(op["train_upper"], 6),
                   heldout_fpb=round(burden["heldout_fpb"], 6),
                   heldout_lo=round(burden["heldout_lo"], 6),
                   heldout_hi=round(burden["heldout_hi"], 6),
                   heldout_keys=burden["heldout_n"],
                   unit_auc=round(a_unit, 4), unit_auc_lo=round(lo, 4),
                   unit_auc_hi=round(hi, 4),
                   flow_auc=round(auc(np.concatenate(ep_all), sc_te), 4),
                   fired=sum(o["fired"] for o in outs),
                   at_floor=sum(o["at_floor"] for o in outs),
                   mean_ebd_pct=round(100 * float(np.mean(ebds)), 4),
                   median_ebd_pct=round(100 * float(np.median(ebds)), 4))
        summary.append(row)
        print(f"{name:<12} thr {thr:<12g} FPB {row['heldout_fpb']:.2%} "
              f"unitAUC {a_unit:.3f} fired {row['fired']}/7 at-floor {row['at_floor']}/7 "
              f"mean {row['mean_ebd_pct']:.2f}% median {row['median_ebd_pct']:.2f}%")

    rnd = [r for r in summary if r["kind"] == "random"]
    if rnd:
        print(f"\nrandom null ({len(rnd)}): fired {sorted(r['fired'] for r in rnd)}; "
              f"at-floor {sorted(r['at_floor'] for r in rnd)}; held-out FPB "
              f"{min(r['heldout_fpb'] for r in rnd):.2%}-{max(r['heldout_fpb'] for r in rnd):.2%}; "
              f"unit AUC median {np.median([r['unit_auc'] for r in rnd]):.3f}")

    for path, data in ((args.out, summary), (args.episodes_out, per_ep)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
