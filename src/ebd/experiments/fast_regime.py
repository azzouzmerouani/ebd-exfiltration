"""The D <= T regime, measured on packets.

    uv run ebd-fast

Every public exfiltration dataset we examined is low-and-slow, so the regime the
bound warns about -- a transfer that finishes inside one emission interval, and
is therefore lost in full before any record consumer can act -- cannot be
exercised on any of them. This driver exercises it on a purpose-built capture.

Three questions, one capture
----------------------------
1. **Does the floor reach 1 when D <= T?**  Measured, not assumed: the floor is
   read off the exporter's own first record for each transfer.
2. **Does chunking help at flow granularity?**  The paper argues it does. Each
   duration is run as one connection and as eight, on the same asset.
3. **Is the floor escapable?**  A packet-driven incremental detector is measured
   on the same traffic at the same false-positive burden.

Outputs ``results/fast_regime.csv``, ``results/fast_regime_escape.csv``, and
``figures/fast_regime.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..realdata.edxgb import build_matrix, derive_rates
from ..realdata.operating_point import choose_threshold, measure_burden
from ..testbed.export import DEFAULT_ACTIVE_TIMEOUTS, export, export_provenance
from ..testbed.incremental import run_over_pcap
from ..testbed.reconstruct import build_timeline


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# Attribution                                                                  #
# --------------------------------------------------------------------------- #

def attribute(records, transfers) -> dict[int, list]:
    """Map exported records to the transfer that produced them.

    Source ports are reused over a long run, so a port alone is ambiguous. The
    key is (port, time): a record belongs to the transfer that used that port
    and whose wall-clock window contains the record's first packet. A small
    tolerance absorbs the gap between the client's clock and the capture's.
    """
    by_port: dict[int, list[tuple[float, float, int]]] = {}
    for i, t in enumerate(transfers):
        for p in t["src_ports"]:
            by_port.setdefault(int(p), []).append(
                (t["t_start"] - 2.0, t["t_end"] + 5.0, i)
            )

    out: dict[int, list] = {i: [] for i in range(len(transfers))}
    unmatched = 0
    for r in records:
        cands = by_port.get(r.src_port, [])
        hit = next((i for lo, hi, i in cands if lo <= r.first_seen <= hi), None)
        if hit is None:
            unmatched += 1
            continue
        out[hit].append(r)
    if unmatched:
        print(f"  ! {unmatched} records could not be attributed (ignored)")
    return out


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    root = _project_root()
    ap = argparse.ArgumentParser(description="The D<=T regime on packets")
    ap.add_argument("--testbed", type=Path, default=root / "data/testbed")
    ap.add_argument("--fpb-budget", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--out", type=Path, default=root / "results/fast_regime.csv")
    ap.add_argument("--escape-out", type=Path,
                    default=root / "results/fast_regime_escape.csv")
    ap.add_argument("--figure", type=Path, default=root / "figures/fast_regime.png")
    ap.add_argument("--rule", choices=("wilson", "point"), default="point",
                    help="operating-point rule (ebd.realdata.operating_point); "
                         "'point' reproduces the archived run, 'wilson' is the "
                         "a-priori rule of the revised paper (M2)")
    ap.add_argument("--meta-out", type=Path, default=None,
                    help="JSON with per-timeout burdens, held-out AUC (M1) and the "
                         "asset-invariance deviations (M4)")
    ap.add_argument("--skip-escape", action="store_true")
    ap.add_argument("--exclude-unreconciled", action="store_true",
                    help="exclude, and list in the metadata, any attack transfer whose "
                         "reconstructed payload is off its planned asset by 1%% or more, "
                         "instead of stopping. Used only for the netem re-recording, whose "
                         "unprivileged capture loses packets at the highest rates.")
    args = ap.parse_args(argv)

    manifest = json.loads((args.testbed / "manifest.json").read_text())
    pcap = args.testbed / "capture.pcap"
    transfers = manifest["transfers"]
    dst_port = manifest["sink_port"]
    attacks = [i for i, t in enumerate(transfers) if t["kind"] == "attack"]
    benign = [i for i, t in enumerate(transfers) if t["kind"] == "benign"]

    print("=" * 82)
    print("  THE D <= T REGIME, MEASURED ON PACKETS")
    print(f"  capture: {pcap.stat().st_size / 1e6:.1f} MB, "
          f"{len(attacks)} attack + {len(benign)} benign transfers")
    print("=" * 82)

    # ---- ground truth ----------------------------------------------------
    print("\n[1/4] reconstructing byte timelines (receiver-side, de-duplicated payload)")
    timelines, unreconciled = {}, []
    for i in attacks:
        t = transfers[i]
        tl = build_timeline(
            str(pcap), {int(p) for p in t["src_ports"]}, dst_port,
            t_lo=t["t_start"] - 2.0, t_hi=t["t_end"] + 5.0,
        )
        recon = abs(tl.total_payload - t["asset_bytes"]) / max(t["asset_bytes"], 1)
        if recon >= 0.01:
            msg = (f"{t['label']}: reconstructed {tl.total_payload} vs planned "
                   f"{t['asset_bytes']} ({recon:.2%}) -- ground truth is not trustworthy")
            if not args.exclude_unreconciled:
                raise AssertionError(msg)
            print(f"      EXCLUDED {msg}")
            unreconciled.append({"label": t["label"], "index": i,
                                 "reconstructed_bytes": tl.total_payload,
                                 "planned_bytes": t["asset_bytes"],
                                 "relative_error": recon})
            continue
        timelines[i] = tl
    excluded_idx = {u["index"] for u in unreconciled}
    attacks = [i for i in attacks if i not in excluded_idx]
    worst = max(abs(timelines[i].total_payload - transfers[i]["asset_bytes"])
                / transfers[i]["asset_bytes"] for i in attacks)
    print(f"      all {len(attacks)} episodes reconcile to within {worst:.4%} "
          f"of the planned asset")

    # ---- per-timeout analysis -------------------------------------------
    rows = []
    meta = {"testbed": str(args.testbed), "rule": args.rule,
            "budget": args.fpb_budget, "conditions": manifest.get("conditions", ""),
            "excluded_labels": manifest.get("excluded_labels", []),
            "reconciliation_worst": worst, "per_timeout": {},
            "timeline_bin_seconds": next(iter(timelines.values())).bin_seconds,
            "unreconciled_excluded": unreconciled}
    for ta in DEFAULT_ACTIVE_TIMEOUTS:
        print(f"\n[2/4] exporting at active_timeout = {int(ta)}s")
        recs = export(pcap, ta)
        att = {i: rs for i, rs in attribute(recs, transfers).items()
               if i not in excluded_idx}
        n_att_recs = sum(len(att[i]) for i in attacks)
        print(f"      {len(recs)} records; {n_att_recs} attributed to attacks")

        # Labels and features over every attributed record.
        idx_rows, idx_owner = [], []
        for i, rs in att.items():
            for r in rs:
                d = dict(r.fields)
                d.update(derive_rates(d))
                idx_rows.append(d)
                idx_owner.append(i)
        cols = [c for c in idx_rows[0] if c not in
                {"src_ip", "dst_ip", "bidirectional_first_seen_ms",
                 "bidirectional_last_seen_ms", "PTR_I", "PTR_O", "BTR_I", "BTR_O"}]
        y = np.array([1 if transfers[o]["kind"] == "attack" else 0
                      for o in idx_owner])
        owner = np.array(idx_owner)

        rng = np.random.default_rng(args.seed)
        b_perm = rng.permutation(benign)
        b_test = set(b_perm[: len(b_perm) // 2].tolist())
        b_train = set(b_perm[len(b_perm) // 2:].tolist())

        thr, fpb, model_all, op, burden, per_session = _fit_operating_point(
            idx_rows, cols, y, owner, attacks, b_train, b_test, args)
        print(f"      operating point ({op['rule']}): threshold={thr:.3f}  "
              f"train FPB={op['train_fpb']:.2%} (upper {op['train_upper']:.2%})  "
              f"held-out session FPB={fpb:.2%} "
              f"[{burden['heldout_lo']:.2%}, {burden['heldout_hi']:.2%}] "
              f"on {burden['heldout_n']} sessions")
        loo_attack_max: dict[int, float] = {}
        loo_attack_scores: list[float] = []

        for i in attacks:
            t, tl = transfers[i], timelines[i]
            rs = att[i]
            if not rs:
                continue
            # Floor: the exporter's own earliest record for this transfer.
            t_first = min(r.last_seen for r in rs)
            floor = tl.fraction_by(t_first)

            # Detector: leave-one-transfer-out, so the scored transfer is unseen.
            m = _train_loo(idx_rows, cols, y, owner, i, b_test)
            sub = [j for j, o in enumerate(idx_owner) if o == i]
            X = build_matrix([idx_rows[j] for j in sub], cols)
            sc = m.predict_proba(X)[:, 1]
            loo_attack_max[i] = float(sc.max()) if len(sc) else float("-inf")
            loo_attack_scores.extend(float(s) for s in sc)
            hit = [rs[k].last_seen for k, s in enumerate(sc) if s >= thr]
            ebd = tl.fraction_by(min(hit)) if hit else 1.0

            rows.append({
                "label": t["label"],
                "active_timeout_s": ta,
                "asset_bytes": t["asset_bytes"],
                "planned_duration_s": t["duration_s"],
                "actual_duration_s": round(t["t_end"] - t["t_start"], 3),
                "n_connections": t["n_connections"],
                "n_records": len(rs),
                "T_over_D": round(ta / max(t["t_end"] - t["t_start"], 1e-9), 5),
                "t_first_record_s": round(t_first - t["t_start"], 3),
                "t_alert_s": round(min(hit) - t["t_start"], 3) if hit else "",
                # how long after the transfer's own end its first record appeared
                # (negative while the transfer is still running)
                "first_record_after_end_s": round(t_first - t["t_end"], 3),
                "floor": round(floor, 6),
                "ebd": round(ebd, 6),
                "gap": round(ebd - floor, 6),
                "fired": bool(hit),
                "threshold": round(thr, 4),
                "session_fpb_heldout": round(fpb, 5),
                "rule": op["rule"],
                "session_fpb_train": round(op["train_fpb"], 5),
                "session_fpb_train_upper": round(op["train_upper"], 5),
                "session_fpb_heldout_lo": round(burden["heldout_lo"], 5),
                "session_fpb_heldout_hi": round(burden["heldout_hi"], 5),
                "heldout_benign_sessions": burden["heldout_n"],
            })

        # M1: how separable is this testbed? Held-out AUC, with each attack
        # transfer scored by the model that never saw it and each held-out
        # benign session by the operating model (which never saw held-out
        # benign). Transfer/session level (loudest record) and record level.
        auc_session, auc_record = _heldout_auc(
            loo_attack_max, loo_attack_scores, per_session, b_test,
            model_all, idx_rows, cols, idx_owner)
        meta["per_timeout"][str(int(ta))] = {
            "threshold": thr, "admissible": op["admissible"],
            "train_sessions": op["train_n"], "train_fpb": op["train_fpb"],
            "train_upper": op["train_upper"],
            "heldout_sessions": burden["heldout_n"], "heldout_alerts": burden["heldout_k"],
            "heldout_fpb": burden["heldout_fpb"], "heldout_lo": burden["heldout_lo"],
            "heldout_hi": burden["heldout_hi"],
            "heldout_auc_session": auc_session, "heldout_auc_record": auc_record,
        }
        print(f"      held-out AUC: session level {auc_session:.4f}, "
              f"record level {auc_record:.4f}")

    meta["export_sources"] = export_provenance()
    meta["asset_invariance"] = _asset_invariance(rows)
    meta["single_connection_cells"] = sum(r["n_connections"] == 1 for r in rows)
    meta["closed_form_check"] = _closed_form_check(rows)

    # ---- escaping the floor ---------------------------------------------
    print("\n[3/4] the incremental detector (outside the record-consumer family)")
    escape = ([] if args.skip_escape else
              _escape(pcap, dst_port, transfers, attacks, benign, timelines, args))

    # ---- report ----------------------------------------------------------
    print("\n[4/4] results\n")
    _report(rows, escape)

    for path, data in ((args.out, rows), (args.escape_out, escape)):
        if not data:
            print(f"  (no rows for {path.name}; skipped)")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"  wrote {path}")

    if args.meta_out is not None:
        args.meta_out.parent.mkdir(parents=True, exist_ok=True)
        args.meta_out.write_text(json.dumps(meta, indent=1, default=float))
        print(f"  wrote {args.meta_out}")

    _plot(rows, escape, args.figure)
    print(f"  wrote {args.figure}")
    return 0


def _heldout_auc(loo_attack_max, loo_attack_scores, per_session, b_test,
                 model, rows, cols, owner):
    from sklearn.metrics import roc_auc_score
    neg_s = [per_session[s] for s in b_test if s in per_session]
    pos_s = list(loo_attack_max.values())
    try:
        auc_s = float(roc_auc_score([1] * len(pos_s) + [0] * len(neg_s), pos_s + neg_s))
    except ValueError:
        auc_s = float("nan")
    sel = [j for j, o in enumerate(owner) if o in b_test]
    neg_r = (model.predict_proba(build_matrix([rows[j] for j in sel], cols))[:, 1].tolist()
             if sel else [])
    try:
        auc_r = float(roc_auc_score([1] * len(loo_attack_scores) + [0] * len(neg_r),
                                    list(loo_attack_scores) + neg_r))
    except ValueError:
        auc_r = float("nan")
    return auc_s, auc_r


def _asset_invariance(rows) -> dict:
    """M4: the 10x-asset repeats against their same-duration counterparts."""
    pairs = []
    for r in rows:
        if r["n_connections"] != 1 or r["asset_bytes"] != 20 * 1024 * 1024:
            continue
        twin = next((q for q in rows if q["n_connections"] == 1
                     and q["asset_bytes"] == 2 * 1024 * 1024
                     and q["planned_duration_s"] == r["planned_duration_s"]
                     and q["active_timeout_s"] == r["active_timeout_s"]), None)
        if twin is not None:
            pairs.append({"active_timeout_s": r["active_timeout_s"],
                          "planned_duration_s": r["planned_duration_s"],
                          "floor_20M": r["floor"], "floor_2M": twin["floor"],
                          "abs_diff_pp": abs(r["floor"] - twin["floor"]) * 100})
    diffs = [p["abs_diff_pp"] for p in pairs]
    return {"pairs": pairs, "n_pairs": len(pairs),
            "max_abs_diff_pp": max(diffs) if diffs else None,
            "exact_agreement": sum(d < 1e-9 for d in diffs)}


def _closed_form_check(rows) -> dict:
    """The measured floor against the closed form min(1, T/D), single connections."""
    ex = []
    for r in rows:
        if r["n_connections"] != 1:
            continue
        closed = min(1.0, r["active_timeout_s"] / max(r["actual_duration_s"], 1e-9))
        ex.append((r["floor"] - closed) * 100)
    return {"n": len(ex), "violations": sum(e < -1e-9 for e in ex),
            "mean_excess_pp": float(np.mean(ex)) if ex else None,
            "max_excess_pp": float(np.max(ex)) if ex else None}


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def _xgb(seed: int):
    from xgboost import XGBClassifier
    return XGBClassifier(
        learning_rate=0.01, max_depth=10, n_estimators=30,
        eval_metric="logloss", random_state=seed, n_jobs=4, tree_method="hist",
    )


def _train_loo(rows, cols, y, owner, exclude_owner: int, b_test: set):
    """Model trained on everything except the scored transfer and held-out benign."""
    keep = np.array([o != exclude_owner and o not in b_test for o in owner])
    X = build_matrix([rows[j] for j in np.nonzero(keep)[0]], cols)
    yy = y[keep]
    pos, neg = max(int(yy.sum()), 1), max(len(yy) - int(yy.sum()), 1)
    m = _xgb(20260815)
    m.set_params(scale_pos_weight=neg / pos)
    m.fit(X, yy)
    return m


def _fit_operating_point(rows, cols, y, owner, attacks, b_train, b_test, args):
    """Choose the threshold on TRAIN benign; measure FPB on HELD-OUT benign.

    Selecting on train and reporting on held-out is what keeps the operating
    point honest: a threshold tuned on the same benign traffic it is then
    scored against would understate the burden.
    """
    keep = np.array([o not in b_test for o in owner])
    X = build_matrix([rows[j] for j in np.nonzero(keep)[0]], cols)
    yy = y[keep]
    pos, neg = max(int(yy.sum()), 1), max(len(yy) - int(yy.sum()), 1)
    m = _xgb(args.seed)
    m.set_params(scale_pos_weight=neg / pos)
    m.fit(X, yy)

    # Score every record once and reduce to a per-session maximum. A session
    # alerts iff its loudest record clears the threshold, so the whole FPB
    # sweep is then a comparison against this vector -- rather than one
    # predict_proba call per session per candidate threshold.
    all_scores = m.predict_proba(build_matrix(rows, cols))[:, 1]
    per_session: dict[int, float] = {}
    for s, o in zip(all_scores, owner):
        o = int(o)
        if s > per_session.get(o, -1.0):
            per_session[o] = float(s)

    # The rule of ebd.realdata.operating_point: with rule="point" this is the
    # archived behaviour (most aggressive grid value whose training burden is
    # within budget); with rule="wilson" the upper 95% Wilson bound must be.
    train_max = [per_session[s] for s in b_train if s in per_session]
    test_max = [per_session[s] for s in b_test if s in per_session]
    op = choose_threshold(train_max, args.fpb_budget, rule=args.rule)
    thr = op["threshold"]
    burden = measure_burden(test_max, thr)
    return thr, burden["heldout_fpb"], m, op, burden, per_session


def _alerts_for(alerts, transfer) -> list[float]:
    """Alert times belonging to ``transfer``, matched on (port, time window).

    Port alone is not enough: the kernel recycles ephemeral ports over a run
    this long, so the same rule used to attribute exported records is used here.
    """
    ports = {int(p) for p in transfer["src_ports"]}
    lo, hi = transfer["t_start"] - 2.0, transfer["t_end"] + 5.0
    return [a.time for a in alerts if a.src_port in ports and lo <= a.time <= hi]


def _escape(pcap, dst_port, transfers, attacks, benign, timelines, args):
    """Sweep the incremental detector and report it at the same FPB budget.

    The setting reported is the one minimising mean EBD *subject to* the same
    false-positive budget the record consumer was held to. Choosing it on the
    same traffic it is then scored on flatters the incremental detector, and we
    state that rather than hide it. The comparison survives the concession
    because the quantity it is being compared against -- the record consumer's
    architectural floor -- is not fitted at all: no choice of knobs, model or
    threshold moves it.
    """
    grid = [(b, r, a)
            for b in (64_000, 256_000, 1_000_000, 1_800_000)
            for r in (20_000, 50_000, 200_000, 800_000)
            for a in (0.0, 1.0, 2.5, 4.0, 8.0)]
    best = None
    for min_bytes, min_rate, min_age in grid:
        alerts = run_over_pcap(str(pcap), dst_port, min_bytes, min_rate, min_age)
        fp = sum(bool(_alerts_for(alerts, transfers[i])) for i in benign)
        fpb = fp / max(len(benign), 1)
        if fpb > args.fpb_budget:
            continue
        ebds = []
        for i in attacks:
            hits = _alerts_for(alerts, transfers[i])
            ebds.append(timelines[i].fraction_by(min(hits)) if hits else 1.0)
        mean = float(np.mean(ebds))
        if best is None or mean < best[0]:
            best = (mean, min_bytes, min_rate, min_age, fpb, alerts)

    if best is None:
        print("      no incremental setting met the FPB budget")
        return []
    mean, min_bytes, min_rate, min_age, fpb, alerts = best
    print(f"      best setting: min_bytes={min_bytes:,}  min_rate={min_rate:,}/s"
          f"  min_age={min_age}s  -> FPB={fpb:.2%}, mean EBD={mean:.2%}")

    out = []
    for i in attacks:
        t, tl = transfers[i], timelines[i]
        hits = _alerts_for(alerts, t)
        out.append({
            "label": t["label"],
            "actual_duration_s": round(t["t_end"] - t["t_start"], 3),
            "n_connections": t["n_connections"],
            "asset_bytes": t["asset_bytes"],
            "incremental_ebd": round(tl.fraction_by(min(hits)) if hits else 1.0, 6),
            "fired": bool(hits),
            "min_bytes": min_bytes,
            "min_rate_bps": min_rate,
            "min_age_s": min_age,
            "session_fpb": round(fpb, 5),
        })
    return out


def _report(rows, escape):
    tas = sorted({r["active_timeout_s"] for r in rows})
    for k in (1, 8):
        print(f"  --- {k} connection{'s' if k > 1 else ''} per transfer "
              f"(asset 2 MiB) ---")
        print(f"  {'D (s)':>8}" + "".join(
            f"{'Ta=' + str(int(t)) + 's':>22}" for t in tas))
        print(f"  {'':>8}" + "".join(f"{'floor':>11}{'EBD':>11}" for _ in tas))
        sub = sorted([r for r in rows if r["n_connections"] == k
                      and r["asset_bytes"] == 2 * 1024 * 1024],
                     key=lambda r: r["actual_duration_s"])
        for d in sorted({r["actual_duration_s"] for r in sub}):
            cells = ""
            for t in tas:
                m = next((r for r in sub if r["actual_duration_s"] == d
                          and r["active_timeout_s"] == t), None)
                cells += (f"{m['floor'] * 100:>10.1f}%{m['ebd'] * 100:>10.1f}%"
                          if m else f"{'-':>11}{'-':>11}")
            print(f"  {d:>8.1f}{cells}")
        print()

    tot = [r for r in rows if r["floor"] >= 0.999]
    print(f"  TOTAL LOSS (floor = 100%): {len(tot)}/{len(rows)} transfer-timeout "
          f"combinations")
    dt = [r for r in rows if r["actual_duration_s"] <= r["active_timeout_s"]
          and r["n_connections"] == 1]
    n_tl = sum(r["floor"] >= 0.999 for r in dt)
    print(f"    of the {len(dt)} single-connection cases with D <= Ta, "
          f"{n_tl} are total losses")

    if escape:
        rec_mean = float(np.mean([r["ebd"] for r in rows]))
        inc_mean = float(np.mean([r["incremental_ebd"] for r in escape]))
        floor_mean = float(np.mean([r["floor"] for r in rows]))
        print(f"\n  ESCAPING THE FLOOR, at matched FPB:")
        print(f"    record consumer   mean EBD {rec_mean:6.2%}  "
              f"(its floor: {floor_mean:6.2%})")
        print(f"    incremental       mean EBD {inc_mean:6.2%}")
        print(f"    -> the incremental detector sits {floor_mean / max(inc_mean, 1e-9):.0f}x "
              f"BELOW the record-consumer floor")


def _plot(rows, escape, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    tas = sorted({r["active_timeout_s"] for r in rows})
    cmap = plt.get_cmap("viridis")
    A2 = 2 * 1024 * 1024

    for ax, k, title in ((axes[0], 1, "one connection"),
                         (axes[1], 8, "eight connections (T1030 chunking)")):
        for j, ta in enumerate(tas):
            sub = sorted([r for r in rows if r["n_connections"] == k
                          and r["active_timeout_s"] == ta
                          and r["asset_bytes"] == A2],
                         key=lambda r: r["actual_duration_s"])
            if not sub:
                continue
            x = [r["active_timeout_s"] / r["actual_duration_s"] for r in sub]
            c = cmap(j / max(len(tas) - 1, 1))
            ax.plot(x, [r["floor"] * 100 for r in sub], "-o", ms=4, lw=1.7,
                    color=c, label=f"floor, $T_a$={int(ta)}s")
            ax.plot(x, [r["ebd"] * 100 for r in sub], "--x", ms=5, lw=1.0,
                    color=c, alpha=0.75)
        ax.axvline(1.0, color="k", ls=":", lw=1.0)
        ax.text(1.06, 52, "$D = T$", fontsize=8, rotation=90, va="center")
        ax.set_xscale("log")
        ax.set_xlabel("$T/D$  (emission interval / transfer duration)")
        ax.set_ylim(-4, 108)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, lw=0.4)
        ax.legend(fontsize=6.5, loc="lower right")
    axes[0].set_ylabel("% of asset gone when the alert fires")

    # Third panel: the escape. The incremental detector has no T, so it is
    # plotted against transfer duration, with the record-consumer floors on the
    # same axis for comparison.
    ax = axes[2]
    for j, ta in enumerate(tas):
        sub = sorted([r for r in rows if r["n_connections"] == 1
                      and r["active_timeout_s"] == ta and r["asset_bytes"] == A2],
                     key=lambda r: r["actual_duration_s"])
        if sub:
            ax.plot([r["actual_duration_s"] for r in sub],
                    [r["floor"] * 100 for r in sub], "-o", ms=3.5, lw=1.4,
                    color=cmap(j / max(len(tas) - 1, 1)),
                    label=f"floor, $T_a$={int(ta)}s")
    if escape:
        e = sorted([r for r in escape if r["n_connections"] == 1
                    and r["asset_bytes"] == A2],
                   key=lambda r: r["actual_duration_s"])
        ax.plot([r["actual_duration_s"] for r in e],
                [r["incremental_ebd"] * 100 for r in e],
                "-s", ms=4, lw=2.3, color="crimson",
                label="incremental detector\n(outside the family)")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_xlabel("transfer duration $D$ (s)")
    ax.set_title("escaping the floor, at matched FPB", fontsize=10)
    ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.legend(fontsize=6.5, loc="upper right")

    fig.suptitle("Total loss when the transfer fits inside one emission interval\n"
                 "solid = architectural floor (measured from the exporter), "
                 "dashed = EDXGB-style record consumer", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
