"""Escaping the architectural floor, on a background built to make it hard.

    uv run ebd-escape

Compares two detectors on identical traffic at an identical false-positive
budget:

* a **record consumer** -- the reproduced EDXGB feature pipeline over nfstream
  records, which Proposition 1 bounds;
* an **incremental detector** -- three counters and a threshold, scoring on
  every packet, which Proposition 1 does not bound because it never waits for a
  record.

The background is the one that makes this a real test: it runs concurrently and
contains long-lived low-rate benign sessions (the hard negative for a
persistence rule) as well as short high-rate bulk (the hard negative for a
volume-and-rate rule). See ``ebd.testbed.run_escape_capture``.

Writes ``results/escape.csv`` and ``figures/escape.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..realdata.edxgb import build_matrix, derive_rates
from ..testbed.export import DEFAULT_ACTIVE_TIMEOUTS, export
from ..testbed.incremental import parse_pcap, replay
from ..testbed.reconstruct import build_timeline
from .fast_regime import _alerts_for, _fit_operating_point, _train_loo, attribute


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    root = _project_root()
    ap = argparse.ArgumentParser(description="Escaping the floor")
    ap.add_argument("--testbed", type=Path, default=root / "data/testbed_escape")
    ap.add_argument("--fpb-budget", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--out", type=Path, default=root / "results/escape.csv")
    ap.add_argument("--figure", type=Path, default=root / "figures/escape.png")
    ap.add_argument("--rule", choices=("wilson", "point"), default="point",
                    help="operating-point rule for BOTH detectors: the record "
                         "consumer's threshold and the incremental setting's "
                         "feasibility (ebd.realdata.operating_point)")
    ap.add_argument("--sweep-out", type=Path, default=None,
                    help="every incremental setting with its burden and EBD, so "
                         "the best feasible one can be read against the whole "
                         "feasible set (DA-3)")
    args = ap.parse_args(argv)
    from ..realdata.operating_point import wilson_interval

    manifest = json.loads((args.testbed / "manifest.json").read_text())
    pcap = args.testbed / "capture.pcap"
    transfers = manifest["transfers"]
    dst_port = manifest["sink_port"]
    attacks = [i for i, t in enumerate(transfers) if t["kind"] == "attack"]
    benign = [i for i, t in enumerate(transfers) if t["kind"] == "benign"]
    longs = [i for i in benign
             if transfers[i]["t_end"] - transfers[i]["t_start"] >= 20.0]

    print("=" * 80)
    print("  ESCAPING THE ARCHITECTURAL FLOOR")
    print(f"  capture {pcap.stat().st_size / 1e6:.1f} MB: {len(attacks)} attacks, "
          f"{len(benign)} benign sessions")
    print(f"  of the benign sessions, {len(longs)} last >= 20 s "
          f"(longest {max(transfers[i]['t_end'] - transfers[i]['t_start'] for i in benign):.0f} s)")
    print("=" * 80)

    # ---- ground truth ----------------------------------------------------
    timelines = {}
    for i in attacks:
        t = transfers[i]
        tl = build_timeline(str(pcap), {int(p) for p in t["src_ports"]}, dst_port,
                            t_lo=t["t_start"] - 2.0, t_hi=t["t_end"] + 5.0)
        recon = abs(tl.total_payload - t["asset_bytes"]) / max(t["asset_bytes"], 1)
        assert recon < 0.01, f"{t['label']}: reconciliation {recon:.2%}"
        timelines[i] = tl
    print(f"\n  all {len(attacks)} episodes reconcile exactly to the planned asset")

    # ---- the record consumer, per active timeout -------------------------
    rec: dict[float, dict[int, tuple[float, float]]] = {}
    rec_burden: dict[float, dict] = {}
    for ta in DEFAULT_ACTIVE_TIMEOUTS:
        recs = export(pcap, ta)
        att = attribute(recs, transfers)
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
        perm = rng.permutation(benign)
        b_test = set(perm[: len(perm) // 2].tolist())
        b_train = set(perm[len(perm) // 2:].tolist())
        thr, fpb, _m, op, burden, _ps = _fit_operating_point(
            idx_rows, cols, y, owner, attacks, b_train, b_test, args)
        rec_burden[ta] = {"threshold": thr, **op, **burden}
        print(f"  record consumer, Ta={int(ta):>3}s: threshold={thr:.3f} "
              f"({op['rule']}; train {op['train_fpb']:.2%}, upper "
              f"{op['train_upper']:.2%}), held-out session FPB={fpb:.2%} "
              f"[{burden['heldout_lo']:.2%}, {burden['heldout_hi']:.2%}] "
              f"on {burden['heldout_n']}")

        per = {}
        for i in attacks:
            rs = att[i]
            if not rs:
                continue
            floor = timelines[i].fraction_by(min(r.last_seen for r in rs))
            m = _train_loo(idx_rows, cols, y, owner, i, b_test)
            sub = [j for j, o in enumerate(idx_owner) if o == i]
            sc = m.predict_proba(build_matrix([idx_rows[j] for j in sub], cols))[:, 1]
            hit = [rs[k].last_seen for k, s in enumerate(sc) if s >= thr]
            per[i] = (floor, timelines[i].fraction_by(min(hit)) if hit else 1.0)
        rec[ta] = per

    # ---- the incremental detector ---------------------------------------
    grid = [(b, r, a)
            for b in (32_000, 64_000, 128_000, 256_000, 512_000)
            for r in (0.0, 5_000.0, 20_000.0, 60_000.0, 200_000.0)
            for a in (0.0, 2.0, 5.0, 10.0, 20.0)]
    packets = parse_pcap(str(pcap), dst_port)
    print(f"\n  sweeping {len(grid)} incremental settings over "
          f"{len(packets):,} decoded packets")
    best = None
    sweep = []
    for min_bytes, min_rate, min_age in grid:
        alerts = replay(packets, min_bytes, min_rate, min_age)
        fp = sum(bool(_alerts_for(alerts, transfers[i])) for i in benign)
        n_b = max(len(benign), 1)
        fpb = fp / n_b
        upper = wilson_interval(fp, n_b)[1]
        feasible = (upper <= args.fpb_budget) if args.rule == "wilson" \
            else (fpb <= args.fpb_budget)
        ebds, below = [], {ta: 0 for ta in DEFAULT_ACTIVE_TIMEOUTS}
        for i in attacks:
            hits = _alerts_for(alerts, transfers[i])
            e = timelines[i].fraction_by(min(hits)) if hits else 1.0
            ebds.append(e)
            for ta in DEFAULT_ACTIVE_TIMEOUTS:
                f = rec[ta].get(i, (float("nan"),))[0]
                if e < f - 1e-9:
                    below[ta] += 1
        mean = float(np.mean(ebds))
        sweep.append({"B_min": min_bytes, "R_min": min_rate, "A_min": min_age,
                      "benign_alerted": fp, "benign_sessions": n_b,
                      "session_fpb": round(fpb, 6), "session_fpb_upper": round(upper, 6),
                      "feasible_point": fpb <= args.fpb_budget,
                      "feasible_wilson": upper <= args.fpb_budget,
                      "mean_ebd": round(mean, 6),
                      **{f"below_floor_Ta{int(ta)}": below[ta]
                         for ta in DEFAULT_ACTIVE_TIMEOUTS}})
        if not feasible:
            continue
        if best is None or mean < best[0]:
            best = (mean, min_bytes, min_rate, min_age, fpb, alerts)

    feas_key = "feasible_wilson" if args.rule == "wilson" else "feasible_point"
    feas = [s for s in sweep if s[feas_key]]
    print(f"\n  feasible settings under the {args.rule} rule: {len(feas)} of {len(sweep)}")
    if feas:
        means = sorted(s["mean_ebd"] for s in feas)
        print(f"    mean EBD over the feasible set: best {means[0]:.1%}, "
              f"median {float(np.median(means)):.1%}, worst {means[-1]:.1%}")
        for ta in DEFAULT_ACTIVE_TIMEOUTS:
            b = [s[f"below_floor_Ta{int(ta)}"] for s in feas]
            print(f"    transfers below the Ta={int(ta)}s floor: median "
                  f"{float(np.median(b)):g}, best {max(b)}, worst {min(b)} of {len(attacks)}")
    if args.sweep_out is not None:
        args.sweep_out.parent.mkdir(parents=True, exist_ok=True)
        with args.sweep_out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
            w.writeheader()
            w.writerows(sweep)
        print(f"    wrote {args.sweep_out}")

    if best is None:
        print("\n  no incremental setting met the FPB budget -- no escape claim")
        return 1
    mean, min_bytes, min_rate, min_age, fpb, alerts = best
    fp_long = sum(bool(_alerts_for(alerts, transfers[i])) for i in longs)
    print(f"\n  incremental: B_min={min_bytes:,}  R_min={min_rate:,.0f}/s  "
          f"A_min={min_age}s")
    print(f"    session FPB={fpb:.2%} over all benign; "
          f"{fp_long}/{len(longs)} of the long-lived benign sessions alerted")

    # ---- report ----------------------------------------------------------
    print(f"\n  {'transfer':<22}{'D (s)':>7}" +
          "".join(f"{'floor Ta=' + str(int(t)):>14}" for t in DEFAULT_ACTIVE_TIMEOUTS) +
          f"{'incremental':>13}")
    print("  " + "-" * (29 + 14 * len(DEFAULT_ACTIVE_TIMEOUTS) + 13))
    rows = []
    for i in sorted(attacks, key=lambda i: transfers[i]["t_end"] - transfers[i]["t_start"]):
        t = transfers[i]
        d = t["t_end"] - t["t_start"]
        hits = _alerts_for(alerts, t)
        inc = timelines[i].fraction_by(min(hits)) if hits else 1.0
        cells = "".join(f"{rec[ta].get(i, (float('nan'),))[0] * 100:>13.1f}%"
                        for ta in DEFAULT_ACTIVE_TIMEOUTS)
        print(f"  {t['label']:<22}{d:>7.1f}{cells}{inc * 100:>12.1f}%")
        row = {"label": t["label"], "duration_s": round(d, 2),
               "asset_bytes": t["asset_bytes"], "incremental_ebd": round(inc, 6),
               "incremental_fired": bool(hits),
               "B_min": min_bytes, "R_min": min_rate, "A_min": min_age,
               "incremental_session_fpb": round(fpb, 5)}
        for ta in DEFAULT_ACTIVE_TIMEOUTS:
            f, e = rec[ta].get(i, (float("nan"), float("nan")))
            row[f"floor_Ta{int(ta)}"] = round(f, 6)
            row[f"record_ebd_Ta{int(ta)}"] = round(e, 6)
        for ta in DEFAULT_ACTIVE_TIMEOUTS:
            b = rec_burden[ta]
            row[f"record_threshold_Ta{int(ta)}"] = b["threshold"]
            row[f"record_fpb_Ta{int(ta)}"] = round(b["heldout_fpb"], 6)
            row[f"record_fpb_lo_Ta{int(ta)}"] = round(b["heldout_lo"], 6)
            row[f"record_fpb_hi_Ta{int(ta)}"] = round(b["heldout_hi"], 6)
        row["rule"] = args.rule
        rows.append(row)

    # headline: how often does the incremental detector beat the floor?
    for ta in DEFAULT_ACTIVE_TIMEOUTS:
        below = sum(1 for r in rows
                    if r["incremental_ebd"] < r[f"floor_Ta{int(ta)}"] - 1e-9)
        mf = np.mean([r[f"floor_Ta{int(ta)}"] for r in rows])
        print(f"\n  Ta={int(ta):>3}s: mean floor {mf:6.1%}, mean incremental EBD "
              f"{np.mean([r['incremental_ebd'] for r in rows]):6.1%}"
              f"  -> below the floor on {below}/{len(rows)} transfers")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {args.out}")
    _plot(rows, args.figure)
    print(f"  wrote {args.figure}")
    return 0


def _plot(rows, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = sorted(rows, key=lambda r: r["duration_s"])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    cmap = plt.get_cmap("viridis")
    tas = [5, 15, 60]
    w = 0.8 / (len(tas) + 1)
    for j, ta in enumerate(tas):
        ax.bar(x - 0.4 + w * (j + 0.5),
               [r[f"floor_Ta{ta}"] * 100 for r in rows], w,
               color=cmap(j / (len(tas) - 1)), label=f"floor, $T_a$={ta}s")
    ax.bar(x - 0.4 + w * (len(tas) + 0.5),
           [r["incremental_ebd"] * 100 for r in rows], w,
           color="crimson", label="incremental detector")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['duration_s']:.0f}s" for r in rows])
    ax.set_xlabel("transfer duration $D$")
    ax.set_ylabel("% of asset gone when the alert fires")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("A detector outside the family, at the same false-positive burden\n"
                 "record-consumer floors vs an incremental trigger", fontsize=10)
    ax.grid(axis="y", alpha=0.3, lw=0.4, which="both")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
