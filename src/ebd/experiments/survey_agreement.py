"""Agreement between the two codings of the survey sample, and the frame by stratum.

    uv run ebd-survey-agreement

Inputs (all in ``data/survey/``):

``coded.csv``          the first coding (unchanged since revision 5).
``second_coding.csv``  a second coding of all 39 papers, made blind to the first
                       from bibliographic identifiers and the written codebook only
                       (see ``protocol.md``, "Second coding (revision 6)").
``frame.csv``          each paper's channel (DNS or other) and scope
                       (exfiltration detector / general intrusion detection /
                       survey), from the second coder's inclusion decision.

Outputs: ``results/survey_agreement.csv`` (Cohen's kappa with bootstrap
intervals, raw agreement, the confusion of units) and ``results/survey_strata.csv``
(record consumers per stratum, under each coding, with Wilson intervals).

Agreement is computed over the papers for which BOTH coders could determine the
unit of inference; papers the second coder could not determine are counted and
reported, never imputed.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from ebd.realdata.operating_point import wilson_interval

RECORD_UNITS = {"flow_record", "session_aggregate", "host_timeseries"}
NON_RECORD_UNITS = {"packet", "payload_dpi"}
UNDETERMINED = "undetermined"
SEED = 20261009
N_BOOT = 10_000


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's (1960) kappa for two raters over the same items.

    Returns 1.0 when both raters use a single identical category throughout
    (observed = expected = 1), the conventional value for perfect agreement.
    """
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two equal-length, non-empty rating lists")
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / (n * n)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def bootstrap_kappa(a: list[str], b: list[str], n_boot: int = N_BOOT,
                    seed: int = SEED) -> tuple[float, float]:
    """Percentile 95% interval for kappa, resampling papers with replacement."""
    rng = np.random.default_rng(seed)
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    n = len(a_arr)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(cohen_kappa(list(a_arr[idx]), list(b_arr[idx])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def is_rc(unit: str) -> str:
    if unit in RECORD_UNITS:
        return "yes"
    if unit in NON_RECORD_UNITS:
        return "no"
    return UNDETERMINED


def main(argv: list[str] | None = None) -> int:
    root = _root()
    ap = argparse.ArgumentParser(description="Survey: inter-coder agreement and strata")
    d = root / "data/survey"
    ap.add_argument("--first", type=Path, default=d / "coded.csv")
    ap.add_argument("--second", type=Path, default=d / "second_coding.csv")
    ap.add_argument("--frame", type=Path, default=d / "frame.csv")
    ap.add_argument("--out", type=Path, default=root / "results/survey_agreement.csv")
    ap.add_argument("--strata-out", type=Path, default=root / "results/survey_strata.csv")
    args = ap.parse_args(argv)

    first = {r["key"]: r for r in csv.DictReader(args.first.open())}
    second = {r["key"]: r for r in csv.DictReader(args.second.open())}
    frame = {r["key"]: r for r in csv.DictReader(args.frame.open())}
    assert set(first) == set(second) == set(frame), "the three files must list the same papers"
    keys = list(first)

    both = [k for k in keys if second[k]["unit_of_inference"] != UNDETERMINED]
    undet = [k for k in keys if k not in both]
    u1 = [first[k]["unit_of_inference"] for k in both]
    u2 = [second[k]["unit_of_inference"] for k in both]
    r1 = [is_rc(u) for u in u1]
    r2 = [is_rc(u) for u in u2]

    k_unit, k_rc = cohen_kappa(u1, u2), cohen_kappa(r1, r2)
    ci_unit, ci_rc = bootstrap_kappa(u1, u2), bootstrap_kappa(r1, r2)
    agree_unit = sum(x == y for x, y in zip(u1, u2))
    agree_rc = sum(x == y for x, y in zip(r1, r2))

    # Cadence: only where both coders call the paper a record consumer.
    rc_both = [k for k, a, b in zip(both, r1, r2) if a == b == "yes"]
    c1 = [first[k]["cadence_stated"] for k in rc_both]
    c2 = [second[k]["cadence_stated"] for k in rc_both]
    k_cad = cohen_kappa(c1, c2)
    ci_cad = bootstrap_kappa(c1, c2)

    rows = [
        ("papers", len(keys), "", "", len(keys)),
        ("second_coder_undetermined", len(undet), "", "", len(keys)),
        ("unit_agreement", agree_unit, "", "", len(both)),
        ("unit_kappa", round(k_unit, 4), round(ci_unit[0], 4), round(ci_unit[1], 4), len(both)),
        ("record_consumer_agreement", agree_rc, "", "", len(both)),
        ("record_consumer_kappa", round(k_rc, 4), round(ci_rc[0], 4), round(ci_rc[1], 4), len(both)),
        ("cadence_kappa_among_rc", round(k_cad, 4), round(ci_cad[0], 4), round(ci_cad[1], 4), len(rc_both)),
        ("cadence_first_yes_or_partial", sum(c in ("yes", "partial") for c in c1), "", "", len(rc_both)),
        ("cadence_second_yes_or_partial", sum(c in ("yes", "partial") for c in c2), "", "", len(rc_both)),
        ("cadence_second_yes", sum(c == "yes" for c in c2), "", "", len(rc_both)),
        ("second_read_fulltext", sum(second[k]["source_level"] == "abstract_and_fulltext" for k in keys), "", "", len(keys)),
        ("second_include_yes", sum(second[k]["include"] == "yes" for k in keys), "", "", len(keys)),
    ]
    # Round 3 (C2): kappa on the in-rule exfiltration detectors, overall and per
    # channel. The pooled kappa above includes 17 out-of-rule papers that both
    # coders agree on trivially, so the paper reports these instead.
    for label, chans in (("exfiltration", ("dns", "other")), ("exfiltration_dns", ("dns",)),
                         ("exfiltration_other", ("other",))):
        ks = [k for k in both if frame[k]["scope"] == "exfiltration" and frame[k]["channel"] in chans]
        a = [is_rc(first[k]["unit_of_inference"]) for k in ks]
        b = [is_rc(second[k]["unit_of_inference"]) for k in ks]
        agree = sum(x == y for x, y in zip(a, b))
        if len(set(a) | set(b)) > 1:
            kk, ci = cohen_kappa(a, b), bootstrap_kappa(a, b)
            rows.append((f"record_consumer_kappa_{label}", round(kk, 4), round(ci[0], 4),
                         round(ci[1], 4), len(ks)))
        rows.append((f"record_consumer_agreement_{label}", agree, "", "", len(ks)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value", "ci_lo", "ci_hi", "n"])
        w.writerows(rows)

    # Strata: record consumers under each coding, surveys excluded.
    # Round 3 (C2): the in-rule sample is the exfiltration detectors; the
    # general-IDS papers are reported separately and are not pooled into "all".
    strata = [("exfiltration", "dns"), ("exfiltration", "other"),
              ("exfiltration", "any"), ("general_nids", "other")]
    srows = []

    def members(scope, channel):
        if channel == "any":
            return [k for k in keys if frame[k]["scope"] == scope]
        return [k for k in keys if frame[k]["scope"] == scope
                and frame[k]["channel"] == channel]

    for scope, channel in strata:
        ks = members(scope, channel)
        f_rc = sum(first[k]["record_consumer"] == "yes" for k in ks)
        det2 = [k for k in ks if second[k]["unit_of_inference"] != UNDETERMINED]
        s_rc = sum(is_rc(second[k]["unit_of_inference"]) == "yes" for k in det2)
        lo1, hi1 = wilson_interval(f_rc, len(ks))
        lo2, hi2 = wilson_interval(s_rc, len(det2)) if det2 else (float("nan"),) * 2
        srows.append(dict(scope=scope, channel=channel, n=len(ks),
                          first_rc=f_rc, first_lo=round(lo1, 4), first_hi=round(hi1, 4),
                          second_determined=len(det2), second_rc=s_rc,
                          second_lo=round(lo2, 4), second_hi=round(hi2, 4)))
    with args.strata_out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
        w.writeheader()
        w.writerows(srows)

    print("=" * 78)
    print("  SURVEY: AUTOMATED SECOND CODING AGAINST THE FIRST")
    print("=" * 78)
    print(f"  papers {len(keys)}; second coder could not determine the unit for "
          f"{len(undet)}: {', '.join(undet)}")
    print(f"  unit of inference : {agree_unit}/{len(both)} agree, kappa {k_unit:.3f} "
          f"[{ci_unit[0]:.3f}, {ci_unit[1]:.3f}]")
    print(f"  record consumer   : {agree_rc}/{len(both)} agree, kappa {k_rc:.3f} "
          f"[{ci_rc[0]:.3f}, {ci_rc[1]:.3f}]")
    print(f"  cadence (both RC) : kappa {k_cad:.3f} [{ci_cad[0]:.3f}, {ci_cad[1]:.3f}] "
          f"on {len(rc_both)}; yes/partial first {rows[7][1]}, second {rows[8][1]}")
    print("\n  unit disagreements:")
    for k, a, b in zip(both, u1, u2):
        if a != b:
            flag = "  <- record-consumer status differs" if is_rc(a) != is_rc(b) else ""
            print(f"    {k:<30} {a:<18} -> {b}{flag}")
    print("\n  strata (surveys excluded):")
    for s in srows:
        print(f"    {s['scope']:<14} {s['channel']:<6} n={s['n']:>2}  first {s['first_rc']}/{s['n']} "
              f"[{s['first_lo']:.3f}, {s['first_hi']:.3f}]   second {s['second_rc']}/"
              f"{s['second_determined']} [{s['second_lo']:.3f}, {s['second_hi']:.3f}]")
    print(f"\n  wrote {args.out} and {args.strata_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
