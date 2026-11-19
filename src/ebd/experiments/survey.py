"""Summarise the coded survey of detector input granularity.

    uv run ebd-survey

Turns ``data/survey/coded.csv`` into the counts the paper quotes, and asserts
the internal consistency of the coding (a paper coded as a record consumer must
have a record-level unit of inference, and vice versa) so a coding slip cannot
silently become a published number.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

RECORD_UNITS = {"flow_record", "session_aggregate", "host_timeseries"}
NON_RECORD_UNITS = {"packet", "payload_dpi"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    root = _project_root()
    ap = argparse.ArgumentParser(description="Coded survey summary")
    ap.add_argument("--coded", type=Path, default=root / "data/survey/coded.csv")
    ap.add_argument("--out", type=Path, default=root / "results/survey_summary.csv")
    args = ap.parse_args(argv)

    rows = list(csv.DictReader(args.coded.open()))

    # Consistency: the two primary columns must not disagree.
    for r in rows:
        unit, is_rc = r["unit_of_inference"], r["record_consumer"] == "yes"
        assert (unit in RECORD_UNITS) == is_rc, (
            f"coding inconsistency on {r['key']}: unit={unit} "
            f"record_consumer={r['record_consumer']}"
        )
        if not is_rc:
            assert r["cadence_stated"] == "n_a", (
                f"{r['key']}: non-consumers must have cadence_stated=n_a"
            )

    n = len(rows)
    units = Counter(r["unit_of_inference"] for r in rows)
    n_rc = sum(r["record_consumer"] == "yes" for r in rows)
    tools = Counter(r["telemetry_tool"] for r in rows if r["record_consumer"] == "yes")
    cadence = Counter(r["cadence_stated"] for r in rows if r["record_consumer"] == "yes")
    years = [int(r["year"]) for r in rows]

    print("=" * 74)
    print("  CODED SURVEY: WHAT DO NETWORK EXFILTRATION DETECTORS CONSUME?")
    print(f"  n = {n} papers, {min(years)}-{max(years)}")
    print("  Keyword-sampled frame, single coder. See data/survey/protocol.md")
    print("=" * 74)

    print("\n  unit of inference")
    for u, c in units.most_common():
        mark = "  <- record consumer" if u in RECORD_UNITS else ""
        print(f"    {u:<20} {c:>3}  ({c / n:>5.1%}){mark}")

    print(f"\n  RECORD CONSUMERS: {n_rc}/{n} ({n_rc / n:.1%})")
    print(f"  non-consumers   : {n - n_rc}/{n} ({(n - n_rc) / n:.1%})")

    print("\n  telemetry pipeline, among record consumers")
    for t, c in tools.most_common():
        print(f"    {t:<26} {c:>3}")

    print("\n  is the telemetry cadence stated? (screening level -- see protocol)")
    for k in ("yes", "partial", "no"):
        c = cadence.get(k, 0)
        print(f"    {k:<10} {c:>3}/{n_rc}")
    named = [r["key"] for r in rows if r["cadence_stated"] in ("yes", "partial")]
    print(f"    the {len(named)} that do: {', '.join(named)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value", "denominator"])
        w.writerow(["papers_coded", n, n])
        w.writerow(["record_consumers", n_rc, n])
        for u, c in units.most_common():
            w.writerow([f"unit_{u}", c, n])
        for k in ("yes", "partial", "no"):
            w.writerow([f"cadence_{k}", cadence.get(k, 0), n_rc])
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
