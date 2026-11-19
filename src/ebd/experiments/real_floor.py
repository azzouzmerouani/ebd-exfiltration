"""Phase C/E: the architectural EBD floor on real exfiltration episodes.

    uv run ebd-real

Takes the PCAP-reconstructed ground truth produced by the Phase B driver and
reports, per episode, the floor that bounds **any** detector which consumes
completed flow records — computed from quantities measured in the data rather
than assumed:

* ``D``  — the episode's span, from its reconstructed byte timeline;
* ``T``  — ``min(active_timeout, first connection's duration)``, because a flow
  record is emitted when its connection closes, not only when the exporter's
  active timeout fires (:mod:`ebd.realdata.cadence_probe`);
* ``V_e`` — de-duplicated application-layer payload, the asset itself, verified
  against the dataset's own byte counts.

Scope, stated up front
----------------------
DAPT2021's exfiltration is **low-and-slow**: the whole six-week campaign moves
~16 MB, and the largest single episode is 3.98 MB spread over 37 hours. With D
in hours and T in seconds, ``min(1, T/D)`` is near zero. So this validates that
the floor formula holds on real traffic **in the drip regime**, and it measures
how much of the asset a real incumbent lets go — but it cannot exercise the
blind spot, which needs ``D <= T``. No public dataset we found contains fast
bulk exfiltration; that absence is reported as a finding rather than papered
over.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_groundtruth(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EBD floor on real episodes")
    parser.add_argument("--groundtruth", type=Path, required=True,
                        help="JSONL produced by the Phase B driver")
    parser.add_argument("--max-divergence", type=float, default=0.05,
                        help="reject episodes whose reconciliation exceeds this")
    parser.add_argument("--active-timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)

    recs = load_groundtruth(args.groundtruth)
    print("=" * 78)
    print("  REAL-DATA EBD FLOOR — DAPT2021 / Unraveled")
    print("  Drip regime only: this validates the formula and measures the")
    print("  incumbent's exposure; it cannot exercise the D <= T blind spot.")
    print("=" * 78)

    print(f"\n{'episode':<34}{'V_e (MB)':>10}{'D (s)':>10}{'recon':>8}{'admit':>7}")
    print("-" * 78)
    admitted, rejected = [], []
    for r in recs:
        link = r["wire_bytes"] + 14 * r["n_all_packets"]
        div = abs(link - r["flow_bytes"]) / max(r["flow_bytes"], 1)
        dur = r["t_end"] - r["t_start"]
        ok = div <= args.max_divergence
        (admitted if ok else rejected).append((r, div, dur))
        print(f"{r['episode_id'][:33]:<34}{r['pcap_bytes']/1e6:>10.3f}{dur:>10.0f}"
              f"{div*100:>7.1f}%{'  yes' if ok else '   NO':>7}")

    print("-" * 78)
    print(f"admitted {len(admitted)} / {len(recs)} episodes"
          f"  (reconciliation within {args.max_divergence:.0%})")
    if rejected:
        print("REJECTED (excluded from every figure below):")
        for r, div, _ in rejected:
            print(f"   {r['episode_id']}  divergence={div*100:.1f}%")

    root = _project_root()
    (root / "results").mkdir(exist_ok=True)
    out = root / "results" / "real_floor.csv"
    import csv

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode_id", "asset_bytes", "duration_s", "reconciliation",
                    "internal_dst", "payload_fraction", "n_packets"])
        for r, div, dur in admitted:
            link = r["wire_bytes"] + 14 * r["n_all_packets"]
            w.writerow([r["episode_id"], r["pcap_bytes"], f"{dur:.1f}", f"{div:.5f}",
                        r["destination_is_internal"],
                        f"{r['pcap_bytes']/max(link,1):.4f}", r["n_packets"]])
    print(f"\nwrote {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
