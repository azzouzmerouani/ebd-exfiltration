"""Loss at containment, V_e(a_e + R)/V_e, for assumed response latencies R.

    python -m ebd.experiments.response_sensitivity

R is not measured in any corpus used in the paper; the values are assumed and
labelled as such.  For each of the seven reconstructed DAPT2021 episodes the
loss is read off the packet-level byte timeline at (i) the floor instant
t_start + T_eff and (ii) the first alert of the reproduced detector under the
strict protocol (results/real_edxgb_detail_strict_wilson.json).  An episode on
which the detector never fires is lost in full whatever R is.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

R_VALUES = (0.0, 300.0, 3600.0, 14400.0)


def frac_at(rec: dict, t: float) -> float:
    idx = int((t - rec["bin_start"]) // rec["bin_seconds"])
    if idx < 0:
        return 0.0
    bins = rec["bins"]
    return min(1.0, sum(bins[: min(idx + 1, len(bins))]) / rec["pcap_bytes"])


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    gt = {json.loads(l)["episode_id"]: json.loads(l)
          for l in (root / "data/realdata/groundtruth.jsonl").read_text().splitlines() if l.strip()}
    det = json.load(open(root / "results/real_edxgb_detail_strict_wilson.json"))["episodes"]
    out = []
    for eid, d in det.items():
        r = gt[eid]
        t0 = d["t_start"]
        for R in R_VALUES:
            floor = frac_at(r, t0 + d["first_record_s"] + R)
            loss = 1.0 if d["alert_s"] is None else frac_at(r, t0 + d["alert_s"] + R)
            out.append(dict(episode_id=eid, duration_h=round((d["t_end"] - t0) / 3600, 2),
                            R_s=int(R), floor_plus_R=round(floor, 6), detector_plus_R=round(loss, 6)))
    path = root / "results/response_sensitivity.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    for R in R_VALUES:
        sel = [o for o in out if o["R_s"] == int(R)]
        mf = sum(o["floor_plus_R"] for o in sel) / len(sel)
        md = sum(o["detector_plus_R"] for o in sel) / len(sel)
        print(f"R={int(R):6d}s  mean floor+R={mf*100:6.2f}%  mean detector+R={md*100:6.2f}%  "
              + "  ".join(f"{o['detector_plus_R']*100:5.1f}" for o in sel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
