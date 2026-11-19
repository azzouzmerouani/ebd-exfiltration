"""Floors under composed telemetry pipelines (rev13 review, R3-1, R3-2, R2-2).

    python -m ebd.experiments.pipeline_profiles

Each profile composes the delays between the end of the first record's flow (or,
for cloud flow logs, the first packet) and the moment a scheduled detection rule
can see that record:

  exporter release  -> collector or delivery -> SIEM ingestion -> rule schedule

Clock-driven stages (collector file rotation, cloud aggregation window, rule
period) are uniform over their phase; the floor is averaged over the phases and
also given at the worst phase. Fixed delays (ingestion, cloud delivery) are
assumptions, stated in PROFILES; nothing here is measured on a deployed SIEM.
No model is involved.

DAPT2021: packet-level byte timelines (groundtruth.jsonl), first record = first
flow to end; whether that flow was closed by FIN (all five port-443 episodes) or
not (the two port-3306 episodes) is taken from results/rev11_audit.json.
DNS corpora: constant-rate closed form min(1, delay / D) per transfer, where the
delay is counted from the first query; for the per-query log profile there is
no flow wait, and for the single-connection (DoT/DoH) profile the connection is
one flow released by the active timeout or at its end.

Writes results/pipeline_profiles.csv.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import statistics as st
from pathlib import Path

from ..realdata.episodes import group_into_episodes
from .cadence_sweep import load_flows

# name: (release of a FIN-closed TCP flow, release of a flow without FIN, rotation
#        or aggregation window W (uniform phase), fixed delivery + ingestion, rule
#        period P (uniform phase), window anchored at first packet?)
PROFILES = {
    # Cisco FNF: FIN/RST-closed flows released at teardown, inactive timeout 15 s;
    # nfcapd rotates files every 300 s; ingestion 60 s and a 5-min rule assumed.
    "netflow_onprem": dict(fin=0.0, nofin=15.0, window=300.0, fixed=60.0, period=300.0,
                           anchor="flow_end", udp=15.0),
    # Zeek conn.log: written at connection close, or after the TCP inactivity
    # timeout (5 min; UDP 1 min); streamed to the SIEM (60 s assumed), 5-min rule.
    "zeek_conn": dict(fin=0.0, nofin=300.0, window=0.0, fixed=60.0, period=300.0,
                      anchor="flow_end", udp=60.0),
    # Cloud flow logs, 10-min aggregation (AWS default): a record per window in
    # which packets were seen; delivery 300 s and a 15-min rule assumed.
    "cloud_600": dict(fin=0.0, nofin=0.0, window=600.0, fixed=300.0, period=900.0,
                      anchor="first_packet", udp=0.0),
    # Cloud flow logs, 1-min aggregation; delivery 300 s and a 5-min rule assumed.
    "cloud_60": dict(fin=0.0, nofin=0.0, window=60.0, fixed=300.0, period=300.0,
                     anchor="first_packet", udp=0.0),
    # Per-query DNS logs (resolver or Zeek dns.log): no flow wait; ingestion 60 s
    # and a 5-min rule assumed. Applies to the DNS corpora only.
    "dns_query_log": dict(fin=0.0, nofin=0.0, window=0.0, fixed=60.0, period=300.0,
                          anchor="first_packet", udp=0.0),
}
STEPS = 24  # phase grid per clock-driven stage


def _phases(window, period):
    ws = [(k + 0.5) / STEPS * window for k in range(STEPS)] if window else [0.0]
    ps = [(k + 0.5) / STEPS * period for k in range(STEPS)] if period else [0.0]
    return [a + b for a, b in itertools.product(ws, ps)], window + period


def _frac(g, t):
    idx = int(math.floor((t - g["bin_start"]) / g["bin_seconds"]))
    if idx < 0:
        return 0.0
    return min(1.0, sum(g["bins"][: idx + 1]) / max(g["pcap_bytes"], 1))


def _fkey(f):
    return (f.src_ip, f.src_port, f.dst_ip, f.dst_port, int(round(f.start * 1000)))


def _flow_flags(root: Path) -> dict:
    """FIN and RST packet counts of every exfiltration flow record (nfstream
    features of data/realdata/train_flows.csv), keyed like ``_fkey``."""
    out = {}
    with (root / "data/realdata/train_flows.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r.get("Stage", "").strip() != "Data Exfiltration":
                continue
            k = (r["src_ip"], int(float(r["src_port"])), r["dst_ip"], int(float(r["dst_port"])),
                 int(float(r["bidirectional_first_seen_ms"])))
            out[k] = (int(float(r["bidirectional_fin_packets"])), int(float(r["bidirectional_rst_packets"])))
    return out


def main(profiles: dict | None = None, out_name: str = "pipeline_profiles.csv") -> int:
    PROFILES_ = PROFILES if profiles is None else profiles
    root = Path(__file__).resolve().parents[3]
    gt = {}
    for line in (root / "data/realdata/groundtruth.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            gt[d["episode_id"]] = d
    eps = {e.episode_id: e for e in group_into_episodes(load_flows(root / "data/realdata/exfil_flows.csv"))
           if e.episode_id in gt}
    # rev11_audit: the first records of all five port-443 episodes contain FIN,
    # those of the two port-3306 episodes do not
    audit = json.loads((root / "results/rev11_audit.json").read_text())["first_record"]
    assert all((v["fin"] > 0) == k.startswith("443") for k, v in audit.items())
    flags = _flow_flags(root) if any(p.get("per_flow") for p in PROFILES_.values()) else {}
    rows = []
    for prof, p in PROFILES_.items():
        if prof == "dns_query_log" or p.get("dns_only"):
            continue
        for eid, e in eps.items():
            g = gt[eid]
            flows = sorted(e.flows, key=lambda f: f.start)
            if p["anchor"] == "flow_end" and p.get("per_flow"):
                # first record actually released: min over flows of end + release,
                # with the release chosen per flow from its own FIN/RST counts
                def _rel(f):
                    fin, rst = flags.get(_fkey(f), (0, 0))
                    return p["fin"] if fin else (p.get("rst", p["fin"]) if rst else p["nofin"])
                t0 = min(f.start + f.duration + _rel(f) for f in flows)
            elif p["anchor"] == "flow_end":
                rel = p["fin"] if e.dst_port == 443 else p["nofin"]
                t0 = min(f.start + f.duration for f in flows) + rel
            else:
                t0 = g["t_start"]
            offs, worst = _phases(p["window"], p["period"])
            v = [_frac(g, t0 + p["fixed"] + o) for o in offs]
            rows.append(dict(corpus="dapt2021", profile=prof, unit=eid, duration_s=g["t_end"] - g["t_start"],
                             floor_mean_phase=sum(v) / len(v),
                             floor_worst_phase=_frac(g, t0 + p["fixed"] + worst)))
    dns = []
    for path, corpus in ((root / "results/cic_dns_split/cic_floors.csv", "cic"),
                         (root / "results/dns_v42/dns_floors.csv", "dns_v42")):
        for r in csv.DictReader(path.open()):
            if r["consumer"] != "flow":
                continue
            dns.append((corpus, r["episode"], float(r["duration_s"]), int(r["n_ports"])))
    for prof, p in PROFILES_.items():
        for corpus, eid, D, nports in dns:
            if prof == "dns_query_log" or nports > 1 or prof.startswith("cloud"):
                base = p["udp"] if not prof.startswith("cloud") else 0.0
            elif "maxlife" in p:  # measured profiles: released after the UDP idle
                # timeout at the end of the tunnel, or at the exporter's maximum lifetime
                base = min(D + p["udp"], p["maxlife"])
            else:  # single-port tunnel at the client edge: one flow, released at its
                # end or by the 1800 s active timeout; Zeek has no active timeout
                # and writes conn.log after the UDP inactivity timeout
                base = D + p["udp"] if prof.startswith("zeek_conn") else min(D, 1800.0)
            offs, worst = _phases(p["window"], p["period"])
            v = [min(1.0, (base + p["fixed"] + o) / D) for o in offs]
            rows.append(dict(corpus=corpus, profile=prof, unit=eid, duration_s=D,
                             floor_mean_phase=sum(v) / len(v),
                             floor_worst_phase=min(1.0, (base + p["fixed"] + worst) / D)))
    # encrypted upstream (DoT/DoH from a forwarding resolver): one persistent
    # connection shared by all clients, whose records are cut by the 1800 s active
    # timeout at a phase unrelated to the tunnel (uniform), then collector,
    # ingestion and rule as netflow_onprem. Attribution to the client is assumed.
    p = PROFILES_.get("netflow_onprem", PROFILES["netflow_onprem"])
    for corpus, eid, D, nports in dns:
        grid = lambda w: [(k + 0.5) / 12 * w for k in range(12)]
        offs3 = [a + b + c for a, b, c in itertools.product(grid(1800.0), grid(p["window"]),
                                                           grid(p["period"]))]
        v = [min(1.0, (p["fixed"] + o) / D) for o in offs3]
        rows.append(dict(corpus=corpus, profile="dns_encrypted_single_connection", unit=eid, duration_s=D,
                         floor_mean_phase=sum(v) / len(v),
                         floor_worst_phase=min(1.0, (p["fixed"] + 1800.0 + p["window"] + p["period"]) / D)))
    out = root / "results" / out_name
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    for corpus in ("dapt2021", "cic", "dns_v42"):
        for prof in list(PROFILES_) + ["dns_encrypted_single_connection"]:
            sub = [r for r in rows if r["corpus"] == corpus and r["profile"] == prof]
            if not sub:
                continue
            m = [r["floor_mean_phase"] for r in sub]
            wv = [r["floor_worst_phase"] for r in sub]
            print(f"{corpus:<9} {prof:<32} n={len(sub):2d} mean {100*st.mean(m):6.2f}% "
                  f"median {100*st.median(m):6.2f}%  worst-phase mean {100*st.mean(wv):6.2f}%  "
                  f"total loss {sum(x >= 0.999 for x in m)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
