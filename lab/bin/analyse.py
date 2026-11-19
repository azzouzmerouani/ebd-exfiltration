"""Analyse the lab pipeline-delay measurement.

For every transfer of /opt/lab/run/transfers.jsonl and every pipeline, find the
first record covering the transfer and the times at which it became available:

  release   exporter sends the record (Filebeat receive time, event.created;
            for Zeek the time Filebeat read the conn.log line)
  file      nfcapd closes the 300 s file that holds it (NetFlow collector path)
  ingest    Elasticsearch indexes it (event.ingested)
  alert     a Kibana detection rule creates its alert (two rules: default
            event-time query, and timestamp_override = event.ingested)

and read the floor V_e(t)/V_e off the transfer's own byte timeline, measured
from the router capture (vh.pcap). Writes /opt/lab/out/delays.csv and prints a
summary. No model is involved.
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics as st
import struct
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

RUN = "/opt/lab/run"
OUT = "/opt/lab/out"
VICTIM, PEER = "10.10.1.2", "10.10.2.2"


def es(path, body=None):
    creds = dict(l.strip().split("=", 1) for l in open(f"{RUN}/creds"))
    cmd = ["curl", "-s", "-u", f"elastic:{creds['ELASTIC_PW']}", "-H", "Content-Type: application/json",
           f"http://127.0.0.1:9200/{path}"]
    if body is not None:
        cmd += ["-d", json.dumps(body)]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def ts(s):
    if s is None:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def scroll(index, query, fields):
    out, after = [], None
    while True:
        body = {"size": 5000, "query": query, "_source": fields, "sort": [{"_doc": "asc"}]}
        if after:
            body["search_after"] = after
        r = es(f"{index}/_search", body)
        hits = r.get("hits", {}).get("hits", [])
        if not hits:
            return out
        out += [h["_source"] for h in hits]
        after = hits[-1]["sort"]


def get(d, dotted):
    for k in dotted.split("."):
        if isinstance(d, dict) and k in d:
            d = d[k]
        elif isinstance(d, dict) and dotted in d:
            return d[dotted]
        else:
            return d.get(dotted) if isinstance(d, dict) else None
    return d


def pcap_timelines(path, ports):
    """Per destination port: list of (t, payload bytes) victim -> peer."""
    tl = defaultdict(list)
    with open(path, "rb") as fh:
        gh = fh.read(24)
        endian = "<" if gh[:4] in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
        nano = gh[:4] in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
        while True:
            h = fh.read(16)
            if len(h) < 16:
                break
            sec, frac, incl, _ = struct.unpack(endian + "IIII", h)
            data = fh.read(incl)
            t = sec + frac / (1e9 if nano else 1e6)
            if len(data) < 34 or data[12:14] != b"\x08\x00":
                continue
            ip = data[14:]
            ihl = (ip[0] & 15) * 4
            total = struct.unpack("!H", ip[2:4])[0]
            proto = ip[9]
            src = ".".join(map(str, ip[12:16])); dst = ".".join(map(str, ip[16:20]))
            if src != VICTIM or dst != PEER or len(ip) < ihl + 8:
                continue
            l4 = ip[ihl:]
            dport = struct.unpack("!H", l4[2:4])[0]
            if dport not in ports:
                continue
            if proto == 6:
                pay = total - ihl - (l4[12] >> 4) * 4
            elif proto == 17:
                pay = total - ihl - 8
            else:
                continue
            if pay > 0:
                tl[dport].append((t, pay))
    return tl


def floor_at(timeline, total, t):
    if t is None:
        return None
    return min(1.0, sum(b for (x, b) in timeline if x <= t) / total)


def main():
    os.makedirs(OUT, exist_ok=True)
    transfers = [json.loads(l) for l in open(f"{RUN}/transfers.jsonl")]
    transfers = [t for t in transfers if t["id"] < 900]
    port_of = {}
    for t in transfers:
        port_of[t["id"]] = (20000 if t["kind"].startswith("tcp") else 30000) + t["id"]
    tl = pcap_timelines("/opt/lab/pcap/vh.pcap", set(port_of.values()))

    q = {"bool": {"filter": [{"term": {"source.ip": VICTIM}}, {"term": {"destination.ip": PEER}}]}}
    flows = scroll("lab-flows,lab-zeek", q, ["exporter_profile", "destination.port", "source.port",
                                              "event.created", "event.ingested", "event.start", "event.end",
                                              "@timestamp", "network.bytes", "zeek"])
    alerts = scroll(".alerts-security.alerts-default", {"match_all": {}},
                    ["@timestamp", "kibana.alert.rule.name", "kibana.alert.original_time",
                     "destination.port", "exporter_profile", "source.ip"])

    first = {}  # (profile, dport) -> first record dict
    for f in flows:
        prof = f.get("exporter_profile"); dport = get(f, "destination.port")
        if dport is None:
            continue
        created = ts(get(f, "event.created")) or ts(f.get("@timestamp"))
        rec = dict(created=created, ingested=ts(get(f, "event.ingested")), end=ts(get(f, "event.end")),
                   start_evt=ts(get(f, "event.start")))
        if prof == "zeek":
            z = f.get("zeek", {})
            zts, zdur = z.get("ts"), z.get("duration") or 0.0
            rec["end"] = (zts + zdur) if zts is not None else None
            rec["created"] = ts(f.get("@timestamp"))
            rec["start_evt"] = zts
        k = (prof, int(dport))
        if k not in first or rec["created"] < first[k]["created"]:
            first[k] = rec
    first_alert = {}
    for a in alerts:
        k = (a.get("kibana.alert.rule.name") or get(a, "kibana.alert.rule.name"), a.get("exporter_profile"),
             get(a, "destination.port"))
        t = ts(a["@timestamp"])
        if k[2] is not None and (k not in first_alert or t < first_alert[k]):
            first_alert[k] = t

    runs = defaultdict(list)
    for l in open(f"{OUT}/rule_runs.jsonl"):
        r = json.loads(l); runs[r["rule"]].append((ts(r["start"]), ts(r["end"])))
    for k in runs:
        runs[k].sort()

    def rule_alert(rule, ingested, event_time, lookback=360.0, refresh=1.0):
        """Alert time of a 5-min rule run with look-back, derived from the measured
        run times: the first run that starts after the record is searchable; the
        record is found only if its event time lies within that run's look-back."""
        for (s0, e0) in runs[rule]:
            if s0 >= ingested + refresh:
                return e0 if event_time >= s0 - lookback else None
        return None

    rows = []
    for t in transfers:
        dport = port_of[t["id"]]; series = sorted(tl.get(dport, []))
        if not series:
            continue
        total = sum(b for _, b in series); t0 = series[0][0]; t1 = series[-1][0]
        for prof in ("e1", "e2", "zeek"):
            r = first.get((prof, dport))
            row = dict(id=t["id"], kind=t["kind"], label=None, bytes_payload=total, D=t1 - t0,
                       pipeline=prof, t_start=t0)
            if r is None:
                row.update(found=False); rows.append(row); continue
            fin = r["created"]
            file_close = (math.floor(fin / 300.0) + 1) * 300.0 if prof in ("e1", "e2") else None
            a_def = first_alert.get(("lab-exfil-peer-default", prof, dport))
            a_ing = first_alert.get(("lab-exfil-peer-ingested", prof, dport))
            d_ing = rule_alert("lab-exfil-peer-ingested", r["ingested"], r["ingested"])
            d_evt = rule_alert("lab-exfil-peer-default", r["ingested"], r.get("start_evt") or fin)
            row.update(found=True,
                       alert_ing_derived=(d_ing - t0) if d_ing else None,
                       alert_ing_observed_minus_derived=(a_ing - d_ing) if (a_ing and d_ing) else None,
                       alert_flowstart_derived=(d_evt - t0) if d_evt else None,
                       missed_flowstart=d_evt is None,
                       floor_alert_ing_derived=floor_at(series, total, d_ing) if d_ing else 1.0,
                       floor_alert_flowstart_derived=floor_at(series, total, d_evt) if d_evt else 1.0,
                       first_flow_end=(r["end"] - t0) if r["end"] else None,
                       release=fin - t0, release_after_flow_end=(fin - r["end"]) if r["end"] else None,
                       file_close=(file_close - t0) if file_close else None,
                       ingest=(r["ingested"] - t0) if r["ingested"] else None,
                       ingest_lag=(r["ingested"] - fin) if r["ingested"] else None,
                       alert_default=(a_def - t0) if a_def else None,
                       alert_ingested=(a_ing - t0) if a_ing else None,
                       floor_flow_end=floor_at(series, total, r["end"]),
                       floor_release=floor_at(series, total, fin),
                       floor_file=floor_at(series, total, file_close),
                       floor_ingest=floor_at(series, total, r["ingested"]),
                       floor_alert_default=floor_at(series, total, a_def) if a_def else 1.0,
                       floor_alert_ingested=floor_at(series, total, a_ing) if a_ing else 1.0,
                       missed_default=a_def is None, missed_ingested=a_ing is None)
            rows.append(row)
    plan = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(f"{RUN}/plan.jsonl")}
    for r in rows:
        r["label"] = plan.get(r["id"])
    keys = sorted({k for r in rows for k in r})
    with open(f"{OUT}/delays.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT}/delays.csv ({len(rows)} rows, {len(transfers)} transfers)")
    def summ(vals):
        v = [x for x in vals if x is not None]
        if not v:
            return "n/a"
        return f"n={len(v)} med {st.median(v):.1f} min {min(v):.1f} max {max(v):.1f}"
    for prof in ("e1", "e2", "zeek"):
        for lab in sorted({r["label"] for r in rows}):
            sub = [r for r in rows if r["pipeline"] == prof and r["label"] == lab and r.get("found")]
            if not sub:
                continue
            print(f"{prof:5s} {lab:11s} release-after-end {summ([r['release_after_flow_end'] for r in sub])} | "
                  f"ingest lag {summ([r['ingest_lag'] for r in sub])} | alert(ing)-start {summ([r['alert_ingested'] for r in sub])} | "
                  f"floor alert(ing) mean {100*st.mean(r['floor_alert_ingested'] for r in sub):.1f}% | "
                  f"missed default {sum(r['missed_default'] for r in sub)}/{len(sub)}")


if __name__ == "__main__":
    sys.exit(main())
