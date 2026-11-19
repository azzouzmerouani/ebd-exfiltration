"""Architectural floor on CIC-Bell-DNS-EXF-2021 (second DNS corpus, revision 12).

    PYTHONPATH=src python3 scripts/cic_dns_floor.py --root <folder with the CIC pcaps> --out results/cic_dns

Standard library only. The corpus contains complete DNSExfiltrator transfers
(base64URL encoding, 500 ms throttling, 255-byte requests, 63-character labels)
of six file types in a light and a heavy variant, plus benign DNS
(Mahdavifar et al., ICCNS 2021). Unlike the first DNS corpus, the attack
captures are not truncated prefixes, which is the point of adding it.

For every attack capture the script parses the DNS queries, builds episodes
with the same rules as the first corpus (one client and registered tunnel
domain per capture, split on silence > 300 s), and measures the floor of the
per-query, flow-record (active 1800 s, idle 15 s) and session (300 s)
consumers, plus a sweep of the active timeout, the idle timeout and the session
window. Detectors are not trained here: on this corpus only the floor is
measured.

Captures are classified from their path: a path containing "benign" is benign;
otherwise a path containing "attack" (or one of the file-type words) is an
attack capture, with family = the file type (audio, compressed, exe, image,
text, video) and variant = heavy or light. The classification is printed before
anything is measured; check it.

Progress is checkpointed per capture in <out>/cic_state.json, so the script can
be stopped and re-run: finished captures are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ebd.dns.engine import (Episode, Query, bootstrap_ci, check_invariants,  # noqa: E402
                            clopper_pearson, family_bootstrap, measure_floor, wilson)

PCAP_EXT = (".pcap", ".pcapng", ".cap", ".pcap.gz", ".pcapng.gz")
TYPES = ("audio", "compressed", "exe", "image", "text", "video")
ACTIVE, IDLE, WINDOW, GAP = 1800.0, 15.0, 300.0, 300.0
ACTIVE_GRID = (30.0, 60.0, 120.0, 300.0, 600.0, 1800.0)
IDLE_GRID = (5.0, 15.0, 30.0, 60.0, 120.0)
SESSION_GRID = (60.0, 300.0, 600.0, 1800.0)
CONSUMERS = ("per-query", "flow", "session")


def parse_headers(path: Path, rel: str):
    """DNS queries from packet headers only.

    The CIC captures were recorded with a snapshot length of 96 bytes, so query
    names are cut and cannot be decoded. The UDP length field is intact, so each
    query's size on the wire is exact. The asset unit is the DNS message length
    minus its fixed 12-byte header and 4-byte question trailer; the tunnel
    suffix, constant per query, is the per-query overhead whose removal the
    engine reports as the A5 sensitivity. The tunnel is the (client, server) pair
    that carries the most query bytes in the capture.
    """
    import struct
    rows, stats = [], {"file": rel, "packets": 0, "udp53": 0, "snaplen": None}
    with path.open("rb") as fh:
        gh = fh.read(24)
        magic = struct.unpack("<I", gh[:4])[0]
        end = "<" if magic in (0xA1B2C3D4, 0xA1B23C4D) else ">"
        nano = magic in (0xA1B23C4D, 0x4D3CB2A1)
        stats["snaplen"] = struct.unpack(end + "I", gh[16:20])[0]
        while True:
            h = fh.read(16)
            if len(h) < 16:
                break
            ts, frac, incl, _orig = struct.unpack(end + "IIII", h)
            b = fh.read(incl)
            stats["packets"] += 1
            if len(b) < 14:
                continue
            et = struct.unpack("!H", b[12:14])[0]
            if et == 0x0800 and len(b) >= 34:
                ihl = (b[14] & 15) * 4; proto = b[23]
                src = ".".join(map(str, b[26:30])); dst = ".".join(map(str, b[30:34])); l4 = 14 + ihl
            elif et == 0x86DD and len(b) >= 54:
                proto = b[20]; src = b[22:38].hex(); dst = b[38:54].hex(); l4 = 54
            else:
                continue
            if proto != 17 or len(b) < l4 + 8:
                continue
            sp, dp, ul = struct.unpack("!HHH", b[l4:l4 + 6])
            if dp != 53:
                continue
            stats["udp53"] += 1
            t = ts + frac / (1e9 if nano else 1e6)
            q0 = l4 + 8 + 12          # first label of the query name (may be cut by the snaplen)
            label = ""
            if len(b) > q0:
                n = b[q0]
                label = b[q0 + 1:q0 + 1 + n].decode("ascii", "replace") if 0 < n < 64 else ""
                if label.lower() == "init" and len(b) > q0 + 1 + n:
                    n2 = b[q0 + 1 + n]
                    label += "." + b[q0 + 2 + n:q0 + 2 + n + n2].decode("ascii", "replace")
            rows.append((t, src, dst, sp, max(0, ul - 8 - 16), label))
    by_pair = {}
    for r in rows:
        by_pair.setdefault((r[1], r[2]), []).append(r)
    if not by_pair:
        return [], stats
    pair = max(by_pair, key=lambda k: sum(x[4] for x in by_pair[k]))
    qs = [Query(ts=t, src=s_, dst=d_, sport=sp, qname=lab, qtype=0, payload=pl, tunnel=d_, file=rel)
          for (t, s_, d_, sp, pl, lab) in sorted(by_pair[pair])]
    stats.update(tunnel_pair=f"{pair[0]}->{pair[1]}", tunnel_queries=len(qs),
                 other_queries=len(rows) - len(qs), tunnel_share_bytes=sum(q.payload for q in qs) / max(1, sum(r[4] for r in rows)),
                 t_first=qs[0].ts if qs else None, t_last=qs[-1].ts if qs else None)
    return qs, stats


def split_episodes(qs, tool, family, rel, gap_s, min_queries=20, min_payload=512):
    eps, run = [], []
    for q in qs:
        if run and q.ts - run[-1].ts > gap_s:
            eps.append(run); run = []
        run.append(q)
    if run:
        eps.append(run)
    out = []
    for r in eps:
        ep = Episode("", tool, r[0].src, r[0].dst, r, family, {"file": rel})
        if ep.n_queries >= min_queries and ep.asset >= min_payload and ep.duration > 0:
            ep.eid = f"{tool}|{rel}|{ep.src}->{ep.tunnel}|{ep.t0:.3f}"
            out.append(ep)
    return out


def classify(root: Path, p: Path) -> dict:
    rel = str(p.relative_to(root))
    low = rel.lower()
    if "benign" in low and not any(t in Path(low).name for t in TYPES):
        return {"kind": "benign", "rel": rel}
    ftype = next((t for t in TYPES if t in Path(low).name), None) or next((t for t in TYPES if t in low), None)
    if "attack" in low or ftype:
        variant = "heavy" if "heavy" in low else "light" if "light" in low else "unspecified"
        return {"kind": "attack", "rel": rel, "family": ftype or "unknown", "variant": variant}
    return {"kind": "unknown", "rel": rel}


def dedup_chunks(qs):
    """Keep the first query of each DNSExfiltrator chunk (rev12 review, R2-3).

    DNSExfiltrator sends an ``init`` request and then one request per chunk whose
    first label is the decimal chunk index; a repeated index is a retry. The
    first label survives the 96-byte snapshot length. Queries whose first label
    is neither ``init`` nor a decimal index are kept and counted.
    """
    seen, out, st_ = set(), [], {"dup": 0, "init": 0, "other_label": 0, "chunks": 0, "max_index": -1}
    for q in qs:
        lab = q.qname
        if lab.isdigit():
            i = int(lab)
            if i in seen:
                st_["dup"] += 1
                continue
            seen.add(i); st_["chunks"] += 1; st_["max_index"] = max(st_["max_index"], i)
        elif lab.lower().startswith("init"):
            st_["init"] += 1
            # init.<base32("filename|nbChunks")>: decode what the snaplen left
            import base64
            enc = lab.split(".", 1)[1] if "." in lab else ""
            enc = enc[: len(enc) // 8 * 8]
            try:
                st_["init_decoded"] = base64.b32decode(enc.upper()).decode("utf-8", "replace")
            except Exception:
                st_["init_decoded"] = ""
            st_["init_label"] = lab
        else:
            st_["other_label"] += 1
        out.append(q)
    st_["missing_below_max"] = st_["max_index"] + 1 - st_["chunks"] if st_["max_index"] >= 0 else 0
    st_["kept_payload"] = sum(q.payload for q in out)
    return out, st_


def split_at_init(qs):
    """Split a capture into transfers at each DNSExfiltrator ``init`` request.

    Some light captures hold the same file sent several times; each transfer
    starts with an ``init`` query, and its chunk indices restart at 0.
    Queries before the first ``init`` are kept as their own segment.
    """
    segs, cur = [], []
    for q in qs:
        if q.qname.lower().startswith("init") and cur:
            segs.append(cur); cur = []
        cur.append(q)
    if cur:
        segs.append(cur)
    return segs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/cic_dns"))
    ap.add_argument("--list-only", action="store_true", help="print the classification and stop")
    ap.add_argument("--max-files", type=int, default=0, help="process at most N new captures, then stop")
    ap.add_argument("--dedup", action="store_true",
                    help="drop repeated DNSExfiltrator chunk indices before measuring (use a separate --out)")
    ap.add_argument("--split-init", action="store_true",
                    help="treat each DNSExfiltrator init request as the start of a new transfer")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in a.root.rglob("*") if p.is_file() and p.name.lower().endswith(PCAP_EXT))
    desc = [classify(a.root, p) for p in files]
    print(f"{len(files)} capture files under {a.root}")
    for d in desc:
        size = (a.root / d["rel"]).stat().st_size / 1e6
        print(f"  {d['kind']:7s} {d.get('family', ''):11s} {d.get('variant', ''):11s} {size:9.1f} MB  {d['rel']}")
    if a.list_only:
        return 0

    state_path = a.out / "cic_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": {}, "floors": [], "sweep": [], "parse": []}
    done_now = 0
    for d in desc:
        if d["kind"] != "attack" or d["rel"] in state["done"]:
            continue
        if a.max_files and done_now >= a.max_files:
            break
        t0 = time.time()
        qs, ps = parse_headers(a.root / d["rel"], d["rel"])
        ps["raw_payload"] = sum(q.payload for q in qs)
        segs = split_at_init(qs) if a.split_init else [qs]
        ps["transfers"] = len(segs)
        if a.dedup:
            new, agg = [], {}
            for seg in segs:
                seg2, ds = dedup_chunks(seg)
                new.append(seg2)
                for k, v in ds.items():
                    if k == "max_index":
                        agg.setdefault("max_index_per_transfer", []).append(v)
                    elif k == "init_decoded":
                        agg.setdefault("init_decoded_per_transfer", []).append(v)
                    elif isinstance(v, (int, float)):
                        agg[k] = agg.get(k, 0) + v
                    else:
                        agg.setdefault(k, v)
            segs = new
            ps.update({f"dedup_{k}": v for k, v in agg.items()})
        ps.update(kind="attack", family=d["family"], variant=d["variant"])
        state["parse"].append(ps)
        tool = "dnsexfiltrator-cic"
        eps = [e for seg in segs
               for e in split_episodes(seg, tool, f"{d['family']}-{d['variant']}", d["rel"], GAP)]
        notes = []
        for ep in eps:
            ep.meta.update(variant=d["variant"], ftype=d["family"])
            fl = {}
            for c in CONSUMERS:
                f = measure_floor(ep, c, t_active=ACTIVE, t_idle=IDLE, window_s=WINDOW, capture_end=ep.t1)
                if f:
                    f.update(variant=d["variant"], ftype=d["family"])
                    fl[c] = f
                    state["floors"].append(f)
            notes += check_invariants(ep, fl, strict=False)
            for ta in ACTIVE_GRID:
                f = measure_floor(ep, "flow", t_active=ta, t_idle=IDLE, capture_end=ep.t1)
                if f: f.update(param=ta, axis="flow active timeout"); state["sweep"].append(f)
            for ti in IDLE_GRID:
                f = measure_floor(ep, "flow", t_active=ACTIVE, t_idle=ti, capture_end=ep.t1)
                if f: f.update(param=ti, axis="flow idle timeout"); state["sweep"].append(f)
            for w in SESSION_GRID:
                f = measure_floor(ep, "session", window_s=w)
                if f: f.update(param=w, axis="session window"); state["sweep"].append(f)
        state["done"][d["rel"]] = {"episodes": len(eps), "queries": len(qs), "seconds": round(time.time() - t0, 1),
                                    "invariant_notes": notes[:5]}
        done_now += 1
        state_path.write_text(json.dumps(state, default=str))
        print(f"  done {d['rel']}: {len(qs):,} queries, {len(eps)} episodes, {time.time() - t0:.0f} s", flush=True)

    todo = [d["rel"] for d in desc if d["kind"] == "attack" and d["rel"] not in state["done"]]
    if todo:
        print(f"{len(todo)} attack captures still to process; re-run to continue.")
        return 0

    # ---- outputs
    def write(name, rows):
        if not rows:
            return
        keys = sorted({k for r in rows for k in r})
        with (a.out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    write("cic_floors.csv", state["floors"])
    write("cic_cadence_sweep.csv", state["sweep"])
    write("cic_parse_stats.csv", state["parse"])

    summ = {"captures": len(state["done"]), "episodes": sum(v["episodes"] for v in state["done"].values())}
    for c in CONSUMERS:
        v = [r["floor_measured"] for r in state["floors"] if r["consumer"] == c]
        if not v:
            continue
        k = sum(x >= 0.999999 for x in v)
        by_f = {}
        for r in state["floors"]:
            if r["consumer"] == c:
                by_f.setdefault(r["family"], []).append(r["floor_measured"])
        fm, flo, fhi = family_bootstrap(by_f, n_boot=10_000, seed=20260825)
        lo, hi = bootstrap_ci(v, n_boot=10_000, seed=20260825)
        summ[c] = {"n": len(v), "mean": st.mean(v), "median": st.median(v), "boot": [lo, hi],
                   "total_loss": k, "total_loss_wilson": wilson(k, len(v)), "total_loss_exact": clopper_pearson(k, len(v)),
                   "family_mean": fm, "family_boot": [flo, fhi], "n_families": len(by_f),
                   "families_total_loss": sum(1 for x in by_f.values() if min(x) >= 0.999999)}
    flow = [r for r in state["floors"] if r["consumer"] == "flow"]
    if flow:
        summ["flow_by_variant"] = {var: {"n": len(g), "mean": st.mean(x["floor_measured"] for x in g),
                                         "median_duration_s": st.median(x["duration_s"] for x in g),
                                         "total_loss": sum(x["floor_measured"] >= 0.999999 for x in g)}
                                   for var in sorted({r["variant"] for r in flow})
                                   for g in [[r for r in flow if r["variant"] == var]]}
        summ["flow_excess_over_closed_form_pp"] = {"mean": st.mean(r["excess_pp"] for r in flow),
                                                   "min": min(r["excess_pp"] for r in flow),
                                                   "max": max(r["excess_pp"] for r in flow)}
        summ["ports_per_episode_median"] = st.median(r["n_ports"] for r in flow)
    sw = {}
    for r in state["sweep"]:
        sw.setdefault(f"{r['axis']}@{r['param']:g}", []).append(r["floor_measured"])
    summ["sweep_mean"] = {k: st.mean(v) for k, v in sorted(sw.items())}
    summ["sweep_total_loss"] = {k: sum(x >= 0.999999 for x in v) for k, v in sorted(sw.items())}
    (a.out / "cic_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
