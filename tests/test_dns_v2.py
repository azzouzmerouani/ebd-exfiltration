"""Tests for the v2 behaviours. Each one targets a way real lab captures could
silently corrupt the episode set or the floor."""
from dataclasses import replace
import struct, socket, sys
from ebd.dns.engine import (Query, build_episodes, benign_units, measure_floor,
                            registered_domain, unique_info_trace, ibhh_threshold,
                            ibhh_alert_time, fraction_lost_by, wilson_upper, pick_threshold)
from ebd.dns.pcap_io import parse_pcap

FAILS = []
def check(name, cond, d=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + d) if d else ''}")
    if not cond: FAILS.append(name)

ALPH = "abcdefghijklmnopqrstuvwxyz234567"
def tunnel(n, rate, file, src="10.0.0.5", t0=1000.0, tunnel_dom="t.exf-lab.net",
           paylen=40, ppq=True, overhead_prefix=""):
    out = []
    for i in range(n):
        lab = overhead_prefix + (f"{i:06d}" + ALPH * 3)[:paylen]
        out.append(Query(ts=t0 + i / rate, src=src, dst="9.9.9.9",
                         sport=(40000 + i) if ppq else 40000,
                         qname=f"{lab}.{tunnel_dom}", qtype=16, file=file))
    return out

print("\n=== 1. client IPs reused across capture files must NOT merge episodes ===")
# same client, same tunnel, OVERLAPPING timestamps, two different capture files
qs = tunnel(100, 1.0, "a_up.pcapng") + tunnel(100, 1.0, "b_up.pcapng", t0=1030.0)
eps = build_episodes(qs, "tool", gap_s=300)
check("two files -> two episodes", len(eps) == 2, f"{len(eps)}")
check("each episode is one file's 100 queries", sorted(e.n_queries for e in eps) == [100, 100],
      str([e.n_queries for e in eps]))
merged = build_episodes([replace(q, file="") for q in qs], "tool", gap_s=300)
check("(control) without file scoping they would merge", len(merged) == 1 and merged[0].n_queries == 200,
      f"{[e.n_queries for e in merged]}")

print("\n=== 2. background DNS inside a tool capture never becomes an episode ===")
bg = []
for i in range(300):     # a chatty OS: lots of queries, many unique names, small payload
    bg.append(Query(ts=1000.0 + i * 0.4, src="10.0.0.5", dst="9.9.9.9", sport=50000 + i,
                    qname=f"x{i%57}.update.vendor-cdn.com", qtype=1, file="a_up.pcapng"))
log = []
eps = build_episodes(tunnel(120, 1.0, "a_up.pcapng") + bg, "tool", gap_s=300, log=log)
check("only the tunnel survives", len(eps) == 1 and eps[0].tunnel == "t.exf-lab.net",
      f"{[(e.tunnel, e.n_queries) for e in eps]}")
check("the drop is logged, not silent", log and log[0]["n_background_domains"] == 1,
      f"{log[0] if log else None}")

print("\n=== 3. a capture that saw both legs (stub->resolver, resolver->auth) counts once ===")
log = []
two_leg = tunnel(120, 1.0, "c_up.pcapng", src="10.0.0.5") + \
          tunnel(120, 1.0, "c_up.pcapng", src="10.0.0.53", t0=1000.02)
eps = build_episodes(two_leg, "tool", gap_s=300, log=log)
check("one episode, not two", len(eps) == 1, f"{len(eps)}")
check("duplicate vantage recorded", log[0]["n_duplicate_vantage"] == 1 and
      log[0]["duplicate_vantage_max_share"] > 0.9, f"share={log[0]['duplicate_vantage_max_share']:.2f}")

print("\n=== 4. A5 unit sensitivity: per-query overhead is measured, not assumed ===")
ep = build_episodes(tunnel(100, 1.0, "d_up.pcapng"), "tool", gap_s=300)[0]
f = measure_floor(ep, "flow", t_active=1800.0, t_idle=15.0)
check("no overhead -> both units agree exactly", abs(f["unit_delta_pp"]) < 1e-9,
      f"delta={f['unit_delta_pp']:.4f} pp")
# polls: every 4th query is a 12-char ack carrying no data
mixed = []
for i, q in enumerate(tunnel(100, 1.0, "e_up.pcapng")):
    if i % 4 == 3:
        q = replace(q, qname=f"ack{i:06d}xyz.t.exf-lab.net")
    mixed.append(q)
ep2 = build_episodes(mixed, "tool", gap_s=300)[0]
f2 = measure_floor(ep2, "flow", t_active=1800.0, t_idle=15.0)
check("overhead estimated from the shortest queries", f2["overhead_est"] == 12, f"h={f2['overhead_est']}")
check("unit delta is reported and small for interleaved polls", abs(f2["unit_delta_pp"]) < 2.0,
      f"delta={f2['unit_delta_pp']:+.3f} pp")

print("\n=== 5. registered domain heuristic ===")
check("plain", registered_domain("aGVsbG8.t.exf-lab.net") == "exf-lab.net")
check("two-letter ccTLD with SLD", registered_domain("x.y.example.co.uk") == "example.co.uk")
check("short names", registered_domain("localhost") == "localhost")

print("\n=== 6. benign units are cut exactly like malicious ones ===")
bq = [Query(ts=1000.0 + i, src="10.9.0.2", dst="9.9.9.9", sport=50000 + i,
            qname=d, qtype=1, file="u1.pcap")
      for i, d in enumerate(["www.google.com", "mail.google.com", "api.github.com",
                             "www.google.com", "cdn.github.com"] * 4)]
bu = benign_units(bq, gap_s=300)
check("one unit per (file, client, registered domain)", len(bu) == 2,
      f"{sorted((e.tunnel, e.n_queries) for e in bu)}")

print("\n=== 7. ibHH-style accumulator counts UNIQUE subdomain characters ===")
rep = [Query(ts=float(i), src="a", dst="b", sport=1, qname="www.example.com", qtype=1)
       for i in range(50)]
tr = unique_info_trace(rep, window_s=60)
check("a repeated name adds nothing after the first", tr[-1][1] == 3, f"{tr[-1]}")
ep3 = build_episodes(tunnel(200, 2.0, "f_up.pcapng"), "tool", gap_s=300)[0]
tr3 = unique_info_trace(ep3.queries, window_s=60)
check("a tunnel accumulates fast", tr3[30][1] > 30 * 40, f"after 31 queries: {tr3[30][1]} chars")
ev = [(float(t), 0) for t in range(0, 200, 1)]
win = unique_info_trace([Query(ts=float(i), src="a", dst="b", sport=1,
                               qname=f"u{i}.example.com", qtype=1) for i in range(200)], window_s=10)
check("the window evicts old names", max(v for _, v in win) <= 11 * 4, f"max={max(v for _, v in win)}")

print("\n=== 8. ibHH threshold calibrated on benign STREAMS with the Wilson rule ===")
streams = [[Query(ts=float(j), src="c", dst="r", sport=1, qname=f"w{j}.site{s}.com", qtype=1)
            for j in range(3 + s % 5)] for s in range(400)]
thr, meta = ibhh_threshold(streams, window_s=60, budget=0.01)
check("threshold respects the upper bound", meta["train_upper"] <= 0.01,
      f"thr={thr:.0f} upper={meta['train_upper']:.4f} n={meta['n_benign_streams']}")
t_alert = ibhh_alert_time(ep3, thr, window_s=60)
ebd = fraction_lost_by(ep3, t_alert)
pq_floor = measure_floor(ep3, "per-query")["floor_measured"]
check("ibHH-style EBD sits at or above the per-query floor", ebd >= pq_floor - 1e-12,
      f"EBD={ebd*100:.2f}%  per-query floor={pq_floor*100:.2f}%")
check("…and well below the flow floor here", ebd < measure_floor(ep3, "flow", t_active=1800.0,
      t_idle=15.0)["floor_measured"], "statistical delay < architectural delay")

print("\n=== 9. pcapng with TWO interfaces at different timestamp resolutions ===")
def dnsq(name):
    o = struct.pack("!HHHHHH", 7, 0x0100, 1, 0, 0, 0)
    for l in name.split("."): o += bytes([len(l)]) + l.encode()
    return o + b"\x00" + struct.pack("!HH", 16, 1)
def frame(name):
    m = dnsq(name); udp = struct.pack("!HHHH", 40000, 53, 8 + len(m), 0) + m
    ip = struct.pack("!BBHHHBBH", 0x45, 0, 20 + len(udp), 0, 0, 64, 17, 0) + \
         socket.inet_aton("10.0.0.5") + socket.inet_aton("9.9.9.9")
    return b"\x00" * 12 + b"\x08\x00" + ip + udp
def idb(linktype, tsres_exp):
    opt = struct.pack("<HHB", 9, 1, tsres_exp) + b"\x00" * 3 + struct.pack("<HH", 0, 0)
    body = struct.pack("<HHI", linktype, 0, 262144) + opt
    return struct.pack("<II", 1, 12 + len(body)) + body + struct.pack("<I", 12 + len(body))
def epb(iface, ticks, f):
    pad = (4 - len(f) % 4) % 4
    body = struct.pack("<IIIII", iface, ticks >> 32, ticks & 0xFFFFFFFF, len(f), len(f)) + f + b"\x00" * pad
    return struct.pack("<II", 6, 12 + len(body)) + body + struct.pack("<I", 12 + len(body))
with open("/tmp/two_if.pcapng", "wb") as fh:
    fh.write(struct.pack("<IIIHHq", 0x0A0D0D0A, 28, 0x1A2B3C4D, 1, 0, -1) + struct.pack("<I", 28))
    fh.write(idb(1, 6)); fh.write(idb(1, 9))                 # iface0: us, iface1: ns
    fh.write(epb(0, 1_700_000_000_000_000, frame("a.t.x.net")))
    fh.write(epb(1, 1_700_000_001_000_000_000, frame("b.t.x.net")))
q9, st9 = parse_pcap("/tmp/two_if.pcapng")
check("both packets parsed", len(q9) == 2, f"{len(q9)} err={st9['error']}")
check("per-interface timestamp resolution honoured", len(q9) == 2 and abs((q9[1].ts - q9[0].ts) - 1.0) < 1e-6,
      f"dt={q9[1].ts - q9[0].ts if len(q9)==2 else None}")

print("\n=== 10. Wilson operating point degrades safely when the budget is unreachable ===")
thr10, m10 = pick_threshold([0.99] * 50, budget=0.01)
check("unreachable budget -> never alerts, and says why", thr10 > 1.0 and "reason" in m10, str(m10))

print("\n" + "=" * 70)
print("ALL V2 TESTS PASSED" if not FAILS else "FAILURES: " + ", ".join(FAILS))
sys.exit(1 if FAILS else 0)
