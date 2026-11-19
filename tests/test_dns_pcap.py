"""Round-trip test: synthesise captures with stdlib only, parse them back,
and confirm the recovered floors equal values computable by hand."""
import gzip, socket, struct, sys
from ebd.dns.pcap_io import parse_pcap
from ebd.dns.engine import build_episodes, measure_floor, check_invariants

FAILS = []
def check(name, cond, d=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + d) if d else ''}")
    if not cond: FAILS.append(name)

# ---------- synthesis (stdlib only) ----------
def dns_query(txid, name, qtype=16):
    out = struct.pack("!HHHHHH", txid, 0x0100, 1, 0, 0, 0)
    for lab in name.split("."):
        out += bytes([len(lab)]) + lab.encode()
    return out + b"\x00" + struct.pack("!HH", qtype, 1)

def udp4(src, dst, sport, dport, payload):
    udp = struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload
    tot = 20 + len(udp)
    ip = struct.pack("!BBHHHBBH", 0x45, 0, tot, 0, 0, 64, 17, 0) + \
         socket.inet_aton(src) + socket.inet_aton(dst)
    return ip + udp

def udp6(src, dst, sport, dport, payload):
    udp = struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload
    ip = struct.pack("!IHBB", 0x60000000, len(udp), 17, 64) + \
         socket.inet_pton(socket.AF_INET6, src) + socket.inet_pton(socket.AF_INET6, dst)
    return ip + udp

def eth(payload, ethertype=0x0800):
    return b"\x00\x01\x02\x03\x04\x05" + b"\x06\x07\x08\x09\x0a\x0b" + \
           struct.pack("!H", ethertype) + payload

def sll(payload, ethertype=0x0800):
    return struct.pack("!HHH", 0, 1, 6) + b"\x00" * 8 + struct.pack("!H", ethertype) + payload

def write_pcap(path, frames, linktype=1, gz=False):
    op = gzip.open if gz else open
    with op(path, "wb") as fh:
        fh.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 262144, linktype))
        for ts, f in frames:
            sec, usec = int(ts), int(round((ts - int(ts)) * 1e6))
            fh.write(struct.pack("<IIII", sec, usec, len(f), len(f)) + f)

def write_pcapng(path, frames, linktype=1):
    with open(path, "wb") as fh:
        shb = struct.pack("<IIIHHq", 0x0A0D0D0A, 28, 0x1A2B3C4D, 1, 0, -1) + struct.pack("<I", 28)
        fh.write(shb)
        idb_body = struct.pack("<HHI", linktype, 0, 262144)
        fh.write(struct.pack("<II", 0x00000001, 12 + len(idb_body)) + idb_body +
                 struct.pack("<I", 12 + len(idb_body)))
        for ts, f in frames:
            pad = (4 - len(f) % 4) % 4
            us = int(round(ts * 1e6))
            # EPB body: iface, ts_hi, ts_lo, caplen, origlen, data, pad
            body = struct.pack("<IIIII", 0, us >> 32, us & 0xFFFFFFFF, len(f), len(f)) + f + b"\x00" * pad
            blen = 12 + len(body)
            fh.write(struct.pack("<II", 0x00000006, blen) + body + struct.pack("<I", blen))

def frames(n=120, rate=2.0, tunnel="t.exfil-lab.net", port_per_query=True,
           t0=1700000000.0, wrap=eth, mk=udp4, src="10.1.2.3", dst="9.9.9.9"):
    out = []
    for i in range(n):
        lab = (f"{i:05d}" + "abcdefghijklmnopqrstuvwxyz234567" * 3)[:45]
        msg = dns_query(1000 + i, f"{lab}.{tunnel}")
        pkt = mk(src, dst, (40000 + i) if port_per_query else 40000, 53, msg)
        et = 0x86DD if mk is udp6 else 0x0800
        out.append((t0 + i / rate, wrap(pkt, et) if wrap else pkt))
    return out

# ---------- tests ----------
print("\n=== A. classic pcap / Ethernet / IPv4 round-trip ===")
write_pcap("/tmp/a.pcap", frames())
qs, st = parse_pcap("/tmp/a.pcap")
check("120 queries recovered", len(qs) == 120, f"{len(qs)} from {st['packets']} packets")
check("0 malformed, no error", st["malformed"] == 0 and st["error"] is None, str(st["error"]))
ep = build_episodes(qs, "synth", gap_s=60)[0]
check("tunnel inferred", ep.tunnel == "t.exfil-lab.net", ep.tunnel)
check("payload 45 chars/query", all(q.payload == 45 for q in ep.queries))
check("asset = 5400", ep.asset == 5400, str(ep.asset))
check("D = 59.5 s", abs(ep.duration - 59.5) < 1e-3, f"{ep.duration:.4f}")

floors = {c: measure_floor(ep, c, t_active=1800.0, t_idle=15.0, window_s=300.0)
          for c in ("per-query", "flow", "session")}
print("      consumer     measured   closed-form    T_eff   records")
for c in ("per-query", "flow", "session"):
    f = floors[c]
    print(f"      {c:11s} {f['floor_measured']*100:7.2f}%  {f['floor_closed_form']*100:9.2f}%"
          f"  {f['t_eff_s']:7.2f}s {f['n_records']:8d}")
check("per-query floor = 1/120", abs(floors["per-query"]["floor_measured"] - 1/120) < 1e-6)
check("flow floor = 31/120 (idle 15 s @ 2 Hz)",
      abs(floors["flow"]["floor_measured"] - 31/120) < 1e-6,
      f"{floors['flow']['floor_measured']:.5f}")
check("session floor = 100%", floors["session"]["floor_measured"] == 1.0)
check("ordering per-query < flow < session",
      floors["per-query"]["floor_measured"] < floors["flow"]["floor_measured"] < 1.0001)
check("invariants clean", check_invariants(ep, floors) == [])

print("\n=== B. container / link-layer / IP-version robustness ===")
write_pcap("/tmp/b.pcap.gz", frames(n=40), gz=True)
q_gz, _ = parse_pcap("/tmp/b.pcap.gz");         check("gzip pcap", len(q_gz) == 40, str(len(q_gz)))
write_pcapng("/tmp/c.pcapng", frames(n=40))
q_ng, s_ng = parse_pcap("/tmp/c.pcapng");       check("pcapng", len(q_ng) == 40, f"{len(q_ng)} err={s_ng['error']}")
write_pcap("/tmp/d.pcap", frames(n=40, wrap=sll), linktype=113)
q_sll, _ = parse_pcap("/tmp/d.pcap");           check("Linux SLL", len(q_sll) == 40, str(len(q_sll)))
write_pcap("/tmp/e.pcap", frames(n=40, wrap=None), linktype=101)
q_raw, _ = parse_pcap("/tmp/e.pcap");           check("raw IP", len(q_raw) == 40, str(len(q_raw)))
write_pcap("/tmp/f.pcap", frames(n=40, mk=udp6, src="2001:db8::1", dst="2001:4860:4860::8888"))
q_v6, _ = parse_pcap("/tmp/f.pcap");            check("IPv6", len(q_v6) == 40, str(len(q_v6)))

print("\n=== C. a truncated / corrupt tail loses the tail, not the file ===")
raw = open("/tmp/a.pcap", "rb").read()
open("/tmp/g.pcap", "wb").write(raw[: len(raw) // 2] + b"\xff\xff\xff")
q_tr, s_tr = parse_pcap("/tmp/g.pcap")
check("partial file still yields its readable prefix", 40 < len(q_tr) < 120, f"{len(q_tr)} queries")

print("\n=== D. benign background is not mistaken for a tunnel ===")
ben = []
for i, host in enumerate(["www.google.com", "cdn.cloudflare.net", "mail.yahoo.co.jp",
                          "api.github.com", "static.wikipedia.org"] * 8):
    ben.append((1700000000.0 + i * 3.0, eth(udp4("10.1.2.9", "9.9.9.9", 40000 + i, 53,
                                                 dns_query(i, host, 1)))))
write_pcap("/tmp/h.pcap", ben)
q_b, _ = parse_pcap("/tmp/h.pcap")
eps_b = build_episodes(q_b, "benign", gap_s=60, min_queries=20, min_payload=512)
check("40 benign queries parsed", len(q_b) == 40, str(len(q_b)))
check("no common tunnel suffix across unrelated domains",
      all(e.tunnel in ("", "com", "net", "org", "jp") for e in eps_b) or not eps_b,
      f"{[e.tunnel for e in eps_b]}")

print("\n=== E. single-port tunnel: the flow never closes -> total loss ===")
write_pcap("/tmp/i.pcap", frames(n=120, port_per_query=False))
q_sp, _ = parse_pcap("/tmp/i.pcap")
ep_sp = build_episodes(q_sp, "synth", gap_s=60)[0]
f_sp = measure_floor(ep_sp, "flow", t_active=1800.0, t_idle=15.0)
check("single-port flow floor = 100%", f_sp["floor_measured"] == 1.0,
      f"{f_sp['floor_measured']*100:.1f}%  distinct ports={ep_sp.n_ports}")
check("port-per-query leaks 25.8%, single-port leaks 100%",
      abs(floors['flow']['floor_measured'] - 31/120) < 1e-6 and f_sp['floor_measured'] == 1.0)

print("\n" + "=" * 68)
print("ALL PASSED" if not FAILS else "FAILURES: " + ", ".join(FAILS))
sys.exit(1 if FAILS else 0)
