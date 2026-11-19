"""v4, module level: malformed-message reasons and samples in the parser, and the address cache.
(The notebook-level v4 checks -- linear-time featuriser, per-source ibHH calibration, active-timeout sweep,
benign-read configuration, timing hooks -- need the notebook and live with its generator.)"""
import os, struct, socket, sys, tempfile
from ebd.dns.pcap_io import parse_pcap, malformed_reason, parse_dns_question, _ntop

FAILS = []
def check(name, cond, d=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + d) if d else ''}")
    if not cond: FAILS.append(name)

# ---------------------------------------------------------------- A. malformed reasons
print("\n=== malformed-message reasons ===")
HDR = struct.pack("!HHHHHH", 1, 0x0100, 1, 0, 0, 0)
CASES = {
    "shorter than a DNS header": b"\x00\x01\x02",
    "question count is 0": struct.pack("!HHHHHH", 1, 0x0100, 0, 0, 0, 0) + b"\x03www\x00\x00\x01\x00\x01",
    "name runs past the end of the message": HDR + b"\x01a",
    "label runs past the end of the message": HDR + b"\x20abc",
    "reserved label bits (0x40/0x80)": HDR + b"\x80xxxxxxxx",
    "compression pointer inside the question": HDR + b"\xc0\x0c\x00\x01\x00\x01",
    "question cut before QTYPE/QCLASS": HDR + b"\x01a\x00",
    "name longer than 255 octets": HDR + b"\x01a" * 130 + b"\x00\x00\x01\x00\x01",     # v4.2: was "more than 128 labels" (a name that long is over 255 octets first)
}
for why, msg in CASES.items():
    check(f"rejected by the parser: {why}", parse_dns_question(msg) is None)
    check(f"classified as: {why}", malformed_reason(msg) == why, malformed_reason(msg))
good = HDR + b"\x03www\x07example\x03com\x00\x00\x01\x00\x01"
check("a well-formed question is not malformed", parse_dns_question(good) is not None)

def udp(src, dst, sport, dport, payload):
    u = struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload
    return struct.pack("!BBHHHBBH", 0x45, 0, 20 + len(u), 0, 0, 64, 17, 0) + \
        socket.inet_aton(src) + socket.inet_aton(dst) + u
ETH = lambda ip: b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x08\x00" + ip
frames = []
t = 1.7e9
for i in range(5): frames.append((t + i, ETH(udp("10.0.0.1", "10.0.0.2", 40000 + i, 53, good))))
for i, (why, msg) in enumerate(CASES.items()):
    for rep in range(5 if i == 0 else 1):                              # one reason is more common
        frames.append((t + 10 + i * 10 + rep, ETH(udp("10.0.0.1", "10.0.0.2", 41000 + i, 53, msg))))
PCAP = os.path.join(tempfile.mkdtemp(), "v4.pcap")
with open(PCAP, "wb") as fh:
    fh.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 262144, 1))
    for ts, f in frames:
        fh.write(struct.pack("<IIII", int(ts), int((ts % 1) * 1e6), len(f), len(f)) + f)
qs, st = parse_pcap(PCAP)
check("good queries kept, bad ones counted separately", len(qs) == 5 and st["malformed"] == 5 + len(CASES) - 1,
      f"{len(qs)} kept, {st['malformed']} malformed")
check("reason counts add up to the malformed total",
      sum(st["malformed_reason_counts"].values()) == st["malformed"])
check("the most common reason is first in the text summary",
      st["malformed_reasons"].startswith("shorter than a DNS header: 5"), st["malformed_reasons"][:60])
check("at most 3 samples are kept per reason",
      all(len(v) <= 3 for v in st["malformed_samples"].values()) and
      len(st["malformed_samples"]["shorter than a DNS header"]) == 3)
check("a sample carries protocol, ports, length and the first bytes",
      st["malformed_samples"]["question count is 0"][0].startswith("udp 41001->53 len=")
      and "0001010000000000" in st["malformed_samples"]["question count is 0"][0])

# ---------------------------------------------------------------- address cache
print("\n=== address text is cached ===")
a4 = _ntop(bytes([10, 1, 2, 3])); b4 = _ntop(bytes([10, 1, 2, 3]))
check("IPv4 text is right", a4 == "10.1.2.3", a4)
check("the same address is ONE string object (memory: one per host, not one per query)", a4 is b4)
a6 = _ntop(socket.inet_pton(socket.AF_INET6, "2001:db8::1"))
check("IPv6 text is right", a6 == "2001:db8::1", a6)
check("a bytearray key works (an unhashable buffer must not crash the parser)", _ntop(bytearray([10, 1, 2, 3])) == "10.1.2.3")
check("an address of the wrong length falls back to hex, never raises", _ntop(b"\x01\x02\x03") == "010203")
qs2, _ = parse_pcap(PCAP)
check("queries from one host share one source-address string", len({id(q.src) for q in qs2}) == 1, f"{len({id(q.src) for q in qs2})} objects for {len(qs2)} queries")

print(f"\n{'ALL PASSED' if not FAILS else 'FAILED: ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
