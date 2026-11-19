"""v4.2, module level: what the 7 Oct measurement run (v4.1) exposed on the benign side.

Four defects, none touching a floor (floors use no benign data):
  1. the DNS name decoder recursed once per compression pointer and its label counter restarted on every call,
     so a name whose pointer leads back to itself recursed until Python gave up -- and the per-capture handler
     then ended the WHOLE capture (4 of the 13 university captures read on 7 Oct);
  2. the operating-point search ran over a grid whose lowest point (0.005) was already inside the 1% budget in
     54 of 54 folds, so the budget never bound;
  3. a DNS-over-TCP message whose 2-byte length arrived in a segment of its own was read from the wrong offset;
  4. (notebook level, not tested here) the memory limit's source was not recorded.
These checks pin down what changed and what did not: the same names on every message v4.1 could decode, a
capture that survives a hostile packet, and an operating point that is exact.
"""
import math, os, random, struct, socket, sys, tempfile

from ebd.dns import pcap_io as P
from ebd.dns import engine as E

FAILS = []
def check(name, cond, d=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + d) if d else ''}")
    if not cond: FAILS.append(name)

# ------------------------------------------------------------------ reference: the v4.1 decoder, verbatim
def qname_v41(msg, pos):
    labels, seen = [], 0
    while True:
        if pos >= len(msg):
            return None
        n = msg[pos]
        if n == 0:
            return (".".join(labels), pos + 1)
        if n & 0xC0 == 0xC0:
            if pos + 2 > len(msg):
                return None
            ptr = struct.unpack("!H", msg[pos:pos + 2])[0] & 0x3FFF
            if ptr >= pos or seen > 8:
                return None
            tail = qname_v41(msg, ptr)
            if tail is None:
                return None
            labels.append(tail[0])
            return (".".join(labels), pos + 2)
        pos += 1
        if pos + n > len(msg) or n > 63:
            return None
        labels.append(msg[pos:pos + n].decode("latin-1"))
        pos += n
        seen += 1
        if seen > 128:
            return None

HDR = struct.pack("!HHHHHH", 7, 0x0100, 1, 0, 0, 0)

def qname_ref(msg, pos, wire=1):
    """The v4.1 decoder plus the two RFC 1035 rules v4.2 added: no pointer into the 12-byte header, and at most 255
    octets for a name. (v4.1's recursion restarts its label counter, so wire is carried in explicitly.)"""
    labels, seen = [], 0
    while True:
        if pos >= len(msg):
            return None
        n = msg[pos]
        if n == 0:
            return (".".join(labels), pos + 1)
        if n & 0xC0 == 0xC0:
            if pos + 2 > len(msg):
                return None
            ptr = struct.unpack("!H", msg[pos:pos + 2])[0] & 0x3FFF
            if ptr >= pos or ptr < 12 or seen > 8:
                return None
            tail = qname_ref(msg, ptr, wire)
            if tail is None:
                return None
            labels.append(tail[0])
            return (".".join(labels), pos + 2)
        pos += 1
        if pos + n > len(msg) or n > 63:
            return None
        wire += n + 1
        if wire > 255:
            return None
        labels.append(msg[pos:pos + n].decode("latin-1"))
        pos += n
        seen += 1
        if seen > 128:
            return None

def ptr(to):
    return bytes([0xC0 | (to >> 8), to & 255])

# ------------------------------------------------------------------ A. the decoder
print("\n=== A. the iterative name decoder ===")
rng = random.Random(1)
def labels_bytes(k, lo=1, hi=9):
    return b"".join(bytes([n]) + bytes(rng.randrange(97, 123) for _ in range(n)) for n in [rng.randrange(lo, hi) for _ in range(k)])

def rand_case(header_ptrs=False):
    """-> (message, offset to decode from). The name usually starts at 12, like a question; the 'dict' and 'chain' cases put
    earlier names in front of it and start later, so that the pointers really are followed (a pointer in a question
    has nothing earlier to lead to)."""
    k = rng.choice(["labels", "tail", "junk", "chain", "root", "dict", "dict", "dict"])
    if k == "labels":
        return HDR + labels_bytes(rng.randrange(1, 6), 1, 20) + b"\x00\x00\x10\x00\x01", 12
    if k == "tail":                                         # a pointer after the root: never reached from offset 12
        return HDR + b"\x03abc\x00" + labels_bytes(rng.randrange(0, 4)) + ptr(rng.randrange(12, 40)) + b"\x00\x10\x00\x01", 12
    if k == "junk":
        return HDR + bytes(rng.randrange(256) for _ in range(rng.randrange(0, 90))), 12
    if k == "root":
        return HDR + b"\x00\x00\x10\x00\x01", 12
    if k == "chain":                                        # pointers that each lead to an earlier pointer, ending at a name
        m = bytearray(HDR + b"\x03abc\x00"); prev = 12
        for _ in range(rng.randrange(1, 12)):
            here = len(m); m += ptr(prev); prev = here
        return bytes(m) + ptr(prev) + b"\x00\x10\x00\x01", len(m)          # decode from the last pointer
    m = bytearray(HDR); starts = [12]                       # "dict": earlier names, then a name that ends in a pointer
    for _ in range(rng.randrange(1, 4)):
        starts.append(len(m)); m += labels_bytes(rng.randrange(1, 5)) + b"\x00"
    pos = len(m)
    m += labels_bytes(rng.randrange(0, 4))
    if header_ptrs:
        to = rng.choice(starts + [rng.randrange(0, len(m) + 4) for _ in range(3)])        # anywhere, header and forward included
    else:
        to = rng.choice(starts + [rng.randrange(12, len(m) + 4) for _ in range(3)])
    return bytes(m) + ptr(to) + b"\x00\x10\x00\x01", pos

def qname_v41_traced(msg, pos, trace):
    """qname_v41 (verbatim logic) that also records whether it followed a pointer into the header and how many wire
    octets the labels it read take, so that a difference with v4.2 can be checked against the two new rules."""
    labels, seen = [], 0
    while True:
        if pos >= len(msg):
            return None
        n = msg[pos]
        if n == 0:
            return (".".join(labels), pos + 1)
        if n & 0xC0 == 0xC0:
            if pos + 2 > len(msg):
                return None
            ptr_ = struct.unpack("!H", msg[pos:pos + 2])[0] & 0x3FFF
            if ptr_ >= pos or seen > 8:
                return None
            if ptr_ < 12:
                trace["header_pointer"] = True
            trace["jumps"] = trace.get("jumps", 0) + 1
            tail = qname_v41_traced(msg, ptr_, trace)
            if tail is None:
                return None
            labels.append(tail[0])
            return (".".join(labels), pos + 2)
        pos += 1
        if pos + n > len(msg) or n > 63:
            return None
        trace["wire"] = trace.get("wire", 1) + n + 1
        labels.append(msg[pos:pos + n].decode("latin-1"))
        pos += n
        seen += 1
        if seen > 128:
            return None

def compare_with_v41(gen, n_msgs):
    same = explained = unexplained = loops = 0; first = None
    compare_with_v41.jumped = 0
    for _ in range(n_msgs):
        m, pos0 = gen()
        tr = {}
        try:
            a = qname_v41_traced(m, pos0, tr)
        except RecursionError:                              # v4.1 on a pointer loop; v4.2 must reject it
            loops += 1
            if P._qname(m, pos0) is None: same += 1
            else: unexplained += 1; first = first or (m.hex(), "RecursionError", P._qname(m, pos0))
            continue
        b = P._qname(m, pos0)
        new_rule = a is not None and (tr.get("header_pointer") or tr.get("wire", 1) > 255)
        if new_rule:
            if b is None: explained += 1
            else: unexplained += 1; first = first or (m.hex(), a, b)
        elif a == b:
            same += 1
            if a is not None and tr.get("jumps"): compare_with_v41.jumped += 1      # a name decoded THROUGH a pointer, identically
        else: unexplained += 1; first = first or (m.hex(), a, b)
    return same, explained, unexplained, loops, first

same, explained, unexplained, loops, first = compare_with_v41(rand_case, 60000)
check("same name and end offset as v4.1 on 60,000 cases (labels, junk, root, chains, earlier names plus a pointer; loops included), "
      "except where v4.1 followed a pointer into the header (the rule v4.2 added)", unexplained == 0 and loops > 100 and compare_with_v41.jumped > 5000,
      f"{same} same ({compare_with_v41.jumped} decoded through a pointer), {explained} rejected under the new rule, "
      f"{loops} loops (v4.1: RecursionError), {unexplained} unexplained {first or ''}")
same, explained, unexplained, loops, first = compare_with_v41(lambda: rand_case(header_ptrs=True), 60000)
check("... and on 60,000 cases whose pointers may lead anywhere, the header and forward targets included",
      unexplained == 0 and explained > 1000 and same > 1000, f"{same} same, {explained} rejected under the new rule, {loops} loops, {unexplained} unexplained {first or ''}")
bad_ref = 0
for _ in range(60000):
    m, pos0 = rand_case(header_ptrs=True)
    try: want = qname_ref(m, pos0)
    except RecursionError: want = None
    if P._qname(m, pos0) != want: bad_ref += 1
check("v4.2 equals the RFC-rule reference (v4.1 + the two rules) on those cases", bad_ref == 0, f"{bad_ref} differ")
hm = HDR + b"\x01a\xc0\x00" + b"\x00\x10\x00\x01"        # 'a' then a pointer to offset 0: header bytes
check("a pointer into the header: v4.1 reads the header as labels, v4.2 rejects it",
      qname_v41(hm, 12) is not None and P._qname(hm, 12) is None and P.parse_dns_question(hm) is None)
for m in (HDR + b"\x07example\x03com\x00\x00\x10\x00\x01",
          HDR + b"\x03abc\x00\xc0\x0c",                       # a name that is one pointer (to a name)
          HDR + b"\x00"):                                     # the root alone
    check(f"known name {P._qname(m, 12)}", qname_v41(m, 12) == P._qname(m, 12))
check("a trailing root after a jump is kept, as in v4.1 ('abc.')",
      P._qname(HDR + b"\x03abc\x00\x03xyz\xc0\x0c", 17) == qname_v41(HDR + b"\x03abc\x00\x03xyz\xc0\x0c", 17))

def long_name(octets):
    """A name of exactly this many wire octets (root included), as labels of up to 63 bytes."""
    labs, left = [], octets - 1
    while left > 0:
        take = min(64, left)
        if take == 1:                                       # cannot end on a bare length byte: shorten the previous label
            labs[-1] = labs[-1][:-1]; take = 2
        labs.append(b"x" * (take - 1)); left -= take
    return b"".join(bytes([len(l)]) + l for l in labs) + b"\x00"
for octets, ok in ((254, True), (255, True), (256, False), (300, False)):
    nm = long_name(octets)
    check(f"a name of {octets} octets on the wire is {'decoded' if ok else 'rejected (RFC 1035: at most 255)'}",
          len(nm) == octets and ((P._qname(HDR + nm, 12) is not None) == ok) and (qname_ref(HDR + nm, 12) is not None) == ok)
check("... v4.1 took the 256- and 300-octet names", qname_v41(HDR + long_name(256), 12) is not None and qname_v41(HDR + long_name(300), 12) is not None)
check("... and the malformed reason says why", P.malformed_reason(HDR + long_name(300) + b"\x00\x10\x00\x01") == "name longer than 255 octets")

loop = HDR + b"\x01a\xc0\x0c" + b"\x00\x10\x00\x01"        # 'a' then a pointer back to its own start
try:
    qname_v41(loop, 12); old = "returned"
except RecursionError:
    old = "RecursionError"
check("a looping pointer: v4.1 raised RecursionError (the bug)", old == "RecursionError", old)
check("a looping pointer: v4.2 rejects the name", P._qname(loop, 12) is None and P.parse_dns_question(loop) is None)

def chain(n):                                              # n pointers, each to the one before; the first to 'abc'
    m = bytearray(HDR + b"\x03abc\x00"); prev = 12
    for _ in range(n):
        here = len(m); m += ptr(prev); prev = here
    return bytes(m), prev
for n, ok in ((1, True), (128, True), (129, False), (5000, False)):
    m, last = chain(n)
    got = P._qname(m, last)
    check(f"a chain of {n} pointers is {'decoded' if ok else 'rejected, not fatal'}",
          (got is not None and got[0] == "abc") if ok else got is None)

# ------------------------------------------------------------------ pcap helpers
def dns_q(txid, name, qtype=16):
    out = struct.pack("!HHHHHH", txid & 0xFFFF, 0x0100, 1, 0, 0, 0)
    for lab in name.split("."):
        out += bytes([len(lab)]) + lab.encode()
    return out + b"\x00" + struct.pack("!HH", qtype, 1)

def eth(ip):
    return b"\x00\x01\x02\x03\x04\x05" + b"\x06\x07\x08\x09\x0a\x0b" + b"\x08\x00" + ip

def udp_frame(src, dst, sport, dport, payload):
    u = struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload
    return eth(struct.pack("!BBHHHBBH", 0x45, 0, 20 + len(u), 0, 0, 64, 17, 0) + socket.inet_aton(src) + socket.inet_aton(dst) + u)

def tcp_frame(src, dst, sport, dport, payload, seq=1):
    t = struct.pack("!HHIIBBHHH", sport, dport, seq & 0xFFFFFFFF, 1, 0x50, 0x18, 65535, 0, 0) + payload
    return eth(struct.pack("!BBHHHBBH", 0x45, 0, 20 + len(t), 0, 0, 64, 6, 0) + socket.inet_aton(src) + socket.inet_aton(dst) + t)

def write(path, frames):
    with open(path, "wb") as fh:
        fh.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 262144, 1))
        for ts, f in frames:
            fh.write(struct.pack("<IIII", int(ts), int(round((ts % 1) * 1e6)), len(f), len(f)) + f)

TMP = tempfile.mkdtemp(prefix="v42_")
def pcap(name, frames):
    p = os.path.join(TMP, name); write(p, frames); return p

C, S = "10.1.2.3", "9.9.9.9"
def udp_queries(n, t0=1.7e9, sport0=40000, tag="u"):
    return [(t0 + i * 0.1, udp_frame(C, S, sport0 + i, 53, dns_q(1000 + i, f"{tag}{i:04d}.t.example.org"))) for i in range(n)]

# ------------------------------------------------------------------ B. a hostile packet no longer costs the capture
print("\n=== B. a packet that raises is counted and skipped; a capture ends only when its container fails ===")
good = udp_queries(40)
frames = good[:20] + [(1.7e9 + 2.05, udp_frame(C, S, 41000, 53, loop))] + good[20:]
qs, st = P.parse_pcap(pcap("loop.pcap", frames))
check("a looping-pointer packet in the middle: all 40 queries survive", len(qs) == 40 and st["error"] is None, f"{len(qs)} queries")
check("... and it is counted as malformed, once", st["malformed"] == 1 and st["malformed_c2s"] == 1 and st["packet_errors"] == 0)

real = P.parse_dns_question
def flaky(msg):
    if msg[12:16] == b"\x01a\xc0\x0c":
        raise ValueError("synthetic failure")
    return real(msg)
P.parse_dns_question = flaky
try:
    qs, st = P.parse_pcap(pcap("loop2.pcap", frames))
finally:
    P.parse_dns_question = real
check("a decoder exception: counted, sampled, the capture goes on",
      len(qs) == 40 and st["packet_errors"] == 1 and st["error"] is None and "synthetic failure" in st["packet_error_samples"],
      f"{len(qs)} queries, packet_errors {st['packet_errors']}, '{st['packet_error_samples']}'")

def memory_hog(msg):
    if msg[12:16] == b"\x01a\xc0\x0c":
        raise MemoryError("out of memory")
    return real(msg)
P.parse_dns_question = memory_hog
try:
    qs, st = P.parse_pcap(pcap("loop3.pcap", frames))
finally:
    P.parse_dns_question = real
check("a MemoryError is not swallowed per packet: the capture ends and says so",
      len(qs) == 20 and (st["error"] or "").startswith("MemoryError") and st["packet_errors"] == 0, f"{len(qs)} queries, error {st['error']}")

p = pcap("whole.pcap", good)
raw = open(p, "rb").read()
cut = os.path.join(TMP, "cut.pcap"); open(cut, "wb").write(raw[:-30])         # the last packet is cut short
qs, st = P.parse_pcap(cut)
check("a pcap cut mid-packet keeps every complete packet", len(qs) == 39, f"{len(qs)} of 40, error {st['error']}")
qs, st = P.parse_pcap(os.path.join(TMP, "nonexistent.pcap"))
check("an unreadable file is reported, not raised", qs == [] and st["error"] is not None, str(st["error"]))

# ------------------------------------------------------------------ C. DNS over TCP, length prefix in its own segment
print("\n=== C. DNS over TCP: a 2-byte length sent on its own ===")
m1 = dns_q(77, "abc.t.exfil-lab.net")
pre1 = struct.pack("!H", len(m1))
t0 = 1.7e9
frames = [(t0, tcp_frame(C, S, 50000, 53, pre1 + m1))]                                                   # together
qs, st = P.parse_pcap(pcap("tcp_a.pcap", frames))
check("prefix and message in one segment: read at offset 2, as in v4.1",
      [q.qname for q in qs] == ["abc.t.exfil-lab.net"] and st["tcp_split_read"] == 0 and st["malformed"] == 0
      and st["tcp_prefix_seen"] == 0)

frames = [(t0, tcp_frame(C, S, 50001, 53, pre1, seq=1000)), (t0 + .001, tcp_frame(C, S, 50001, 53, m1, seq=1002))]   # apart
qs, st = P.parse_pcap(pcap("tcp_b.pcap", frames))
check("prefix alone, then the message: recovered from offset 0",
      [q.qname for q in qs] == ["abc.t.exfil-lab.net"] and st["tcp_split_read"] == 1 and st["malformed"] == 0
      and st["tcp_prefix_seen"] == 1 and st["tcp_prefix_dropped"] == 0)

frames = [(t0, tcp_frame(C, S, 50002, 53, pre1, seq=2000)), (t0 + .0005, tcp_frame(C, S, 50002, 53, b"", seq=2002)),     # an ACK between
          (t0 + .001, tcp_frame(C, S, 50002, 53, m1, seq=2002))]
qs, st = P.parse_pcap(pcap("tcp_c.pcap", frames))
check("a bare ACK between the prefix and the message does not lose the prefix",
      len(qs) == 1 and st["tcp_split_read"] == 1)

ma, mb = dns_q(1, "aaaa.t.exfil-lab.net"), dns_q(2, "bbbbbbbbbbbbbbbb.t.exfil-lab.net")
frames = [(t0, tcp_frame(C, S, 50003, 53, struct.pack("!H", len(ma)), seq=3000)),
          (t0 + .0001, tcp_frame(C, S, 50004, 53, struct.pack("!H", len(mb)), seq=4000)),
          (t0 + .001, tcp_frame(C, S, 50003, 53, ma, seq=3002)), (t0 + .002, tcp_frame(C, S, 50004, 53, mb, seq=4002))]
qs, st = P.parse_pcap(pcap("tcp_d.pcap", frames))
check("two connections interleaved: each prefix goes with its own message",
      sorted(q.qname for q in qs) == ["aaaa.t.exfil-lab.net", "bbbbbbbbbbbbbbbb.t.exfil-lab.net"] and st["tcp_split_read"] == 2)

frames = [(t0, tcp_frame(C, S, 50005, 53, struct.pack("!H", 999), seq=5000)), (t0 + .001, tcp_frame(C, S, 50005, 53, pre1 + m1, seq=5002))]
qs, st = P.parse_pcap(pcap("tcp_e.pcap", frames))
check("a prefix that does not match the next segment's length is ignored (v4.1 reading), and counted as dropped",
      [q.qname for q in qs] == ["abc.t.exfil-lab.net"] and st["tcp_split_read"] == 0 and st["tcp_prefix_dropped"] == 1)

frames = [(t0 + i * .01, tcp_frame(C, S, 20000 + i, 53, struct.pack("!H", len(m1)), seq=7000)) for i in range(5000)]
frames.append((t0 + 100, tcp_frame(C, S, 29999, 53, m1, seq=7002)))
frames[-2:-1] = [(t0 + 99, tcp_frame(C, S, 29999, 53, struct.pack("!H", len(m1)), seq=7000))]
qs, st = P.parse_pcap(pcap("tcp_f.pcap", frames))
check("5,000 unanswered prefixes: bounded memory, and a later pair is still recovered", len(qs) == 1 and st["tcp_split_read"] == 1,
      f"{len(qs)} queries")
check("... and the evicted prefixes are counted, not silently lost", st["tcp_prefix_seen"] == 5000 and st["tcp_prefix_dropped"] > 0,
      f"seen {st['tcp_prefix_seen']}, dropped {st['tcp_prefix_dropped']}")

r = dns_q(5, "resp.t.example.org"); r = r[:2] + b"\x81\x80" + r[4:]               # a response
frames = [(t0, tcp_frame(S, C, 53, 50006, struct.pack("!H", len(r)), seq=6000)), (t0 + .001, tcp_frame(S, C, 53, 50006, r, seq=6002))]
qs, st = P.parse_pcap(pcap("tcp_g.pcap", frames))
check("a response sent in two segments is not a query", qs == [] and st["malformed"] == 0)

# ---- v4.2 review: the stray 2-byte segment. Reading an ordinary [length][message] buffer from offset 0 parses as the
# root name (byte 10, the high byte of ARCOUNT, is 0), so a remembered value equal to the next segment's length used to
# replace a real query with ''.
qq = dns_q(0x1234, "qqqqqqqq.example.org")
ordinary = struct.pack("!H", len(qq)) + qq                                          # one ordinary segment, [length][message]
stray = struct.pack("!H", len(ordinary))                                            # a bare 2-byte segment equal to its length
reading = P.parse_dns_question(ordinary)
check("(the trap) offset 0 of an ordinary buffer does parse, as a root name", reading is not None and reading[0] == "",
      f"{reading}")
frames = [(t0, tcp_frame(C, S, 50007, 53, stray, seq=8000)), (t0 + .01, tcp_frame(C, S, 50007, 53, ordinary, seq=90000))]
qs, st = P.parse_pcap(pcap("tcp_h.pcap", frames))
check("a stray 2-byte segment, then a segment of that length elsewhere in the stream: the real query, not ''",
      [q.qname for q in qs] == ["qqqqqqqq.example.org"] and st["tcp_split_read"] == 0 and st["tcp_prefix_dropped"] == 1,
      f"{[q.qname for q in qs]}, split reads {st['tcp_split_read']}")
frames = [(t0, tcp_frame(C, S, 50008, 53, stray, seq=8000)), (t0 + .01, tcp_frame(C, S, 50008, 53, ordinary, seq=8002))]
qs, st = P.parse_pcap(pcap("tcp_i.pcap", frames))
check("... even when the segment follows at exactly the next sequence number (the question count rules offset 0 out)",
      [q.qname for q in qs] == ["qqqqqqqq.example.org"] and st["tcp_split_read"] == 0,
      f"{[q.qname for q in qs]}, split reads {st['tcp_split_read']}")
frames = [(t0, tcp_frame(C, S, 50009, 53, pre1, seq=9000)), (t0 + .001, tcp_frame(C, S, 50009, 53, m1, seq=9500))]
qs, st = P.parse_pcap(pcap("tcp_j.pcap", frames))
check("the message at the wrong sequence number is not the one the prefix announced",
      st["tcp_split_read"] == 0 and st["tcp_prefix_dropped"] == 1 and st["tcp_prefix_seen"] == 1)
frames = [(t0, tcp_frame(C, S, 50010, 53, pre1, seq=0xFFFFFFFF)), (t0 + .001, tcp_frame(C, S, 50010, 53, m1, seq=1))]
qs, st = P.parse_pcap(pcap("tcp_k.pcap", frames))
check("sequence numbers wrap at 2^32", [q.qname for q in qs] == ["abc.t.exfil-lab.net"] and st["tcp_split_read"] == 1)
frames = [(t0, tcp_frame(C, S, 50011, 53, pre1, seq=100)), (t0 + .0005, tcp_frame(C, S, 50011, 53, pre1, seq=100)),   # retransmitted
          (t0 + .001, tcp_frame(C, S, 50011, 53, m1, seq=102))]
qs, st = P.parse_pcap(pcap("tcp_l.pcap", frames))
check("a retransmitted prefix does not lose the pairing (and is counted)",
      [q.qname for q in qs] == ["abc.t.exfil-lab.net"] and st["tcp_split_read"] == 1 and st["tcp_prefix_seen"] == 2 and st["tcp_prefix_dropped"] == 1)
m2 = dns_q(78, "def.t.exfil-lab.net")
frames = [(t0, tcp_frame(C, S, 50012, 53, pre1, seq=100)), (t0 + .001, tcp_frame(C, S, 50012, 53, m1, seq=102)),
          (t0 + .002, tcp_frame(C, S, 50012, 53, struct.pack("!H", len(m2)), seq=102 + len(m1))),
          (t0 + .003, tcp_frame(C, S, 50012, 53, m2, seq=104 + len(m1)))]
qs, st = P.parse_pcap(pcap("tcp_m.pcap", frames))
check("pipelined [p1][m1][p2][m2] in four segments: both recovered",
      [q.qname for q in qs] == ["abc.t.exfil-lab.net", "def.t.exfil-lab.net"] and st["tcp_split_read"] == 2 and st["tcp_prefix_dropped"] == 0)
frames = [(t0, tcp_frame(C, S, 50013, 53, pre1, seq=100)), (t0 + .001, tcp_frame(C, S, 50013, 53, bytes(len(m1)), seq=102))]
qs, st = P.parse_pcap(pcap("tcp_n.pcap", frames))
check("a prefix followed by a segment of the right length and sequence number that is not DNS: malformed, once, no query",
      qs == [] and st["malformed"] == 1)

# ------------------------------------------------------------------ D. malformed_c2s
print("\n=== D. malformed payloads are split by direction ===")
junk = bytes(random.Random(5).randrange(256) for _ in range(120))
frames = udp_queries(10) + [(t0 + 5, udp_frame(C, S, 40500, 53, junk)), (t0 + 6, udp_frame(S, C, 53, 40500, junk))]
qs, st = P.parse_pcap(pcap("junk.pcap", frames))
check("one bad payload to port 53, one from port 53", st["malformed"] == 2 and st["malformed_c2s"] == 1 and len(qs) == 10,
      f"malformed {st['malformed']}, to 53: {st['malformed_c2s']}")
check("port53 counts both directions (unchanged denominator)", st["port53"] == 12)

# ------------------------------------------------------------------ E. the operating point
print("\n=== E. the operating point is exact ===")
def f32(x): return struct.unpack("<f", struct.pack("<f", x))[0]
budget = .01
for n in (500, 5000, 100000):
    r = random.Random(n)
    sc = [r.random() ** 8 for _ in range(n)]
    thr, meta = E.pick_threshold(sc, budget)
    k = sum(1 for x in sc if x >= thr)
    check(f"n = {n:>6,}: the threshold is an observed score" + ("" if meta["k_allowed"] >= 1 else " (k = 0: above every one)"),
          thr in set(sc) if meta["k_allowed"] >= 1 else thr > max(sc))
    check(f"n = {n:>6,}: burden inside the budget, and one more alert would break it ({k} alert)",
          E.wilson_upper(k, n) <= budget < E.wilson_upper(k + 1, n))
    check(f"n = {n:>6,}: the budget bound exactly (limit 'budget', k_allowed == alerts, {meta['budget_used']:.1%} of it used)",
          meta["budget_bound"] and meta["limit"] == "budget" and meta["k_allowed"] == k and meta["budget_used"] <= 1.0
          and (n < 5000 or meta["budget_used"] > 0.9))      # at n = 500 one alert is a big step: k_allowed is 0, 76% used

sc = [0.1] * 3000 + [0.7] * 200 + [0.9] * 5
thr, meta = E.pick_threshold(sc, budget)
k = sum(1 for x in sc if x >= thr)
check("a tie group straddling the limit: the whole group stays silent, and the meta says it was a tie plateau",
      thr > 0.7 and E.wilson_upper(k, len(sc)) <= budget and meta["budget_bound"] and meta["limit"] == "tie plateau" and k < meta["k_allowed"],
      f"thr={thr}, alerts {k}, k_allowed {meta['k_allowed']}")

thr, meta = E.pick_threshold([0.0] * 5000, budget)
check("all benign scores zero: the budget is not what limits sensitivity", not meta["budget_bound"] and meta["limit"] == "zero scores"
      and 0.0 < thr < 1e-30 and f32(thr) == thr and f32(thr) > 0.0, f"thr={thr!r}")
sc = [0.0] * 4990 + [0.3] * 10
thr, meta = E.pick_threshold(sc, budget)
check("zeros plus 10 positives, inside the budget: all 10 alert, and the budget is not what limits sensitivity",
      (not meta["budget_bound"]) and sum(1 for x in sc if x >= thr) == 10 and 10 <= meta["k_allowed"], f"k_allowed {meta['k_allowed']}")
sc = [0.0] * 4900 + [0.3] * 100
thr, meta = E.pick_threshold(sc, budget)
check("zeros plus 100 positives, over the budget: the tie group of positives stays silent",
      meta["budget_bound"] and thr > 0.3 and sum(1 for x in sc if x >= thr) == 0, f"k_allowed {meta['k_allowed']}")

sc = [float("nan")] * 10 + [0.0] * 4000 + [0.2] * 990
thr, meta = E.pick_threshold(sc, budget)
check("NaN scores are counted and never alert", meta["n_nan"] == 10 and thr == thr and not (float("nan") >= thr)
      and sum(1 for x in sc if x == x and x >= thr) <= meta["k_allowed"], f"thr={thr}, n_nan {meta['n_nan']}")
thr, meta = E.pick_threshold([float("nan")] * 5000, budget)
check("all-NaN scores behave like all-zero scores (nothing can alert; the budget is not what limits)",
      meta["n_nan"] == 5000 and not meta["budget_bound"] and meta["limit"] == "zero scores" and 0.0 < thr < 1e-30)

thr, meta = E.pick_threshold([0.99] * 100, budget)
check("100 benign units cannot certify the budget: nothing alerts, and 'the budget bound' is not claimed",
      thr == math.inf and meta["reason"] == "budget unreachable" and meta["budget_bound"] is None and meta["limit"] == "not certifiable"
      and meta["budget_used"] > 1.0 and meta["train_burden"] == 0.0, f"budget used {meta['budget_used']:.2f}")
check("... and an infinite threshold stays silent in float32", not (f32(1.0) >= thr))
m0 = E.pick_threshold([], budget)
check("no benign units at all: nothing alerts (inf, like every other 'alert on nothing')", m0[0] == math.inf and m0[1]["budget_bound"] is None
      and m0[1]["limit"] == "not certifiable")
thr, meta = E.pick_threshold([0.1] * 10, 1.0)
check("a budget of 100% admits everything", thr == 0.0 and not meta["budget_bound"])

r = random.Random(3)
s32 = [f32(r.betavariate(0.05, 5)) for _ in range(50000)]                 # scores as a float32 model emits them
thr, meta = E.pick_threshold(s32, budget)
a_double = sum(1 for x in s32 if x >= thr); a_single = sum(1 for x in s32 if f32(x) >= f32(thr))
check("float32 scores and a float64 threshold agree on who alerts", a_double == a_single == round(meta["train_burden"] * len(s32)),
      f"{a_double} / {a_single}")
pivot = sorted(s32)[len(s32) - meta["k_allowed"] - 1]
check("the (k+1)-th highest score stays silent", pivot < thr)

def grid_v41(scores, budget, grid):
    import bisect
    s = sorted(scores); n = len(s)
    for t in sorted(grid):
        k = n - bisect.bisect_left(s, t); u = E.wilson_upper(k, n)
        if u <= budget: return (t, k / n, u)
    return (1.0 + 1e-9, 0.0, 0.0)
bad = 0
grid = [i / 200.0 for i in range(200, -1, -1)]
for seed in range(200):
    r = random.Random(seed); n = r.choice([400, 1000, 4000])
    sc = [r.random() ** r.choice([1, 3, 8]) for _ in range(n)]
    t_new, m_new = E.pick_threshold(sc, budget, grid=grid)
    t_old, b_old, u_old = grid_v41(sc, budget, grid)
    if not (t_new == t_old and abs(m_new["train_burden"] - b_old) < 1e-15 and abs(m_new["train_upper"] - u_old) < 1e-15): bad += 1
check("grid= reproduces the v4.1 search exactly (200 random cases)", bad == 0, f"{bad} differ")
r = random.Random(11)
sc = [r.random() * 0.004 for _ in range(20000)]                      # a model that scores every benign unit below 0.004
t_grid, m_grid = E.pick_threshold(sc, budget, grid=grid)
t_exact, m_exact = E.pick_threshold(sc, budget)
check("the v4.1 blind spot: on the grid the search stops at 0.005 and the budget never bound",
      t_grid == 0.005 and not m_grid["budget_bound"] and m_grid["train_burden"] == 0.0 and m_grid["budget_used"] < 0.2,
      f"thr={t_grid}, budget_bound={m_grid['budget_bound']}, burden {m_grid['train_burden']}")
check("... the exact rule goes far below it and uses the budget", t_exact < 0.004 and m_exact["budget_bound"] and 0.005 < m_exact["train_burden"] <= budget,
      f"thr={t_exact:.2e}, burden {m_exact['train_burden']:.4f}")
sc = [0.0] * 990 + [0.6] * 10
t_grid, m_grid = E.pick_threshold(sc, budget, grid=grid)
check("on the grid, when benign scores reach past the lowest setting, the budget does bind", t_grid > 0.6 and m_grid["budget_bound"], f"thr={t_grid}")

for n in (400, 1000, 10000):
    kk = E.max_alerts_within(n, budget)
    check(f"max_alerts_within({n}) is the largest k inside the budget ({kk})", E.wilson_upper(kk, n) <= budget < E.wilson_upper(kk + 1, n))
check("max_alerts_within is -1 when even zero alerts cannot be certified", E.max_alerts_within(100, budget) == -1)

# ------------------------------------------------------------------ F. family-level bootstrap
print("\n=== F. the family-weighted mean and its two-stage bootstrap ===")
m, lo, hi = E.family_bootstrap({"a": [1.0, 1.0], "b": [0.0], "c": [0.0, 0.0, 0.0, 0.0]}, n_boot=500)
check("every family has the same weight (1/3, not 2/7)", abs(m - 1 / 3) < 1e-12, f"{m:.4f}")
check("the interval brackets the point estimate", lo <= m <= hi, f"[{lo:.3f}, {hi:.3f}]")
m, lo, hi = E.family_bootstrap({"a": [0.4, 0.4], "b": [0.4], "c": [0.4] * 5}, n_boot=300)
check("no variation, no interval", lo == hi == m == 0.4 or abs(lo - 0.4) < 1e-12 and abs(hi - 0.4) < 1e-12)
a = E.family_bootstrap({"a": [1.0, 2.0], "b": [3.0], "c": [5.0, 6.0, 7.0]}, n_boot=400, seed=9)
b = E.family_bootstrap({"a": [1.0, 2.0], "b": [3.0], "c": [5.0, 6.0, 7.0]}, n_boot=400, seed=9)
check("deterministic given the seed", a == b)
m, lo, hi = E.family_bootstrap({"only": [1.0, 3.0]})
check("a single family has a mean and no interval", m == 2.0 and math.isnan(lo) and math.isnan(hi))
check("nothing in, NaN out", math.isnan(E.family_bootstrap({})[0]))
wide = E.family_bootstrap({"a": [0.0], "b": [10.0], "c": [0.0], "d": [10.0]}, n_boot=2000)
flat = E.family_bootstrap({"a": [0.0, 10.0, 0.0, 10.0], "b": [0.0, 10.0, 0.0, 10.0], "c": [0.0, 10.0, 0.0, 10.0], "d": [0.0, 10.0, 0.0, 10.0]}, n_boot=2000)
check("families that differ give a wider interval than the same spread inside every family", (wide[2] - wide[1]) > (flat[2] - flat[1]),
      f"{wide[2] - wide[1]:.2f} vs {flat[2] - flat[1]:.2f}")

# ------------------------------------------------------------------ G. the v4.2 review: threshold edge cases
print("\n=== G. pick_threshold after the review: tie plateaus, +inf, tiny values, odd inputs ===")
TINY = E._TINY_F32
try:
    import numpy as np
except ImportError:
    np = None

def next_up_ref(x):
    """The smallest float32 strictly above x (numpy's nextafter), as a float."""
    if x >= E._MAX_F32: return math.inf
    if x < -E._MAX_F32: return -E._MAX_F32
    f = np.float32(x)
    return float(f) if float(f) > x else float(np.nextafter(f, np.float32(np.inf)))
if np is not None:
    r = random.Random(4)
    xs = [0.0, 1e-50, 1e-46, 7.0e-46, 7.1e-46, 1.4e-45, 2.8e-45, 1e-40, 0.3, 0.6, 1.0, 3.0e38, 3.4e38, 3.5e38,
          -1e-50, -1.4e-45, -0.3, -3.4e38, -1e39] + [r.random() for _ in range(3000)] + [r.random() ** 40 for _ in range(3000)] \
         + [-r.random() ** 5 for _ in range(500)] + [float(np.float32(r.random())) for _ in range(3000)]
    bad = [x for x in xs if E._next_up_f32(x) != next_up_ref(x) or not E._next_up_f32(x) > x]
    check(f"_next_up_f32 equals numpy's nextafter on {len(xs):,} values, denormals, zero and negatives included", not bad, f"{bad[:3]}")
check("_next_up_f32 of a value below the smallest float32 is the smallest float32, not NaN (v4.2 review: it was NaN)",
      E._next_up_f32(1e-50) == TINY and E._next_up_f32(0.0) == TINY and E._next_up_f32(math.inf) == math.inf)
thr, meta = E.pick_threshold([1e-50] * 1000, budget)
check("scores below float32's range: a real threshold and a burden of 0, not NaN", thr == TINY and meta["train_burden"] == 0.0)

thr, meta = E.pick_threshold([0.1] * 16800 + [0.7] * 3000 + [0.9] * 200, budget)
check("(review 2) a tie plateau is reported as one: 0 of a budget of 172 alerts used, not 'the budget was binding'",
      meta["limit"] == "tie plateau" and meta["k_allowed"] == 172 and meta["train_burden"] == 0.0 and meta["budget_used"] < 0.03
      and thr > 0.9, f"limit {meta['limit']}, budget used {meta['budget_used']:.3f}")
thr, meta = E.pick_threshold([0.5] * 990 + [math.inf] * 10, budget)
check("(review 4) +inf scores cannot be silenced: the burden is reported honestly and flagged over budget",
      meta["limit"] == "over budget" and meta["budget_bound"] is None and meta["budget_used"] > 1.0 and abs(meta["train_burden"] - 0.01) < 1e-12,
      f"limit {meta['limit']}, burden {meta['train_burden']}")
thr, meta = E.pick_threshold([math.inf] * 100, budget)
check("... and with too few units, inf scores are not reported as 'zero alerts'", thr == math.inf and meta["train_burden"] == 1.0 and meta["limit"] == "not certifiable")

try:
    E.pick_threshold([0.1] * 1000, float("nan")); nan_budget = "accepted"
except ValueError:
    nan_budget = "ValueError"
check("(review 6) a NaN budget raises ValueError", nan_budget == "ValueError", nan_budget)
thr, meta = E.pick_threshold((x for x in [0.1] * 990 + [0.9] * 10), budget)
check("(review 6) a generator of scores is read once, not turned into 'no benign units'", meta["n_benign"] == 1000 and meta["limit"] != "not certifiable")
thr, meta = E.pick_threshold([0.1] * 1000, budget, grid=[])
check("(review 6) an empty grid is 'unreachable', as in v4.1, not a ValueError from min()", thr == math.inf and meta["reason"] == "budget unreachable")
sc = [-5.0 + 3.0 * random.Random(i).random() for i in range(5000)]                   # log-odds, outside [0, 1]
thr, meta = E.pick_threshold(sc, budget)
check("(review 6) scores below zero use the general rule (v4.2 first cut: 0 alerts with 36 allowed)",
      sum(1 for x in sc if x >= thr) == meta["k_allowed"] == 36 and meta["limit"] == "budget", f"thr {thr:.4f}")

# an independent reference: scan k upward with Wilson, then check the answer against every observed score
def reference(scores, budget):
    s = sorted((-math.inf if x != x else x) for x in scores); n = len(s)
    kmax = -1
    for k in range(0, n + 1):
        if E.wilson_upper(k, n) <= budget: kmax = k
        else: break
    return s, n, kmax
bad = []; classes = {}
for seed in range(400):
    r = random.Random(1000 + seed)
    n = r.choice([381, 400, 500, 1000, 3000, 8000]); kind = r.choice(["cont", "ties", "zeros", "rf", "mixed", "nan", "tiny"])
    if kind == "cont": sc = [r.random() ** r.choice([1, 3, 8]) for _ in range(n)]
    elif kind == "ties": sc = [r.choice([0.1, 0.3, 0.7, 0.9]) for _ in range(n)]
    elif kind == "zeros":
        pz = r.choice([.5, .99, .9995]); sc = [0.0 if r.random() < pz else r.random() for _ in range(n)]
    elif kind == "rf": sc = [r.randrange(0, 301) / 300 if r.random() < .3 else 0.0 for _ in range(n)]
    elif kind == "mixed": sc = [r.random() ** 8 for _ in range(n // 2)] + [0.25] * (n - n // 2)
    elif kind == "nan": sc = [float("nan") if r.random() < .3 else r.random() ** 4 for _ in range(n)]
    else: sc = [r.random() * 1e-3 for _ in range(n)]
    thr, meta = E.pick_threshold(sc, budget)
    s, n_, kmax = reference(sc, budget)
    if kmax < 0:
        ok = thr == math.inf and meta["limit"] == "not certifiable"
    else:
        alerts = sum(1 for x in sc if x == x and x >= thr)
        ok = alerts <= kmax and meta["k_allowed"] == kmax and abs(meta["train_burden"] * n - alerts) < 1e-6 and meta["train_upper"] <= budget
        # the next lower distinct observed score would put more than kmax units over (unless zeros, or everything fits)
        lower = [x for x in sorted(set(s)) if x < thr and x > -math.inf]
        if lower and meta["limit"] in ("budget", "tie plateau"):
            below = max(lower)
            ok = ok and sum(1 for x in s if x >= below) > kmax
        if meta["limit"] == "budget": ok = ok and alerts == kmax
        if meta["limit"] == "tie plateau": ok = ok and alerts < kmax
        if meta["limit"] == "zero scores": ok = ok and thr == TINY
    classes[meta["limit"]] = classes.get(meta["limit"], 0) + 1
    if not ok: bad.append((seed, kind, n, meta["limit"]))
check("400 random score sets (continuous, ties, zeros, RF-like, NaN, tiny): alerts <= k_allowed, the next score down would break the budget, "
      "and the limit label is the right one", not bad, f"{bad[:3]}; labels seen {classes}")
check("... the labels cover budget, tie plateau and zero scores", {"budget", "tie plateau", "zero scores"} <= set(classes), str(classes))

summ = E.limit_summary([{"limit": "budget", "budget_used": 0.98}, {"limit": "tie plateau", "budget_used": 0.02},
                        {"limit": "not certifiable", "budget_used": 1.26}, {"limit": "zero scores", "budget_used": 0.0}])
check("limit_summary counts the labels and leaves 'not certifiable' out of the budget-used range",
      summ["budget"] == 1 and summ["tie plateau"] == 1 and summ["not certifiable"] == 1 and summ["zero scores"] == 1 and summ["folds"] == 4
      and summ["used_max"] == 0.98 and summ["used_min"] == 0.0 and abs(summ["used_median"] - 0.02) < 1e-12, str(summ))

# ------------------------------------------------------------------ H. intervals
print("\n=== H. the family-level intervals ===")
order_a = {"a": [0.0, 0.5], "b": [1.0], "c": [0.2, 0.4, 0.9], "d": [0.1], "e": [0.7, 0.8]}
order_b = {k: order_a[k] for k in ["e", "c", "a", "d", "b"]}
check("(review 8) family_bootstrap does not depend on the order of the dict", E.family_bootstrap(order_a, n_boot=2000) == E.family_bootstrap(order_b, n_boot=2000))
for bad_v in (float("nan"), float("inf")):
    try:
        E.family_bootstrap({"a": [0.1, bad_v], "b": [0.2]}, n_boot=50); got = "returned"
    except ValueError as e:
        got = "ValueError" if "'a'" in str(e) else "ValueError without the family"
    check(f"(review 8) a non-finite value ({bad_v}) stops with the family named, instead of returning (nan, x, nan)", got == "ValueError", got)

known_t = {(0.975, 1): 12.706204736, (0.975, 2): 4.302652730, (0.975, 3): 3.182446305, (0.975, 7): 2.364624252,
           (0.975, 10): 2.228138852, (0.975, 30): 2.042272456, (0.95, 7): 1.894578605, (0.995, 7): 3.499483297, (0.9, 5): 1.475884049}
worst = max(abs(E.t_quantile(p, df) - v) for (p, df), v in known_t.items())
check("t_quantile reproduces published critical values to 1e-8", worst < 1e-8, f"worst {worst:.2e}")
check("t_quantile is symmetric and 0 at the median", E.t_quantile(0.5, 4) == 0.0 and abs(E.t_quantile(0.025, 7) + E.t_quantile(0.975, 7)) < 1e-12)
try:
    import scipy.stats as sst
    worst = max(abs(E.t_quantile(p, df) - sst.t.ppf(p, df)) / max(1.0, abs(sst.t.ppf(p, df)))
                for df in list(range(1, 41)) + [60, 120, 1000] for p in (0.6, 0.9, 0.95, 0.975, 0.99, 0.995, 0.9995))
    check("t_quantile equals scipy's to 1e-9 over df 1-40, 60, 120, 1000", worst < 1e-9, f"worst {worst:.2e}")
    worst = 0.0
    for n_ in (1, 2, 3, 8, 18, 24, 54, 381, 2000):
        for k_ in sorted({0, 1, 2, 3, n_ // 2, n_ - 1, n_}):
            if 0 <= k_ <= n_:
                lo_, hi_ = E.clopper_pearson(k_, n_); ci = sst.binomtest(k_, n_).proportion_ci(confidence_level=0.95, method="exact")
                worst = max(worst, abs(lo_ - ci.low), abs(hi_ - ci.high))
    check("clopper_pearson equals scipy's exact interval to 1e-9 (n up to 2,000)", worst < 1e-9, f"worst {worst:.2e}")
except ImportError:
    print("  (scipy not installed: the t and exact-interval checks against it are skipped; the published values above still ran)")
lo_, hi_ = E.clopper_pearson(3, 8)
check("3 of 8 families: exact interval [8.5%, 75.5%]", abs(lo_ - 0.08523) < 1e-4 and abs(hi_ - 0.75514) < 1e-4, f"[{lo_:.4f}, {hi_:.4f}]")
check("0 of n and n of n have the closed forms 1-(a/2)^(1/n) and (a/2)^(1/n)",
      abs(E.clopper_pearson(0, 8)[1] - (1 - 0.025 ** (1 / 8))) < 1e-12 and abs(E.clopper_pearson(8, 8)[0] - 0.025 ** (1 / 8)) < 1e-12
      and E.clopper_pearson(0, 8)[0] == 0.0 and E.clopper_pearson(8, 8)[1] == 1.0)
check("the exact interval contains the Wilson interval's point and is no narrower at k = 3, n = 8",
      E.clopper_pearson(3, 8)[1] - E.clopper_pearson(3, 8)[0] > E.wilson(3, 8)[1] - E.wilson(3, 8)[0])

m, lo, hi = E.family_t_interval({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]})
h = 3.182446305 * math.sqrt(sum((x - 2.5) ** 2 for x in (1, 2, 3, 4)) / 3) / 2
check("family_t_interval: mean of the family means +- t(0.975, k-1) * sd / sqrt(k)", abs(m - 2.5) < 1e-12 and abs(lo - (2.5 - h)) < 1e-6 and abs(hi - (2.5 + h)) < 1e-6,
      f"[{lo:.4f}, {hi:.4f}]")
m, lo, hi = E.family_t_interval({"a": [0.0, 1.0], "b": [0.5], "c": [0.5, 0.5, 0.5]})
check("... every family counts once, whatever its size", abs(m - 0.5) < 1e-12 and abs(lo - 0.5) < 1e-12 and abs(hi - 0.5) < 1e-12)
m, lo, hi = E.family_t_interval({"a": [0.0], "b": [0.0], "c": [1.0], "d": [1.0], "e": [1.0]}, clip=(0.0, 1.0))
check("... clip keeps a fraction inside [0, 1]", 0.0 <= lo < m < hi <= 1.0 and hi == 1.0, f"[{lo:.3f}, {hi:.3f}]")
check("... one family has a mean and no interval; none gives NaN", E.family_t_interval({"a": [1.0, 3.0]})[0] == 2.0
      and math.isnan(E.family_t_interval({"a": [1.0, 3.0]})[1]) and math.isnan(E.family_t_interval({})[0]))

# coverage at the size of the study, so that the label in the notebook is a measurement and not a hope
print("  (coverage simulation: 8 families, sizes 4,4,3,3,2,2,1,1; 1,000 repetitions; a 2,000-repetition run gave 0.907 and 0.950)")
sizes = [4, 4, 3, 3, 2, 2, 1, 1]
cov_b = cov_t = 0; reps = 1000
for rep_ in range(reps):
    rr = random.Random(7000 + rep_)
    centres = [rr.gauss(0.4, 0.15) for _ in sizes]
    fams = {f"f{i}": [c + rr.gauss(0, 0.10) for _ in range(sz)] for i, (c, sz) in enumerate(zip(centres, sizes))}
    target = 0.4
    _, lb, hb = E.family_bootstrap(fams, n_boot=500, seed=rep_)
    _, lt, ht = E.family_t_interval(fams)
    cov_b += lb <= target <= hb; cov_t += lt <= target <= ht
check(f"percentile cluster bootstrap at 8 families covers visibly less than 95% ({cov_b / reps:.3f}); the t interval is near 95% ({cov_t / reps:.3f})",
      cov_b / reps < 0.93 and 0.92 <= cov_t / reps <= 0.98 and cov_t - cov_b > 0.02 * reps)

print(f"\n{'ALL PASSED' if not FAILS else 'FAILED: ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
