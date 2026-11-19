"""v3 parser hardening: every link type the three source labs could have used, pcapng
block variants, IPv6 extension headers, and the diagnostics that make an unreadable
capture visible instead of silently empty."""
import socket, struct, sys
from ebd.dns.pcap_io import parse_pcap
from ebd.dns.engine import build_episodes

FAILS = []
def check(name, cond, d=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + d) if d else ''}")
    if not cond: FAILS.append(name)

def dns_query(txid, name, qtype=16):
    out = struct.pack("!HHHHHH", txid, 0x0100, 1, 0, 0, 0)
    for lab in name.split("."):
        out += bytes([len(lab)]) + lab.encode()
    return out + b"\x00" + struct.pack("!HH", qtype, 1)

def ip4_udp(src, dst, sport, payload):
    udp = struct.pack("!HHHH", sport, 53, 8 + len(payload), 0) + payload
    return struct.pack("!BBHHHBBH", 0x45, 0, 20 + len(udp), 0, 0, 64, 17, 0) + \
        socket.inet_aton(src) + socket.inet_aton(dst) + udp

def ip6_udp_hbh(src, dst, sport, payload):
    """IPv6 with a Hop-by-Hop options header in front of UDP."""
    udp = struct.pack("!HHHH", sport, 53, 8 + len(payload), 0) + payload
    hbh = bytes([17, 0]) + b"\x01\x04\x00\x00\x00\x00"          # next=UDP, len=0 (8 bytes), PadN
    body = hbh + udp
    return struct.pack("!IHBB", 0x60000000, len(body), 0, 64) + \
        socket.inet_pton(socket.AF_INET6, src) + socket.inet_pton(socket.AF_INET6, dst) + body

def packets(n=120, mk=ip4_udp, src="10.1.2.3", dst="9.9.9.9"):
    out = []
    for i in range(n):
        lab = (f"{i:05d}" + "abcdefghijklmnopqrstuvwxyz234567" * 3)[:45]
        out.append((1.7e9 + i / 2.0, mk(src, dst, 40000 + i, dns_query(1000 + i, f"{lab}.t.exfil-lab.net"))))
    return out

ETH = lambda ip, et=0x0800: b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b" + struct.pack("!H", et) + ip
SLL1 = lambda ip: struct.pack("!HHH", 0, 1, 6) + b"\x00" * 8 + b"\x08\x00" + ip
SLL2 = lambda ip: b"\x08\x00\x00\x00" + struct.pack("!IHBB", 2, 1, 0, 6) + b"\x00" * 8 + ip
def NFLOG(ip):
    tlv_hdr = struct.pack("<HH", 8, 1) + b"\x08\x00\x00\x00"                 # NFULA_PACKET_HDR
    pay = struct.pack("<HH", 4 + len(ip), 9) + ip + b"\x00" * ((4 - len(ip) % 4) % 4)
    return bytes([2, 0]) + b"\x00\x00" + tlv_hdr + pay

def write_pcap(path, frames, linktype_field):
    with open(path, "wb") as fh:
        fh.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 262144, linktype_field))
        for ts, f in frames:
            fh.write(struct.pack("<IIII", int(ts), int(round((ts % 1) * 1e6)), len(f), len(f)) + f)

def write_pcapng(path, frames, linktype, block="epb"):
    with open(path, "wb") as fh:
        fh.write(struct.pack("<IIIHHq", 0x0A0D0D0A, 28, 0x1A2B3C4D, 1, 0, -1) + struct.pack("<I", 28))
        idb = struct.pack("<HHI", linktype, 0, 262144)
        fh.write(struct.pack("<II", 1, 12 + len(idb)) + idb + struct.pack("<I", 12 + len(idb)))
        for ts, f in frames:
            pad = b"\x00" * ((4 - len(f) % 4) % 4)
            us = int(round(ts * 1e6))
            if block == "epb":
                body, bt = struct.pack("<IIIII", 0, us >> 32, us & 0xFFFFFFFF, len(f), len(f)) + f + pad, 6
            elif block == "pb":
                body, bt = struct.pack("<HHIIII", 0, 0, us >> 32, us & 0xFFFFFFFF, len(f), len(f)) + f + pad, 2
            else:
                body, bt = struct.pack("<I", len(f)) + f + pad, 3
            fh.write(struct.pack("<II", bt, 12 + len(body)) + body + struct.pack("<I", 12 + len(body)))

print("\n=== link types ===")
for name, wrap, lt in [("Ethernet", ETH, 1), ("Linux cooked v1 (113)", SLL1, 113),
                       ("Linux cooked v2 (276, tcpdump -i any on libpcap >= 1.10)", SLL2, 276),
                       ("raw IP (101)", lambda ip: ip, 101), ("raw IPv4 (228)", lambda ip: ip, 228),
                       ("NFLOG (239)", NFLOG, 239)]:
    write_pcap("/tmp/v3.pcap", [(t, wrap(p)) for t, p in packets()], lt)
    qs, st = parse_pcap("/tmp/v3.pcap")
    check(f"{name}: 120 queries", len(qs) == 120, f"{len(qs)}; {st['linktypes']}")

write_pcapng("/tmp/v3.pcapng", [(t, SLL2(p)) for t, p in packets()], 276)
qs, st = parse_pcap("/tmp/v3.pcapng")
check("pcapng + Linux cooked v2: 120 queries", len(qs) == 120, st["linktypes"])

write_pcap("/tmp/v3.pcap", [(t, ETH(p) + b"\xaa\xbb\xcc\xdd") for t, p in packets()], 0x14000001)
qs, st = parse_pcap("/tmp/v3.pcap")
check("FCS bits in the link-type field are masked off", len(qs) == 120, f"{len(qs)}; {st['linktypes']}")

print("\n=== pcapng block variants ===")
write_pcapng("/tmp/v3.pcapng", [(t, ETH(p)) for t, p in packets()], 1, block="pb")
qs, st = parse_pcap("/tmp/v3.pcapng")
check("obsolete Packet Blocks read, timestamps kept", len(qs) == 120 and abs(qs[1].ts - qs[0].ts - .5) < 1e-6,
      f"{len(qs)}")
write_pcapng("/tmp/v3.pcapng", [(t, ETH(p)) for t, p in packets()], 1, block="spb")
qs, st = parse_pcap("/tmp/v3.pcapng")
check("Simple Packet Blocks (no timestamp) are counted, not mis-timed",
      len(qs) == 0 and st["no_timestamp"] == 120, f"{st['no_timestamp']}")

print("\n=== IPv6 extension headers ===")
v6 = packets(mk=ip6_udp_hbh, src="2001:db8::5", dst="2001:db8::53")
write_pcap("/tmp/v3.pcap", [(t, ETH(p, 0x86DD)) for t, p in v6], 1)
qs, st = parse_pcap("/tmp/v3.pcap")
check("UDP behind a Hop-by-Hop header is found", len(qs) == 120 and qs[0].src == "2001:db8::5",
      f"{len(qs)}")

print("\n=== diagnostics ===")
write_pcap("/tmp/v3.pcap", [(t, b"\xde\xad" + p) for t, p in packets()], 147)
qs, st = parse_pcap("/tmp/v3.pcap")
check("unknown link type: reported, never guessed",
      len(qs) == 0 and st["unsupported_linktype"] == 120 and "UNSUPPORTED(147)" in st["linktypes"],
      st["linktypes"])
check("first bytes recorded for the diagnosis", st["first_bytes"].startswith("dead45"), st["first_bytes"][:12])
write_pcap("/tmp/v3.pcap", [(t, ETH(p, 0x0806)) for t, p in packets(20)], 1)
qs, st = parse_pcap("/tmp/v3.pcap")
check("non-IP frames counted as such", st["non_ip"] == 20 and not qs)
open("/tmp/v3.txt", "w").write("this is not a capture\n")
qs, st = parse_pcap("/tmp/v3.txt")
check("a non-capture file is an error, not an empty capture", st["error"] and "no packets" in st["error"],
      str(st["error"]))

print("\n=== episode log carries what the diagnosis needs ===")
write_pcap("/tmp/v3.pcap", [(t, ETH(p)) for t, p in packets()], 1)
log = []
eps = build_episodes(parse_pcap("/tmp/v3.pcap", file_tag="x_up.pcapng")[0], "t", gap_s=300.0, log=log)
dec = [r for r in log if "tunnel_domain" in r][0]
check("decision row: queries in file / in the kept group / payload kept",
      dec["n_queries_file"] == 120 and dec["n_queries_kept"] == 120 and dec["payload_kept"] == 120 * 45,
      str({k: dec[k] for k in ("n_queries_file", "n_queries_kept", "payload_kept", "n_groups")}))

print(f"\n{'ALL PASSED' if not FAILS else 'FAILED: ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
