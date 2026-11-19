"""PCAP/PCAPNG -> DNS query records.

Pure standard library. Kaggle notebooks frequently run with internet disabled,
so depending on dpkt/scapy being installable is a way for the run to die four
hours in. Nothing here needs them.

Link types read: Ethernet (VLAN, QinQ, PPPoE), Linux cooked capture v1 and v2
(what `tcpdump -i any` writes on old and new libpcap), BSD/OpenBSD loopback, raw IP
(101, 12, 14, 228, 229) and NFLOG. Any other link type is counted per file and
reported, never guessed: a capture the parser cannot read must show up as unreadable,
not as a capture that happens to contain no DNS.
"""
from __future__ import annotations

import gzip
import os
import socket
import struct
from collections import Counter
from typing import Iterator

from ebd.dns.engine import Query

# link types (tcpdump.org/linktypes.html)
DLT_NULL, DLT_EN10MB, DLT_RAW, DLT_LOOP, DLT_LINUX_SLL = 0, 1, 101, 108, 113
DLT_IPV4, DLT_IPV6, DLT_NFLOG, DLT_LINUX_SLL2 = 228, 229, 239, 276
DLT_RAW_ALT = {12, 14}
LINKTYPE_NAMES = {0: "BSD loopback", 1: "Ethernet", 12: "raw IP", 14: "raw IP",
                  101: "raw IP", 108: "OpenBSD loopback", 113: "Linux cooked",
                  228: "raw IPv4", 229: "raw IPv6", 239: "NFLOG", 276: "Linux cooked v2"}
SUPPORTED_LINKTYPES = set(LINKTYPE_NAMES)
_IP_ETHERTYPES = (0x0800, 0x86DD)


def linktype_name(lt: int) -> str:
    return LINKTYPE_NAMES.get(lt, f"UNSUPPORTED({lt})")


def _open(path: str):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


# --------------------------------------------------------------------------- #
# Container readers                                                             #
# --------------------------------------------------------------------------- #

def _iter_pcap(fh) -> Iterator[tuple[float, bytes, int]]:
    hdr = fh.read(24)
    if len(hdr) < 24:
        return
    magic = struct.unpack("<I", hdr[:4])[0]
    if magic in (0xA1B2C3D4, 0xA1B23C4D):
        endian, nano = "<", magic == 0xA1B23C4D
    elif magic in (0xD4C3B2A1, 0x4D3CB2A1):
        endian, nano = ">", magic == 0x4D3CB2A1
    else:
        return
    # low 16 bits are the link type; the high bits may carry FCS information
    linktype = struct.unpack(endian + "I", hdr[20:24])[0] & 0xFFFF
    div = 1e9 if nano else 1e6
    ph = struct.Struct(endian + "IIII")
    while True:
        rec = fh.read(16)
        if len(rec) < 16:
            return
        sec, frac, caplen, _ = ph.unpack(rec)
        data = fh.read(caplen)
        if len(data) < caplen:
            return
        yield (sec + frac / div, data, linktype)


def _iter_pcapng(fh) -> Iterator[tuple[float, bytes, int]]:
    """PCAPNG. Each Enhanced Packet Block names its interface; linktype and
    timestamp resolution are per interface, so they are tracked per interface
    rather than taken from whichever IDB happened to come last."""
    endian = "<"
    ifaces: list[tuple[int, float]] = []          # (linktype, ticks per second)
    last_ts = None
    while True:
        head = fh.read(8)
        if len(head) < 8:
            return
        btype = struct.unpack(endian + "I", head[:4])[0]
        if btype == 0x0A0D0D0A:                              # SHB: new section
            bom = fh.read(4)
            if len(bom) < 4:
                return
            endian = "<" if struct.unpack("<I", bom)[0] == 0x1A2B3C4D else ">"
            blen = struct.unpack(endian + "I", head[4:8])[0]
            fh.read(max(0, blen - 12))                       # 8 (head) + 4 (BOM) consumed
            ifaces = []                                      # interface ids reset per section
            continue
        blen = struct.unpack(endian + "I", head[4:8])[0]
        if blen < 12:
            return
        body = fh.read(blen - 12)
        fh.read(4)
        if btype == 0x00000001 and len(body) >= 8:           # IDB
            linktype = struct.unpack(endian + "H", body[:2])[0]
            tsres = 1e6
            opts, i = body[8:], 0
            while i + 4 <= len(opts):
                ocode, olen = struct.unpack(endian + "HH", opts[i:i + 4])
                val = opts[i + 4:i + 4 + olen]
                if ocode == 9 and val:                       # if_tsresol
                    r = val[0]
                    tsres = float(2 ** (r & 0x7F)) if r & 0x80 else float(10 ** r)
                if ocode == 0:
                    break
                i += 4 + olen + ((4 - olen % 4) % 4)
            ifaces.append((linktype, tsres))
        elif btype == 0x00000006 and len(body) >= 20:        # EPB
            iface, hi, lo, caplen = struct.unpack(endian + "IIII", body[:16])
            lt, res = ifaces[iface] if iface < len(ifaces) else (DLT_EN10MB, 1e6)
            last_ts = ((hi << 32) | lo) / res
            yield (last_ts, body[20:20 + caplen], lt)
        elif btype == 0x00000002 and len(body) >= 20:        # obsolete Packet Block
            iface, _, hi, lo, caplen = struct.unpack(endian + "HHIII", body[:16])
            lt, res = ifaces[iface] if iface < len(ifaces) else (DLT_EN10MB, 1e6)
            last_ts = ((hi << 32) | lo) / res
            yield (last_ts, body[20:20 + caplen], lt)
        elif btype == 0x00000003 and len(body) >= 4:         # Simple Packet Block: no timestamp
            lt = ifaces[0][0] if ifaces else DLT_EN10MB
            yield (None, body[4:], lt)                       # caller counts and skips it


def iter_packets(path: str) -> Iterator[tuple[float, bytes, int]]:
    with _open(path) as fh:
        magic = fh.read(4)
        fh.seek(0)
        if magic == b"\x0a\x0d\x0d\x0a":
            yield from _iter_pcapng(fh)
        else:
            yield from _iter_pcap(fh)


# --------------------------------------------------------------------------- #
# Layers                                                                        #
# --------------------------------------------------------------------------- #

def _nflog_ip(buf: bytes) -> tuple[bytes, int] | None:
    """NFLOG: 4-byte header, then TLVs (host byte order); the IP packet is NFULA_PAYLOAD (9)."""
    off = 4
    while off + 4 <= len(buf):
        tlen, ttype = struct.unpack("<HH", buf[off:off + 4])
        if tlen < 4:
            return None
        if ttype & 0x7FFF == 9:
            return (buf[off + 4:off + tlen], 0)
        off += (tlen + 3) & ~3
    return None


def _l3(buf: bytes, linktype: int) -> tuple[bytes, int] | None:
    """-> (buf, offset of the IP header), or None when the frame carries no IP.
    Unknown link types return None: the caller counts them instead of guessing."""
    if linktype == DLT_EN10MB:
        if len(buf) < 14:
            return None
        et = struct.unpack("!H", buf[12:14])[0]
        off = 14
        while et in (0x8100, 0x88A8, 0x9100) and len(buf) >= off + 4:   # VLAN / QinQ tags
            et = struct.unpack("!H", buf[off + 2:off + 4])[0]
            off += 4
        if et == 0x8864 and len(buf) >= off + 8:                       # PPPoE session
            ppp = struct.unpack("!H", buf[off + 6:off + 8])[0]
            return (buf, off + 8) if ppp in (0x0021, 0x0057) else None
        return (buf, off) if et in _IP_ETHERTYPES else None
    if linktype == DLT_LINUX_SLL:                    # cooked v1: 16 bytes, protocol at 14
        if len(buf) < 16:
            return None
        et = struct.unpack("!H", buf[14:16])[0]
        return (buf, 16) if et in _IP_ETHERTYPES else None
    if linktype == DLT_LINUX_SLL2:                   # cooked v2: 20 bytes, protocol at 0
        if len(buf) < 20:
            return None
        et = struct.unpack("!H", buf[0:2])[0]
        return (buf, 20) if et in _IP_ETHERTYPES else None
    if linktype in (DLT_NULL, DLT_LOOP):
        return (buf, 4) if len(buf) >= 4 else None
    if linktype in (DLT_RAW, DLT_IPV4, DLT_IPV6) or linktype in DLT_RAW_ALT:
        return (buf, 0)
    if linktype == DLT_NFLOG:
        return _nflog_ip(buf)
    return None


def _l4(buf: bytes, off: int):
    """-> (src, dst, proto, payload) or None."""
    if len(buf) < off + 1:
        return None
    ver = buf[off] >> 4
    if ver == 4:
        if len(buf) < off + 20:
            return None
        ihl = (buf[off] & 0x0F) * 4
        if ihl < 20 or len(buf) < off + ihl:
            return None
        proto = buf[off + 9]
        frag = struct.unpack("!H", buf[off + 6:off + 8])[0]
        if frag & 0x1FFF:                       # non-first fragment: no L4 header
            return None
        tot = struct.unpack("!H", buf[off + 2:off + 4])[0]
        end = off + tot if 0 < tot <= len(buf) - off else len(buf)
        return (buf[off + 12:off + 16], buf[off + 16:off + 20], proto, buf[off + ihl:end])
    if ver == 6:
        if len(buf) < off + 40:
            return None
        plen = struct.unpack("!H", buf[off + 4:off + 6])[0]
        end = off + 40 + plen if 0 < plen <= len(buf) - off - 40 else len(buf)
        nh, p = buf[off + 6], off + 40
        for _ in range(8):                                  # walk extension headers
            if nh in (0, 43, 60):                           # hop-by-hop, routing, dest opts
                if len(buf) < p + 2:
                    return None
                nh, p = buf[p], p + (buf[p + 1] + 1) * 8
            elif nh == 44:                                  # fragment header
                if len(buf) < p + 8:
                    return None
                if struct.unpack("!H", buf[p + 2:p + 4])[0] & 0xFFF8:
                    return None                             # non-first fragment
                nh, p = buf[p], p + 8
            elif nh == 51:                                  # authentication header
                if len(buf) < p + 2:
                    return None
                nh, p = buf[p], p + (buf[p + 1] + 2) * 4
            else:
                break
        if p > end:
            return None
        return (buf[off + 8:off + 24], buf[off + 24:off + 40], nh, buf[p:end])
    return None


MAX_NAME_HOPS = 128      # compression pointers followed in one name; real queries use none, so this only bounds garbage
MAX_NAME_OCTETS = 255    # RFC 1035 2.3.4: a name is at most 255 octets on the wire, length bytes and root included
HEADER_LEN = 12          # a compression pointer leads to a name, and no name starts inside the header


def _qname(msg: bytes, pos: int) -> tuple[str, int] | None:
    """Decode a DNS name. Returns (name, next_offset).

    Iterative since v4.2. v4.1 called itself once per compression pointer and restarted its label counter on
    every call, so a long chain of backward pointers ran into Python's recursion limit, and the per-capture
    handler then ended the WHOLE capture: four of the thirteen university captures of the 7 Oct run stopped
    that way (at 6.0, 21.8, 29.4 and 51.8 minutes). The names accepted are the same as before: only backward
    jumps, at most 8 labels before a jump, at most 128 labels in a stretch of the name. What is new is a cap on
    the number of jumps, which no real name comes near, and two rules of RFC 1035 that v4.1 did not enforce and
    no real packet breaks: a pointer cannot lead into the 12-byte header (v4.1 decoded header bytes as labels),
    and a name is at most 255 octets (v4.1 took longer ones). Both only ever turn garbage away."""
    labels, seen, hops, end, wire = [], 0, 0, None, 1      # wire: octets the name takes uncompressed (root byte counted)
    n_msg = len(msg)
    while True:
        if pos >= n_msg:
            return None
        n = msg[pos]
        if n == 0:
            if hops and seen == 0:              # the stretch after the last jump was the root alone (v4.1 appended "" for it)
                labels.append("")
            return (".".join(labels), pos + 1 if end is None else end)
        if n & 0xC0 == 0xC0:                    # compression pointer
            if pos + 2 > n_msg:
                return None
            ptr = ((n & 0x3F) << 8) | msg[pos + 1]
            if ptr >= pos or ptr < HEADER_LEN or seen > 8:     # only backward jumps, never into the header; bound the chase
                return None
            if end is None:
                end = pos + 2                   # the name ends at its first pointer, wherever the pointer leads
            hops += 1
            if hops > MAX_NAME_HOPS:
                return None
            pos, seen = ptr, 0
            continue
        pos += 1
        if pos + n > n_msg or n > 63:
            return None
        wire += n + 1
        if wire > MAX_NAME_OCTETS:
            return None
        labels.append(msg[pos:pos + n].decode("latin-1"))
        pos += n
        seen += 1
        if seen > 128:
            return None


def malformed_reason(msg: bytes) -> str:
    """Why parse_dns_question() rejected a port-53 payload. Diagnosis only: it is called once
    per rejected message, never on the hot path. (The 4 Oct smoke run had 35% of the port-53
    messages in one university capture rejected with no way to say why.)"""
    if len(msg) < 12:
        return "shorter than a DNS header"
    if struct.unpack("!H", msg[4:6])[0] < 1:
        return "question count is 0"
    pos, seen, wire = 12, 0, 1
    while True:
        if pos >= len(msg):
            return "name runs past the end of the message"
        n = msg[pos]
        if n == 0:
            return "question cut before QTYPE/QCLASS"
        if n & 0xC0 == 0xC0:
            return "compression pointer inside the question"
        if n & 0xC0:
            return "reserved label bits (0x40/0x80)"
        if pos + 1 + n > len(msg):
            return "label runs past the end of the message"
        pos += 1 + n
        seen += 1
        wire += n + 1
        if wire > MAX_NAME_OCTETS:
            return "name longer than 255 octets"
        if seen > 128:                                 # (cannot be reached while names are capped at 255 octets: a bound)
            return "more than 128 labels"


def parse_dns_question(msg: bytes):
    """-> (qname, qtype, is_query) or None."""
    if len(msg) < 12:
        return None
    flags, qd = struct.unpack("!HH", msg[2:6])
    if qd < 1:
        return None
    got = _qname(msg, 12)
    if got is None:
        return None
    name, p = got
    if p + 4 > len(msg):
        return None
    qtype = struct.unpack("!H", msg[p:p + 2])[0]
    return (name, qtype, (flags >> 15) == 0)


# --------------------------------------------------------------------------- #
# Public entry point                                                            #
# --------------------------------------------------------------------------- #

_NTOP_CACHE: dict[bytes, str] = {}


def _ntop(a: bytes) -> str:
    """Address text. Cached, so the few thousand distinct hosts in a capture are ONE string object
    each instead of one per query (a 6-million-query benign corpus holds ~130 B less per query)."""
    key = bytes(a)
    s = _NTOP_CACHE.get(key)
    if s is None:
        try:
            s = socket.inet_ntop(socket.AF_INET if len(key) == 4 else socket.AF_INET6, key)
        except Exception:
            s = key.hex()
        if len(_NTOP_CACHE) < 1_000_000:
            _NTOP_CACHE[key] = s
    return s


def parse_pcap(path: str, max_packets: int | None = None,
               queries_only: bool = True, file_tag: str | None = None
               ) -> tuple[list[Query], dict]:
    """Extract DNS queries from one capture.

    The returned stats carry the denominators that make a partially-read or
    skipped file visible in the results rather than silently absent.

    v4.2: a packet that raises is counted and skipped, and so is a message the decoder cannot take; neither ends
    the capture any more (v4.1 lost four university captures to one exception each). Only a failure of the
    container reader itself, a corrupt tail, ends a capture, and then `error` says so.

    DNS over TCP: a message normally follows its 2-byte length in the same segment (offset 2, as before). Some
    clients send the length on its own first. A bare 2-byte segment is remembered per direction together with the
    sequence number of the byte after it, and the next segment of that connection is taken for the message only if
    it starts at exactly that sequence number AND is exactly as long as the prefix announced AND parses as a DNS
    message with one question; its message is then read from offset 0 (`tcp_split_read`). Anything else falls back
    to offset 2 as in v4.1. The checks keep a stray 2-byte segment from turning the next ordinary `[length][message]`
    segment into an empty name (the v4.2 review's counter-example: a root name parses from offset 0 of almost any
    such buffer). `tcp_prefix_seen` counts the remembered prefixes and `tcp_prefix_dropped` those that were replaced,
    evicted or not followed by a matching segment. Not handled, as in v4.1: two prefixed messages in one segment
    (only the first is read) and a message split over three segments, which lose TCP queries, and retransmissions,
    which repeat one.
    `malformed_c2s` counts the rejected payloads sent TO port 53: the denominator that matters for queries is
    those plus the queries parsed, since a rejected payload from port 53 is a response or a stream fragment.
    """
    qs: list[Query] = []
    file_tag = file_tag if file_tag is not None else os.path.basename(path)
    st = {"file": file_tag, "path": path, "packets": 0, "port53": 0, "dns": 0,
          "udp53": 0, "malformed": 0, "malformed_c2s": 0, "tcp_split_read": 0,
          "tcp_prefix_seen": 0, "tcp_prefix_dropped": 0,
          "packet_errors": 0, "packet_error_samples": "",
          "t_first": None, "t_last": None,
          "truncated": False, "error": None, "linktypes": "", "unsupported_linktype": 0,
          "non_ip": 0, "no_timestamp": 0, "first_bytes": "",
          "malformed_reasons": "", "malformed_reason_counts": {}, "malformed_samples": {}}
    lts: Counter = Counter()
    bad_why: Counter = Counter()
    bad_samples: dict[str, list[str]] = {}
    split: dict = {}              # (src, dst, sport, dport) -> (length announced by a bare 2-byte TCP segment, seq after it)
    errs: list[str] = []
    try:
        for ts, buf, lt in iter_packets(path):
            st["packets"] += 1
            lts[lt] += 1
            if st["packets"] == 1:
                st["first_bytes"] = buf[:40].hex()
            if ts is None:                                    # pcapng Simple Packet Block
                st["no_timestamp"] += 1
                continue
            if st["t_first"] is None:
                st["t_first"] = ts
            st["t_last"] = ts
            if max_packets and st["packets"] > max_packets:
                st["truncated"] = True
                break
            try:
                if lt not in SUPPORTED_LINKTYPES:
                    st["unsupported_linktype"] += 1
                    continue
                l3 = _l3(buf, lt)
                if l3 is None:
                    st["non_ip"] += 1
                    continue
                got = _l4(*l3)
                if got is None:
                    continue
                src, dst, proto, pay = got
                cands = None
                if proto == 17:                               # UDP
                    if len(pay) < 8:
                        continue
                    sport, dport = struct.unpack("!HH", pay[:4])
                    msg = pay[8:]
                elif proto == 6:                              # TCP
                    if len(pay) < 20:
                        continue
                    sport, dport = struct.unpack("!HH", pay[:4])
                    if 53 not in (sport, dport):
                        continue
                    data = pay[(pay[12] >> 4) * 4:]
                    key = (src, dst, sport, dport)
                    seq = struct.unpack("!I", pay[4:8])[0]
                    if len(data) == 2:                        # a length prefix sent on its own
                        st["tcp_prefix_seen"] += 1
                        if len(split) >= 4096:                # a flood of bare prefixes must not grow without bound
                            st["tcp_prefix_dropped"] += len(split)
                            split.clear()
                        if key in split:
                            st["tcp_prefix_dropped"] += 1     # replaced before its message came
                        split[key] = ((data[0] << 8) | data[1], (seq + 2) & 0xFFFFFFFF)
                        continue
                    if len(data) < 3:                         # a bare ACK or a stub: leave a remembered prefix alone
                        continue
                    pending = split.pop(key, None)
                    msg = data[2:]
                    if pending is not None:
                        if pending == (len(data), seq):       # the prefix came alone: the message starts here
                            cands = (data, msg)
                        else:
                            st["tcp_prefix_dropped"] += 1     # the next segment was not the message it announced
                else:
                    continue
                if 53 not in (sport, dport) or not msg:
                    continue
                st["port53"] += 1
                q = None
                if cands:
                    q = parse_dns_question(cands[0])          # offset 0 first when the prefix arrived on its own ...
                    if q is not None and struct.unpack("!H", cands[0][4:6])[0] != 1:
                        q = None                              # (a query has one question; anything else is a mis-pairing)
                    if q is not None:
                        st["tcp_split_read"] += 1
                if q is None:
                    q = parse_dns_question(msg)               # ... otherwise after the prefix, as in v4.1
                if q is None:
                    st["malformed"] += 1
                    if dport == 53:
                        st["malformed_c2s"] += 1
                    shown = cands[0] if cands else msg
                    why = malformed_reason(shown)
                    bad_why[why] += 1
                    smp = bad_samples.setdefault(why, [])
                    if len(smp) < 3:                         # first bytes + enough context to recognise it
                        smp.append(f"{'udp' if proto == 17 else 'tcp'} {sport}->{dport} len={len(shown)} "
                                   f"{shown[:40].hex()}")
                    continue
                name, qtype, is_q = q
                if queries_only and not is_q:
                    continue
                st["dns"] += 1
                st["udp53"] += 1 if proto == 17 else 0
                qs.append(Query(ts=float(ts), src=_ntop(src), dst=_ntop(dst), sport=int(sport),
                                qname=name, qtype=int(qtype), file=file_tag))
            except MemoryError:
                raise
            except Exception as e:                            # this packet only; the capture goes on
                st["packet_errors"] += 1
                if len(errs) < 3:
                    errs.append(f"{type(e).__name__}: {str(e)[:60]} at packet {st['packets']}")
    except Exception as e:                                     # a corrupt tail must not
        st["error"] = f"{type(e).__name__}: {e}"               # lose the packets before it
    st["linktypes"] = ", ".join(f"{linktype_name(k)}: {v:,}" for k, v in lts.most_common())
    st["malformed_reasons"] = "; ".join(f"{k}: {v:,}" for k, v in bad_why.most_common())
    st["malformed_reason_counts"] = dict(bad_why)
    st["malformed_samples"] = bad_samples
    st["packet_error_samples"] = "; ".join(errs)
    if st["packets"] == 0 and st["error"] is None:
        st["error"] = "no packets read (not a pcap/pcapng file, or empty)"
    return qs, st
