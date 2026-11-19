"""Minimal streaming PCAP reader for byte-timeline ground truth.

Phase B needs one thing from 28 GB of packet capture: for each exfiltration
episode, *how many application-layer bytes had left by time t*. That is a tiny
projection of the capture, so the reader streams packets and keeps only
``(timestamp, payload_length)`` for the 5-tuples of interest — no packets are
buffered and nothing is written to disk. The same code therefore works against
a local file or an HTTP response object.

Why application-layer bytes
---------------------------
EBD is denominated in the asset that actually left, so the count must exclude
Ethernet/IP/TCP headers and, for TCP, must not double-count retransmissions the
way a naive packet-length sum would. We take the TCP/UDP *payload* length, and
for TCP we track sequence numbers so a retransmitted segment is not counted
twice. This is the receiver-side standard the synthetic work used, applied to
real packets.

Deliberately dependency-free: scapy/dpkt are not needed for a projection this
narrow, and avoiding them keeps the parse auditable — every offset below is
from the pcap/Ethernet/IP/TCP specs directly.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Callable, Iterator

# Canonical pcap magic values. The writer stores them in its *native* byte
# order, so a reader that always unpacks little-endian sees the canonical value
# for a little-endian file and the byte-swapped value for a big-endian one.
_MAGIC_US = 0xA1B2C3D4          # microsecond resolution
_MAGIC_NS = 0xA1B23C4D          # nanosecond resolution
_MAGIC_US_SWAPPED = 0xD4C3B2A1  # same, seen from the opposite endianness
_MAGIC_NS_SWAPPED = 0x4D3CB2A1

LINKTYPE_ETHERNET = 1
LINKTYPE_RAW = 101
LINKTYPE_LINUX_SLL = 113
LINKTYPE_LINUX_SLL2 = 276

PROTO_TCP = 6
PROTO_UDP = 17


class PcapFormatError(ValueError):
    """The stream is not a pcap we can parse."""


@dataclass(frozen=True)
class Packet:
    ts: float           # epoch seconds
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: int
    payload_len: int    # application-layer bytes (the asset actually stolen)
    seq: int | None     # TCP sequence number, for dedup; None for UDP
    wire_len: int = 0   # IP total length, i.e. what a flow exporter counts


def _ip_str(buf, off: int) -> str:
    return f"{buf[off]}.{buf[off+1]}.{buf[off+2]}.{buf[off+3]}"


def _l3_offset(buf, off: int, end: int, linktype: int) -> int:
    """Offset of the IPv4 header inside a link-layer frame, or -1.

    Works on absolute offsets into the read buffer. Nothing is sliced or
    copied — important because the buffer is a bytearray that must stay
    resizable while packets are being yielded.
    """
    if linktype == LINKTYPE_RAW:
        return off
    if linktype == LINKTYPE_ETHERNET:
        if end - off < 14:
            return -1
        etype = struct.unpack_from("!H", buf, off + 12)[0]
        p = off + 14
        while etype in (0x8100, 0x88A8, 0x9100):    # VLAN / QinQ
            if end - p < 4:
                return -1
            etype = struct.unpack_from("!H", buf, p + 2)[0]
            p += 4
        return p if etype == 0x0800 else -1
    if linktype == LINKTYPE_LINUX_SLL:
        if end - off < 16:
            return -1
        etype = struct.unpack_from("!H", buf, off + 14)[0]
        return off + 16 if etype == 0x0800 else -1
    if linktype == LINKTYPE_LINUX_SLL2:
        if end - off < 20:
            return -1
        etype = struct.unpack_from("!H", buf, off)[0]
        return off + 20 if etype == 0x0800 else -1
    return -1


def _parse_at(buf, off: int, end: int, ts: float) -> Packet | None:
    """Parse IPv4 + TCP/UDP starting at absolute offset ``off``.

    Only header fields and lengths are read — the payload itself is never
    touched, since EBD needs how many bytes moved, not which bytes.
    """
    if end - off < 20:
        return None
    ver_ihl = buf[off]
    if ver_ihl >> 4 != 4:                       # IPv6 absent from this capture
        return None
    ihl = (ver_ihl & 0x0F) * 4
    if ihl < 20 or end - off < ihl:
        return None
    total_len = struct.unpack_from("!H", buf, off + 2)[0]
    proto = buf[off + 9]
    src = _ip_str(buf, off + 12)
    dst = _ip_str(buf, off + 16)
    # Trust the IP header's own length rather than the captured length: a
    # snaplen-truncated packet still reports the true size on the wire.
    ip_payload_len = max(total_len - ihl, 0)
    l4 = off + ihl

    if proto == PROTO_TCP:
        if end - l4 < 20:
            return None
        sport, dport = struct.unpack_from("!HH", buf, l4)
        seq = struct.unpack_from("!I", buf, l4 + 4)[0]
        doff = (buf[l4 + 12] >> 4) * 4
        if doff < 20:
            return None
        return Packet(ts, src, sport, dst, dport, proto,
                      max(ip_payload_len - doff, 0), seq, total_len)

    if proto == PROTO_UDP:
        if end - l4 < 8:
            return None
        sport, dport = struct.unpack_from("!HH", buf, l4)
        ulen = struct.unpack_from("!H", buf, l4 + 4)[0]
        return Packet(ts, src, sport, dst, dport, proto, max(ulen - 8, 0), None,
                      total_len)

    return None


# Plausible capture-timestamp window, used to re-find record boundaries after
# corruption. Wide enough for any realistic capture, tight enough that random
# bytes almost never pass.
_TS_MIN = 1_262_304_000     # 2010-01-01
_TS_MAX = 1_893_456_000     # 2030-01-01


@dataclass
class ParseStats:
    """What the parse actually saw — including what it could not read.

    These captures are not clean: records appear whose declared length runs
    past the next valid record header, with timestamp discontinuities across
    the damage. Recovering the readable remainder is the right call, but the
    loss has to be *measured*, because bytes lost here understate the
    reconstructed asset and therefore flatter every EBD computed from it.
    """

    packets: int = 0
    bytes_read: int = 0
    resyncs: int = 0
    resync_bytes_skipped: int = 0
    truncated_tail: bool = False

    @property
    def loss_fraction(self) -> float:
        return self.resync_bytes_skipped / max(self.bytes_read, 1)


def iter_packets(
    stream: BinaryIO,
    keep: Callable[[str, int, str, int, int], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    stats: ParseStats | None = None,
    chunk: int = 1 << 22,
    stop_after_ts: float | None = None,
) -> Iterator[Packet]:
    """Yield parsed packets from a (possibly damaged) pcap stream.

    ``keep(src_ip, src_port, dst_ip, dst_port, proto)`` filters before the
    packet object is built, so a multi-GB capture costs one pass and no memory.
    ``stats`` is filled in as parsing proceeds; pass one in to learn how much of
    the file had to be skipped.

    ``stop_after_ts`` ends the scan once the capture has moved conclusively past
    that time. Records are chronological, so this bounds the read by *time*
    rather than by a guessed byte offset — which is the safe direction: guessing
    an end offset can silently truncate an episode, and a truncated episode
    understates its own asset.

    On a record header that cannot be true (implausible timestamp, zero or
    over-snaplen length), the reader scans forward byte-by-byte for the next
    header that is self-consistent, rather than aborting the file or blindly
    trusting the length. Everything skipped is counted.
    """
    st = stats if stats is not None else ParseStats()
    buf = bytearray()

    def fill(n: int) -> bool:
        while len(buf) < n:
            block = stream.read(chunk)
            if not block:
                return False
            buf.extend(block)
            st.bytes_read += len(block)
        return True

    if not fill(24):
        raise PcapFormatError("stream shorter than a pcap global header")
    (magic,) = struct.unpack_from("<I", buf, 0)
    if magic in (_MAGIC_US, _MAGIC_NS):
        endian, nano = "<", magic == _MAGIC_NS
    elif magic in (_MAGIC_US_SWAPPED, _MAGIC_NS_SWAPPED):
        endian, nano = ">", magic == _MAGIC_NS_SWAPPED
    else:
        raise PcapFormatError(f"bad pcap magic 0x{magic:08X} (pcapng is not supported)")
    linktype = struct.unpack_from(endian + "I", buf, 20)[0]
    snaplen = struct.unpack_from(endian + "I", buf, 16)[0] or 262144
    ph = struct.Struct(endian + "IIII")
    frac_div = 1e9 if nano else 1e6
    frac_max = 1_000_000_000 if nano else 1_000_000

    del buf[:24]
    pos = 0
    past = 0

    def plausible(at: int) -> bool:
        if at + 16 > len(buf):
            return False
        ts, fr, incl, orig = ph.unpack_from(buf, at)
        return (
            _TS_MIN <= ts <= _TS_MAX
            and fr < frac_max
            and 0 < incl <= snaplen
            and orig >= incl
        )

    while True:
        # keep a working window; compact what has been consumed
        if pos > (1 << 23):
            del buf[:pos]
            pos = 0
        if len(buf) - pos < 16 and not fill(pos + 16):
            st.truncated_tail = len(buf) - pos > 0
            return

        if not plausible(pos):
            # ---- resync: find the next self-consistent record header ----
            start = pos
            found = False
            while True:
                if len(buf) - pos < 64 and not fill(pos + 64):
                    st.resyncs += 1
                    st.resync_bytes_skipped += len(buf) - start
                    st.truncated_tail = True
                    return
                limit = len(buf) - 16
                while pos <= limit:
                    if plausible(pos):
                        found = True
                        break
                    pos += 1
                if found:
                    break
                if pos > (1 << 23):
                    del buf[:pos]
                    pos = 0
                    start = 0
            st.resyncs += 1
            st.resync_bytes_skipped += pos - start
            continue

        ts_sec, ts_frac, incl, _orig = ph.unpack_from(buf, pos)
        if len(buf) - pos < 16 + incl and not fill(pos + 16 + incl):
            # The declared body runs past the end of the data. Either the file
            # genuinely ends here, or this header is lying about its length —
            # look for a later valid header before giving up on the remainder.
            hit = -1
            nxt = pos + 17
            limit = len(buf) - 16
            while nxt <= limit:
                if plausible(nxt):
                    hit = nxt
                    break
                nxt += 1
            if hit == -1:
                st.truncated_tail = True
                return
            st.resyncs += 1
            st.resync_bytes_skipped += hit - pos
            pos = hit
            continue
        # A record whose body swallows the next valid header is damaged; treat
        # its declared length as untrustworthy rather than consuming the body.
        if not (plausible(pos + 16 + incl) or len(buf) - (pos + 16 + incl) < 16):
            nxt = pos + 17
            scan_end = min(pos + 16 + incl, len(buf) - 16)
            hit = -1
            while nxt <= scan_end:
                if plausible(nxt):
                    hit = nxt
                    break
                nxt += 1
            if hit != -1:
                st.resyncs += 1
                st.resync_bytes_skipped += (pos + 16 + incl) - hit
                incl = hit - (pos + 16)

        st.packets += 1
        if progress is not None and st.packets % 2_000_000 == 0:
            progress(st.bytes_read, st.packets)

        body = pos + 16
        body_end = body + incl
        pos = body_end
        l3 = _l3_offset(buf, body, body_end, linktype)
        if l3 < 0:
            continue
        ts = ts_sec + ts_frac / frac_div
        if stop_after_ts is not None and ts > stop_after_ts:
            # require a run of out-of-window records so one stray timestamp
            # (these captures do contain them) cannot end the scan early
            past += 1
            if past >= 5000:
                return
        else:
            past = 0
        pkt = _parse_at(buf, l3, body_end, ts)
        if pkt is None:
            continue
        if keep is not None and not keep(
            pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port, pkt.protocol
        ):
            continue
        yield pkt


# --------------------------------------------------------------------------- #
# Byte-timeline accumulation                                                  #
# --------------------------------------------------------------------------- #

def find_record_offset(
    fetch: Callable[[int, int], bytes],
    size: int,
    target_ts: float,
    probe: int = 1 << 16,
) -> int:
    """Byte offset of the first record at or after ``target_ts``.

    A pcap is chronologically ordered, so a timestamp can be located by binary
    search instead of a linear scan. ``fetch(offset, length)`` returns that byte
    range (an HTTP Range request in practice). Each probe reads ``probe`` bytes
    and resyncs to the first self-consistent record header inside it, so the
    search never needs record alignment to begin with.

    This is what makes a short episode inside a multi-GB capture cheap: roughly
    ``log2(size/probe)`` probes of 64 KB, rather than downloading the file.
    Returns 0 if the target precedes the capture, or ``size`` if it follows it.
    """
    def first_ts_at(off: int) -> tuple[float, int] | None:
        blob = fetch(off, probe)
        if len(blob) < 32:
            return None
        for i in range(0, len(blob) - 16):
            ts, fr, incl, orig = struct.unpack_from("<IIII", blob, i)
            if (_TS_MIN <= ts <= _TS_MAX and fr < 1_000_000
                    and 0 < incl <= 262144 and orig >= incl):
                # require a second consistent record to avoid a chance match
                j = i + 16 + incl
                if j + 16 <= len(blob):
                    t2, f2, i2, o2 = struct.unpack_from("<IIII", blob, j)
                    if not (_TS_MIN <= t2 <= _TS_MAX and f2 < 1_000_000
                            and 0 < i2 <= 262144 and o2 >= i2):
                        continue
                return ts + fr / 1e6, off + i
        return None

    lo, hi = 24, max(size - probe, 24)
    best = 24
    for _ in range(48):
        if lo >= hi:
            break
        mid = (lo + hi) // 2
        got = first_ts_at(mid)
        if got is None:
            hi = mid - probe if mid - probe > lo else lo
            continue
        ts, off = got
        if ts < target_ts:
            lo = off + 1
            best = off
        else:
            hi = mid
    return max(best - probe, 24)


@dataclass
class TimelineBuilder:
    """Accumulate application-layer bytes per time bin, in one direction.

    TCP retransmissions are suppressed by remembering which sequence numbers
    have already been counted per connection: re-sending bytes does not
    exfiltrate them twice, and counting them twice would understate a
    detector's EBD by inflating the denominator.
    """

    bin_seconds: float = 1.0
    bins: dict[int, int] = field(default_factory=dict)
    seen_seq: dict[tuple, set] = field(default_factory=dict)
    total_bytes: int = 0        # application-layer payload (EBD denominator)
    wire_bytes: int = 0         # IP-layer bytes, comparable to a flow exporter
    n_packets: int = 0
    n_all_packets: int = 0      # incl. pure ACKs, which carry no asset
    t_min: float | None = None
    t_max: float | None = None
    duplicate_bytes: int = 0

    def add(self, pkt: Packet) -> None:
        # Every packet in the direction counts toward the wire total, because
        # that is what the dataset's own byte column measures; only payload
        # counts toward the asset.
        self.n_all_packets += 1
        self.wire_bytes += pkt.wire_len
        if pkt.payload_len <= 0:
            return
        if pkt.seq is not None:
            conn = (pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port)
            seqs = self.seen_seq.setdefault(conn, set())
            if pkt.seq in seqs:
                self.duplicate_bytes += pkt.payload_len
                return
            seqs.add(pkt.seq)
        self.n_packets += 1
        self.total_bytes += pkt.payload_len
        self.t_min = pkt.ts if self.t_min is None else min(self.t_min, pkt.ts)
        self.t_max = pkt.ts if self.t_max is None else max(self.t_max, pkt.ts)
        b = int(pkt.ts // self.bin_seconds)
        self.bins[b] = self.bins.get(b, 0) + pkt.payload_len

    def timeline(self) -> tuple[list[float], list[int]]:
        """Return (bin_start_times, bytes_in_bin), ordered by time."""
        if not self.bins:
            return [], []
        lo, hi = min(self.bins), max(self.bins)
        times = [(lo + i) * self.bin_seconds for i in range(hi - lo + 1)]
        vals = [self.bins.get(lo + i, 0) for i in range(hi - lo + 1)]
        return times, vals

    def bytes_by(self, t: float) -> int:
        """Application-layer bytes that had arrived at or before time ``t``."""
        cutoff = int(t // self.bin_seconds)
        return sum(v for b, v in self.bins.items() if b <= cutoff)
