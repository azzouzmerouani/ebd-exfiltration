"""Phase B: reconstruct per-episode byte timelines from the raw captures.

For each exfiltration episode identified from the labeled flow records, this
streams the corresponding capture once and accumulates the application-layer
bytes travelling in the exfiltration direction, binned in time. The result is
the ground truth EBD is measured against: *how much of the asset had actually
left by any given instant*.

Vantage point
-------------
Each episode is reconstructed from **one** capture — the host capture the
episode's labeled flows came from. Using two vantage points (host and gateway)
for the same episode would count the same bytes twice, and picking the vantage
per-episode from its own label provenance keeps the choice consistent and
auditable rather than ad hoc.

For exfiltration leaving the network, the gateway is the truer "bytes that
actually left" observation point; the source-host capture is an upper bound on
it, and equal whenever the transfer completed. Where the destination is
internal, the host capture is the correct vantage outright. Both cases are
recorded so the distinction survives into the report.

Reconciliation, and the definition that makes it exact
------------------------------------------------------
The reconstruction is checked against the flow records' own ``src2dst_bytes``.
The two count different things, and conflating them produces a spurious ~46 %
divergence on a *perfectly* reconstructed episode:

* ``src2dst_bytes`` is a **link-layer** total over **every** packet in the
  direction, pure ACKs included;
* the asset is **de-duplicated application-layer payload** — what the attacker
  actually stole, excluding framing and retransmissions.

Comparing like for like (IP bytes plus the 14-byte Ethernet header, over all
packets) reconciles exactly. Measured on the first episode:

    packets in direction   19,912  vs exporter's src2dst_packets  19,912   (exact)
    IP bytes                2.072 MB
    + 14 B x 19,912         0.279 MB
    = link-layer            2.351 MB  vs exporter's src2dst_bytes 2.351 MB  (exact)
    payload (the asset)     1.257 MB  -- 53 % of what crossed the wire

So the reconstruction is verifiably identical to what the exporter saw, and
**EBD is denominated in the 1.257 MB of payload**, not the 2.351 MB of traffic:
framing is not stolen data. Any episode whose reconciliation is not near-exact
is reported rather than silently filtered.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from .pcapreader import (
    ParseStats,
    TimelineBuilder,
    find_record_offset,
    iter_packets,
)
from .source import BUCKET_URL


# --------------------------------------------------------------------------- #
# Ranged, retrying HTTP access to the bucket                                   #
# --------------------------------------------------------------------------- #

class RangeStream:
    """A forward-only stream over an HTTP object, resumable after a stall.

    S3 connections here stall often enough that a single long-lived GET is not
    viable: a stalled socket blocks until its timeout with no progress. This
    reads in bounded windows and reissues a Range request from the last byte
    successfully consumed, so a stall costs seconds rather than the run.
    """

    def __init__(self, key: str, start: int = 0, end: int | None = None,
                 window: int = 1 << 24, timeout: float = 60.0, retries: int = 6):
        self.url = BUCKET_URL + urllib.parse.quote(key)
        self.pos = start
        self.end = end
        self.window = window
        self.timeout = timeout
        self.retries = retries
        self._buf = b""
        self._off = 0
        self._eof = False

    def _pull(self) -> bool:
        if self._eof:
            return False
        hi = self.pos + self.window - 1
        if self.end is not None:
            if self.pos > self.end:
                self._eof = True
                return False
            hi = min(hi, self.end)
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={self.pos}-{hi}"}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    blob = r.read()
                if not blob:
                    self._eof = True
                    return False
                self._buf, self._off = blob, 0
                self.pos += len(blob)
                return True
            except Exception as e:                       # noqa: BLE001
                last = e
                time.sleep(min(2 ** attempt, 20))
        raise OSError(f"range fetch failed after {self.retries} tries: {last}")

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = 1 << 20
        out = bytearray()
        while len(out) < n:
            if self._off >= len(self._buf):
                if not self._pull():
                    break
            take = min(n - len(out), len(self._buf) - self._off)
            out += self._buf[self._off: self._off + take]
            self._off += take
        return bytes(out)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fetch_range(key: str, offset: int, length: int, timeout: float = 60.0,
                retries: int = 5) -> bytes:
    """One Range request, retried — used by the timestamp binary search."""
    url = BUCKET_URL + urllib.parse.quote(key)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={offset}-{offset + length - 1}"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:                            # noqa: BLE001
            last = e
            time.sleep(min(2 ** attempt, 15))
    raise OSError(f"range fetch failed: {last}")


def object_size(key: str, timeout: float = 60.0) -> int:
    req = urllib.request.Request(BUCKET_URL + urllib.parse.quote(key), method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(r.headers["Content-Length"])


class _PrefixedStream:
    """The capture's real 24-byte global header, followed by a byte range.

    ``iter_packets`` needs the global header to learn endianness, link type and
    snaplen, but a ranged read starts mid-file. Splicing the true header in
    front lets the reader work on a slice exactly as it would on the whole file;
    its resync then finds the first intact record boundary inside the slice.
    """

    def __init__(self, key: str, start: int, end: int):
        self._head = fetch_range(key, 0, 24)
        self._body = RangeStream(key, start=start, end=end)
        self._head_off = 0

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = 1 << 20
        out = b""
        if self._head_off < len(self._head):
            take = min(n, len(self._head) - self._head_off)
            out = self._head[self._head_off: self._head_off + take]
            self._head_off += take
            n -= take
        if n > 0:
            out += self._body.read(n)
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@dataclass
class EpisodeGroundTruth:
    """Reconstructed byte timeline for one episode."""

    episode_id: str
    pcap_key: str
    src_ip: str
    dst_ip: str
    dst_port: int
    destination_is_internal: bool
    t_start: float
    t_end: float
    pcap_bytes: int                  # de-duplicated application-layer bytes (the asset)
    wire_bytes: int                  # IP-layer bytes in the same direction
    flow_bytes: int                  # dataset's own src2dst_bytes for the episode
    n_packets: int                   # payload-bearing packets
    n_all_packets: int               # every packet in the direction, incl. pure ACKs
    duplicate_bytes: int
    bin_seconds: float
    bin_start: float
    bins: list[int] = field(default_factory=list)
    parse: dict = field(default_factory=dict)

    ETH_HEADER = 14

    @property
    def reconstructed_link_bytes(self) -> int:
        """IP bytes plus the Ethernet header the exporter also counted."""
        return self.wire_bytes + self.ETH_HEADER * self.n_all_packets

    @property
    def divergence(self) -> float:
        """Reconciliation figure, compared like-for-like.

        The dataset's ``src2dst_bytes`` is a **link-layer** count over *every*
        packet in the direction; our asset figure is de-duplicated
        application-layer payload. Comparing those two directly is meaningless
        (it shows ~46% on a perfectly reconstructed episode, all of it framing).
        The honest check adds back IP+Ethernet framing and compares that.
        """
        denom = max(self.flow_bytes, 1)
        return abs(self.reconstructed_link_bytes - self.flow_bytes) / denom

    @property
    def payload_fraction(self) -> float:
        """Share of the transferred bytes that was actual asset, not framing."""
        return self.pcap_bytes / max(self.reconstructed_link_bytes, 1)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def cumulative(self) -> list[int]:
        out, run = [], 0
        for v in self.bins:
            run += v
            out.append(run)
        return out

    def bytes_by(self, t: float) -> int:
        """Bytes that had left by absolute time ``t``."""
        if not self.bins:
            return 0
        idx = int((t - self.bin_start) // self.bin_seconds)
        if idx < 0:
            return 0
        idx = min(idx, len(self.bins) - 1)
        return sum(self.bins[: idx + 1])

    def nebd_for_alert(self, alert_time: float | None) -> float:
        """Normalised EBD if a detector alerts at ``alert_time`` (None = never)."""
        total = self.pcap_bytes
        if total <= 0:
            return 0.0
        if alert_time is None:
            return 1.0
        return min(1.0, self.bytes_by(alert_time) / total)


def _is_internal(ip: str) -> bool:
    return ip.startswith(("10.", "192.168.")) or (
        ip.startswith("172.") and 16 <= int(ip.split(".")[1]) <= 31
    )


def reconstruct(
    episode,
    pcap_key: str,
    flow_bytes: int,
    bin_seconds: float = 1.0,
    time_margin: float = 15.0,
    open_stream=None,
    intervals: list[tuple[float, float]] | None = None,
) -> EpisodeGroundTruth:
    """Stream ``pcap_key`` and rebuild ``episode``'s outbound byte timeline.

    Packets are kept when they travel from the episode's source to its
    destination and port **and** fall inside one of the episode's constituent
    flow windows, each widened by ``time_margin``.

    Matching per-flow rather than against one span from first to last matters:
    a beaconing episode is mostly idle, so a single span would sweep in
    unrelated traffic to the same peer and overstate the asset. On the 23-minute
    episode that error was +16.7%.
    """
    src, dst, dport = episode.src_ip, episode.dst_ip, episode.dst_port
    lo, hi = episode.t_start - time_margin, episode.t_end + time_margin
    if intervals is None:
        intervals = [(f.start, f.end) for f in episode.flows]
    windows = sorted((a - time_margin, b + time_margin) for a, b in intervals)
    # merge overlapping windows so the membership test stays cheap
    merged: list[list[float]] = []
    for a, b in windows:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    import bisect
    starts = [w[0] for w in merged]

    def in_window(t: float) -> bool:
        i = bisect.bisect_right(starts, t) - 1
        return i >= 0 and t <= merged[i][1]

    def keep(s, sp, d, dp, pr):
        return s == src and d == dst and dp == dport

    tb = TimelineBuilder(bin_seconds=bin_seconds)
    st = ParseStats()

    if open_stream is not None:
        stream_cm = open_stream(pcap_key)
        byte_lo = byte_hi = None
    else:
        # Seek to the episode's window instead of streaming the whole capture:
        # a 23-minute episode inside a 4 GB day costs a handful of 64 KB probes.
        size = object_size(pcap_key)
        try:
            byte_lo = find_record_offset(
                lambda o, n: fetch_range(pcap_key, o, n), size, lo
            )
        except Exception:                       # corrupt probes: start at the top
            byte_lo = 24
        # Only the START is located by offset. The END is bounded by timestamp
        # while streaming: a guessed end offset can silently truncate an episode,
        # and a truncated episode understates its own asset.
        byte_hi = size - 1
        stream_cm = _PrefixedStream(pcap_key, byte_lo, byte_hi)

    with stream_cm as stream:
        for pkt in iter_packets(stream, keep=keep, stats=st, stop_after_ts=hi):
            if in_window(pkt.ts):
                tb.add(pkt)

    times, vals = tb.timeline()
    return EpisodeGroundTruth(
        episode_id=episode.episode_id,
        pcap_key=pcap_key,
        src_ip=src,
        dst_ip=dst,
        dst_port=dport,
        destination_is_internal=_is_internal(dst),
        t_start=tb.t_min if tb.t_min is not None else episode.t_start,
        t_end=tb.t_max if tb.t_max is not None else episode.t_end,
        pcap_bytes=tb.total_bytes,
        wire_bytes=tb.wire_bytes,
        flow_bytes=flow_bytes,
        n_packets=tb.n_packets,
        n_all_packets=tb.n_all_packets,
        duplicate_bytes=tb.duplicate_bytes,
        bin_seconds=bin_seconds,
        bin_start=times[0] if times else 0.0,
        bins=vals,
        parse={
            "packets": st.packets,
            "bytes_read": st.bytes_read,
            "resyncs": st.resyncs,
            "resync_bytes_skipped": st.resync_bytes_skipped,
            "loss_fraction": st.loss_fraction,
            "truncated_tail": st.truncated_tail,
        },
    )
