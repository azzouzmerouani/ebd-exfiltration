"""Reconstruct the receiver-side byte timeline from a capture.

Applies the same discipline as :mod:`ebd.realdata.groundtruth` used on DAPT2021,
so the testbed measurement and the real-data measurement mean the same thing:

* **receiver-side** -- a byte counts when it lands at the destination, not when
  it is queued at the source;
* **payload, not wire** -- TCP payload only; link-layer framing, headers, pure
  ACKs and retransmissions cross the wire but are not part of the asset;
* **de-duplicated** -- a retransmitted segment delivers no new asset, so bytes
  are accounted by TCP sequence-number coverage, not by summing segment lengths.

De-duplication is what makes ``V_e(t)`` a *cumulative asset* curve rather than a
traffic curve, and it is the reason the paper reports payload as 32-93% of the
link-layer count on real episodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import dpkt

# Ethernet header length. The Linux loopback interface presents DLT_EN10MB
# framing, so wire-byte accounting adds this to the IP total length.
_LINK_HDR = 14


@dataclass
class Timeline:
    """A binned cumulative byte timeline for one flow direction."""

    bin_start: float
    bin_seconds: float
    bins: list[int] = field(default_factory=list)
    total_payload: int = 0
    wire_bytes: int = 0
    n_packets: int = 0
    duplicate_bytes: int = 0

    def cumulative_by(self, t: float) -> int:
        """Payload bytes delivered at or before absolute time ``t``."""
        if not self.bins:
            return 0
        idx = int((t - self.bin_start) // self.bin_seconds)
        if idx < 0:
            return 0
        return int(sum(self.bins[: min(idx + 1, len(self.bins))]))

    def fraction_by(self, t: float) -> float:
        if self.total_payload <= 0:
            return 0.0
        return min(1.0, self.cumulative_by(t) / self.total_payload)


def _add(tl: Timeline, ts: float, nbytes: int) -> None:
    idx = int((ts - tl.bin_start) // tl.bin_seconds)
    if idx < 0:
        return
    while len(tl.bins) <= idx:
        tl.bins.append(0)
    tl.bins[idx] += nbytes


#: Timeline resolution for the controlled captures. ``fraction_by(t)`` counts the
#: whole bin containing ``t``, so the floor can be overstated by up to one bin of
#: delivery; at 1 s that is large for transfers lasting a few seconds (it produced
#: the ~25% "plateau" of the chunked cells in revision 6). 10 ms makes it negligible.
TESTBED_BIN_SECONDS = 0.01


def build_timeline(
    pcap_path: str,
    src_port_set: set[int],
    dst_port: int,
    bin_seconds: float = TESTBED_BIN_SECONDS,
    t_lo: float | None = None,
    t_hi: float | None = None,
) -> Timeline:
    """Byte timeline for traffic from ``src_port_set`` to ``dst_port``.

    ``t_lo``/``t_hi`` bound the attribution window. They are **not** optional in
    practice: over a run long enough to matter, the kernel recycles ephemeral
    source ports, so a port alone attributes some later transfer's bytes to this
    episode. That failure is silent and inflates the asset -- here it inflated
    one 2 MiB episode by 37% -- which is why the driver asserts reconciliation
    against the planned asset before computing any EBD from a timeline.

    Sequence-number coverage is tracked per connection so retransmissions do not
    inflate the asset. The interval set is kept coalesced, which keeps this
    linear in practice for the in-order transfers the testbed generates.
    """
    tl: Timeline | None = None
    seen: dict[int, list[tuple[int, int]]] = {}   # src_port -> merged [lo,hi)

    with open(pcap_path, "rb") as fh:
        for ts, buf in dpkt.pcap.Reader(fh):
            if t_lo is not None and ts < t_lo:
                continue
            if t_hi is not None and ts > t_hi:
                continue
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                ip = eth.data
                if not isinstance(ip, dpkt.ip.IP):
                    continue
                tcp = ip.data
                if not isinstance(tcp, dpkt.tcp.TCP):
                    continue
            except Exception:
                continue

            if tcp.dport != dst_port or tcp.sport not in src_port_set:
                continue

            if tl is None:
                tl = Timeline(bin_start=float(int(ts)), bin_seconds=bin_seconds)

            # Payload length is derived from the IP/TCP *headers*, never from
            # len(tcp.data). The capture is taken with a small snaplen so the
            # trace stays tractable, which truncates the payload bytes but
            # leaves ip.len exact. Using len(tcp.data) here silently reports a
            # fraction of the asset -- a mistake that costs a whole run.
            ip_hl = ip.hl * 4
            tcp_hl = tcp.off * 4
            n = int(ip.len) - ip_hl - tcp_hl

            tl.n_packets += 1
            tl.wire_bytes += _LINK_HDR + int(ip.len)

            if n <= 0:
                continue

            lo, hi = int(tcp.seq), int(tcp.seq) + n
            ivals = seen.setdefault(tcp.sport, [])
            new = _novel_bytes(ivals, lo, hi)
            tl.duplicate_bytes += n - new
            if new:
                tl.total_payload += new
                _add(tl, ts, new)

    return tl or Timeline(bin_start=0.0, bin_seconds=bin_seconds)


def _novel_bytes(ivals: list[tuple[int, int]], lo: int, hi: int) -> int:
    """Bytes in ``[lo, hi)`` not already covered; inserts the interval."""
    novel = hi - lo
    for a, b in ivals:
        if b <= lo or a >= hi:
            continue
        novel -= min(hi, b) - max(lo, a)
    novel = max(0, novel)

    ivals.append((lo, hi))
    ivals.sort()
    merged: list[tuple[int, int]] = []
    for a, b in ivals:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    ivals[:] = merged
    return novel
