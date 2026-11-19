"""An incremental detector that is *not* a record consumer, and beats the floor.

Why this exists
---------------
The architectural floor is a negative result about a family. On its own it tells
a defender what not to expect, not what to build. This module is the
constructive counterpart: a detector outside the family, measured on the same
traffic, at the same false-positive burden, so the comparison is like for like.

The escape is architectural, not statistical. The floor exists because a record
consumer has no score until a record exists. A detector that maintains state per
connection and re-scores on every packet has a score at every instant, so
nothing in Eq. (3) applies to it. It is deliberately *simpler* than the
classifier it is compared against -- three counters and a threshold, no learned
model -- which is the point: the gain comes from where it sits in the pipeline,
not from model capacity. If a stronger detector inside the family cannot beat a
weaker one outside it, the boundary is architectural.

The rule
--------
For each connection, maintain bytes-so-far ``b`` and age ``a``. Alert when

    b >= B_min   and   b / max(a, eps) >= R_min   and   a >= A_min

i.e. sustained outbound volume, at sustained rate, sustained for long enough.
The alert instant is the packet that first satisfies all three, not the
connection's teardown.

The third condition is not padding. The benign background here deliberately
contains hard negatives -- short high-rate bulk transfers whose volume *and*
instantaneous rate overlap the attacks -- so volume and rate alone cannot
separate them at any threshold, by construction. Persistence is the only clean
discriminator, which is the same design principle the synthetic harness of
Sec. 6 uses. A detector denied that dimension is not being tested, it is being
handicapped.

One consequence to state up front: for transfers so short that they finish
before ``A_min``, this detector does not fire at all, and its EBD is 1. It does
not beat the floor there, because in this traffic mix nothing could -- benign
sessions of the same volume, rate and duration exist. The escape it demonstrates
is in *earliness on detectable transfers*, not in detecting the undetectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import dpkt

_LINK_HDR = 14


@dataclass
class _ConnState:
    t_first: float = 0.0
    payload: int = 0
    alerted: bool = False


@dataclass(frozen=True)
class Alert:
    src_port: int
    time: float
    payload_at_alert: int


@dataclass
class IncrementalDetector:
    """Packet-driven cumulative-volume trigger.

    ``min_bytes`` is the volume that must have moved, ``min_rate`` the sustained
    bytes-per-second and ``min_age`` the elapsed connection lifetime, all
    measured on the outbound direction only.
    """

    min_bytes: int
    min_rate: float
    min_age: float = 0.0
    conns: dict[int, _ConnState] = field(default_factory=dict)
    alerts: list[Alert] = field(default_factory=list)

    def observe(self, src_port: int, ts: float, payload: int,
                is_syn: bool = False) -> Alert | None:
        """Feed one packet. ``is_syn`` starts a fresh connection on this port.

        Resetting on SYN is not a detail: the kernel recycles ephemeral ports,
        so state keyed on the port alone accumulates bytes across unrelated
        connections and fires on volume that no single connection ever moved.
        """
        st = self.conns.get(src_port)
        if st is None or is_syn:
            st = _ConnState(t_first=ts)
            self.conns[src_port] = st
        st.payload += payload
        if st.alerted:
            return None
        age = max(ts - st.t_first, 1e-6)
        if (st.payload >= self.min_bytes
                and st.payload / age >= self.min_rate
                and age >= self.min_age):
            st.alerted = True
            a = Alert(src_port=src_port, time=ts, payload_at_alert=st.payload)
            self.alerts.append(a)
            return a
        return None


def parse_pcap(pcap: str, dst_port: int) -> list[tuple[float, int, int, bool]]:
    """Decode once into ``(ts, src_port, payload_len, is_syn)`` tuples.

    Sweeping the detector's operating point means replaying the same trace tens
    of times; decoding it once and replaying from memory turns a several-minute
    sweep into a few seconds, and guarantees every setting sees byte-identical
    input.

    Payload length is taken from the IP/TCP headers, as elsewhere, so the small
    capture snaplen does not distort it.
    """
    out: list[tuple[float, int, int, bool]] = []
    with open(pcap, "rb") as fh:
        for ts, buf in dpkt.pcap.Reader(fh):
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
            if tcp.dport != dst_port:
                continue
            is_syn = bool(tcp.flags & dpkt.tcp.TH_SYN) and not (
                tcp.flags & dpkt.tcp.TH_ACK)
            n = int(ip.len) - ip.hl * 4 - tcp.off * 4
            if n <= 0 and not is_syn:
                continue
            out.append((float(ts), int(tcp.sport), max(n, 0), is_syn))
    return out


def replay(packets, min_bytes: int, min_rate: float,
           min_age: float = 0.0) -> list[Alert]:
    """Run the detector over pre-parsed packets; return every alert, in order.

    Alerts are a list rather than keyed by source port, because ports are
    recycled: the caller attributes each alert to a transfer by
    ``(port, time window)``, the same rule used for the exported records.
    """
    det = IncrementalDetector(min_bytes=min_bytes, min_rate=min_rate,
                              min_age=min_age)
    for ts, sport, n, is_syn in packets:
        det.observe(sport, ts, n, is_syn=is_syn)
    return det.alerts


def run_over_pcap(
    pcap: str, dst_port: int, min_bytes: int, min_rate: float,
    min_age: float = 0.0,
) -> list[Alert]:
    """Convenience wrapper: parse then replay. Prefer the two-step form in a sweep."""
    return replay(parse_pcap(pcap, dst_port), min_bytes, min_rate, min_age)
