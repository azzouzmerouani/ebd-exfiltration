"""Measure the record-emission interval ``T`` from the data, never assume it.

``EBD_floor = min(1, T/D)`` is only as trustworthy as ``T``, and ``T`` is the
parameter the detection literature does not report. Assuming a tool's documented
default is exactly the mistake that produced the CICFlowMeter "5 s" confusion
(that figure is an active/idle *feature* parameter, not an export cadence), so
here ``T`` is measured from the flow records themselves.

Two different quantities, both real
-----------------------------------
**Active timeout** — the exporter's ceiling on how long a single flow may stay
open before a record is forced out. It shows up as a hard ceiling in the flow
duration distribution. For DAPT2021: the longest of 7,522 exfiltration flows is
1798.25 s and none exceeds 1800 s, which pins nfstream's ``active_timeout`` at
its 1800 s default.

**Effective T** — the time until the *first record covering an episode* actually
becomes available, which is what bounds a detector. A flow record is emitted
when its connection **closes**, not only when the active timeout fires, so

    T_eff = min(active_timeout, duration of the episode's first connection)

For an episode carried by one long-lived connection these coincide. For an
episode built from many short connections — a beacon, or a transfer chunked to
evade volume thresholds — the connection length dominates and ``T_eff`` is far
smaller than the active timeout.

The consequence is worth stating, because it cuts against intuition:
**chunking a transfer into many short connections shortens the architectural
blind spot.** T1030-style chunking evades size-based detection but hands a
flow-record consumer its first evidence sooner. The attacker's best move against
*this* detector class is one long, fast connection, not many small ones.
"""

from __future__ import annotations

from dataclasses import dataclass

# nfstream's documented default; used only as the ceiling to clamp against,
# after the measurement below confirms it.
NFSTREAM_ACTIVE_TIMEOUT_S = 1800.0


@dataclass(frozen=True)
class CadenceMeasurement:
    """What the flow records say about the exporter's emission behaviour."""

    n_flows: int
    max_duration: float
    n_over_ceiling: int
    inferred_active_timeout: float

    def report(self) -> str:
        return (
            f"flows={self.n_flows}  max_duration={self.max_duration:.2f}s  "
            f"flows_over_{self.inferred_active_timeout:.0f}s={self.n_over_ceiling}\n"
            f"  -> inferred active_timeout = {self.inferred_active_timeout:.0f}s"
        )


def measure_active_timeout(
    flows, candidates: tuple[float, ...] = (60.0, 120.0, 300.0, 1800.0),
    tolerance: float = 0.01,
) -> CadenceMeasurement:
    """Infer the exporter's active timeout from the flow duration ceiling.

    The smallest candidate that no flow meaningfully exceeds is the timeout: a
    flow simply cannot outlive it. Reporting the count above the ceiling makes
    the inference falsifiable rather than a guess.
    """
    durs = [f.duration for f in flows]
    if not durs:
        return CadenceMeasurement(0, 0.0, 0, NFSTREAM_ACTIVE_TIMEOUT_S)
    mx = max(durs)
    inferred = NFSTREAM_ACTIVE_TIMEOUT_S
    for c in sorted(candidates):
        if mx <= c * (1.0 + tolerance):
            inferred = c
            break
    over = sum(1 for d in durs if d > inferred * (1.0 + tolerance))
    return CadenceMeasurement(len(durs), mx, over, inferred)


def effective_T(episode, active_timeout: float = NFSTREAM_ACTIVE_TIMEOUT_S) -> float:
    """Time until the episode's first flow record becomes available.

    A record is emitted when its flow **ends**, so the first record an episode
    yields is the one whose flow ends earliest — which is *not* necessarily the
    flow that started first. An episode carrying many parallel connections can
    have a long-running first connection while a later, shorter one closes well
    before it and emits its record first.

    Getting this wrong understates how early a detector could act, and shows up
    immediately as a negative gap (a detector apparently alerting before any
    record could exist), which is why the sign of the gap is worth asserting.
    """
    if not episode.flows:
        return active_timeout
    earliest_end = min(f.end for f in episode.flows)
    t = earliest_end - episode.t_start
    if t <= 0:
        return 0.0
    return min(active_timeout, t)


def floor_for(episode, active_timeout: float = NFSTREAM_ACTIVE_TIMEOUT_S) -> float:
    """``min(1, T_eff/D)`` — the floor as a fraction of the episode's DURATION.

    Correct as stated, but note what it is not: a fraction of *bytes*. The two
    coincide only when the transfer runs at constant rate. For a bursty or
    beaconing episode use :func:`byte_floor`, which reads the actual asset
    accrued by ``T_eff`` off the reconstructed timeline. Comparing this
    time-fraction against a byte-denominated detector result produces
    nonsense — including negative "gaps", where a detector appears to fire
    before the first record could exist.
    """
    d = episode.duration
    if d <= 0:
        return 1.0
    return min(1.0, effective_T(episode, active_timeout) / d)


def byte_floor(
    episode,
    bin_start: float,
    bin_seconds: float,
    bins,
    total_bytes: int,
    active_timeout: float = NFSTREAM_ACTIVE_TIMEOUT_S,
) -> float:
    """Fraction of the **asset** already gone when the first record exists.

    This is the floor EBD must be compared against, because EBD is denominated
    in bytes. The earliest a flow-record consumer can alert is when its first
    record is emitted, at ``t_start + T_eff``; whatever the reconstructed
    timeline says had left by then is lost to *any* such detector.

    Because a real detector's alert time is never earlier than that instant, the
    measured EBD is always >= this value — the gap cannot be negative, which is
    what makes it a usable invariant.
    """
    if total_bytes <= 0 or not bins:
        return 0.0
    t_first = episode.t_start + effective_T(episode, active_timeout)
    idx = int((t_first - bin_start) // bin_seconds)
    if idx < 0:
        return 0.0
    leaked = sum(bins[: min(idx + 1, len(bins))])
    return min(1.0, leaked / total_bytes)
