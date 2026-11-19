"""Export cadence: the architectural EBD floor of flow-based detectors.

The incumbent exfiltration detectors are **batch per-flow classifiers**: they
consume a completed flow record (EDXGB reads CICFlowMeter rows; NEDS reads
aggregated per-protocol session metadata) and emit one decision per record. They
have no notion of a score at time *t* — their first opportunity to say anything
about a transfer is when the telemetry pipeline *emits a record* for it.

That single fact bounds how early they can possibly be, independently of how good
their machine learning is.

The bound
---------
Let ``T`` be the record-emission interval (the exporter's active/flow timeout) and
``D`` the duration of the exfiltration. A flow-based detector sees nothing until
the first record exists, so even a **perfect** one — correct on the very first
record, zero false positives — cannot alert before ``min(T, D)``. Hence

    EBD_floor / asset  =  min(1, T / D)      (constant-rate transfer)

with two consequences that matter:

* **D <= T  =>  EBD = 100 %.** A transfer that finishes inside one emission
  interval produces its first record only at flow teardown — after every byte has
  already left. No flow-based detector can do better than total loss.
* The bound is **rate-shaped and inverted**: the *faster* the theft, the smaller
  ``D``, and the closer the floor gets to total loss. The architecture is most
  blind exactly where damage accrues fastest.

Why this is a floor, not an estimate
------------------------------------
It is computed against an **oracle** incumbent (:class:`OracleFlowDetector`) that
fires on its first available record with perfect accuracy. Real detectors need
evidence and make mistakes, so they can only do worse. Nothing here depends on any
particular model's quality — which is what makes the claim hard to attack.

Refinements that *help* the incumbent are modelled honestly, so the floor stays
conservative: an idle gap longer than the idle timeout tears the flow down early
and emits a record sooner, and a transfer split across many short-lived
connections emits one record per connection. Both are represented by
:func:`emission_windows`.

Published defaults used in the paper
------------------------------------
``CICFlowMeter`` 120 s flow timeout (the tool behind SCVIC-APT-2021; its 5 s
"activity timeout" parameterises active/idle burst *features*, not export).
NetFlow/IPFIX active timeout is commonly 60 s (Cisco IOS default, and the usual
recommendation) but defaults to 1800 s on Cisco IOS XR — a 30x spread in a
parameter the detection literature does not report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detectors.base import BaseDetector
from .episode import Episode

# Record-emission intervals (seconds) documented for real exporters.
CADENCE_PRESETS: dict[str, float] = {
    "cicflowmeter-120s": 120.0,   # CICFlowMeter flow timeout (SCVIC-APT-2021)
    "netflow-60s": 60.0,          # Cisco IOS default / common recommendation
    "netflow-300s": 300.0,        # frequently deployed compromise
    "iosxr-1800s": 1800.0,        # Cisco IOS XR default active timeout
}


# --------------------------------------------------------------------------- #
# Emission model                                                              #
# --------------------------------------------------------------------------- #

def emission_windows(
    episode: Episode,
    flow_timeout: float,
    idle_timeout: float | None = None,
) -> list[int]:
    """Window indices at which a flow record becomes available to a detector.

    A record is emitted when any of these happens:

    * **active/flow timeout** — the flow has been open for ``flow_timeout``
      seconds; the exporter flushes a record and the flow continues;
    * **idle timeout** — no bytes have arrived for ``idle_timeout`` seconds; the
      flow is torn down and flushed (a new flow starts on the next active
      window). ``None`` disables idle teardown;
    * **end of episode** — the transfer finishes and the final record is flushed.

    Returns 0-based window indices, ascending and unique. Indices are the windows
    at whose *end* the record exists, matching
    :meth:`Episode.bytes_leaked_by`.
    """
    dt = episode.dt
    n = episode.n_windows
    out: list[int] = []

    flow_open = False
    flow_start = 0          # window index where the current flow began
    last_active = -1        # last window index with bytes

    for k in range(n):
        active = episode.window_bytes[k] > 0

        if active and not flow_open:
            flow_open = True
            flow_start = k

        if flow_open:
            # idle teardown: gap since last activity exceeded the idle timeout
            if (
                idle_timeout is not None
                and not active
                and last_active >= 0
                and (k - last_active) * dt >= idle_timeout
            ):
                out.append(k)
                flow_open = False
                continue

            # active timeout: flow has been open long enough to be flushed
            if (k - flow_start + 1) * dt >= flow_timeout:
                out.append(k)
                flow_start = k + 1      # exporter resets the active clock

        if active:
            last_active = k

    if n and (not out or out[-1] != n - 1):
        out.append(n - 1)               # final record at teardown

    return sorted(set(i for i in out if 0 <= i < n))


# --------------------------------------------------------------------------- #
# Oracle incumbent                                                            #
# --------------------------------------------------------------------------- #

class OracleFlowDetector(BaseDetector):
    """A perfect flow-based detector: correct on its first available record.

    Upper bound on every batch per-flow detector. It knows the answer, never
    false-alarms, and fires the instant a record exists — so the EBD it achieves
    is a *floor* that no real flow-based detector can beat. Scores are 1.0 from
    the first emission onward and 0.0 before, so the standard EBD harness (which
    alerts on the first window crossing a threshold) reads it correctly.
    """

    def __init__(self, flow_timeout: float, idle_timeout: float | None = None) -> None:
        self.flow_timeout = flow_timeout
        self.idle_timeout = idle_timeout
        self.name = f"oracle-flow@{flow_timeout:g}s"
        self._fire_at: int | None = None
        self._k = 0

    def reset(self) -> None:
        self._fire_at = None
        self._k = 0

    def update(self, window_bytes: float, episode: Episode) -> float:
        if self._fire_at is None:
            em = emission_windows(episode, self.flow_timeout, self.idle_timeout)
            self._fire_at = em[0] if em else episode.n_windows
        score = 1.0 if self._k >= self._fire_at else 0.0
        self._k += 1
        return score


# --------------------------------------------------------------------------- #
# Floor computation                                                           #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CadenceFloor:
    """Best-case EBD for any flow-based detector on one episode."""

    episode_id: str
    rate: str
    duration_s: float
    cadence_s: float
    first_record_window: int
    ebd_mb: float
    nebd: float          # fraction of the asset lost, in [0, 1]
    total_loss: bool     # True when the whole asset is gone before any record


def cadence_floor(
    episode: Episode,
    cadence_s: float,
    idle_timeout: float | None = None,
) -> CadenceFloor:
    """EBD floor for ``episode`` at record-emission interval ``cadence_s``."""
    em = emission_windows(episode, cadence_s, idle_timeout)
    first = em[0] if em else episode.n_windows - 1
    ebd = episode.bytes_leaked_by(first)
    total = episode.total_bytes
    nebd = (ebd / total) if total > 0 else 0.0
    return CadenceFloor(
        episode_id=episode.episode_id,
        rate=episode.scenario.rate if episode.scenario else "?",
        duration_s=episode.duration,
        cadence_s=cadence_s,
        first_record_window=first,
        ebd_mb=ebd,
        nebd=nebd,
        total_loss=nebd >= 0.999,
    )


def theoretical_floor(duration_s: float, cadence_s: float) -> float:
    """Closed form ``min(1, T/D)`` for a constant-rate transfer.

    Provided so the simulated floor can be checked against the algebra the paper
    states, rather than the two drifting apart.
    """
    if duration_s <= 0:
        return 1.0
    return float(min(1.0, cadence_s / duration_s))
