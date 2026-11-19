"""Sweep the architectural floor across the telemetry cadence ``T``.

Why this exists
---------------
Section "Discussion" of the paper recommends that the field report and optimise
``T``. A recommendation is not a result. This module produces the curve that
makes the recommendation actionable: for each real episode, the oracle floor as
a function of the exporter's active timeout.

The generalised first-record time
---------------------------------
The paper's Eq. (4) reads

    T_eff = min( T_active , min_f  t_end(f) - t_start(e) )

which is correct *for the timeout the capture was actually exported at*, because
the observed ``t_end(f)`` already embeds that timeout. It cannot be re-used to
ask "what if the exporter had been configured differently", since lowering the
active timeout **splits** long flows and therefore changes ``t_end(f)``.

The counterfactual needs the per-flow form. Under a hypothetical active timeout
``Ta`` a flow observed to span ``[s, e]`` emits its first record at

    min( s + Ta , e )                      -- timeout fires, or connection closes

whichever happens first, so

    T_eff(e; Ta) = min_{f in e}  min( s_f + Ta , e_f )  -  t_start(e)      (4')

Eq. (4') reduces to Eq. (4) at the timeout the data was exported with, which is
asserted as a regression check in :func:`effective_T_at`. It is strictly more
general: it is defined for any ``Ta``, including values below the one the
capture used.

One honest limitation, stated in the paper alongside the result: splitting a
flow at ``s + Ta`` changes the *features* of the record a detector would see, so
this sweep is a sweep of the **oracle floor**, not of a particular detector's
measured EBD. Re-measuring the detector at each ``Ta`` would require re-exporting
the raw capture and re-training. The floor is exactly the quantity that is
model-independent, which is what makes the counterfactual legitimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

# Candidate active timeouts, in seconds. Chosen to span the range actually
# observed in deployed exporters and analysis tools:
#   15    aggressive tuning (below every documented default)
#   60    Juniper flow-active-timeout default
#   120   CICFlowMeter flow timeout
#   300   common collector-side aggregation window
#   600   conservative enterprise setting
#   1800  Cisco IOS Flexible NetFlow / IOS XR / nfstream default
DEFAULT_TIMEOUTS: tuple[float, ...] = (15.0, 60.0, 120.0, 300.0, 600.0, 1800.0)


def effective_T_at(episode, active_timeout: float) -> float:
    """Eq. (4'): time until the episode's first record exists, under ``Ta``.

    Returns seconds after ``episode.t_start``. Always in ``[0, Ta]``.
    """
    if not episode.flows:
        return active_timeout
    t0 = episode.t_start
    first = min(min(f.start + active_timeout, f.end) for f in episode.flows)
    return max(0.0, first - t0)


def byte_floor_at(
    episode,
    bins: Sequence[float],
    bin_start: float,
    bin_seconds: float,
    total_bytes: float,
    active_timeout: float,
) -> float:
    """Fraction of the asset already delivered when the first record exists.

    This is the floor EBD must be compared against: it is denominated in bytes,
    read off the reconstructed timeline, not in duration.
    """
    if total_bytes <= 0 or not len(bins):
        return 0.0
    t_first = episode.t_start + effective_T_at(episode, active_timeout)
    idx = int((t_first - bin_start) // bin_seconds)
    if idx < 0:
        return 0.0
    leaked = float(sum(bins[: min(idx + 1, len(bins))]))
    return min(1.0, leaked / float(total_bytes))


def time_floor_at(episode, active_timeout: float) -> float:
    """``min(1, T_eff/D)`` -- the closed form, as a fraction of DURATION.

    Reported beside the byte floor so the reader can see how far the
    constant-rate closed form of Eq. (3) departs from the byte-denominated
    quantity on bursty real traffic. It is *not* the number EBD is compared
    against; see the module docstring.
    """
    d = episode.duration
    if d <= 0:
        return 1.0
    return min(1.0, effective_T_at(episode, active_timeout) / d)


@dataclass(frozen=True)
class SweepRow:
    episode_id: str
    active_timeout: float
    t_eff: float
    duration: float
    byte_floor: float
    time_floor: float
    asset_bytes: int

    @property
    def ratio_T_over_D(self) -> float:
        return self.t_eff / self.duration if self.duration > 0 else float("inf")


def sweep(
    episode,
    bins: Sequence[float],
    bin_start: float,
    bin_seconds: float,
    total_bytes: int,
    timeouts: Sequence[float] = DEFAULT_TIMEOUTS,
) -> list[SweepRow]:
    """Floor for one episode across the candidate cadences."""
    rows: list[SweepRow] = []
    for ta in timeouts:
        rows.append(
            SweepRow(
                episode_id=episode.episode_id,
                active_timeout=float(ta),
                t_eff=effective_T_at(episode, ta),
                duration=episode.duration,
                byte_floor=byte_floor_at(
                    episode, bins, bin_start, bin_seconds, total_bytes, ta
                ),
                time_floor=time_floor_at(episode, ta),
                asset_bytes=int(total_bytes),
            )
        )
    return rows
