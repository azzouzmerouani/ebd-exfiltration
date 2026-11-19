"""Session granularity: the second half of the record-consumer family.

The paper's floor holds for any consumer of completed records, with ``T`` read as
the emission interval of *whichever record the detector consumes*. Flow-granularity
consumers (EDXGB) see one record per transport connection. Session-granularity
consumers (NEDS) see one record per aggregation window, whose features are
averages over the connections it summarises.

Because an aggregate cannot exist before the connections it summarises have
ended, ``T_session >= T_flow`` always. This module measures how much larger.

Emission model, chosen to be generous to the session consumer
-------------------------------------------------------------
A real idle-gap sessioniser cannot know a session has ended until it has waited
out the idle timeout ``S``, so its record is emitted at ``last_flow_end + S``.
We emit at ``last_flow_end`` instead -- the earliest instant the record could
conceivably exist. Every session-granularity number in this paper is therefore a
*lower bound* on that family's floor, which makes the conclusion conservative:
the real blind spot is at least this deep, and probably deeper.

The chunking corollary, finally testable
----------------------------------------
The paper predicts that splitting a transfer into many short connections
*shortens* the blind spot at flow granularity (each connection emits on close)
and does *not* -- and may lengthen it -- at session granularity (the chunks are
summarised into one record, and extra chunks can extend the window). That
prediction was flagged as unmeasured. :func:`granularity_table` measures it.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

# Session idle timeouts to report. A session aggregator must wait out a silence
# before deciding a session has ended; these span plausible settings.
DEFAULT_SESSION_GAPS: tuple[float, ...] = (60.0, 300.0, 600.0, 1800.0)


@dataclass(frozen=True)
class SessionGroup:
    """A set of flows summarised into one record."""

    flows: list
    t_first: float
    t_last: float

    @property
    def n_flows(self) -> int:
        return len(self.flows)

    @property
    def emission_time(self) -> float:
        """Earliest instant this aggregate could be emitted."""
        return self.t_last


def sessionize(flows: Sequence, gap_seconds: float) -> list[SessionGroup]:
    """Split flows into aggregation groups on silences longer than ``gap_seconds``.

    Same rule the paper uses to build episodes, applied one level up: a session
    aggregator groups connections that are not separated by a silence longer
    than its idle timeout.
    """
    if not flows:
        return []
    ordered = sorted(flows, key=lambda f: (f.start, f.end))
    groups: list[list] = [[ordered[0]]]
    running_end = ordered[0].end
    for f in ordered[1:]:
        if f.start - running_end > gap_seconds:
            groups.append([f])
            running_end = f.end
        else:
            groups[-1].append(f)
            running_end = max(running_end, f.end)
    return [
        SessionGroup(
            flows=g,
            t_first=min(x.start for x in g),
            t_last=max(x.end for x in g),
        )
        for g in groups
    ]


def effective_T_session(episode, gap_seconds: float) -> float:
    """Time until the episode's first *session* record exists."""
    groups = sessionize(episode.flows, gap_seconds)
    if not groups:
        return 0.0
    first = min(g.emission_time for g in groups)
    return max(0.0, first - episode.t_start)


def byte_floor_session(
    episode,
    bins: Sequence[float],
    bin_start: float,
    bin_seconds: float,
    total_bytes: float,
    gap_seconds: float,
) -> float:
    """Fraction of the asset gone when the first session record exists."""
    if total_bytes <= 0 or not len(bins):
        return 0.0
    t_first = episode.t_start + effective_T_session(episode, gap_seconds)
    idx = int((t_first - bin_start) // bin_seconds)
    if idx < 0:
        return 0.0
    leaked = float(sum(bins[: min(idx + 1, len(bins))]))
    return min(1.0, leaked / float(total_bytes))


def n_records(episode, gap_seconds: float) -> int:
    """How many session records the episode yields. One means total loss."""
    return len(sessionize(episode.flows, gap_seconds))
