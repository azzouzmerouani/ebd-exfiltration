"""Group labeled flow records into distinct exfiltration *episodes*.

Why this exists
---------------
A labeled dataset reports **flows**, not incidents. One exfiltration event
becomes many flow rows for two independent reasons:

* **The exporter splits it.** CICFlowMeter closes a flow at its flow timeout
  (120 s by default), so a transfer lasting ten minutes is emitted as several
  consecutive records — this is exactly the mechanism the EBD floor is about.
* **The transfer uses several connections.** A bulk upload opened with parallel
  streams gets a fresh source port per TCP connection, so one logical transfer
  appears as several simultaneous flows.

Counting flow rows as if they were incidents would inflate n by an order of
magnitude and make a handful of events look like a benchmark. EBD is defined
per *episode* — bytes lost before the alert, out of the bytes that episode
exfiltrated — so the episode is the only correct unit.

Grouping rule
-------------
Flows are grouped by the **(source IP, destination IP, destination port)**
triple, deliberately *excluding* the source port, since parallel connections of
one transfer differ only in that field. Within a group, flows are sorted by
start time and split into a new episode whenever the gap between the end of the
previous flow and the start of the next exceeds ``gap_seconds``.

The default gap is one flow timeout (120 s). The justification is mechanical
rather than arbitrary: when the exporter splits a *continuous* transfer, the
next record begins immediately, so within-episode gaps are near zero; a silence
longer than the exporter's own timeout is therefore better explained by "a
separate transfer later" than by "the same transfer, still running". Callers
can widen or narrow it, and :func:`sensitivity` reports how much the episode
count depends on the choice — which is the honest thing to publish alongside n.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_GAP_SECONDS = 120.0


@dataclass(frozen=True)
class FlowRecord:
    """One labeled flow row, reduced to the fields grouping and EBD need.

    ``start`` is epoch seconds; ``duration`` is seconds. ``fwd_bytes`` is the
    source -> destination byte count, which for an exfiltration flow is the
    direction the data leaves in. ``bwd_bytes`` is the reverse direction, kept
    because the incumbent detector's own logic keys on the asymmetry between
    the two.
    """

    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: int
    start: float
    duration: float
    fwd_bytes: int
    bwd_bytes: int
    fwd_packets: int = 0
    bwd_packets: int = 0
    stage: str = ""
    activity: str = ""
    source_file: str = ""

    @property
    def end(self) -> float:
        return self.start + max(self.duration, 0.0)

    @property
    def group_key(self) -> tuple[str, str, int]:
        """Grouping key: source port is excluded on purpose (see module doc)."""
        return (self.src_ip, self.dst_ip, self.dst_port)


@dataclass(frozen=True)
class ExfilEpisode:
    """A set of flow records treated as one exfiltration event."""

    episode_id: str
    flows: list[FlowRecord] = field(default_factory=list)

    @property
    def src_ip(self) -> str:
        return self.flows[0].src_ip

    @property
    def dst_ip(self) -> str:
        return self.flows[0].dst_ip

    @property
    def dst_port(self) -> int:
        return self.flows[0].dst_port

    @property
    def t_start(self) -> float:
        return min(f.start for f in self.flows)

    @property
    def t_end(self) -> float:
        return max(f.end for f in self.flows)

    @property
    def duration(self) -> float:
        """Wall-clock span of the episode, in seconds — the ``D`` of min(1, T/D)."""
        return self.t_end - self.t_start

    @property
    def fwd_bytes(self) -> int:
        """Flow-reported outbound bytes.

        This is the dataset's own number, used for the reconciliation check
        against the PCAP reconstruction. It is *not* the ground truth for EBD —
        the reconstructed byte timeline is.
        """
        return sum(f.fwd_bytes for f in self.flows)

    @property
    def n_flows(self) -> int:
        return len(self.flows)

    @property
    def is_multi_flow(self) -> bool:
        return len(self.flows) > 1

    @property
    def source_files(self) -> set[str]:
        return {f.source_file for f in self.flows if f.source_file}

    def five_tuples(self) -> set[tuple[str, int, str, int, int]]:
        """Distinct 5-tuples, for selecting this episode's packets from a PCAP."""
        return {
            (f.src_ip, f.src_port, f.dst_ip, f.dst_port, f.protocol)
            for f in self.flows
        }


def group_into_episodes(
    flows: list[FlowRecord],
    gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> list[ExfilEpisode]:
    """Group flow records into episodes. See module docstring for the rule.

    Flows are grouped by ``(src_ip, dst_ip, dst_port)`` and split whenever the
    silence between one flow's end and the next flow's start exceeds
    ``gap_seconds``. Overlapping or back-to-back flows always stay together.

    Returned episodes are ordered by start time; each carries a stable id built
    from its key and start time.
    """
    if gap_seconds < 0:
        raise ValueError("gap_seconds must be non-negative")

    by_key: dict[tuple[str, str, int], list[FlowRecord]] = {}
    for f in flows:
        by_key.setdefault(f.group_key, []).append(f)

    episodes: list[ExfilEpisode] = []
    for key, group in by_key.items():
        group = sorted(group, key=lambda f: (f.start, f.end))
        current: list[FlowRecord] = [group[0]]
        # running end-of-episode; max() so a long flow keeps the window open
        running_end = group[0].end

        for f in group[1:]:
            if f.start - running_end > gap_seconds:
                episodes.append(_make_episode(key, current))
                current = [f]
                running_end = f.end
            else:
                current.append(f)
                running_end = max(running_end, f.end)
        episodes.append(_make_episode(key, current))

    return sorted(episodes, key=lambda e: e.t_start)


def _make_episode(key: tuple[str, str, int], flows: list[FlowRecord]) -> ExfilEpisode:
    src, dst, port = key
    t0 = min(f.start for f in flows)
    return ExfilEpisode(episode_id=f"{src}->{dst}:{port}@{t0:.3f}", flows=list(flows))


def sensitivity(
    flows: list[FlowRecord],
    gaps: tuple[float, ...] = (30.0, 60.0, 120.0, 300.0, 600.0),
) -> dict[float, int]:
    """Episode count as a function of the gap threshold.

    Publish this next to n. If the count is stable across a wide range of
    thresholds, the grouping is not an artifact of the chosen value; if it
    swings wildly, that is a caveat the reader is entitled to.
    """
    return {g: len(group_into_episodes(flows, g)) for g in gaps}
