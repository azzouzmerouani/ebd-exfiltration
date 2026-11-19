"""Real-traffic validation of the architectural EBD floor.

Everything else in this package works on synthetic byte timelines. This
subpackage works on a public capture — DAPT2021 / "Unraveled" (ASU), which
distributes both labeled CICFlowMeter flow records *and* the raw packet
captures they were derived from, over a public S3 bucket with no
authentication.

The distinction matters: the flow CSVs are what an incumbent detector consumes,
and the PCAP is the only thing that can supply *ground truth* for EBD (how many
bytes had actually left by a given instant). Measuring one against the other is
the whole point.
"""

from .episodes import (
    FlowRecord,
    ExfilEpisode,
    group_into_episodes,
    DEFAULT_GAP_SECONDS,
)

__all__ = [
    "FlowRecord",
    "ExfilEpisode",
    "group_into_episodes",
    "DEFAULT_GAP_SECONDS",
]
