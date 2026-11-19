"""Episode model: a receiver-side byte-arrival timeline for one session.

An *episode* is the unit of evaluation. For an attack episode it captures a
single exfiltration attempt; for a benign episode it captures ordinary traffic
that a detector must not fire on.

Ground truth is measured at the *receiver, application layer* — the number of
bytes that have actually landed at the destination by the end of each window.
This is what the EBD metric is denominated in: bytes that have irreversibly
left the network. Packet-layer counts are avoided on purpose (retransmits,
padding and header overhead all inflate them relative to the asset actually
stolen).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class ReconciliationError(ValueError):
    """Raised when receiver bytes diverge from the sender by more than tolerance.

    Episodes that fail reconciliation must not enter the benchmark database:
    if we cannot trust the byte ground truth, every EBD figure computed from
    the episode is meaningless.
    """


@dataclass(frozen=True)
class ScenarioCell:
    """One cell of the adversary matrix (rate x chunking x destination)."""

    rate: str  # "burst" | "fast" | "throttled" | "drip"
    chunking: str  # "single" | "chunked" | "many-small"
    destination: str  # e.g. "cloud-a" | "cloud-b" | "self-host"

    @property
    def key(self) -> str:
        return f"{self.rate}/{self.chunking}/{self.destination}"


@dataclass(frozen=True)
class Episode:
    """A per-window receiver-side byte timeline plus its label and provenance.

    Attributes
    ----------
    episode_id:
        Stable identifier.
    label:
        ``True`` for an exfiltration (attack) episode, ``False`` for benign.
    window_bytes:
        Bytes that arrived at the receiver during each window, in order. Length
        is the number of windows. Units are bytes; the synthetic generator uses
        megabytes-as-float, which is fine as long as every episode shares a unit.
    dt:
        Window duration in seconds (default 1.0).
    scenario:
        The adversary matrix cell for attack episodes; ``None`` for benign.
    sender_total:
        Bytes the *sender* reports having transmitted, used only for
        reconciliation. ``None`` skips the check.
    """

    episode_id: str
    label: bool
    window_bytes: np.ndarray
    dt: float = 1.0
    scenario: ScenarioCell | None = None
    sender_total: float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        wb = np.asarray(self.window_bytes, dtype=float)
        if wb.ndim != 1:
            raise ValueError("window_bytes must be 1-D")
        if np.any(wb < 0):
            raise ValueError("window_bytes must be non-negative")
        # frozen dataclass: bypass the setattr guard once, deliberately.
        object.__setattr__(self, "window_bytes", wb)

    # -- geometry -------------------------------------------------------------

    @property
    def n_windows(self) -> int:
        return int(self.window_bytes.shape[0])

    @property
    def duration(self) -> float:
        return self.n_windows * self.dt

    @property
    def cumulative_bytes(self) -> np.ndarray:
        """Cumulative bytes landed at the receiver at the *end* of each window."""
        return np.cumsum(self.window_bytes)

    @property
    def total_bytes(self) -> float:
        """Full asset exposure if the episode runs to completion undetected."""
        return float(self.window_bytes.sum())

    # -- EBD support ----------------------------------------------------------

    def bytes_leaked_by(self, alert_window: int | None) -> float:
        """Bytes gone by the time an alert is raised.

        ``alert_window`` is the 0-based index of the window during which the
        detector fired. Bytes that arrived *up to and including* that window are
        counted as leaked — the alert cannot un-send what already arrived.

        ``None`` means the detector never fired: the full asset is lost.
        A negative index is treated as "fired before any data moved" (0 bytes).
        """
        if alert_window is None:
            return self.total_bytes
        if alert_window < 0:
            return 0.0
        k = min(alert_window, self.n_windows - 1)
        return float(self.cumulative_bytes[k])

    # -- integrity ------------------------------------------------------------

    def reconcile(self, tol: float = 0.01) -> None:
        """Verify receiver bytes match the sender within ``tol`` (default 1%).

        Raises :class:`ReconciliationError` on divergence so a bad capture is
        rejected loudly rather than silently polluting the benchmark.
        """
        if self.sender_total is None:
            return
        recv = self.total_bytes
        denom = max(self.sender_total, 1e-9)
        divergence = abs(recv - self.sender_total) / denom
        if divergence > tol:
            raise ReconciliationError(
                f"episode {self.episode_id}: receiver={recv:.3f} vs "
                f"sender={self.sender_total:.3f} "
                f"({divergence:.2%} > {tol:.0%} tolerance)"
            )

    # -- feature access -------------------------------------------------------

    def rate_series(self) -> np.ndarray:
        """Instantaneous byte-arrival rate per window (bytes / second)."""
        return self.window_bytes / self.dt
