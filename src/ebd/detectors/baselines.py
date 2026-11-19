"""Simple baseline detectors to exercise the harness.

These are intentionally naive. Their job in the pilot is to populate the
benchmark with detectors that trade off earliness and false positives in
*different* ways, so the EBD-vs-F1 ranking machinery has something to rank.
The six real detectors from the benchmark spec replace them once lab data
exists; they all satisfy the same :class:`BaseDetector` contract.
"""

from __future__ import annotations

from ..episode import Episode
from .base import BaseDetector


class VolumeThreshold(BaseDetector):
    """Score = bytes in the current window.

    Fires when a single window is unusually large. Fast on bursts, blind to slow
    drips that never spike a single window.
    """

    name = "volume"

    def update(self, window_bytes: float, episode: Episode) -> float:
        return float(window_bytes)


class EWMARate(BaseDetector):
    """Score = exponentially weighted moving average of window bytes.

    Smooths the rate over recent windows, so a sustained moderate transfer
    accumulates a high score even if no single window spikes.
    """

    name = "ewma"

    def __init__(self, alpha: float = 0.3) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._ewma: float | None = None

    def reset(self) -> None:
        self._ewma = None

    def update(self, window_bytes: float, episode: Episode) -> float:
        if self._ewma is None:
            self._ewma = float(window_bytes)
        else:
            self._ewma = self.alpha * float(window_bytes) + (1 - self.alpha) * self._ewma
        return self._ewma


class CumulativeBytes(BaseDetector):
    """Score = total bytes sent so far this episode.

    Catches everything eventually (score only grows), but only crosses a fixed
    threshold *after* a fixed volume has already left — structurally late, which
    is exactly the kind of detector EBD should punish and F1 should not.
    """

    name = "cumulative"

    def __init__(self) -> None:
        self._total = 0.0

    def reset(self) -> None:
        self._total = 0.0

    def update(self, window_bytes: float, episode: Episode) -> float:
        self._total += float(window_bytes)
        return self._total
