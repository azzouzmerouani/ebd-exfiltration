"""Change-detection baseline: CUSUM on the byte-arrival rate.

A CUSUM (cumulative sum) detector accumulates how far the recent rate runs
*above* a benign reference, and resets to zero whenever the rate falls back.
Unlike a single-window volume threshold it does not need any one window to
spike; a sustained moderate transfer piles up drift and eventually trips. That
makes it stronger than :class:`VolumeThreshold` on slow drips and weaker than a
rate-blind cumulative counter on bursts — a genuinely different point in the
earliness/false-positive trade-off, which is what the ranking machinery needs.

The reference rate ``k`` is a slack term: rates within ``k`` of the baseline are
treated as noise and do not accumulate. It is set from a benign sample via
:meth:`fit`; absent that, a conservative default is used.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..episode import Episode
from .base import BaseDetector


class CusumRate(BaseDetector):
    """One-sided CUSUM on per-window byte-arrival rate."""

    name = "cusum"

    def __init__(self, k: float | None = None) -> None:
        self.k = k              # slack (MB/window); learned in fit() if None
        self._s = 0.0

    def fit(self, benign: Sequence[Episode], attack: Sequence[Episode]) -> "CusumRate":
        """Set the slack to benign mean + 3 sigma of the per-window rate."""
        rates = np.concatenate(
            [ep.rate_series() for ep in benign if ep.n_windows > 0]
        ) if benign else np.array([0.0])
        self.k = float(rates.mean() + 3.0 * rates.std())
        return self

    def reset(self) -> None:
        self._s = 0.0

    def update(self, window_bytes: float, episode: Episode) -> float:
        k = self.k if self.k is not None else 1.0
        rate = float(window_bytes) / episode.dt
        self._s = max(0.0, self._s + (rate - k))
        return self._s
