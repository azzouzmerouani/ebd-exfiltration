"""The online detector contract every benchmark detector implements.

Detection is treated as a *streaming* problem: windows arrive one at a time and
the detector must produce a score for the current window using only what it has
seen so far. This is what makes early detection meaningful — a detector that
needs the whole episode before scoring can never alert early by construction.

All six benchmark detectors (and the ECTS trigger layer) speak this interface,
so the harness can drive any of them identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..episode import Episode


class BaseDetector(ABC):
    """Base class for online, causal exfiltration detectors.

    Contract
    --------
    * :meth:`reset` is called once before each episode; clear all per-episode
      state here.
    * :meth:`update` is called once per window, in order, and returns a scalar
      anomaly score for the current window. Higher = more suspicious. It must
      depend only on windows seen so far (no peeking ahead).

    A detector may read static, non-causal fields of ``episode`` (e.g. an asset
    size known at session start) but must not index ``episode.window_bytes``
    beyond the current window.
    """

    name: str = "base"

    def reset(self) -> None:  # noqa: B027 - intentional no-op default
        """Clear per-episode state. Override if the detector is stateful."""

    @abstractmethod
    def update(self, window_bytes: float, episode: Episode) -> float:
        """Return the anomaly score for the current window.

        Parameters
        ----------
        window_bytes:
            Bytes that arrived at the receiver during this window.
        episode:
            The episode being scored, for access to static context only.
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"
