"""A trained, persistence-aware base detector.

The naive baselines score a window from its bytes alone, so they cannot tell a
*fast attack* from a *fast benign* transfer — and an ECTS trigger built on such a
base gains nothing from byte-denomination, because it accelerates false positives
as fast as true ones. The byte-vs-time control (D5) only becomes meaningful on a
base whose probability already separates the classes.

:class:`WindowClassifier` is that base. It learns a logistic model over streaming,
causal features — instantaneous and smoothed rate, how long the rate has stayed
*elevated* (the persistence signal that distinguishes a sustained exfiltration
session from a short benign burst), cumulative volume, elapsed time, and recent
elevation fraction. It is also a legitimate stand-alone detector: a "trained
classifier" baseline that should out-detect the naive ones.

Training uses scikit-learn; scoring is a hand-rolled numpy sigmoid over the stored
coefficients, so per-window inference is cheap and has no per-call library
overhead in the tight streaming loop.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np

from ..episode import Episode
from .base import BaseDetector

_RECENT = 10           # window of the "recent elevation fraction" feature
_ALPHA = 0.3           # EWMA smoothing for the rate feature
_N_FEATURES = 6


class WindowClassifier(BaseDetector):
    """Logistic classifier over streaming per-window features (P(attack))."""

    name = "classifier"

    def __init__(self, alpha: float = _ALPHA, recent: int = _RECENT) -> None:
        self.alpha = alpha
        self.recent = recent
        # learned parameters (set in fit)
        self._elev: float = 1.0
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        # streaming state (set in reset)
        self.reset()

    # -- streaming feature extraction ----------------------------------------

    def reset(self) -> None:
        self._t = 0
        self._cum = 0.0
        self._ewma = 0.0
        self._run = 0
        self._recent_elev: deque[int] = deque(maxlen=self.recent)

    def _step(self, window_bytes: float, dt: float) -> np.ndarray:
        """Advance state by one window and return its feature vector."""
        rate = float(window_bytes) / dt
        self._ewma = rate if self._t == 0 else self.alpha * rate + (1 - self.alpha) * self._ewma
        elevated = rate > self._elev
        self._run = self._run + 1 if elevated else 0
        self._cum += float(window_bytes)
        self._t += 1
        self._recent_elev.append(1 if elevated else 0)
        frac = sum(self._recent_elev) / len(self._recent_elev)
        return np.array(
            [rate, self._ewma, float(self._run), np.log1p(self._cum), float(self._t), frac]
        )

    # -- training ------------------------------------------------------------

    def fit(self, benign: Sequence[Episode], attack: Sequence[Episode]) -> "WindowClassifier":
        """Learn the elevation threshold and the logistic model.

        The elevation threshold is set a few times above the benign median rate
        and bounded, so a low-and-slow drip (~1 MB/window) still counts as
        elevated while ordinary light background does not — otherwise the
        persistence feature would be blind to exactly the attacks that need it.
        """
        benign = list(benign)
        attack = list(attack)
        rates = np.concatenate([ep.rate_series() for ep in benign if ep.n_windows]) \
            if benign else np.array([0.1])
        self._elev = float(np.clip(np.median(rates) * 4.0, 0.2, 0.9))

        X, y = [], []
        for ep in attack:
            self.reset()
            for k in range(ep.n_windows):
                X.append(self._step(ep.window_bytes[k], ep.dt))
                y.append(1)
        for ep in benign:
            self.reset()
            for k in range(ep.n_windows):
                X.append(self._step(ep.window_bytes[k], ep.dt))
                y.append(0)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(X)
        model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            scaler.transform(X), y
        )
        # freeze parameters for fast numpy inference
        self._mean = scaler.mean_.copy()
        self._scale = scaler.scale_.copy()
        self._coef = model.coef_.ravel().copy()
        self._intercept = float(model.intercept_[0])
        self.reset()
        return self

    # -- online scoring ------------------------------------------------------

    def update(self, window_bytes: float, episode: Episode) -> float:
        if self._coef is None:
            raise RuntimeError("WindowClassifier.update called before fit()")
        feat = self._step(window_bytes, episode.dt)
        z = float(((feat - self._mean) / self._scale) @ self._coef + self._intercept)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
