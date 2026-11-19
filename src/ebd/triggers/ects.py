"""ECTS trigger: cost-sensitive optimal stopping for exfiltration alerts.

Early Classification of Time Series (ECTS) is the optimal-stopping view of
detection: at every window the trigger chooses to *alert now* or *wait for more
evidence*, minimising expected total cost. Waiting buys confidence (fewer false
positives) but incurs a **delay cost**. The whole paper turns on what that delay
cost is denominated in.

Two variants share one myopic stopping rule and differ *only* in the delay cost:

* **byte cost** (the contribution) — waiting one more window costs the bytes
  expected to leak during it, i.e. it scales with the *current transfer rate*.
  The trigger therefore fires quickly on fast transfers and tolerates waiting on
  slow ones, because a slow transfer is leaking little per window.
* **time cost** (the D5 control) — waiting one more window costs a fixed amount,
  regardless of how fast data is moving. This is the ordinary early-classification
  objective. It is rate-blind.

Myopic rule
-----------
Let ``p`` be the calibrated probability the current window is exfiltration and
``d`` the delay cost of waiting one more window. Alerting risks a false-positive
cost ``C_fp`` if the episode is benign; waiting risks the delay cost if it is an
attack. One-step lookahead gives: alert when

    (1 - p) * C_fp  <=  p * d      <=>      p >= C_fp / (C_fp + d).

We expose the signed **margin** ``p - C_fp/(C_fp + d)`` as the detector score, so
the standard EBD harness can sweep a threshold around 0 to trace the operating
curve. Threshold 0 is the myopically optimal operating point; the sweep exists
only to draw the EBD-FPB trade-off.

For the byte variant ``d`` depends on the window's rate, so the cutoff is shaped
by rate at *every* operating point; for the time variant ``d`` is constant, so
the cutoff is flat and the trigger is rate-blind — which is precisely the
behaviour the D5 experiment is designed to separate from the byte variant.
"""

from __future__ import annotations

import numpy as np

from ..detectors.base import BaseDetector
from ..episode import Episode


class ProbabilityCalibrator:
    """Map a detector's raw scores to calibrated attack probabilities.

    A monotonic logistic fit on ``(score, is-attack-window)`` pairs. Kept
    deliberately simple and 1-D: the calibrator's only job is to turn an
    arbitrary anomaly score into something the cost rule can treat as P(attack).
    Degenerate training data (one class only) falls back to a constant.
    """

    def __init__(self) -> None:
        self._model = None
        self._const: float | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "ProbabilityCalibrator":
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        labels = np.asarray(labels, dtype=int)
        if labels.min() == labels.max():
            # only one class present; return its base rate
            self._const = float(labels.mean())
            return self
        from sklearn.linear_model import LogisticRegression

        self._model = LogisticRegression(max_iter=1000)
        self._model.fit(scores, labels)
        return self

    def predict_proba(self, score: float) -> float:
        if self._const is not None:
            return self._const
        if self._model is None:
            raise RuntimeError("calibrator not fitted")
        p = self._model.predict_proba(np.array([[score]]))[0, 1]
        return float(p)


class ECTSTrigger(BaseDetector):
    """Optimal-stopping trigger wrapping a base detector.

    Parameters
    ----------
    base:
        The online detector whose scores are calibrated into probabilities.
    cost_model:
        ``"byte"`` for the rate-aware byte-denominated delay cost (contribution)
        or ``"time"`` for the fixed per-window delay cost (control).
    lam:
        Delay-cost weight. Larger ``lam`` makes waiting more expensive, so the
        trigger fires earlier.
    c_fp:
        False-positive cost. Larger ``c_fp`` makes alerting more expensive, so
        the trigger waits for higher confidence.
    """

    def __init__(
        self,
        base: BaseDetector,
        cost_model: str = "byte",
        lam: float = 1.0,
        c_fp: float = 1.0,
    ) -> None:
        if cost_model not in ("byte", "time", "byte-floored"):
            raise ValueError("cost_model must be 'byte', 'time' or 'byte-floored'")
        self.base = base
        self.cost_model = cost_model
        self.lam = float(lam)
        self.c_fp = float(c_fp)
        self.name = f"ects-{cost_model}[{base.name}]"
        self._cal = ProbabilityCalibrator()
        self._rate_scale = 1.0
        self._fitted = False

    # -- training -------------------------------------------------------------

    def calibrate(
        self,
        attack_episodes,
        benign_episodes,
    ) -> "ECTSTrigger":
        """Fit score->probability calibration and the rate normaliser.

        Attack windows with data movement are positives; every benign window is
        a negative. The rate scale is the median rate of active attack windows,
        so ``lam`` means the same thing for the byte and time variants (both are
        "cost per typical active window").
        """
        scores, labels, rates = [], [], []
        for ep in attack_episodes:
            self.base.reset()
            for k in range(ep.n_windows):
                s = self.base.update(ep.window_bytes[k], ep)
                active = ep.window_bytes[k] > 0
                scores.append(s)
                labels.append(1 if active else 0)
                if active:
                    rates.append(ep.window_bytes[k] / ep.dt)
        for ep in benign_episodes:
            self.base.reset()
            for k in range(ep.n_windows):
                s = self.base.update(ep.window_bytes[k], ep)
                scores.append(s)
                labels.append(0)

        self._cal.fit(np.array(scores), np.array(labels))
        self._rate_scale = float(np.median(rates)) if rates else 1.0
        if self._rate_scale <= 0:
            self._rate_scale = 1.0
        self._fitted = True
        return self

    # -- online scoring -------------------------------------------------------

    def reset(self) -> None:
        self.base.reset()

    @staticmethod
    def _cutoff_for(c_fp: float, d: float) -> float:
        return c_fp / (c_fp + d) if (c_fp + d) > 0 else 1.0

    def _cutoff(self, window_bytes: float, episode: Episode) -> float:
        """Probability cutoff at which the myopic rule says "alert now".

        * ``time``          — fixed delay cost ``lam``: a rate-blind cutoff.
        * ``byte``          — delay cost scales with the current rate, so the
          cutoff drops on fast transfers *and rises on slow ones* (the flaw D5
          exposes: it under-reacts to low-and-slow and, being low on any fast
          flow, fires on fast benign bursts too).
        * ``byte-floored``  — the decoupled fix predicted by the D5 kernel: rate
          may only ever *lower* the cutoff (advance the alert), never raise it
          above the time-cost cutoff. The detection bar is therefore never
          stricter than ``time`` — it cannot under-detect a drip — while a fast
          transfer can still trip earlier. Rate advances the stop; it never
          suppresses a detection the base would otherwise make.
        """
        rate = (window_bytes / episode.dt) / self._rate_scale
        d_byte = self.lam * rate
        d_time = self.lam
        if self.cost_model == "time":
            return self._cutoff_for(self.c_fp, d_time)
        if self.cost_model == "byte":
            return self._cutoff_for(self.c_fp, d_byte)
        # byte-floored: min cutoff == max aggressiveness, but never above time's
        return min(
            self._cutoff_for(self.c_fp, d_byte),
            self._cutoff_for(self.c_fp, d_time),
        )

    def update(self, window_bytes: float, episode: Episode) -> float:
        if not self._fitted:
            raise RuntimeError("call calibrate() before scoring")
        raw = self.base.update(window_bytes, episode)
        p = self._cal.predict_proba(raw)
        # signed margin: >= 0 means the myopic rule says "alert now"
        return p - self._cutoff(window_bytes, episode)
