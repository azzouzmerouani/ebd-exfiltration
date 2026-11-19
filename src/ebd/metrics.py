"""EBD metrics: damage-denominated evaluation of exfiltration detectors.

The paper's argument is that classification metrics (F1, PR-AUC) rank detectors
by *whether* they eventually decide correctly, while for exfiltration the thing
that matters is *how much data left before the alert*. This module implements
both families so the two rankings can be compared directly (hypothesis H1).

Central quantity
----------------
**EBD — Exfiltrated Bytes before Detection.** For an attack episode and a
detector that alerts at some window, EBD is the receiver-side bytes already
landed by that window (:meth:`Episode.bytes_leaked_by`). Never firing costs the
full asset. Lower is better.

EBD alone is trivially gamed: a detector that alerts on window 0 of everything
scores EBD = 0. So EBD is always reported *paired* with a false-positive burden
(FPB) measured on benign episodes, and detectors are compared along the
resulting EBD-FPB trade-off curve — the damage-denominated analogue of an ROC.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from scipy.stats import kendalltau

from .episode import Episode


# --------------------------------------------------------------------------- #
# Alert model                                                                  #
# --------------------------------------------------------------------------- #

def first_alert_window(scores: np.ndarray, threshold: float) -> int | None:
    """Index of the first window whose score meets ``threshold``, else ``None``.

    A detector is treated as an online trigger: the alert is the *first* moment
    the streaming score crosses the operating threshold. Later windows are
    irrelevant — you cannot un-raise an alarm.
    """
    scores = np.asarray(scores, dtype=float)
    hits = np.nonzero(scores >= threshold)[0]
    return int(hits[0]) if hits.size else None


# --------------------------------------------------------------------------- #
# EBD                                                                          #
# --------------------------------------------------------------------------- #

def compute_ebd(episode: Episode, alert_window: int | None) -> float:
    """Exfiltrated Bytes before Detection for one attack episode.

    Thin wrapper over :meth:`Episode.bytes_leaked_by`, named for the metric so
    call sites read as the paper does.
    """
    return episode.bytes_leaked_by(alert_window)


def normalized_ebd(episode: Episode, alert_window: int | None) -> float:
    """EBD as a fraction of the asset in [0, 1] (``nEBD``).

    Comparable across episodes of different asset sizes; 0 = caught before any
    data moved, 1 = whole asset gone.
    """
    total = episode.total_bytes
    if total <= 0:
        return 0.0
    return compute_ebd(episode, alert_window) / total


@dataclass(frozen=True)
class EBDOperatingPoint:
    """One point on the EBD-FPB trade-off curve, at a fixed threshold."""

    threshold: float
    mean_ebd: float          # mean EBD over attack episodes (bytes)
    mean_nebd: float         # mean normalized EBD over attack episodes ([0,1])
    detection_rate: float    # fraction of attack episodes ever alerted on
    fpb: float               # false-positive burden on benign episodes
    fp_per_hour: float       # benign false alerts per benign-hour


def _score_episode(detector, episode: Episode) -> np.ndarray:
    """Run an online detector across an episode and collect per-window scores."""
    detector.reset()
    scores = np.empty(episode.n_windows, dtype=float)
    for k in range(episode.n_windows):
        scores[k] = detector.update(episode.window_bytes[k], episode)
    return scores


def ebd_fpb_curve(
    detector,
    attack_episodes: Sequence[Episode],
    benign_episodes: Sequence[Episode],
    thresholds: Sequence[float],
) -> list[EBDOperatingPoint]:
    """Sweep operating thresholds and trace the EBD-FPB curve.

    For each threshold:
      * on every attack episode, find the first alert window and its EBD;
      * on every benign episode, record whether *any* window alerted (FPB) and
        the number of benign alert-hours.

    ``fpb`` is the fraction of benign episodes with at least one alert.
    ``fp_per_hour`` normalises false alerts by benign observation time so
    detectors evaluated on different benign volumes stay comparable.
    """
    attack_scores = [_score_episode(detector, ep) for ep in attack_episodes]
    benign_scores = [_score_episode(detector, ep) for ep in benign_episodes]
    benign_hours = sum(ep.duration for ep in benign_episodes) / 3600.0

    points: list[EBDOperatingPoint] = []
    for thr in thresholds:
        ebds, nebds, detected = [], [], 0
        for ep, sc in zip(attack_episodes, attack_scores):
            aw = first_alert_window(sc, thr)
            ebds.append(compute_ebd(ep, aw))
            nebds.append(normalized_ebd(ep, aw))
            detected += aw is not None

        n_benign_fp = 0
        benign_fp_events = 0
        for ep, sc in zip(benign_episodes, benign_scores):
            aw = first_alert_window(sc, thr)
            if aw is not None:
                n_benign_fp += 1
                benign_fp_events += int(np.count_nonzero(sc >= thr))

        n_attack = max(len(attack_episodes), 1)
        n_benign = max(len(benign_episodes), 1)
        points.append(
            EBDOperatingPoint(
                threshold=float(thr),
                mean_ebd=float(np.mean(ebds)) if ebds else 0.0,
                mean_nebd=float(np.mean(nebds)) if nebds else 0.0,
                detection_rate=detected / n_attack,
                fpb=n_benign_fp / n_benign,
                fp_per_hour=(benign_fp_events / benign_hours) if benign_hours else 0.0,
            )
        )
    return points


def ebd_at_fpb(points: Sequence[EBDOperatingPoint], max_fpb: float) -> EBDOperatingPoint:
    """Best (lowest mean EBD) operating point whose FPB stays within budget.

    This is the single-number summary used to rank detectors: fix an acceptable
    false-positive burden, then ask how little data each detector lets leak. If
    no operating point satisfies the budget, the least-bad one is returned.
    """
    feasible = [p for p in points if p.fpb <= max_fpb]
    pool = feasible or list(points)
    return min(pool, key=lambda p: p.mean_ebd)


# --------------------------------------------------------------------------- #
# Classification metrics (the incumbent ranking)                               #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ClassificationScore:
    threshold: float
    precision: float
    recall: float
    f1: float


def episode_classification(
    detector,
    attack_episodes: Sequence[Episode],
    benign_episodes: Sequence[Episode],
    thresholds: Sequence[float],
) -> ClassificationScore:
    """Best-F1 episode-level classification score across thresholds.

    An episode counts as *predicted attack* if any window alerts. This is the
    conventional framing the exfiltration literature reports — deliberately
    blind to *when* within the episode the alert fired.
    """
    attack_scores = [_score_episode(detector, ep) for ep in attack_episodes]
    benign_scores = [_score_episode(detector, ep) for ep in benign_episodes]

    best = ClassificationScore(threshold=float("nan"), precision=0, recall=0, f1=-1.0)
    for thr in thresholds:
        tp = sum(first_alert_window(sc, thr) is not None for sc in attack_scores)
        fn = len(attack_scores) - tp
        fp = sum(first_alert_window(sc, thr) is not None for sc in benign_scores)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        if f1 > best.f1:
            best = ClassificationScore(float(thr), precision, recall, f1)
    return best


# --------------------------------------------------------------------------- #
# H1: do the two rankings disagree?                                            #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RankingComparison:
    detectors: list[str]
    f1_rank: dict[str, int]     # 1 = best by F1
    ebd_rank: dict[str, int]    # 1 = best (lowest) mean EBD
    kendall_tau: float
    p_value: float
    diverges: bool              # True if tau < threshold (H1 supported)

    def as_table(self) -> str:
        rows = ["detector            F1-rank   EBD-rank",
                "-------------------------------------"]
        for d in self.detectors:
            rows.append(f"{d:<18}  {self.f1_rank[d]:>6}   {self.ebd_rank[d]:>7}")
        rows.append("-------------------------------------")
        verdict = ("tau below 0.5: divergent by the H1 criterion" if self.diverges
                   else "tau at or above 0.5: agree by the H1 criterion")
        rows.append(f"Kendall tau = {self.kendall_tau:+.3f}  p = {self.p_value:.3f}  -> {verdict}")
        return "\n".join(rows)


def _rank(values: dict[str, float], ascending: bool) -> dict[str, int]:
    """Dense 1-based ranking; ``ascending`` True means smaller value ranks 1."""
    ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=not ascending)
    return {name: i + 1 for i, (name, _) in enumerate(ordered)}


def compare_rankings(
    f1_by_detector: dict[str, float],
    ebd_by_detector: dict[str, float],
    tau_threshold: float = 0.5,
) -> RankingComparison:
    """Compare the F1 ranking against the EBD ranking (pre-registered H1).

    H1 (registered before running): if Kendall's tau between the two rankings is
    **below ``tau_threshold`` (0.5)** the rankings diverge — F1 winners are not
    EBD winners, so the field has been optimising the wrong target. tau at or
    above 0.5 falsifies H1.

    Higher F1 is better; *lower* EBD is better, so EBD is ranked ascending. tau
    is computed on the raw values (higher-F1, and negated-EBD so both point the
    same way) to keep the sign interpretable: tau ~ +1 means the orderings agree.
    """
    detectors = sorted(f1_by_detector)
    if set(detectors) != set(ebd_by_detector):
        raise ValueError("f1 and ebd dicts must cover the same detectors")

    f1_rank = _rank(f1_by_detector, ascending=False)  # higher F1 -> rank 1
    ebd_rank = _rank(ebd_by_detector, ascending=True)  # lower EBD -> rank 1

    f1_vals = [f1_by_detector[d] for d in detectors]
    # negate EBD so "better" is larger on both axes -> tau>0 == agreement
    ebd_vals = [-ebd_by_detector[d] for d in detectors]
    tau, p = kendalltau(f1_vals, ebd_vals)
    tau = float(tau) if tau == tau else 0.0  # nan -> 0 (all-tied edge case)
    p = float(p) if p == p else 1.0

    return RankingComparison(
        detectors=detectors,
        f1_rank=f1_rank,
        ebd_rank=ebd_rank,
        kendall_tau=tau,
        p_value=p,
        diverges=tau < tau_threshold,
    )
