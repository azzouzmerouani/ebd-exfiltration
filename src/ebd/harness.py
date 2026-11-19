"""Benchmark orchestration: run every detector, rank two ways, test H1.

This ties the pieces together. For each detector it:

1. trains it on the *train* split — CUSUM learns a benign slack, the ECTS
   triggers calibrate their score->probability map and rate scale, plain
   baselines need nothing;
2. traces the detector's EBD-FPB trade-off on the *test* split by sweeping a
   per-detector threshold grid (scores live on different scales, so the grid is
   derived from each detector's own score distribution);
3. reads off two summaries at a shared false-positive budget — the incumbent
   **best-F1** and the damage-denominated **mean EBD**;
4. compares the F1 ranking against the EBD ranking (pre-registered **H1**).

Nothing here is exfiltration tooling; it is pure evaluation over in-memory
episode arrays and is safe to run anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .episode import Episode
from .metrics import (
    RankingComparison,
    _score_episode,
    compare_rankings,
    ebd_at_fpb,
    ebd_fpb_curve,
    episode_classification,
)


# --------------------------------------------------------------------------- #
# Threshold grids                                                             #
# --------------------------------------------------------------------------- #

def auto_thresholds(
    detector,
    episodes: Sequence[Episode],
    n: int = 160,
) -> np.ndarray:
    """A per-detector threshold grid spanning that detector's score range.

    Scores are pooled across every window of every episode; the grid is the set
    of quantiles of those scores, plus a sentinel just above the maximum so the
    "never fires" operating point (full-asset EBD, zero FPB) is always on the
    curve. Quantile spacing keeps points where the scores actually are, which a
    linear grid would miss for the heavy-tailed volume detectors.
    """
    pooled = np.concatenate(
        [_score_episode(detector, ep) for ep in episodes if ep.n_windows > 0]
    )
    if pooled.size == 0:
        return np.array([0.0, 1.0])
    qs = np.linspace(0.0, 1.0, n)
    grid = np.unique(np.quantile(pooled, qs))
    hi = float(pooled.max())
    span = hi - float(pooled.min())
    sentinel = hi + (span if span > 0 else abs(hi) + 1.0) * 0.01 + 1e-9
    return np.append(grid, sentinel)


# --------------------------------------------------------------------------- #
# Training dispatch                                                            #
# --------------------------------------------------------------------------- #

def train_detector(
    detector,
    train_attack: Sequence[Episode],
    train_benign: Sequence[Episode],
) -> None:
    """Train a detector using whichever protocol it exposes.

    The base contract has no ``fit``; different detector families train
    differently, so we dispatch by capability rather than force one signature:
    ECTS triggers expose ``calibrate(attack, benign)`` and may wrap a trainable
    base (which must be fitted *first*), CUSUM and the trained classifier expose
    ``fit(benign, attack)``, plain baselines expose neither and are used as-is.
    """
    if hasattr(detector, "calibrate"):
        base = getattr(detector, "base", None)
        if base is not None and hasattr(base, "fit"):
            base.fit(train_benign, train_attack)
        detector.calibrate(train_attack, train_benign)
    elif hasattr(detector, "fit"):
        detector.fit(train_benign, train_attack)


# --------------------------------------------------------------------------- #
# Per-detector evaluation                                                      #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DetectorResult:
    name: str
    best_f1: float
    f1_precision: float
    f1_recall: float
    mean_ebd: float          # bytes (MB) at the FPB budget
    mean_nebd: float         # fraction of asset in [0, 1] at the FPB budget
    detection_rate: float
    fpb: float
    asset_mb: float

    @property
    def asset_saved(self) -> float:
        """Fraction of the asset saved on average: ``1 - nEBD``."""
        return max(0.0, 1.0 - self.mean_nebd)


def evaluate_detector(
    detector,
    test_attack: Sequence[Episode],
    test_benign: Sequence[Episode],
    max_fpb: float,
    asset_mb: float,
) -> DetectorResult:
    thresholds = auto_thresholds(detector, list(test_attack) + list(test_benign))
    curve = ebd_fpb_curve(detector, test_attack, test_benign, thresholds)
    op = ebd_at_fpb(curve, max_fpb)
    clf = episode_classification(detector, test_attack, test_benign, thresholds)
    return DetectorResult(
        name=detector.name,
        best_f1=clf.f1,
        f1_precision=clf.precision,
        f1_recall=clf.recall,
        mean_ebd=op.mean_ebd,
        mean_nebd=op.mean_nebd,
        detection_rate=op.detection_rate,
        fpb=op.fpb,
        asset_mb=asset_mb,
    )


# --------------------------------------------------------------------------- #
# Full benchmark                                                               #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BenchmarkReport:
    results: list[DetectorResult]
    ranking: RankingComparison
    max_fpb: float

    def to_frame(self):
        """Results as a pandas DataFrame, sorted by EBD (best first)."""
        import pandas as pd

        rows = [
            {
                "detector": r.name,
                "best_F1": round(r.best_f1, 4),
                "mean_EBD_MB": round(r.mean_ebd, 2),
                "nEBD": round(r.mean_nebd, 4),
                "asset_saved": round(r.asset_saved, 4),
                "detect_rate": round(r.detection_rate, 4),
                "FPB": round(r.fpb, 4),
                "F1_rank": self.ranking.f1_rank[r.name],
                "EBD_rank": self.ranking.ebd_rank[r.name],
            }
            for r in self.results
        ]
        df = pd.DataFrame(rows).sort_values("mean_EBD_MB").reset_index(drop=True)
        return df


def run_benchmark(
    detectors: Sequence,
    train_attack: Sequence[Episode],
    train_benign: Sequence[Episode],
    test_attack: Sequence[Episode],
    test_benign: Sequence[Episode],
    max_fpb: float = 0.05,
    asset_mb: float = 300.0,
) -> BenchmarkReport:
    """Train, evaluate and rank every detector; return the full report."""
    results: list[DetectorResult] = []
    for det in detectors:
        train_detector(det, train_attack, train_benign)
        results.append(
            evaluate_detector(det, test_attack, test_benign, max_fpb, asset_mb)
        )

    f1_by = {r.name: r.best_f1 for r in results}
    ebd_by = {r.name: r.mean_ebd for r in results}
    ranking = compare_rankings(f1_by, ebd_by)
    return BenchmarkReport(results=results, ranking=ranking, max_fpb=max_fpb)
