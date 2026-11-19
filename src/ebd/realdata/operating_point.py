"""The operating-point rule used for every detector in the paper.

EBD is degenerate without a stated false-positive burden (Sec. 3.2), so every
detector is run at a threshold fixed *a priori* on training benign data:

* a threshold is **admissible** when the upper 95% Wilson bound of the burden it
  produces on the *training* benign units lies within the budget;
* the most aggressive admissible threshold on the grid is used;
* the burden it actually produces is then **measured** on held-out benign units
  and reported with its own Wilson interval, whatever it turns out to be.

The rule never looks at held-out data, and it derives its headroom from the
number of benign units actually observed rather than from a guessed constant.
It replaces the point-estimate rule of earlier drafts, which put the operating
point exactly on the budget boundary where ordinary sampling variation carried
it over (Sec. 9.2). ``rule="point"`` keeps that older rule available so the
archived numbers can be reproduced and compared.

A *unit* is whatever the detector's false positives are counted over: a benign
session for the flow-record detectors of Sec. 7, a benign session *record* for
the session-granularity detector of Sec. 7.9. A unit alerts when its loudest
score clears the threshold.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

Z95 = 1.959963984540054

#: The threshold grid used throughout Sec. 7 and Sec. 9 (0.05, 0.10, ..., 0.95).
DEFAULT_GRID = np.unique(np.round(np.linspace(0.05, 0.95, 19), 3))


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Two-sided Wilson score interval for ``k`` successes in ``n`` trials."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson_upper(k: int, n: int, z: float = Z95) -> float:
    return wilson_interval(k, n, z)[1]


def unit_max_scores(unit_scores: Iterable[np.ndarray]) -> np.ndarray:
    """Loudest score per unit; an empty unit can never alert."""
    return np.array([float(np.max(s)) if len(s) else -np.inf for s in unit_scores],
                    dtype=float)


def choose_threshold(train_unit_max: Sequence[float], budget: float,
                     rule: str = "wilson",
                     grid: Sequence[float] = DEFAULT_GRID) -> dict:
    """The most aggressive admissible threshold on ``grid`` (ascending).

    Returns the threshold together with the training burden it produces and the
    upper Wilson bound of that burden. ``admissible`` is False only when no grid
    value meets the rule, in which case the top of the grid is returned and the
    caller must report the operating point as failing its budget.
    """
    if rule not in ("wilson", "point"):
        raise ValueError(f"unknown operating-point rule {rule!r}")
    m = np.asarray(train_unit_max, dtype=float)
    n = int(len(m))
    for t in sorted(float(x) for x in grid):
        k = int((m >= t).sum())
        p = k / n if n else 0.0
        upper = wilson_upper(k, n)
        ok = upper <= budget if rule == "wilson" else p <= budget
        if ok:
            return dict(threshold=t, rule=rule, budget=budget, admissible=True,
                        train_k=k, train_n=n, train_fpb=p, train_upper=upper,
                        grid_bottom_binds=bool(t <= min(float(x) for x in grid)))
    t = float(max(grid))
    k = int((m >= t).sum())
    return dict(threshold=t, rule=rule, budget=budget, admissible=False,
                train_k=k, train_n=n, train_fpb=k / n if n else 0.0,
                train_upper=wilson_upper(k, n), grid_bottom_binds=False)


def extended_grid(train_unit_max: Sequence[float],
                  base: Sequence[float] = DEFAULT_GRID) -> np.ndarray:
    """``base`` extended below its minimum, so the grid floor does not bind.

    Adds 20 log-spaced values from 1e-4 to min(base) and every distinct finite
    training score (thresholding at a score value is the finest step the rule
    can distinguish). If ``grid_bottom_binds`` is still True on this grid, the
    detector meets the budget at any threshold and its operating point is not
    determined by the rule; report it as such.
    """
    m = np.asarray(train_unit_max, dtype=float)
    lo = float(min(base))
    extra = np.geomspace(1e-4, lo, 20)
    scores = np.unique(m[np.isfinite(m)])
    scores = scores[(scores > 0) & (scores < lo)]
    return np.unique(np.round(np.concatenate([np.asarray(base, float), extra, scores]), 8))


def measure_burden(heldout_unit_max: Sequence[float], threshold: float) -> dict:
    """Burden actually produced on held-out units, with its Wilson interval."""
    m = np.asarray(heldout_unit_max, dtype=float)
    n = int(len(m))
    k = int((m >= threshold).sum())
    lo, hi = wilson_interval(k, n)
    return dict(heldout_k=k, heldout_n=n, heldout_fpb=k / n if n else 0.0,
                heldout_lo=lo, heldout_hi=hi)


def quantile_grid(train_unit_max: Sequence[float],
                  base: Sequence[float] = DEFAULT_GRID) -> np.ndarray:
    """``base`` plus every distinct finite training score in (0, 1].

    Thresholding at a score value is the finest step the rule can resolve, so
    this grid removes the coarseness of the 0.05 steps anywhere in the range
    (e.g. between 0.35 and 0.40), not only below the grid floor.
    """
    m = np.asarray(train_unit_max, dtype=float)
    scores = np.unique(m[np.isfinite(m)])
    scores = scores[(scores > 0) & (scores <= 1)]
    return np.unique(np.round(np.concatenate([np.asarray(base, float), scores,
                                              np.geomspace(1e-4, float(min(base)), 20)]), 8))


def training_negatives(rows: list[dict], policy: str) -> list[dict]:
    """Rows kept as training data under a negatives policy.

    ``all`` (the original behaviour) keeps every non-exfiltration row, so
    attack flows of other stages train as negatives; on DAPT2021 these include
    gateway-vantage records of the scored exfiltration connections.
    ``benign`` drops every attack row of another stage, so the only negatives
    are benign flows.
    """
    if policy == "all":
        return rows
    if policy != "benign":
        raise ValueError(policy)
    keep = ("Benign", "Normal", "", "Data Exfiltration")
    return [r for r in rows if r.get("Stage", "").strip() in keep]
