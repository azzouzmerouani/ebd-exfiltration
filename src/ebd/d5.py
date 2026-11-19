"""Experiment D5 — the byte-vs-time control, isolated.

D5 is the experiment the paper lives or dies on. Both ECTS triggers share one
base scorer and one calibration; they differ *only* in the delay-cost
denomination. If the byte-denominated trigger does not save bytes over the
time-denominated one, the contribution collapses to "we applied ECTS to
exfiltration". So it is run in isolation, cleanly, and early.

Protocol
--------
1. Fit and calibrate both triggers on the **train** split (shared base scorer).
2. Match each trigger to the **same false-positive budget** on *train* benign —
   EBD is meaningless unless the alarms cost the same. Selecting the operating
   point on train, not test, keeps the comparison honest.
3. Measure EBD **per episode, paired** on *test* attacks (identical episodes for
   both), and the achieved FPB on *test* benign.
4. Break EBD down **by rate** — the axis where byte-denomination is expected to
   act — and run a **paired bootstrap** on the per-episode difference.

The verdict is pre-registered: byte-cost wins if the mean paired saving (time EBD
minus byte EBD) is positive *and* its bootstrap CI excludes zero. Reporting the
per-rate split matters as much as the aggregate: the expected shape is a saving
concentrated on fast transfers and a wash (or small loss) on drips, and that
shape is itself the finding, whatever the aggregate sign.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .episode import Episode
from .harness import auto_thresholds
from .metrics import _score_episode, compute_ebd, first_alert_window, normalized_ebd

RATE_ORDER = ("burst", "fast", "throttled", "drip")


# --------------------------------------------------------------------------- #
# Operating-point matching                                                    #
# --------------------------------------------------------------------------- #

def benign_fpb(detector, benign: Sequence[Episode], threshold: float) -> float:
    """Fraction of benign episodes with at least one alert at ``threshold``."""
    scores = [_score_episode(detector, ep) for ep in benign]
    hits = sum(first_alert_window(sc, threshold) is not None for sc in scores)
    return hits / max(len(benign), 1)


def match_fpb_threshold(
    detector, benign: Sequence[Episode], target_fpb: float, grid: np.ndarray
) -> float:
    """Smallest (most aggressive) threshold whose benign FPB stays within budget.

    FPB is non-increasing in the threshold, so the first grid point that meets the
    budget is the most aggressive feasible operating point — the one that lets the
    trigger fire as early as the false-positive budget allows.
    """
    scores = [_score_episode(detector, ep) for ep in benign]
    for thr in np.sort(grid):
        fpb = sum(first_alert_window(sc, thr) is not None for sc in scores) / max(len(scores), 1)
        if fpb <= target_fpb:
            return float(thr)
    return float(np.max(grid))


# --------------------------------------------------------------------------- #
# Per-episode EBD at a fixed operating point                                  #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EpisodeEBD:
    rate: str
    ebd_mb: float
    nebd: float
    fired: bool


def episode_ebds(detector, attacks: Sequence[Episode], threshold: float) -> list[EpisodeEBD]:
    out: list[EpisodeEBD] = []
    for ep in attacks:
        sc = _score_episode(detector, ep)
        aw = first_alert_window(sc, threshold)
        rate = ep.scenario.rate if ep.scenario else "?"
        out.append(EpisodeEBD(rate, compute_ebd(ep, aw), normalized_ebd(ep, aw), aw is not None))
    return out


# --------------------------------------------------------------------------- #
# The comparison                                                              #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RateCell:
    rate: str
    n: int
    ebd_mb_byte: float
    ebd_mb_time: float
    nebd_byte: float
    nebd_time: float
    det_byte: float
    det_time: float

    @property
    def saving_mb(self) -> float:
        return self.ebd_mb_time - self.ebd_mb_byte


@dataclass(frozen=True)
class D5Result:
    base_name: str
    target_fpb: float
    fpb_byte: float
    fpb_time: float
    det_byte: float
    det_time: float
    mean_ebd_mb_byte: float
    mean_ebd_mb_time: float
    paired_nebd_saving: float     # mean(time nEBD - byte nEBD); >0 favours byte
    ci_lo: float
    ci_hi: float
    frac_byte_better: float
    per_rate: list[RateCell]

    @property
    def supported(self) -> bool:
        """Pre-registered verdict: positive saving with a CI clear of zero."""
        return self.paired_nebd_saving > 0 and self.ci_lo > 0

    def report(self) -> str:
        lines = [
            f"D5  byte-vs-time control   (base = {self.base_name})",
            "=" * 60,
            f"FPB budget           : {self.target_fpb:.0%}"
            f"   (achieved: byte {self.fpb_byte:.1%}, time {self.fpb_time:.1%})",
            f"detection rate       : byte {self.det_byte:.1%}   time {self.det_time:.1%}",
            f"mean EBD (MB)        : byte {self.mean_ebd_mb_byte:7.2f}   "
            f"time {self.mean_ebd_mb_time:7.2f}",
            "",
            "per-rate mean EBD (MB)   [saving = time - byte, >0 favours byte]",
            f"  {'rate':<10}{'n':>4}{'byte':>10}{'time':>10}{'saving':>10}{'det%':>8}",
        ]
        for c in self.per_rate:
            lines.append(
                f"  {c.rate:<10}{c.n:>4}{c.ebd_mb_byte:>10.2f}{c.ebd_mb_time:>10.2f}"
                f"{c.saving_mb:>+10.2f}{c.det_byte*100:>7.0f}%"
            )
        verdict = "SUPPORTED" if self.supported else "NOT supported"
        lines += [
            "",
            f"paired nEBD saving   : {self.paired_nebd_saving:+.4f}"
            f"   95% CI [{self.ci_lo:+.4f}, {self.ci_hi:+.4f}]",
            f"episodes byte<=time  : {self.frac_byte_better:.0%}",
            f"pre-registered D5    : {verdict}  "
            f"(byte {'beats' if self.supported else 'does not beat'} time on this run)",
        ]
        return "\n".join(lines)


def _bootstrap_ci(diffs: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    if diffs.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, diffs.size, diffs.size)
        means[b] = diffs[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def d5_compare(
    byte_trigger,
    time_trigger,
    train_attack: Sequence[Episode],
    train_benign: Sequence[Episode],
    test_attack: Sequence[Episode],
    test_benign: Sequence[Episode],
    target_fpb: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
) -> D5Result:
    """Run the full D5 comparison. Both triggers must share a pre-fitted base."""
    byte_trigger.calibrate(train_attack, train_benign)
    time_trigger.calibrate(train_attack, train_benign)

    pool = list(train_benign) + list(test_attack)
    t_byte = match_fpb_threshold(byte_trigger, train_benign, target_fpb,
                                 auto_thresholds(byte_trigger, pool))
    t_time = match_fpb_threshold(time_trigger, train_benign, target_fpb,
                                 auto_thresholds(time_trigger, pool))

    byte_recs = episode_ebds(byte_trigger, test_attack, t_byte)
    time_recs = episode_ebds(time_trigger, test_attack, t_time)

    nebd_byte = np.array([r.nebd for r in byte_recs])
    nebd_time = np.array([r.nebd for r in time_recs])
    diffs = nebd_time - nebd_byte           # >0 means byte leaked less
    ci_lo, ci_hi = _bootstrap_ci(diffs, n_boot, seed)

    per_rate: list[RateCell] = []
    for rate in RATE_ORDER:
        bi = [r for r in byte_recs if r.rate == rate]
        ti = [r for r in time_recs if r.rate == rate]
        if not bi:
            continue
        per_rate.append(
            RateCell(
                rate=rate,
                n=len(bi),
                ebd_mb_byte=float(np.mean([r.ebd_mb for r in bi])),
                ebd_mb_time=float(np.mean([r.ebd_mb for r in ti])),
                nebd_byte=float(np.mean([r.nebd for r in bi])),
                nebd_time=float(np.mean([r.nebd for r in ti])),
                det_byte=float(np.mean([r.fired for r in bi])),
                det_time=float(np.mean([r.fired for r in ti])),
            )
        )

    base_name = getattr(getattr(byte_trigger, "base", None), "name", "?")
    return D5Result(
        base_name=base_name,
        target_fpb=target_fpb,
        fpb_byte=benign_fpb(byte_trigger, test_benign, t_byte),
        fpb_time=benign_fpb(time_trigger, test_benign, t_time),
        det_byte=float(np.mean([r.fired for r in byte_recs])) if byte_recs else 0.0,
        det_time=float(np.mean([r.fired for r in time_recs])) if time_recs else 0.0,
        mean_ebd_mb_byte=float(np.mean([r.ebd_mb for r in byte_recs])) if byte_recs else 0.0,
        mean_ebd_mb_time=float(np.mean([r.ebd_mb for r in time_recs])) if time_recs else 0.0,
        paired_nebd_saving=float(diffs.mean()) if diffs.size else 0.0,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        frac_byte_better=float(np.mean(diffs >= 0)) if diffs.size else 0.0,
        per_rate=per_rate,
    )
