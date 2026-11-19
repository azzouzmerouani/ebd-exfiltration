"""Synthetic episode generator for the pilot benchmark.

No lab captures exist yet (failure points C3/C4: an isolated network is a
prerequisite, not a given). Until they do, the harness is exercised on
*synthetic* receiver-side byte timelines whose shape is controlled by the same
adversary matrix that will drive the real rclone runs.

What this is and is not
-----------------------
This generator is a **methodology test**, not evidence. It lets us prove the
end-to-end pipeline runs — episodes -> detectors -> EBD/FPB/F1 -> H1/D5 -> and
that the metrics behave as designed. It says **nothing** about how real detectors
rank on real traffic. Every number produced from synthetic data must carry that
caveat; the empirical claim in the paper stands or falls on lab captures.

The one honest signal
---------------------
In a byte-only timeline the sole realistic discriminator between exfiltration and
a benign bulk transfer (a backup, an upload) is **persistence**: exfiltration is a
*sustained* session, benign bulk is a *short* burst. So attacks are generated as
sustained sessions (``dur_min..dur_max`` windows) and benign bulk as short bursts
(``<= benign_bulk_dur_max`` windows). A detector that learns persistence can
separate the classes; one that looks only at instantaneous rate or single-window
volume cannot — which is exactly the regime that makes the byte-vs-time control
(D5) meaningful. The ``rate`` axis still sets per-window throughput, so it remains
the axis along which the byte- and time-denominated triggers separate.

Units are megabytes-as-float with ``dt = 1 s`` throughout, consistent with
:class:`ebd.episode.Episode`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .episode import Episode
from .scenarios import ScenarioSpec, build_matrix


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DatasetConfig:
    """Knobs for the synthetic dataset. Defaults give a few hundred episodes."""

    dur_min: int = 20                  # attacks are sustained sessions: >= dur_min windows
    dur_max: int = 200                 # ...up to dur_max windows
    dt: float = 1.0                    # window width, seconds
    replicates_per_cell: int = 4       # attack episodes per adversary-matrix cell
    n_benign_light: int = 160          # light benign flows (easy negatives)
    n_benign_bulk: int = 60            # bulk benign flows (hard negatives)
    benign_bulk_dur_max: int = 10      # benign bulk is a SHORT burst (the whole point)
    benign_bulk_rate_lo: float = 4.0   # benign bulk per-window rate range (MB)
    benign_bulk_rate_hi: float = 24.0
    seed: int = 20260807


@dataclass(frozen=True)
class Dataset:
    """A generated dataset with a reproducible train/test split.

    The split is by episode. ECTS calibration and base-detector training use only
    the *train* half; every reported metric uses only the *test* half.
    """

    attack: list[Episode]
    benign: list[Episode]
    config: DatasetConfig
    _split_frac: float = field(default=0.5)

    def split(self) -> tuple[list[Episode], list[Episode], list[Episode], list[Episode]]:
        """Return ``(train_attack, train_benign, test_attack, test_benign)``."""
        rng = np.random.default_rng(self.config.seed + 999)
        train_attack, test_attack = _halve(self.attack, self._split_frac, rng)
        train_benign, test_benign = _halve(self.benign, self._split_frac, rng)
        return train_attack, train_benign, test_attack, test_benign

    @property
    def median_asset_mb(self) -> float:
        """Median attack total, for display only (assets are emergent here)."""
        return float(np.median([e.total_bytes for e in self.attack])) if self.attack else 0.0


def _halve(items: list[Episode], frac: float, rng: np.random.Generator):
    idx = rng.permutation(len(items))
    cut = int(round(len(items) * frac))
    train = [items[i] for i in idx[:cut]]
    test = [items[i] for i in idx[cut:]]
    return train, test


# --------------------------------------------------------------------------- #
# Attack shaping (duration-first: sustained sessions)                          #
# --------------------------------------------------------------------------- #

def _normalize_mean(prof: np.ndarray, mean_rate: float) -> np.ndarray:
    """Scale a non-negative profile so its mean equals ``mean_rate`` (MB/window).

    Keeps the per-window rate on the scenario's axis while preserving the shape
    imposed by the chunking strategy.
    """
    prof = np.clip(prof, 0.0, None)
    m = prof.mean()
    if m <= 0:
        return np.full(len(prof), mean_rate)
    return prof * (mean_rate / m)


def _attack_profile(
    rng: np.random.Generator, mb_per_s: float, chunking: str, duration: int
) -> np.ndarray:
    """Per-window exfil bytes (MB) for one sustained attack of ``duration`` windows.

    The mean per-window rate tracks ``mb_per_s`` (the rate axis); ``chunking``
    reshapes *how* the bytes are spread within the session:

    * ``single``      — one steady stream, mild multiplicative jitter.
    * ``chunked``     — parallel chunks: bursts above the nominal rate separated
      by short idle gaps (a duty cycle).
    * ``many-small``  — many small files: heavy-tailed per-window sizes with
      frequent idle gaps between files.
    """
    n = int(max(3, duration))
    if chunking == "single":
        prof = rng.lognormal(mean=0.0, sigma=0.15, size=n)
    elif chunking == "chunked":
        base = rng.lognormal(mean=0.0, sigma=0.2, size=n)
        duty = (np.arange(n) % 4) != 3          # 3 windows on, 1 off
        prof = np.where(duty, base * (4.0 / 3.0), 0.0)
    elif chunking == "many-small":
        sizes = rng.pareto(a=2.5, size=n) + 0.05
        gaps = rng.random(n) < 0.25             # ~1 in 4 windows idle
        prof = np.where(gaps, 0.0, sizes)
    else:  # pragma: no cover - guarded by matrix construction
        raise ValueError(f"unknown chunking {chunking!r}")
    return _normalize_mean(prof, mb_per_s)


def generate_attack_episode(
    rng: np.random.Generator, spec: ScenarioSpec, cfg: DatasetConfig, rep: int
) -> Episode:
    duration = int(rng.integers(cfg.dur_min, cfg.dur_max + 1))
    prof = _attack_profile(rng, spec.mb_per_s, spec.cell.chunking, duration)
    total = float(prof.sum())
    return Episode(
        episode_id=f"atk/{spec.cell.key}/r{rep}",
        label=True,
        window_bytes=prof,
        dt=cfg.dt,
        scenario=spec.cell,
        sender_total=total,   # synthetic: receiver == sender, reconciliation is exact
        meta={"mb_per_s": spec.mb_per_s, "duration": duration, "asset_mb": total},
    )


# --------------------------------------------------------------------------- #
# Benign shaping                                                              #
# --------------------------------------------------------------------------- #

def generate_benign_light(rng: np.random.Generator, cfg: DatasetConfig, i: int) -> Episode:
    n = int(rng.integers(8, 45))
    prof = rng.gamma(shape=1.5, scale=0.05, size=n)     # small MB/window
    prof = np.clip(prof, 0.0, None)
    return Episode(
        episode_id=f"ben-light/{i}",
        label=False,
        window_bytes=prof,
        dt=cfg.dt,
        sender_total=float(prof.sum()),
        meta={"kind": "light"},
    )


def generate_benign_bulk(rng: np.random.Generator, cfg: DatasetConfig, i: int) -> Episode:
    """A legitimate bulk transfer — the hard negative.

    High per-window rate, overlapping the fast attacks, so a rate or single-window
    volume threshold cannot separate it from an attack without paying in false
    positives. The one thing it is *not* is sustained: it is a **short burst**
    (``<= benign_bulk_dur_max`` windows). Persistence is therefore the only clean
    separator — which is precisely the signal the trained base detector must learn
    for the byte-vs-time control to be meaningful.
    """
    n = int(rng.integers(2, cfg.benign_bulk_dur_max + 1))
    rate = float(rng.uniform(cfg.benign_bulk_rate_lo, cfg.benign_bulk_rate_hi))
    prof = _normalize_mean(rate * rng.lognormal(mean=0.0, sigma=0.25, size=n), rate)
    return Episode(
        episode_id=f"ben-bulk/{i}",
        label=False,
        window_bytes=prof,
        dt=cfg.dt,
        sender_total=float(prof.sum()),
        meta={"kind": "bulk", "rate": rate, "duration": n},
    )


# --------------------------------------------------------------------------- #
# Assembly                                                                    #
# --------------------------------------------------------------------------- #

def make_dataset(cfg: DatasetConfig | None = None) -> Dataset:
    """Generate the full synthetic dataset (deterministic given ``cfg.seed``)."""
    cfg = cfg or DatasetConfig()
    rng = np.random.default_rng(cfg.seed)

    attack: list[Episode] = []
    for spec in build_matrix():
        for rep in range(cfg.replicates_per_cell):
            ep = generate_attack_episode(rng, spec, cfg, rep)
            ep.reconcile()   # reject anything that does not conserve bytes
            attack.append(ep)

    benign: list[Episode] = []
    for i in range(cfg.n_benign_light):
        benign.append(generate_benign_light(rng, cfg, i))
    for i in range(cfg.n_benign_bulk):
        benign.append(generate_benign_bulk(rng, cfg, i))
    for ep in benign:
        ep.reconcile()

    return Dataset(attack=attack, benign=benign, config=cfg)
