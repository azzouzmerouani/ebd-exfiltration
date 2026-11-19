"""Unit tests for the synthetic episode generator and reconciliation."""

from __future__ import annotations

import numpy as np
import pytest

from ebd.episode import Episode, ReconciliationError
from ebd.scenarios import build_matrix, matrix_size
from ebd.synth import DatasetConfig, make_dataset


def test_matrix_is_36_cells():
    assert matrix_size() == 36
    assert len(build_matrix()) == 36


def test_dataset_is_deterministic_and_reconciles():
    cfg = DatasetConfig(replicates_per_cell=1, n_benign_light=10, n_benign_bulk=5)
    a = make_dataset(cfg)
    b = make_dataset(cfg)
    # deterministic given the seed
    assert [e.episode_id for e in a.attack] == [e.episode_id for e in b.attack]
    assert np.allclose(a.attack[0].window_bytes, b.attack[0].window_bytes)
    # every generated episode conserves bytes (else make_dataset would have raised)
    for ep in a.attack + a.benign:
        ep.reconcile()


def test_attacks_are_sustained_and_reconcile():
    cfg = DatasetConfig(replicates_per_cell=2)
    ds = make_dataset(cfg)
    for ep in ds.attack:
        # sustained sessions: duration lives in the configured band
        assert cfg.dur_min <= ep.n_windows <= cfg.dur_max
        assert ep.total_bytes == pytest.approx(ep.sender_total, rel=1e-9)
    totals = np.array([ep.total_bytes for ep in ds.attack])
    assert totals.std() > 1.0  # emergent assets vary


def test_benign_bulk_is_a_short_burst():
    cfg = DatasetConfig(replicates_per_cell=1)
    ds = make_dataset(cfg)
    bulk = [e for e in ds.benign if e.meta.get("kind") == "bulk"]
    assert bulk
    # the whole point: bulk is short, attacks are sustained -> persistence separates
    assert max(e.n_windows for e in bulk) <= cfg.benign_bulk_dur_max
    assert np.median([e.n_windows for e in ds.attack]) > cfg.benign_bulk_dur_max


def test_rate_axis_orders_per_window_throughput():
    cfg = DatasetConfig(replicates_per_cell=1)
    ds = make_dataset(cfg)

    def mean_rate(level):
        eps = [e for e in ds.attack if e.scenario.rate == level]
        return np.mean([e.window_bytes.mean() / e.dt for e in eps])

    # burst > fast > throttled > drip in per-window rate
    assert mean_rate("burst") > mean_rate("fast") > mean_rate("throttled") > mean_rate("drip")


def test_labels_are_set():
    ds = make_dataset(DatasetConfig(replicates_per_cell=1, n_benign_light=5, n_benign_bulk=5))
    assert all(e.label for e in ds.attack)
    assert all(not e.label for e in ds.benign)


def test_reconcile_rejects_divergent_capture():
    ep = Episode(
        episode_id="bad",
        label=True,
        window_bytes=np.array([10.0, 10.0]),
        sender_total=100.0,  # sender claims 100, receiver saw 20 -> 80% divergence
    )
    with pytest.raises(ReconciliationError):
        ep.reconcile()
