"""Session records for the Sec. 7.9 detector: grouping, aggregation, emission."""

from __future__ import annotations

import numpy as np
import pytest

from ebd.experiments.session_detector import session_features, sessionize_rows


def _arrays(spans):
    starts = np.array([s for s, _ in spans], dtype=float)
    ends = np.array([s + d for s, d in spans], dtype=float)
    return starts, ends


def test_split_on_silence_measured_from_the_running_end():
    # a long flow keeps the session open: the third flow starts 500 s after the
    # second flow STARTED but only 50 s after the first flow ENDED.
    spans = [(0.0, 900.0), (400.0, 10.0), (950.0, 5.0), (2000.0, 1.0)]
    starts, ends = _arrays(spans)
    sid = sessionize_rows([("a", "b", 1)] * 4, starts, ends, gap_seconds=300.0)
    assert sid[0] == sid[1] == sid[2]
    assert sid[3] != sid[0]


def test_keys_never_merge_and_invalid_rows_are_dropped():
    starts = np.array([0.0, 1.0, np.nan])
    ends = np.array([1.0, 2.0, np.nan])
    keys = [("a", "b", 1), ("a", "c", 1), ("a", "b", 1)]
    sid = sessionize_rows(keys, starts, ends, gap_seconds=1e9)
    assert sid[0] != sid[1]
    assert sid[2] == -1


def test_session_features_aggregate_as_documented():
    spans = [(0.0, 10.0), (20.0, 5.0), (100.0, 1.0)]
    starts, ends = _arrays(spans)
    X = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]], dtype=np.float32)
    sums = np.array([[100, 1, 101, 2, 1, 3]] * 3, dtype=float)
    sid = np.array([0, 0, 1])
    uniq, F, t_first, t_last, n = session_features(X, sid, starts, ends, sums)
    assert list(uniq) == [0, 1] and list(n) == [2, 1]
    # per-column means
    assert F[0, 0] == pytest.approx(2.0) and F[0, 1] == pytest.approx(20.0)
    # extras: n_flows, mean inter-flow time (between starts), span, then sums
    assert F[0, 2] == 2 and F[0, 3] == pytest.approx(20.0) and F[0, 4] == pytest.approx(25.0)
    assert F[0, 5] == pytest.approx(200.0)
    # emission is credited to the end of the last flow summarised
    assert t_last[0] == pytest.approx(25.0) and t_first[1] == pytest.approx(100.0)
    assert F[1, 3] == 0.0                      # one flow: no inter-flow time
