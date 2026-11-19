"""The a-priori operating-point rule (ebd.realdata.operating_point)."""

from __future__ import annotations

import numpy as np
import pytest

from ebd.realdata.operating_point import (
    DEFAULT_GRID, choose_threshold, measure_burden, unit_max_scores, wilson_interval,
    wilson_upper,
)


@pytest.mark.parametrize("k,n,lo,hi", [
    (5, 7, 0.359, 0.918),       # Table 2's 5 of 7 at floor
    (28, 39, 0.562, 0.835),     # the coded sample, Sec. 5
    (7, 7, 0.646, 1.000),       # all seven, Sec. 7.9
    (0, 600, 0.000, 0.00636),   # zero alerts on 600 held-out benign sessions
])
def test_wilson_matches_the_values_quoted_in_the_paper(k, n, lo, hi):
    a, b = wilson_interval(k, n)
    assert a == pytest.approx(lo, abs=6e-4)
    assert b == pytest.approx(hi, abs=6e-4)


def test_wilson_degenerate_inputs():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert 0.0 <= wilson_upper(3, 10) <= 1.0


def test_grid_is_the_papers():
    assert len(DEFAULT_GRID) == 19
    assert DEFAULT_GRID[0] == pytest.approx(0.05) and DEFAULT_GRID[-1] == pytest.approx(0.95)


def test_point_and_wilson_rules_on_a_worked_example():
    # 1000 benign units. Up to 0.45, 11 units alert (1.1%): both rules refuse.
    # At 0.50 four alert (0.4%): the point rule accepts, but the upper Wilson
    # bound of 4/1000 is 1.02%, so the Wilson rule refuses. At 0.60 two alert:
    # upper bound 0.73%, accepted.
    m = np.concatenate([np.full(989, 0.10), np.full(7, 0.45), np.full(2, 0.55),
                        np.full(2, 0.97)])
    point = choose_threshold(m, 0.01, rule="point")
    wil = choose_threshold(m, 0.01, rule="wilson")
    assert point["threshold"] == pytest.approx(0.50)
    assert point["train_k"] == 4 and point["train_upper"] > 0.01
    assert wil["threshold"] == pytest.approx(0.60)
    assert wil["train_k"] == 2 and wil["train_upper"] <= 0.01
    assert wil["admissible"] and point["admissible"]


def test_wilson_is_never_more_aggressive_than_point(rng=np.random.default_rng(7)):
    for _ in range(200):
        m = rng.beta(0.5, 5.0, size=int(rng.integers(50, 3000)))
        p = choose_threshold(m, 0.01, rule="point")
        w = choose_threshold(m, 0.01, rule="wilson")
        assert w["threshold"] >= p["threshold"]
        if w["admissible"]:
            assert w["train_upper"] <= 0.01 + 1e-12


def test_no_admissible_threshold_is_reported_not_hidden():
    m = np.full(100, 0.99)                                 # every unit alerts everywhere
    op = choose_threshold(m, 0.01, rule="wilson")
    assert op["admissible"] is False and op["threshold"] == pytest.approx(0.95)


def test_measure_burden_and_empty_units():
    scores = [np.array([0.2, 0.7]), np.array([0.1]), np.array([])]
    mx = unit_max_scores(scores)
    b = measure_burden(mx, 0.5)
    assert (b["heldout_k"], b["heldout_n"]) == (1, 3)
    assert b["heldout_lo"] <= b["heldout_fpb"] <= b["heldout_hi"]


def test_unknown_rule_is_an_error():
    with pytest.raises(ValueError):
        choose_threshold([0.1], 0.01, rule="median")


def test_grid_bottom_binds_flag():
    import numpy as np
    from ebd.realdata.operating_point import choose_threshold, extended_grid
    m = np.zeros(1000)                     # a detector that never scores a benign unit
    op = choose_threshold(m + 0.01, 0.01, rule="wilson")
    assert op["admissible"] and op["grid_bottom_binds"] and op["threshold"] == 0.05
    g = extended_grid(m + 0.01)
    assert g.min() < 0.05 and 0.05 in g
    op2 = choose_threshold(m + 0.01, 0.01, rule="wilson", grid=g)
    assert op2["threshold"] < 0.05


def test_grid_bottom_not_binding_when_rule_selects_higher():
    import numpy as np
    from ebd.realdata.operating_point import choose_threshold
    m = np.linspace(0, 0.499, 1000)
    op = choose_threshold(m, 0.01, rule="wilson")
    assert op["admissible"] and not op["grid_bottom_binds"] and op["threshold"] == 0.5
