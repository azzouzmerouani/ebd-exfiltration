"""Inter-coder agreement for the survey (ebd.experiments.survey_agreement)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ebd.experiments.survey_agreement import bootstrap_kappa, cohen_kappa, is_rc

ROOT = Path(__file__).resolve().parents[1]


def test_kappa_on_a_hand_computed_example():
    # p_o = 3/4; p_e = 0.5*0.25 + 0.5*0.75 = 0.5; kappa = (0.75-0.5)/(1-0.5) = 0.5
    assert cohen_kappa(list("yynn"), list("ynnn")) == pytest.approx(0.5)


def test_kappa_limits():
    assert cohen_kappa(list("abcab"), list("abcab")) == pytest.approx(1.0)
    assert cohen_kappa(list("aaaa"), list("aaaa")) == 1.0          # one shared category
    # systematic disagreement on a balanced binary item set is -1
    assert cohen_kappa(list("abab"), list("baba")) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_kappa_matches_scikit_learn():
    sk = pytest.importorskip("sklearn.metrics")
    a = list("aabbccabcabcaaab")
    b = list("abbbccabcaacaabb")
    assert cohen_kappa(a, b) == pytest.approx(sk.cohen_kappa_score(a, b))


def test_bootstrap_interval_brackets_the_estimate():
    a = list("aabbccabcabcaaab") * 2
    b = list("abbbccabcaacaabb") * 2
    lo, hi = bootstrap_kappa(a, b, n_boot=500)
    assert lo <= cohen_kappa(a, b) <= hi


def test_record_consumer_mapping():
    assert is_rc("flow_record") == is_rc("session_aggregate") == is_rc("host_timeseries") == "yes"
    assert is_rc("packet") == is_rc("payload_dpi") == "no"
    assert is_rc("undetermined") == "undetermined"


def test_the_three_survey_files_describe_the_same_papers():
    d = ROOT / "data" / "survey"
    keys = [{r["key"] for r in csv.DictReader((d / f).open())}
            for f in ("coded.csv", "second_coding.csv", "frame.csv", "sample_bibliography.csv")]
    assert all(k == keys[0] for k in keys) and len(keys[0]) == 39
    second = list(csv.DictReader((d / "second_coding.csv").open()))
    # blind ids are opaque and unique; every second-coding row carries its evidence
    assert len({r["blind_id"] for r in second}) == 39
    assert all(r["evidence_quote"] and r["evidence_url"].startswith("http") for r in second)
