"""Every number quoted in the manuscript, checked against its artefact.

A paper's numbers drift from its results the moment an experiment is re-run and
a table is not. This test reads ``paper/main.tex`` and the committed CSVs and
asserts they agree, so the drift is a test failure rather than a reviewer's
discovery.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# The paper is the main text plus its Supplementary Material (rev14).
_RAW = (ROOT / "paper" / "main.tex").read_text() + "\n" + (ROOT / "paper" / "supplementary.tex").read_text()

# Searches are done against whitespace-normalised source, so a claim that a
# phrase appears in the paper is not defeated by where LaTeX happened to wrap
# the line. Draft-hygiene checks use the raw text.
TEX = re.sub(r"\s+", " ", _RAW)


def _rows(name: str) -> list[dict]:
    return list(csv.DictReader((ROOT / "results" / name).open()))


def _pct(x) -> float:
    return round(float(x) * 100, 2)


def _tex_has(phrase: str) -> bool:
    return re.sub(r"\s+", " ", phrase) in TEX


# --------------------------------------------------------------------------- #
# Section 5: the coded survey, first and second coding                         #
# --------------------------------------------------------------------------- #

def _survey(name: str) -> list[dict]:
    return list(csv.DictReader((ROOT / "data" / "survey" / name).open()))


def test_first_coding_counts_match_the_text():
    coded = _survey("coded.csv")
    n = len(coded)
    n_rc = sum(r["record_consumer"] == "yes" for r in coded)
    n_flow = sum(r["unit_of_inference"] == "flow_record" for r in coded)
    assert (n, n_rc, n_flow) == (39, 28, 25)
    assert "The author coded the 39 retrieved papers" in TEX
    assert "recording 25 as scoring completed flow records" in TEX
    assert "quote no pooled share" in TEX  # round 3': no pooled 28/39 headline


def test_the_superseded_87_percent_is_not_reused():
    """Revision history lives in the response letter (rev7, EIC M7); the
    superseded figure must not survive in the manuscript."""
    import re
    assert "87.1" not in TEX
    assert not re.search(r"(?<![\d.])87\\%", TEX)
    assert "31 papers" not in TEX


def test_kappa_is_reported_per_stratum():
    m = {r["metric"]: r for r in _rows("survey_agreement.csv")}
    ex, dns = m["record_consumer_kappa_exfiltration"], m["record_consumer_kappa_exfiltration_dns"]
    assert (round(float(ex["value"]), 2), round(float(ex["ci_lo"]), 2), round(float(ex["ci_hi"]), 2)) == (0.76, 0.41, 1.0)
    assert (round(float(dns["value"]), 2), round(float(dns["ci_lo"]), 2)) == (0.36, -0.29)
    assert int(m["record_consumer_agreement_exfiltration"]["value"]) == 15
    assert "28 ($71.8" not in TEX
    assert "$\\kappa = 0.76$ (bootstrap interval $[0.41, 1.00]$)" in TEX
    assert "$\\kappa = 0.36$ $[-0.29, 1.00]$" in TEX


def test_strata_table_matches_the_frame_and_both_codings():
    rows = {(r["scope"], r["channel"]): r for r in _rows("survey_strata.csv")}
    want = {("exfiltration", "dns"): (11, 2, 9, 2), ("exfiltration", "other"): (10, 8, 8, 6),
            ("general_nids", "other"): (17, 17, 17, 17), ("exfiltration", "any"): (21, 10, 17, 8)}
    assert ("all_but_survey", "any") not in rows and ("all_but_dns", "any") not in rows
    for k, (n, f, d2, s2) in want.items():
        r = rows[k]
        assert (int(r["n"]), int(r["first_rc"]), int(r["second_determined"]),
                int(r["second_rc"])) == (n, f, d2, s2), k
    for frag in ("2/11 $[5.1, 47.7]$", "2/9 $[6.3, 54.7]$", "8/10 $[49.0, 94.3]$",
                 "6/8 $[40.9, 92.9]$", "17/17 $[81.6, 100]$",
                 "10/21 $[28.3, 67.6]$", "8/17 $[26.2, 69.0]$"):
        assert re.sub(r"\s+", " ", frag) in TEX, frag
    # round 3 (C2): the in-frame sample of 21 exfiltration detectors is primary;
    # the out-of-rule general-IDS papers are not pooled into "all" rows.
    ex = [rows[("exfiltration", c)] for c in ("dns", "other")]
    assert sum(int(r["n"]) for r in ex) == 21
    assert sum(int(r["first_rc"]) for r in ex) == 10
    assert (sum(int(r["second_rc"]) for r in ex), sum(int(r["second_determined"]) for r in ex)) == (8, 17)
    for stale in ("25/27", "27/38", "25/34", "23/25"):
        assert stale not in TEX, stale


def test_agreement_statistics_match_the_artefact():
    m = {r["metric"]: r for r in _rows("survey_agreement.csv")}
    assert int(m["record_consumer_agreement"]["value"]) == 32
    assert int(m["unit_agreement"]["value"]) == 26
    assert int(m["second_coder_undetermined"]["value"]) == 5
    assert round(float(m["record_consumer_kappa"]["value"]), 2) == 0.85
    assert round(float(m["unit_kappa"]["value"]), 2) == 0.64
    assert round(float(m["cadence_kappa_among_rc"]["value"]), 2) == 0.27
    assert int(m["second_read_fulltext"]["value"]) == 22
    assert "raises $\\kappa$ to $0.85$ $[0.60, 1.00]$" in TEX
    assert "$\\kappa = 0.64$ $[0.42, 0.83]$" in TEX
    assert "9 of the 24" in TEX


def test_frame_scope_comes_from_the_second_coder():
    frame = {r["key"]: r for r in _survey("frame.csv")}
    second = {r["key"]: r for r in _survey("second_coding.csv")}
    for k, f in frame.items():
        if second[k]["include"] == "yes":
            assert f["scope"] == "exfiltration", k
    assert sum(f["scope"] == "survey" for f in frame.values()) == 1
    assert sum(f["scope"] == "general_nids" for f in frame.values()) == 17


# --------------------------------------------------------------------------- #
# Section 7.8: the cadence sweep                                               #
# --------------------------------------------------------------------------- #

def test_cadence_sweep_table_matches_the_csv():
    rows = _rows("cadence_sweep.csv")
    by_ta: dict[float, list[float]] = {}
    for r in rows:
        by_ta.setdefault(float(r["active_timeout_s"]), []).append(
            float(r["byte_floor"]))

    means = {ta: sum(v) / len(v) for ta, v in by_ta.items()}
    # Quoted in the table's mean row and in the prose.
    assert _pct(means[1800.0]) == pytest.approx(0.76, abs=0.005)
    assert _pct(means[15.0]) == pytest.approx(0.33, abs=0.005)
    # The saturation claim: nothing changes between 300 s and 1800 s.
    assert means[300.0] == pytest.approx(means[1800.0])
    assert means[600.0] == pytest.approx(means[1800.0])
    assert "only from $0.76\\%$ to $0.33\\%$" in TEX


def test_cadence_sweep_reproduces_the_published_floors_at_1800s():
    """Table 3's 1800 s column must equal Table 2's floor column."""
    sweep = {r["episode_id"]: float(r["byte_floor"])
             for r in _rows("cadence_sweep.csv")
             if float(r["active_timeout_s"]) == 1800.0}
    gap = {r["episode_id"]: float(r["floor"])
           for r in _rows("real_edxgb_gap_loo.csv")}
    assert set(sweep) == set(gap)
    for eid in sweep:
        assert sweep[eid] == pytest.approx(gap[eid], abs=1e-6), eid


# --------------------------------------------------------------------------- #
# Section 7.5-7.6: Table 2 under the a-priori rule                             #
# --------------------------------------------------------------------------- #

_EP = {  # Table 2's names for the seven episodes
    "10.1.3.17->10.8.10.84:443@1625264949.712": "443 \\#1",
    "10.1.3.17->10.1.5.21:3306@1625199408.918": "3306 \\#1",
    "10.1.3.17->10.1.5.21:3306@1625185759.453": "3306 \\#2",
    "10.1.3.8->10.8.10.84:443@1624318576.594": "443 \\#2",
    "10.1.3.17->10.8.10.84:443@1624665799.794": "443 \\#3",
    "10.1.3.17->10.8.10.84:443@1624784014.250": "443 \\#4",
    "10.1.3.17->10.8.10.84:443@1624863977.550": "443 \\#5",
}


def _fmt_pct(x: float) -> str:
    v = float(x) * 100
    return "100.0\\%" if v >= 99.995 else f"{v:.2f}\\%"


@pytest.mark.parametrize("protocol", ["loo", "strict"])
def test_table2_is_reproduced_exactly_under_both_rules(protocol):
    """The Wilson rule picks the archived threshold; single-threaded reruns
    under both rules reproduce the archived table exactly."""
    arch = _rows(f"real_edxgb_gap_{protocol}.csv")
    for variant in ("wilson", "point_1thread"):
        new = {r["episode_id"]: r for r in _rows(f"real_edxgb_gap_{protocol}_{variant}.csv")}
        for r in arch:
            n = new[r["episode_id"]]
            for k in ("floor", "actual", "gap", "threshold", "session_fpb_heldout",
                      "flow_fp_rate_heldout", "fired"):
                assert n[k] == r[k], (protocol, variant, r["episode_id"], k)


def test_table2_rows_match_the_wilson_csvs():
    loo = {r["episode_id"]: r for r in _rows("real_edxgb_gap_loo_wilson.csv")}
    strict = {r["episode_id"]: r for r in _rows("real_edxgb_gap_strict_wilson.csv")}
    for eid, name in _EP.items():
        lo, st = loo[eid], strict[eid]
        row = (f"{name} & {_fmt_pct(lo['floor'])} & {_fmt_pct(lo['actual'])} & ")
        assert _tex_has(row.replace("\\\\", "\\")) or row in TEX, row
    at_floor = lambda rows: sum(float(r["gap"]) <= 0.01 + 1e-12 for r in rows.values())
    assert at_floor(loo) == 5 and at_floor(strict) == 5


def test_operating_point_and_burden_are_stated():
    r = _rows("real_edxgb_gap_strict_wilson.csv")[0]
    assert r["rule"] == "wilson" and r["admissible"] == "True"
    assert float(r["threshold"]) == pytest.approx(0.40)
    assert round(float(r["session_fpb_train"]) * 100, 2) == 0.15
    assert round(float(r["session_fpb_train_upper"]) * 100, 2) == 0.35
    assert round(float(r["session_fpb_heldout"]) * 100, 2) == 0.45
    assert (round(float(r["session_fpb_heldout_lo"]) * 100, 2),
            round(float(r["session_fpb_heldout_hi"]) * 100, 2)) == (0.27, 0.74)
    assert int(r["heldout_benign_sessions"]) == 3334
    assert "$0.45\\%$ $[0.27, 0.74]$" in TEX
    assert "upper bound $0.35\\%$" in TEX
    assert "$6{,}668$ host--service keys" in TEX and "6{,}669" not in TEX


# --------------------------------------------------------------------------- #
# Section 7.8: four detector families                                          #
# --------------------------------------------------------------------------- #

def _table_row(start: str, after: str = r"\label{tab:multimodel}") -> str:
    """The LaTeX table row that begins with ``start`` (whitespace-normalised),
    searched from the label ``after`` (the four-family table by default)."""
    i = TEX.find(start, max(TEX.find(after), 0))
    assert i >= 0, start
    return TEX[i:TEX.find("\\\\", i)]


def test_multimodel_table_matches_the_wilson_run():
    rows = {r["model"]: r for r in _rows("real_multimodel_summary_wilson.csv")}
    for model in ("XGBoost (default)", "Random Forest", "EDXGB (reproduced)",
                  "Logistic Regression"):
        r = rows[model]
        assert r["rule"] == "wilson"
        line = _table_row(model + ("$^{a}$" if model == "Logistic Regression" else "") + " & ")
        assert f"{float(r['ebd_score_pct']):.2f}" in line, (model, line)
        assert f"{float(r['f1']):.3f}" in line, (model, line)
        fpb = (f"{float(r['fpb_heldout'])*100:.2f}\\% [{float(r['fpb_heldout_lo'])*100:.2f}, "
               f"{float(r['fpb_heldout_hi'])*100:.2f}]")
        assert fpb in line, (model, fpb, line)
        detected = round(float(r["detect_rate"]) * 7)
        assert f"{detected}/7" in line, (model, line)
        # round 3 (R1-M2): admissible = rule met in training AND held-out within budget
        adm = r["admissible"] == "True" and float(r["fpb_heldout"]) <= 0.01
        assert line.rstrip().endswith("yes" if adm else "no"), line
    # EDXGB's row is Table 2's strict column
    strict = _rows("real_edxgb_gap_strict_wilson.csv")
    mean_strict = sum(float(r["actual"]) for r in strict) / len(strict) * 100
    assert float(rows["EDXGB (reproduced)"]["ebd_score_pct"]) == pytest.approx(mean_strict, abs=1e-9)


def test_multimodel_f1_at_the_rule_and_episode_summaries():
    """F1 at the rule's own threshold (episode level) and the median / worst
    nEBD over the seven episodes, as tabulated (EIC M3, M6)."""
    import statistics
    summ = {r["model"]: r for r in _rows("real_multimodel_summary_wilson.csv")}
    per: dict[str, list[float]] = {}
    for r in _rows("real_multimodel_per_episode_wilson.csv"):
        per.setdefault(r["model"], []).append(float(r["actual_pct"]))
    for model, r in summ.items():
        tp = round(float(r["detect_rate"]) * 7)
        fp = round(float(r["fpb_heldout"]) * int(r["heldout_benign_sessions"]))
        f1 = 2 * tp / (2 * tp + fp + (7 - tp))
        line = _table_row(model + ("$^{a}$" if model == "Logistic Regression" else "") + " & ")
        assert f" {f1:.3f} " in line, (model, f1, line)
        med, worst = statistics.median(per[model]), max(per[model])
        fmt = lambda v: f"{v:.2f}" if v < 10 else f"{v:.1f}"
        assert fmt(med) in line and fmt(worst) in line, (model, med, worst, line)
    # round 3 (C1): the pre-specified H1 test is reported with its own verdict
    h1 = (ROOT / "results" / "real_multimodel_h1_wilson.txt").read_text()
    assert "Kendall tau = +0.333" in h1 and ("DIVERGE (H1 supported)" in h1 or "divergent by the H1 criterion" in h1)
    assert "Kendall's $\\tau = +0.33$" in TEX and "H1 therefore holds" not in TEX
    assert "H1 is untested at admissible operating points of the published grid" in TEX
    assert "permutation $p$-value is $0.75$" in TEX
    sens = _rows("real_multimodel_sensitivity_wilson.csv")
    assert len(sens) == 7 and all(r["diverges"] == "True" for r in sens)
    assert "artefact of unequal burdens" not in TEX


def test_matched_burden_comparison_matches_the_sweep():
    sw = {(r["model"], round(float(r["threshold"]), 2)): r
          for r in _rows("real_multimodel_sweep_wilson.csv")}
    pick = lambda m, t: sw[(m, t)]
    rf15, xgb05 = pick("Random Forest", 0.2), pick("XGBoost (default)", 0.05)
    assert round(float(rf15["heldout_fpb"]) * 100, 2) == round(float(xgb05["heldout_fpb"]) * 100, 2) == 0.15
    assert round(float(rf15["mean_ebd_pct"]), 1) == 41.7 and round(float(xgb05["mean_ebd_pct"]), 1) == 27.5
    rf06, xgb06 = pick("Random Forest", 0.35), pick("XGBoost (default)", 0.2)
    assert round(float(rf06["heldout_fpb"]) * 100, 2) == round(float(xgb06["heldout_fpb"]) * 100, 2) == 0.06
    assert round(float(rf06["mean_ebd_pct"]), 1) == 57.5 and round(float(xgb06["mean_ebd_pct"]), 1) == 28.8
    rf48 = pick("Random Forest", 0.1)
    assert round(float(rf48["heldout_fpb"]) * 100, 2) == 0.48 and round(float(rf48["mean_ebd_pct"]), 1) == 20.8
    for frag in ("loses $41.7\\%$ of the asset on average against $27.5\\%$ for XGBoost",
                 "$57.5\\%$ against $28.8\\%$", "($20.8\\%$ at $0.48\\%$)"):
        assert _tex_has(frag.replace("\\n", " ")), frag


# --------------------------------------------------------------------------- #
# Section 7.9: granularity and the session-granularity detector                #
# --------------------------------------------------------------------------- #

def _sd_episodes(gap: float) -> list[dict]:
    return [r for r in _rows("session_detector_episodes.csv") if float(r["gap_s"]) == gap]


def test_session_floor_is_total_loss_at_300s_and_above():
    for gap in (300.0, 600.0, 1800.0):
        sub = _sd_episodes(gap)
        assert len(sub) == 7, gap
        assert all(float(r["floor"]) >= 0.999 for r in sub), gap
        assert all(float(r["ebd"]) >= 0.999 for r in sub), gap
    assert sum(float(r["floor"]) >= 0.999 for r in _sd_episodes(60.0)) == 6


def test_session_detector_never_beats_its_floor_and_fires_on_it():
    fired = []
    for gap in (60.0, 300.0, 600.0, 1800.0):
        sub = _sd_episodes(gap)
        assert all(float(r["ebd"]) >= float(r["floor"]) - 1e-9 for r in sub)
        assert all(abs(float(r["ebd"]) - float(r["floor"])) < 1e-9
                   for r in sub if r["fired"] == "True")
        fired.append(sum(r["fired"] == "True" for r in sub))
    assert fired == [1, 5, 2, 2]
    assert "$1$, $5$, $2$ and $2$ of the seven episodes" in TEX


def test_the_one_episode_with_a_reachable_floor_is_caught_on_it():
    sub = [r for r in _sd_episodes(60.0) if int(r["n_session_records"]) > 1]
    assert len(sub) == 1
    r = sub[0]
    assert int(r["n_session_records"]) == 3 and r["fired"] == "True"
    assert float(r["ebd"]) == pytest.approx(float(r["floor"]), abs=1e-9)
    assert round(float(r["floor"]) * 100, 2) == 1.06
    assert _EP[r["episode_id"]] == "3306 \\#2"
    assert "a floor of $1.06\\%$" in TEX


def test_session_table_matches_the_run():
    for r in _rows("session_detector.csv"):
        gap = int(float(r["gap_s"]))
        n = f"{int(r['heldout_benign_units']):,}".replace(",", "{,}")
        row = _table_row(f"{gap}\\,s & {n} & ", r"\label{tab:sessiondet}")
        fpb = (f"{float(r['heldout_fpb'])*100:.2f}\\% [{float(r['heldout_lo'])*100:.2f}, "
               f"{float(r['heldout_hi'])*100:.2f}]")
        auc = f"{float(r['heldout_auc']):.2f} ({int(r['heldout_episode_sessions'])})"
        assert fpb in row and auc in row, (row, fpb, auc)
        assert row.rstrip().endswith(f"{r['fired']}/7"), row
        assert float(r["threshold"]) == pytest.approx(0.40)
        assert float(r["train_upper"]) <= 0.0008
    # the superseded AUC 0.61-0.63 is explained in the response letter, not here
    assert "A correction." not in TEX and "0.61$--$0.63" not in TEX


def test_all_seven_episodes_are_total_loss_at_session_granularity():
    rows = _rows("granularity.csv")
    for gap in (300.0, 600.0, 1800.0):
        sub = [r for r in rows if float(r["session_gap_s"]) == gap]
        assert len(sub) == 7
        assert all(float(r["session_byte_floor"]) >= 0.999 for r in sub), gap

    at60 = [r for r in rows if float(r["session_gap_s"]) == 60.0]
    assert sum(float(r["session_byte_floor"]) >= 0.999 for r in at60) == 6
    flow = {r["episode_id"]: float(r["flow_byte_floor"]) for r in rows}
    mean_flow = sum(flow.values()) / len(flow)
    assert _pct(mean_flow) == pytest.approx(0.76, abs=0.005)
    assert round(1.0 / mean_flow) == 131


def test_emission_semantics_table_matches_the_run():
    rows = _rows("emission_semantics.csv")
    def mean(sem, p, k="floor"):
        sub = [float(r[k]) for r in rows if r["semantics"] == sem and float(r["param_s"]) == p]
        assert len(sub) == 7
        return 100 * sum(sub) / 7
    assert round(mean("flow", 0), 2) == 0.76 and round(mean("flow", 15), 2) == 7.29
    assert round(mean("flow", 120), 1) == 13.9 and round(mean("flow", 300), 1) == 15.3
    assert round(mean("idle-gap", 60), 1) == 85.9 and round(mean("idle-gap", 120), 1) == 100.0
    assert round(mean("tumbling", 300), 1) == 12.4 and round(mean("polled", 120), 2) == 8.90
    assert round(mean("polled", 120, "floor_worst"), 1) == 13.9
    sil = {r["episode_id"]: float(r["longest_silence_s"]) for r in rows}
    assert max(sil.values()) < 120 and round(max(sil.values()), 1) == 105.4
    # the flow floor at L = 0 is Table 2's floor
    gap = {r["episode_id"]: float(r["floor"]) for r in _rows("real_edxgb_gap_loo_wilson.csv")}
    for r in rows:
        if r["semantics"] == "flow" and float(r["param_s"]) == 0:
            assert float(r["floor"]) == pytest.approx(gap[r["episode_id"]], abs=1e-9)
    assert _tex_has("\\textbf{mean} & & \\textbf{0.76} & \\textbf{7.29} & \\textbf{13.9} & \\textbf{15.3} &")
    ebd = _rows("emission_semantics_ebd.csv")
    loo = lambda L: [r for r in ebd if r["protocol"] == "loo" and float(r["latency_s"]) == L]
    assert sum(r["at_floor"] == "True" for r in loo(15)) == 6
    assert round(100 * sum(float(r["ebd"]) for r in loo(0)) / 7, 1) == 21.4
    assert round(100 * sum(float(r["ebd"]) for r in loo(300)) / 7, 1) == 29.2


def test_session_floor_never_below_flow_floor():
    """T_session >= T_flow is the structural claim; check it held on real data."""
    for r in _rows("granularity.csv"):
        assert float(r["session_byte_floor"]) >= float(r["flow_byte_floor"]) - 1e-9


# --------------------------------------------------------------------------- #
# Section 9: the D <= T regime                                                 #
# --------------------------------------------------------------------------- #

def _fast(name="fast_regime_wilson.csv"):
    return _rows(name)


def test_every_D_leq_T_single_connection_case_is_a_total_loss():
    rows = [r for r in _fast() if r["n_connections"] == "1"]
    dt = [r for r in rows
          if float(r["actual_duration_s"]) <= float(r["active_timeout_s"])]
    assert len(dt) == 17
    assert all(float(r["floor"]) >= 0.999 for r in dt)
    assert all(float(r["ebd"]) >= 0.999 for r in dt)
    assert "(13 at the base asset in Table~\\ref{tab:fast}, 4 among the $10\\times$ repeats) are total losses" in TEX


def test_the_bound_is_never_violated():
    """The measured floor must never fall below min(1, T/D)."""
    ex = []
    for r in _fast():
        if r["n_connections"] != "1":
            continue
        ta, d = float(r["active_timeout_s"]), float(r["actual_duration_s"])
        assert float(r["floor"]) >= min(1.0, ta / d) - 1e-6, r["label"]
        ex.append(float(r["floor"]) - min(1.0, ta / d))
    assert round(100 * sum(ex) / len(ex), 2) == 0.21 and round(100 * max(ex), 1) == 2.5
    assert _tex_has("by $0.21$ percentage points on average (at most\n$2.5$)".replace("\n", " "))


def test_record_consumer_misses_only_at_60s():
    rows = [r for r in _fast() if r["n_connections"] == "1"]
    assert len(rows) == 36
    at = [r for r in rows if abs(float(r["ebd"]) - float(r["floor"])) < 1e-6]
    miss = [r for r in rows if r not in at]
    assert len(at) == 32 and len(miss) == 4
    assert all(float(r["active_timeout_s"]) == 60.0 and r["fired"] == "False" for r in miss)
    assert "in 15 of the 19" in TEX


def test_testbed_separability_and_burdens():
    import json
    m = json.loads((ROOT / "results" / "fast_regime_wilson_meta.json").read_text())
    pt = m["per_timeout"]
    assert m["timeline_bin_seconds"] == 0.01
    s = [round(pt[k]["heldout_auc_session"], 2) for k in ("5", "15", "60")]
    rec = [round(pt[k]["heldout_auc_record"], 2) for k in ("5", "15", "60")]
    assert s == [0.80, 0.89, 0.85] and rec == [0.88, 0.87, 0.77]
    assert [pt[k]["threshold"] for k in ("5", "15", "60")] == [0.6, 0.55, 0.65]
    assert all(pt[k]["train_upper"] <= 0.01 and pt[k]["heldout_fpb"] <= 0.01 for k in pt)
    assert "$0.33\\%$ $[0.09, 1.21]$" in TEX and "$0.17\\%$ $[0.03, 0.94]$" in TEX
    assert "$0.00\\%$ $[0.00, 0.64]$" in TEX
    pm = json.loads((ROOT / "results" / "fast_regime_point_meta.json").read_text())["per_timeout"]
    assert round(pm["5"]["heldout_fpb"] * 100, 2) == 3.50
    assert "0.995" not in TEX


def test_chunking_floor_and_detector():
    ch = [r for r in _fast() if r["n_connections"] == "8"]
    assert len(ch) == 27
    dev = [abs(float(r["floor"]) - min(0.125, float(r["active_timeout_s"])
                                       / float(r["actual_duration_s"]))) for r in ch]
    assert max(dev) < 0.0035
    lost = [r for r in ch if float(r["ebd"]) >= 0.999]
    assert len(lost) == 17 and all(abs(float(r["floor"]) - 0.125) < 1e-6 for r in lost)
    assert "to within $0.34$ percentage points" in TEX and "in 17 of the 27 cells" in TEX


def test_asset_invariance_is_stated_precisely():
    import json
    m = json.loads((ROOT / "results" / "fast_regime_wilson_meta.json").read_text())
    ai = m["asset_invariance"]
    assert round(ai["max_abs_diff_pp"], 2) == 0.07 and ai["exact_agreement"] == 4
    assert "within $0.07$" in TEX


def test_netem_rerecording():
    import json
    m = json.loads((ROOT / "results" / "fast_regime_netem_meta.json").read_text())
    assert [u["label"] for u in m["unreconciled_excluded"]] == ["attack_d4_k1_a20M"]
    assert round(m["unreconciled_excluded"][0]["relative_error"] * 100, 2) == 1.88
    n = {(r["label"], r["active_timeout_s"]): r for r in _fast("fast_regime_netem.csv")}
    p = {(r["label"], r["active_timeout_s"]): r for r in _fast()}
    assert len(n) == 51
    assert max(abs(float(r["floor"]) - float(p[k]["floor"])) for k, r in n.items()) <= 0.0025 + 1e-9
    dt = [r for r in n.values() if r["n_connections"] == "1"
          and float(r["actual_duration_s"]) <= float(r["active_timeout_s"])]
    assert len(dt) == 14 and all(float(r["floor"]) >= 0.999 and float(r["ebd"]) >= 0.999 for r in dt)
    assert all(v["heldout_fpb"] == 0.0 and v["threshold"] == 0.65 for v in m["per_timeout"].values())
    assert all(float(r["ebd"]) >= 0.999 for r in n.values())
    assert "17 reconcile exactly" in TEX and "$1.88\\%$ less than planned" in TEX


# --------------------------------------------------------------------------- #
# Section 10: outside the family                                               #
# --------------------------------------------------------------------------- #

def test_escape_table_matches_the_csv():
    rows = sorted(_rows("escape_wilson.csv"), key=lambda r: float(r["duration_s"]))
    assert len(rows) == 5
    inc = [round(float(r["incremental_ebd"]) * 100, 1) for r in rows]
    assert inc == [100.0, 25.5, 6.3, 100.0, 100.0]
    f60 = [round(float(r["floor_Ta60"]) * 100, 1) for r in rows]
    assert f60 == [100.0, 100.0, 100.0, 46.9, 23.5]
    assert all(float(r[f"record_ebd_Ta{t}"]) == 1.0 for r in rows for t in (5, 15, 60))
    assert "& 25.5 \\\\" in TEX and "& 6.3 \\\\" in TEX
    assert all(float(r["incremental_session_fpb"]) == 0.0 for r in rows)


def test_escape_is_reported_as_bounded_and_selected():
    import statistics
    sw = _rows("escape_sweep.csv")
    fe = [r for r in sw if r["feasible_wilson"] == "True"]
    assert len(sw) == 125 and len(fe) == 40
    me = sorted(float(r["mean_ebd"]) for r in fe)
    assert round(100 * me[0], 1) == 66.4 and round(100 * statistics.median(me), 1) == 89.4
    b60 = [int(r["below_floor_Ta60"]) for r in fe]
    assert statistics.median(b60) == 1 and max(b60) == 2 and min(b60) == 0
    assert "median of $89.4" in TEX and "a median of $1$ of $5$" in TEX


def test_section_9_is_framed_as_instrument_validation():
    assert "instrument validation" in TEX.lower()
    assert "entailed" in TEX


# --------------------------------------------------------------------------- #
# Section 7.6: separability, ablations and the permutation null                #
# --------------------------------------------------------------------------- #

def test_baseline_controls():
    b = {r["variant"]: r for r in _rows("real_baseline.csv")}
    full, nop, rates = b["full"], b["no_ports"], b["rates_only"]
    strict = _rows("real_edxgb_gap_strict_wilson.csv")
    assert int(full["at_floor"]) == 5 and float(full["threshold"]) == 0.4
    assert float(full["mean_ebd_pct"]) == pytest.approx(
        100 * sum(float(r["actual"]) for r in strict) / 7, abs=1e-3)
    assert round(float(full["unit_auc"]), 3) == 0.991 and round(float(full["flow_auc"]), 3) == 0.836
    assert int(nop["at_floor"]) == 2 and int(nop["fired"]) == 6
    assert round(float(nop["median_ebd_pct"]), 2) == 3.46
    assert int(rates["fired"]) == 2
    perm = [r for r in b.values() if r["permuted"] == "True"]
    assert len(perm) == 20 and all(int(r["at_floor"]) == 0 and int(r["fired"]) == 0 for r in perm)
    assert max(float(r["heldout_fpb"]) for r in perm) <= 0.0066
    assert "reaches the floor on no episode" in TEX
    assert "We report no episode-level separability figure" in TEX


def test_timing_only_baselines_and_random_null():
    """Round 3 (R1-M5, DA-3): model-free rules and a burden-matched random null."""
    t = {r["variant"]: r for r in _rows("real_timing_baseline.csv")}
    assert (int(t["bytes_out"]["fired"]), int(t["bytes_out"]["at_floor"])) == (2, 0)
    assert (int(t["bytes_total"]["fired"]), int(t["bytes_total"]["at_floor"])) == (1, 0)
    assert _pct(t["bytes_out"]["heldout_fpb"]) == 0.51 and _pct(t["bytes_total"]["heldout_fpb"]) == 0.66
    rnd = [r for r in t.values() if r["kind"] == "random"]
    assert len(rnd) == 20 and all(int(r["at_floor"]) == 0 for r in rnd)
    fired = [int(r["fired"]) for r in rnd]
    assert min(fired) == 0 and max(fired) == 4
    fpb = [float(r["heldout_fpb"]) for r in rnd]
    assert round(100 * min(fpb), 2) == 0.39 and round(100 * max(fpb), 2) == 1.08
    auc = sorted(float(r["unit_auc"]) for r in rnd)
    assert round(auc[0], 2) == 0.93 and round(auc[-1], 2) == 0.99
    assert "uniform random scores reach $0.93$--$0.99$ (median $0.97$, 20 draws)" in TEX
    assert "random scores on between zero and four episodes at $0.39\\%$--$1.08\\%$" in TEX


def test_h1_stability_files_back_the_quoted_spread():
    """Round 3 (M10): the 20-seed and 15-seed files exist and match the text."""
    import statistics
    full = [float(r["kendall_tau"]) for r in _rows("h1_stability_full.csv")]
    clus = [r for r in _rows("h1_stability_clustered.csv")]
    assert len(full) == 20 and round(statistics.mean(full), 2) == 0.72
    assert round(statistics.stdev(full), 2) == 0.06 and min(full) >= 0.5
    assert len(clus) == 15 and sum(r["diverges"] == "1" for r in clus) == 2


def test_dns_prefix_split():
    """Round 3 (M3): total losses come from 50,000-packet captures."""
    stats = {r["file"]: r for r in _dns("dns_parse_stats.csv")}
    flow = [r for r in _dns("dns_floors.csv") if r["consumer"] == "flow"]
    pre = [r for r in flow if int(stats[r["file"]]["packets"]) == 50000]
    rest = [r for r in flow if int(stats[r["file"]]["packets"]) != 50000]
    assert len(pre) == 12 and len(rest) == 6
    assert all(float(r["floor_measured"]) < 0.999 for r in rest)
    assert sum(float(r["floor_measured"]) >= 0.999 for r in pre) == 9
    assert round(100 * sum(float(r["floor_measured"]) for r in rest) / 6, 1) == 6.2
    assert "the mean flow floor is $6.2\\%$" in TEX


# --------------------------------------------------------------------------- #
# Section 8: the DNS corpus (outputs of the v4.2 run, results/dns_v42)         #
# --------------------------------------------------------------------------- #

def _dns(name: str) -> list[dict]:
    return list(csv.DictReader((ROOT / "results" / "dns_v42" / name).open()))


def _family_means(consumer: str) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for r in _dns("dns_floors.csv"):
        if r["consumer"] == consumer:
            by.setdefault(r["family"], []).append(float(r["floor_measured"]))
    return {f: sum(v) / len(v) for f, v in by.items()}


def test_dns_corpus_size():
    rows = [r for r in _dns("dns_floors.csv") if r["consumer"] == "flow"]
    assert len(rows) == 18 and len({r["family"] for r in rows}) == 8
    assert "18\ntransfers in 8 tool families".replace("\n", " ") in TEX


def test_dns_floors_by_position():
    pq, fl, se = (_family_means(c) for c in ("per-query", "flow", "session"))
    assert round(100 * sum(pq.values()) / 8, 3) == 0.002
    assert round(100 * sum(fl.values()) / 8, 1) == 40.1
    assert all(v >= 0.999 for v in se.values())
    assert sum(v >= 0.999 for v in fl.values()) == 3
    episode_mean = [float(r["floor_measured"]) for r in _dns("dns_floors.csv")
                    if r["consumer"] == "flow"]
    assert round(100 * sum(episode_mean) / 18, 1) == 52.4
    assert "$40.1\\%$ for the\nflow-record consumer".replace("\n", " ") in TEX
    # rev13 review R1-9: no interval over a convenience compilation of families
    assert "40.1 & 100" in TEX and "[0, 81.8]" not in TEX and "[12.8, 75.2]" not in TEX
    assert "compilation, not a sample, so no" in TEX
    summ = {r["consumer"]: r for r in csv.DictReader(open(ROOT / "results/dns_v42/dns_floor_summary.csv"))}
    assert round(100 * float(summ["flow"]["family_boot_lo"]), 1) == 12.8
    assert round(100 * float(summ["flow"]["family_boot_hi"]), 1) == 75.2



def test_dns_family_table_rows():
    fl = _family_means("flow")
    expect = {"dns2tcp": 100, "dnscat2": 100, "iodine": 100, "tcp_over_dns": 17.50,
              "cobaltstrike": 2.03, "dnsexfiltrator": 1.03, "ozymandns": 0.33, "dnspot": 0.01}
    for f, v in expect.items():
        assert round(100 * fl[f], 2) == pytest.approx(v, abs=0.006), f


def test_dns_cadence_sweep():
    by: dict[float, list[float]] = {}
    for r in _dns("dns_cadence_sweep.csv"):
        if r["axis"] == "flow active timeout":
            by.setdefault(float(r["param"]), []).append(float(r["floor_measured"]))
    m = {k: 100 * sum(v) / len(v) for k, v in by.items()}
    assert round(m[1800.0], 1) == 52.4 and round(m[30.0], 1) == 12.9
    assert sum(x >= 0.999 for x in by[1800.0]) == 9 and sum(x >= 0.999 for x in by[30.0]) == 0
    assert "from $52.4\\%$ to $12.9\\%$" in TEX


def test_dns_closed_form_and_invariant():
    ex = [float(r["excess_pp"]) for r in _dns("dns_floors.csv") if r["consumer"] == "flow"]
    assert sum(e < -1e-9 for e in ex) == 6 and round(min(ex), 2) == -1.72
    assert round(sum(ex) / len(ex), 2) == -0.13
    att = _dns("dns_attainment.csv")
    assert all(float(r["ebd"]) >= float(r["floor_measured"]) - 1e-9 for r in att)
    assert len(att) == 54
    assert "all\n54 episode--consumer pairs".replace("\n", " ") in TEX


def test_dns_benign_overshoot_is_reported_as_found():
    ops = _dns("dns_operating_points.csv")
    over = sum(float(r["fpb_university_traffic_lo"]) > 0.01 for r in ops)
    assert len(ops) == 24 and over == 22
    for c, lo, hi in (("per-query", 1.07, 1.13), ("flow", 1.01, 1.13), ("session", 1.33, 1.55)):
        u = [100 * float(r["fpb_university_traffic"]) for r in ops if r["consumer"] == c]
        assert round(min(u), 2) == lo and round(max(u), 2) == hi, c
    assert all(float(r["train_upper"]) <= 0.01 + 1e-9 for r in ops)
    assert "in 22\nof the 24 consumer--fold pairs".replace("\n", " ") in TEX
    assert "did not retune" in TEX

# --------------------------------------------------------------------------- #
# Draft hygiene                                                                #
# --------------------------------------------------------------------------- #

def test_only_author_items_remain_open():
    """Every remaining red marker is something only the author can supply, and
    no number is left pending."""
    body = _RAW.split(r"\begin{document}", 1)[1]
    assert "\\pend{" not in body, "a number still waits for the revision run"
    notes = re.findall(r"\\draftnote\{([^}]*)", body)
    allowed = ("repository URL", "PENDING MAC RUN", "PENDING CIC RUN", "CO-AUTHORS", "AUTHOR: confirm with the publisher")
    assert notes and all(any(a in n for a in allowed) for n in notes), notes


def test_the_cisco_juniper_correction_is_present():
    """The first draft's 60 s figure was Juniper's, not Cisco's."""
    assert "cisco_fnf_cache" in TEX
    assert "juniper_flow_active_timeout" in TEX
    assert "$1800$\\,s on Cisco Flexible NetFlow~\\citep{cisco_fnf_cache}, $60$\\,s in Juniper's" in TEX


def test_unraveled_is_cited_as_the_journal_paper():
    bib = (ROOT / "paper" / "refs.bib").read_text()
    assert "myneni2023unraveled" in bib
    assert "10.1016/j.comnet.2023.109688" in bib
    assert "myneni2023unraveled" in TEX


def test_d5_sign_convention_and_ci_are_backed():
    """Round 3' (M10): the D5 paired difference has a committed source."""
    txt = (ROOT / "results" / "d5_paired_bootstrap.txt").read_text()
    assert "paired nEBD saving   : -0.3390   95% CI [-0.4351, -0.2508]" in txt
    assert "saving = time - byte" in txt
    assert _tex_has("\\mathrm{nEBD}_{\\text{time}} - \\mathrm{nEBD}_{\\text{byte}} = -0.34$")


def test_dns_timeout_sweep_split_by_truncation():
    stats = {r["file"]: int(r["packets"]) for r in _dns("dns_parse_stats.csv")}
    sw = [r for r in _dns("dns_cadence_sweep.csv")
          if r["consumer"] == "flow" and r["axis"] == "flow active timeout"]
    def mean(param, keep):
        v = [float(r["floor_measured"]) for r in sw if float(r["param"]) == param and keep(r)]
        return round(100 * sum(v) / len(v), 1)
    full = lambda r: stats[r["file"]] != 50000
    assert mean(1800.0, lambda r: True) == 52.4 and mean(30.0, lambda r: True) == 12.9
    assert mean(1800.0, full) == 6.2 and mean(30.0, full) == 0.5
    assert "from $6.2\\%$ to $0.5\\%$" in TEX


def test_round5_robustness_and_calibrated_ranks():
    """Round-5 numbers (run_round5.sh) quoted in tab:r12-robust and tab:r12-rank."""
    for frag in ("2--5, med.\\ 4 & 5--6, med.\\ 5 & 0.09--0.78", "2--6, med.\\ 3 & 5--6, med.\\ 5 & 0.09--2.13", "port fields only & 2 (2) & 2 (2) & 0.00\\%",
                 "benign negatives, without port fields & 6 (4)",
                 "XGBoost (default)   & 0.800 & 5.39\\%", "Random Forest       & 0.625 & 20.81\\%",
                 "Logistic Regression$^{a}$ & 0.139 & 86.24\\%", "$+0.67$", "$+1.00$"):
        assert re.sub(r"\s+", " ", frag) in TEX, frag
