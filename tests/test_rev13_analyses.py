"""Numbers of the rev13 analyses (response to the rev12 review) quoted in the paper."""
import csv
import re
import statistics as st
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
TEX = re.sub(r"\s+", " ", (ROOT / "paper/main.tex").read_text() + "\n" + (ROOT / "paper/supplementary.tex").read_text())


def _rows(name):
    return list(csv.DictReader(open(R / name)))


def test_gap_sensitivity_table():
    rows = _rows("gap_sensitivity.csv")
    exp = {120: (0.76, 7.29, 13.9), 300: (0.92, 0.98, 7.82), 600: (0.88, 1.17, 8.46)}
    for g, (l0, l15, l120) in exp.items():
        sub = [r for r in rows if float(r["gap_s"]) == g]
        assert len(sub) == 7
        m = lambda k: 100 * sum(float(r[k]) for r in sub) / 7
        assert m("floor_L0") == pytest.approx(l0, abs=0.006)
        assert m("floor_L15") == pytest.approx(l15, abs=0.006)
        assert m("floor_L120") == pytest.approx(l120, abs=0.06)
    e3 = [r for r in rows if float(r["gap_s"]) == 300 and "1624665799" in r["reference_episode"]][0]
    assert int(e3["other_bytes"]) == 85953
    assert 100 * float(e3["floor_L15"]) == pytest.approx(1.92, abs=0.005)
    assert 100 * float(e3["floor_L120"]) == pytest.approx(3.62, abs=0.005)
    assert "$300$\\,s & 0.92 & 0.98 & 7.82" in TEX


def test_semantics_extended_latency():
    rows = _rows("emission_semantics.csv")
    for L, mean, med in ((600, 16.7, 6.63), (900, 18.4, 8.94), (1800, 27.3, 15.9)):
        v = [float(r["floor"]) for r in rows if r["semantics"] == "flow" and float(r["param_s"]) == L]
        assert 100 * st.mean(v) == pytest.approx(mean, abs=0.06)
        assert 100 * st.median(v) == pytest.approx(med, abs=0.06)
    for W, mean in ((600, 14.2), (900, 15.2)):
        v = [float(r["floor"]) for r in rows if r["semantics"] == "tumbling" and float(r["param_s"]) == W]
        assert 100 * st.mean(v) == pytest.approx(mean, abs=0.06)
    assert "$16.7\\%$, $18.4\\%$ and $27.3\\%$" in TEX


def test_dns_resolver_leg():
    rows = _rows("dns_v42/dns_resolver_leg.csv")
    tl = [r for r in rows if int(r["n_ports"]) == 1 and float(r["floor_client_edge"]) >= 0.999]
    assert len(tl) == 9
    v = [float(r["floor_resolver_leg_closed_form"]) for r in tl]
    assert 100 * min(v) == pytest.approx(7.8, abs=0.05)
    assert 100 * max(v) == pytest.approx(28.3, abs=0.05)
    fam = {}
    for r in rows:
        fam.setdefault(r["family"], []).append(float(r["floor_resolver_leg_closed_form"]))
    assert 100 * st.mean(st.mean(x) for x in fam.values()) == pytest.approx(5.8, abs=0.05)
    assert "$7.8\\%$ to $28.3\\%$" in TEX


def test_cic_numbers_quoted():
    rows = [r for r in _rows("cic_dns_split/cic_floors.csv") if r["consumer"] == "flow"]
    v = [float(r["floor_measured"]) for r in rows]
    assert len(v) == 28
    assert 100 * st.mean(v) == pytest.approx(7.5, abs=0.05)
    assert 100 * st.median(v) == pytest.approx(3.2, abs=0.05)
    cf = lambda L: [min(1.0, (15 + L) / float(r["duration_s"])) for r in rows]
    assert 100 * st.mean(cf(300)) == pytest.approx(56.9, abs=0.05)
    assert 100 * st.mean(cf(900)) == pytest.approx(71.5, abs=0.05)
    assert sum(x >= 0.999 for x in cf(300)) == 11 and sum(x >= 0.999 for x in cf(900)) == 16
    assert "$7.5\\%$ on average" in TEX


def test_round6_out_of_fold_robustness():
    import glob
    R6 = R / "round6"
    def summ(neg, proto, variant):
        rows = []
        for p in glob.glob(str(R6 / f"var_{proto}_{neg}_*[0-9].csv")):
            rows += [r for r in csv.DictReader(open(p)) if r["variant"] == variant]
        assert len(rows) == 10 and all(r["calibration"] == "oof5" for r in rows)
        af = [int(r["at_floor"]) for r in rows]
        fp = [float(r["heldout_fpb"]) for r in rows]
        return min(af), max(af), st.median(af), round(100 * min(fp), 2), round(100 * max(fp), 2)
    assert summ("all", "strict", "full") == (2, 5, 4, 0.09, 0.78)
    assert summ("all", "loo", "full") == (5, 6, 5, 0.09, 0.78)
    assert summ("all", "strict", "no_ports") == (2, 6, 3, 0.09, 2.13)
    assert summ("benign", "strict", "full") == (2, 5, 4, 0.09, 0.57)
    assert summ("benign", "loo", "full")[:3] == (5, 6, 6)
    assert summ("benign", "strict", "no_ports")[:3] == (5, 7, 6)
    bad = 0
    for p in glob.glob(str(R6 / "var_*_*_*[0-9].csv")):
        bad += sum(r["admissible"] != "True" for r in csv.DictReader(open(p)))
    assert bad == 6
    for neg in ("all", "benign"):
        for r in csv.DictReader(open(R6 / f"peer_disjoint_{neg}.csv")):
            assert r["calibration"] == "oof5" and r["grid"] == "quantile"
            assert int(r["at_floor"]) == 0 and int(r["fired"]) == 0


def test_rev14_pipeline_profiles():
    rows = _rows("pipeline_profiles.csv")
    def m(corpus, prof):
        v = [float(r["floor_mean_phase"]) for r in rows if r["corpus"] == corpus and r["profile"] == prof]
        return round(100 * st.mean(v), 1), round(100 * st.median(v), 1)
    assert m("dapt2021", "netflow_onprem") == (15.5, 3.4)
    assert m("dapt2021", "zeek_conn") == (15.0, 5.2)
    assert m("dapt2021", "cloud_600")[0] == 19.5 and m("dapt2021", "cloud_60")[0] == 15.6
    assert m("cic", "netflow_onprem")[0] == 58.0 and m("cic", "dns_query_log")[0] == 47.8
    assert m("cic", "cloud_600")[0] == 71.9 and m("dns_v42", "dns_query_log")[0] == 54.6
    assert "NetFlow on premises & 15.5 (3.4) & 58.0 (74.2) & 65.6 (97.0)" in TEX
    assert "& 15.0 (5.2) & 52.9 (55.7) & 71.7 (100)" in TEX


def test_rev14_cic_prefix_check():
    import ast
    ps = _rows("cic_dns_split/cic_parse_stats.csv")
    match = mismatch = 0
    for p in ps:
        counts = [m + 1 for m in ast.literal_eval(p["dedup_max_index_per_transfer"])]
        inits = ast.literal_eval(p["dedup_init_decoded_per_transfer"])
        for c, lab in zip(counts, inits):
            digits = lab.split("|", 1)[1] if "|" in lab else ""
            if not digits.isdigit():
                continue
            if str(c).startswith(digits):
                match += 1
            else:
                mismatch += 1
    assert (match, mismatch) == (14, 1)


def test_rev14_logistic_regression_tight_tolerance():
    r = {x["model"]: x for x in _rows("real_multimodel_summary_wilson.csv")}["Logistic Regression"]
    assert round(float(r["f1"]), 3) == 0.139 and round(100 * float(r["fpb_heldout"]), 2) == 1.11
    assert "Solved to a tolerance of $10^{-10}$" in TEX


def test_round7_loo_own_operating_points():
    """Leave-one-out models at their own out-of-fold operating points (run_round7.sh)."""
    import glob
    R7 = R / "round7"
    def single(neg):
        return {r["variant"]: r for r in csv.DictReader(open(R7 / f"loo_cal_{neg}.csv"))}
    a, b = single("all"), single("benign")
    pct = lambda x: round(100 * float(x), 2)
    assert (int(a["full"]["at_floor"]), int(a["full"]["at_floor_0pp"])) == (5, 3)
    assert (pct(a["full"]["loo_fpb_min"]), pct(a["full"]["loo_fpb_max"])) == (0.33, 0.60)
    assert (int(a["no_ports"]["at_floor"]), int(a["no_ports"]["at_floor_0pp"])) == (6, 4)
    assert (pct(a["no_ports"]["loo_fpb_min"]), pct(a["no_ports"]["loo_fpb_max"])) == (0.18, 0.63)
    assert (int(a["ports_only"]["at_floor"]), int(a["ports_only"]["at_floor_0pp"])) == (2, 2)
    assert pct(a["ports_only"]["loo_fpb_max"]) == 0.15
    assert (int(b["full"]["at_floor"]), int(b["full"]["at_floor_0pp"])) == (5, 3)
    assert (pct(b["full"]["loo_fpb_min"]), pct(b["full"]["loo_fpb_max"])) == (0.30, 0.66)
    assert (int(b["no_ports"]["at_floor"]), int(b["no_ports"]["at_floor_0pp"])) == (5, 3)
    assert (pct(b["no_ports"]["loo_fpb_min"]), pct(b["no_ports"]["loo_fpb_max"])) == (0.39, 0.90)
    assert all(int(r["loo_admissible"]) == 7 for r in list(a.values()) + list(b.values()))
    rows = [list(csv.DictReader(open(p)))[0] for p in sorted(glob.glob(str(R7 / "var_loo_cal_all_*[0-9].csv")))]
    assert len(rows) == 10
    af = sorted(int(r["at_floor"]) for r in rows)
    assert af == [5] * 9 + [6]
    assert pct(min(float(r["loo_fpb_min"]) for r in rows)) == 0.06
    assert pct(max(float(r["loo_fpb_max"]) for r in rows)) == 1.53
    assert sum(int(r["loo_admissible"]) for r in rows) == 64
    ep = [r for r in csv.DictReader(open(R7 / "loo_cal_all_episodes.csv"))
          if r["variant"] == "full" and "1624665799" in r["episode_id"]][0]
    assert round(100 * float(ep["ebd"]), 2) == 46.09 and round(float(ep["threshold"]), 3) == 0.569
    assert "EDXGB features & 5 (3) & 5 (3) & 0.45\\% & 0.33--0.60\\%" in TEX
    assert "port fields only & 2 (2) & 2 (2) & 0.00\\% & 0.00--0.15\\%" in TEX


def test_rev15_profiles_at_gap_300_and_attain_table():
    rows = _rows("pipeline_profiles_gap.csv")
    def m(G, prof):
        v = [float(r["floor_mean_phase"]) for r in rows if float(r["gap_s"]) == G and r["profile"] == prof]
        assert len(v) == 7
        return 100 * st.mean(v), 100 * st.median(v)
    means = [m(300.0, p)[0] for p in ("netflow_onprem", "zeek_conn", "cloud_600", "cloud_60")]
    meds = [m(300.0, p)[1] for p in ("netflow_onprem", "zeek_conn", "cloud_600", "cloud_60")]
    assert round(min(means), 1) == 8.9 and round(max(means), 1) == 19.4
    assert round(min(meds), 1) == 3.4 and round(max(meds), 1) == 8.1
    assert "means are $8.9\\%$ to $19.4\\%$" in TEX
    ep = {r["episode_id"]: r for r in csv.DictReader(open(R / "round7/loo_cal_all_episodes.csv"))
          if r["variant"] == "full"}
    e3 = [r for k, r in ep.items() if "1624665799" in k][0]
    assert round(100 * float(e3["gap"]), 2) == 45.68
    assert "443 \\#3 & 0.42\\% & 0.42\\% & \\textbf{+0.00} & 46.09\\% & +45.68" in TEX


def test_rev15_lab_pipeline_delays():
    rows = [r for r in csv.DictReader(open(R / "lab_pipeline/delays.csv")) if r["found"] == "True"]
    assert len({r["id"] for r in rows}) == 38 and len(rows) == 114
    f = float
    def med(prof, labels, key):
        v = [f(r[key]) for r in rows if r["pipeline"] == prof and r["label"] in labels and r[key] != ""]
        return round(st.median(v), 1), round(min(v), 1), round(max(v), 1)
    tcp = ("tcp_short", "tcp_med", "tcp_chunk", "tcp_long")
    udp = ("udp_single", "udp_rotate")
    assert med("e2", tcp, "release_after_flow_end") == (1.6, 1.0, 2.5)
    assert med("e2", udp, "release_after_flow_end") == (15.7, 15.4, 16.4)
    assert med("e1", tcp, "release_after_flow_end") == (314.7, 301.4, 359.4)
    assert med("e1", udp, "release_after_flow_end") == (330.2, 301.4, 359.5)
    assert med("zeek", tcp, "release_after_flow_end") == (7.8, 5.6, 15.2)
    assert med("zeek", udp, "release_after_flow_end") == (64.9, 61.1, 68.8)
    w = [f(r["alert_ing_derived"]) - f(r["ingest"]) for r in rows]
    assert (round(st.median(w), 1), round(min(w), 1), round(max(w), 1)) == (172.6, 21.3, 319.7)
    assert sum(r["missed_flowstart"] == "True" for r in rows if r["pipeline"] == "zeek") == 14
    e2 = lambda lab, k: round(100 * st.mean(f(r[k]) for r in rows if r["pipeline"] == "e2" and r["label"] == lab), 1)
    assert e2("tcp_chunk", "floor_alert_ing_derived") == 92.9 and e2("udp_rotate", "floor_alert_ing_derived") == 71.2
    m = list(csv.DictReader(open(R / "pipeline_profiles_measured.csv")))
    mean = lambda c, p: round(100 * st.mean(float(r["floor_mean_phase"]) for r in m if r["corpus"] == c and r["profile"] == p), 1)
    assert mean("dapt2021", "netflow_onprem") == 15.2 and mean("dapt2021", "zeek_conn") == 14.2
    assert mean("dapt2021", "softflowd_default") == 24.0 and mean("cic", "softflowd_default") == 66.4
    assert "softflowd, defaults & release, TCP closed by FIN & -- & $314.7$" in TEX
