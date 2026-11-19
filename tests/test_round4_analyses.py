"""Numbers of the round-4 analyses that the manuscript quotes."""
import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def _rows(name):
    return list(csv.DictReader(open(R / name)))


def test_floor_all_465():
    rows = _rows("real_floor_all.csv")
    assert len(rows) == 465
    total = sum(int(r["fwd_bytes"]) for r in rows)
    w = sum(float(r["flow_floor"]) * int(r["fwd_bytes"]) for r in rows) / total
    assert w == pytest.approx(0.0289, abs=5e-4)
    assert sum(float(r["flow_floor"]) >= 0.999 for r in rows) == 371
    big = [r for r in rows if int(r["fwd_bytes"]) >= 1e5]
    assert len(big) == 7
    assert sum(int(r["fwd_bytes"]) for r in big) / total == pytest.approx(0.964, abs=1e-3)
    assert all(r["reconstructed"] == "True" for r in big)


def test_response_sensitivity_reproduces_table_values_at_R0():
    rows = [r for r in _rows("response_sensitivity.csv") if r["R_s"] == "0"]
    assert sum(float(r["floor_plus_R"]) for r in rows) / 7 == pytest.approx(0.0076, abs=1e-4)
    assert sum(float(r["detector_plus_R"]) for r in rows) / 7 == pytest.approx(0.1846, abs=1e-4)
    means = {}
    for r in _rows("response_sensitivity.csv"):
        means.setdefault(r["R_s"], []).append(float(r["floor_plus_R"]))
    assert sum(means["300"]) / 7 == pytest.approx(0.1529, abs=1e-4)
    assert sum(means["3600"]) / 7 == pytest.approx(0.3292, abs=1e-4)
    assert sum(means["14400"]) / 7 == pytest.approx(0.6285, abs=1e-4)


def test_time_vs_bytes_loo_example():
    rows = {r["episode_id"]: r for r in _rows("time_vs_bytes_loo.csv")}
    e = rows["10.1.3.17->10.8.10.84:443@1624665799.794"]
    assert float(e["latency_s"]) == pytest.approx(85.4, abs=0.05)
    assert float(e["nebd"]) == pytest.approx(0.4601, abs=1e-4)


def test_rev11_audit():
    import json
    a = json.loads((R / "rev11_audit.json").read_text())
    assert a["benign_flows_dst_port_3306"] == 0
    assert all(v == 0 for v in a["benign_flows_to_exfil_peers"].values())
    assert a["other_stage_attack_rows_trained_as_negative"] == 25994
    assert a["feature_identical_negatives_total"] == 7429
    assert a["feature_identical_negatives_by_source"]["192.168.0.11|netgw_Flow_labeled.csv|Lateral Movement"] == 3129
    assert a["at_floor_strict"]["0pp"] == 3 and a["at_floor_loo"]["0pp"] == 4
    assert a["at_floor_strict"]["ratio<=2"] == 3 and a["at_floor_loo"]["ratio<=2"] == 4
    assert a["at_floor_under_latency"]["strict@15.0"] == 4
    assert a["benign_record_inflation"]["15"] == pytest.approx(4.043)
    assert a["benign_record_inflation"]["60"] == pytest.approx(1.654)
    ex = a["cadence_by_stratum"]
    assert ex["exfiltration/other"]["both_rc"] + ex["exfiltration/dns"]["both_rc"] == 7
    assert ex["exfiltration/other"]["second_cadence_yes"] + ex["exfiltration/dns"]["second_cadence_yes"] == 4


def test_peer_rule():
    r = _rows("real_peer_rule.csv")[0]
    assert r["at_floor"] == "7" and r["fired"] == "7"
    assert float(r["heldout_fpb"]) == pytest.approx(0.0003, abs=1e-6)
    assert r["heldout_flagged_flows"] == "14"


def test_cic_dns_second_corpus():
    """Second DNS corpus, split into transfers at each init request and
    de-duplicated by chunk index (rev13; scripts/cic_dns_floor.py --dedup --split-init)."""
    import json
    s = json.loads((R / "cic_dns_split" / "cic_summary.json").read_text())
    assert s["captures"] == 12 and s["episodes"] == 28
    assert s["flow"]["mean"] == pytest.approx(0.0753, abs=5e-4)
    assert s["flow"]["median"] == pytest.approx(0.0320, abs=5e-4)
    assert s["flow"]["total_loss"] == 0 and s["session"]["total_loss"] == 28
    assert s["per-query"]["mean"] == pytest.approx(6.35e-4, abs=1e-5)
    assert s["sweep_mean"]["flow idle timeout@120"] == pytest.approx(0.391, abs=5e-4)
    assert s["sweep_total_loss"]["flow idle timeout@120"] == 6
    assert s["flow_by_variant"]["heavy"]["median_duration_s"] / 3600 == pytest.approx(5.54, abs=0.01)
    assert s["flow_by_variant"]["light"]["median_duration_s"] / 60 == pytest.approx(5.2, abs=0.05)
    rows = [r for r in _rows("cic_dns_split/cic_floors.csv") if r["consumer"] == "flow"]
    assert min(int(r["n_ports"]) for r in rows) == 88 and max(int(r["n_ports"]) for r in rows) == 15325
    assert max(abs(float(r["unit_delta_pp"])) for r in rows) == pytest.approx(0.195, abs=1e-3)
    light = [float(r["floor_measured"]) for r in rows if r["variant"] == "light"]
    assert len(light) == 22 and min(light) == pytest.approx(0.0054, abs=1e-4) and max(light) == pytest.approx(0.350, abs=1e-3)
    ps = _rows("cic_dns_split/cic_parse_stats.csv")
    assert sum(int(p["transfers"]) for p in ps) == 28
    assert sum(int(p["dedup_dup"]) for p in ps) == 4
    assert all(int(p["dedup_missing_below_max"]) == 0 for p in ps)

