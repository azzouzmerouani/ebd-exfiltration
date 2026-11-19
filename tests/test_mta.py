"""The real-malware episodes (Table tab:mta, Supplementary S7) and the numbers quoted from them."""
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/mta/mta_episodes.csv"
if not CSV.exists():
    pytest.skip("data/mta/ is not distributed until the capture publisher confirms its terms (see data/mta/README.md)", allow_module_level=True)
TEX = re.sub(r"\s+", " ", (ROOT / "paper/main.tex").read_text())
TEXN = re.sub(r"(?<=&) +| +(?=&)", " ", TEX)
SCHED = ("Cisco-like NetFlow", "Zeek", "softflowd defaults")


def _rows():
    return list(csv.DictReader(CSV.open()))


def _fmt(v):
    v = float(v)
    return "100" if v == 100 else f"{v:.1f}"


def test_reproducible():
    before = CSV.read_bytes()
    subprocess.run([sys.executable, "-m", "ebd.experiments.mta_episodes"], cwd=ROOT, check=True,
                   env={"PYTHONPATH": str(ROOT / "src")}, capture_output=True)
    assert CSV.read_bytes() == before


def test_table_rows_match_csv():
    for r in _rows():
        kb = f"{int(r['V_bytes']) / 1000:,.1f}".replace(",", "{,}")
        l0 = "100" if float(r["floor_L0"]) == 100 else f"{float(r['floor_L0']):.2f}"
        m = float(r["margin_s"])
        ms = f"$-{abs(m):.2f}$" if m < 0 else f"{m:.2f}"
        ex = r["floor_L0_excl_first"]
        exs = "--" if ex == "" else ("100" if float(ex) == 100 else f"{float(ex):.2f}")
        cells = [r["conns"], kb, f"{float(r['D_s']):.2f}", f"{float(r['T_eff_s']):.2f}", ms, l0, exs,
                 _fmt(r["largest_conn_share"])]
        assert ("& " + " & ".join(cells) + r" \\") in TEXN, (r["family"], cells)


def test_period_table_and_quoted_numbers():
    rows = {(r["family"], r["channel"]): r for r in _rows()}
    lum, amos = rows[("Lumma", "HTTP")], rows[("AMOS (macOS)", "HTTP")]
    for tag in ("netflow", "zeek"):
        cells = [_fmt(lum[f"{tag}_p{p}"]) for p in (0, 10, 30, 60, 300, 900)]
        assert (" & ".join(cells) + r" \\") in TEXN, (tag, cells)
    assert all(amos[f"netflow_p{p}"] == amos[f"zeek_p{p}"] for p in (0, 60, 300, 900))
    assert ("& " + " & ".join(_fmt(amos[f"netflow_p{p}"]) for p in (0, 60, 300, 900)) + r" \\") in TEXN
    others = [r for k, r in rows.items() if k[0] not in ("Lumma", "AMOS (macOS)")]
    assert all(float(r[f"{t}_p{p}"]) == 100 for r in others for t in ("netflow", "zeek", "softflowd")
               for p in (0, 10, 20, 30, 60, 300, 900))
    assert {k[0] for k, r in rows.items() if float(r["T_eff_s"]) >= float(r["D_s"])} == {"AgentTesla", "RedLine", "StealC"}
    snake = sorted(float(r["margin_s"]) for k, r in rows.items() if k[0] == "Snake KeyLogger")
    assert snake == [1.31, 2.46] and "$1.3$ to $2.5$\\,s" in TEX
    assert (lum["netflow_p300_min"], amos["netflow_p300_min"]) == ("2.6", "0.2")
    assert (lum["floor_L0_excl_first"], amos["floor_L0_excl_first"]) == ("0.23", "0.25")
    assert all(r["floor_L0_excl_first"] == "100.0" for k, r in rows.items() if k[0] == "Snake KeyLogger")
    assert round(100 * float(lum["T_eff_s"]) / float(lum["D_s"]), 1) == 17.7 and "$17.7\\%$" in TEX
    at = rows[("AgentTesla", "FTP")]
    assert round(100 * (int(at["V_bytes"]) - int(at["first_conn_bytes"])) / int(at["V_bytes"])) == 35
    assert "at least $79\\%$" not in TEX and "five are lost in full" not in TEX


def test_sweep_matches_paper():
    rows = list(csv.DictReader((ROOT / "results/pipeline_sweep.csv").open()))
    get = lambda c, rel, p, w=0, o=6.4: [r for r in rows if r["corpus"] == c and r["release"] == rel and
                                         int(r["period_s"]) == p and int(r["window_s"]) == w and
                                         float(r["overhead_s"]) == (2.0 if p == 0 else o)][0]
    for c, rel, lab in (("dapt2021", "netflow", "NetFlow"), ("dapt2021", "zeek", "Zeek"),
                        ("dapt2021", "softflowd_default", "softflowd"), ("cic", "netflow", "NetFlow"),
                        ("cic", "zeek", "Zeek"), ("cic", "softflowd_default", "softflowd")):
        cells = [f"{float(get(c, rel, p)['mean']):.1f} ({float(get(c, rel, p)['median']):.1f})" for p in (0, 10, 30, 60, 300, 900)]
        assert (" & ".join(cells)) in TEXN, (c, rel, cells)
    # rev17 values reproduced with files and the 25 s bound
    assert float(get("dapt2021", "netflow", 300, 300, 25.0)["mean"]) == pytest.approx(15.2, abs=0.05)
    assert float(get("dapt2021", "zeek", 300, 0, 25.0)["mean"]) == pytest.approx(14.2, abs=0.05)
    sd = get("dapt2021", "softflowd_default", 300); fw = get("dapt2021", "softflowd_fin_wait", 300)
    idle = get("dapt2021", "softflowd_idle", 300)
    assert (round(float(fw["median"]), 1), round(float(idle["median"]), 1), round(float(sd["median"]), 1)) == (3.2, 30.9, 30.9)
    assert round(float(sd["mean"]), 1) == 23.4 and "$31\\%$" in TEX


def test_lab_observed_alerts():
    rows = [r for r in csv.DictReader((ROOT / "results/lab_pipeline/delays.csv").open()) if r["found"] == "True"]
    assert len(rows) == 114
    obs = [r for r in rows if r["alert_ingested"]]
    d = [abs(float(r["alert_ing_observed_minus_derived"])) for r in obs]
    assert (sum(x <= 25 for x in d), sum(x > 25 for x in d), len(rows) - len(obs)) == (52, 30, 32)
    for pipe, val in (("e2", 74.1), ("zeek", 75.9)):
        sub = [float(r["floor_alert_ingested"]) for r in rows if r["pipeline"] == pipe and r["label"] == "udp_rotate"]
        assert 100 * sum(sub) / len(sub) == pytest.approx(val, abs=0.05)
    ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    runs = [json.loads(l) for l in (ROOT / "results/lab_pipeline/rule_runs.jsonl").open()]
    ing = sorted(ts(r["start"]) - ts(r["sched"]) for r in runs if r["rule"] == "lab-exfil-peer-ingested")
    assert round((ing[11] + ing[12]) / 2, 1) == 1.3 and len(ing) == 24


def test_lab_overhead_direct():
    import json
    from datetime import datetime
    ts = lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()
    runs = sorted((ts(r["sched"]), ts(r["start"])) for r in map(json.loads, (ROOT / "results/lab_pipeline/rule_runs.jsonl").open())
                  if r["rule"] == "lab-exfil-peer-ingested")
    d = [r for r in csv.DictReader((ROOT / "results/lab_pipeline/delays.csv").open())
         if r["found"] == "True" and r["alert_ing_observed_minus_derived"] and abs(float(r["alert_ing_observed_minus_derived"])) <= 25]
    out = []
    for r in d:
        t0 = float(r["t_start"]); ing = t0 + float(r["ingest"])
        nxt = [s for s, st in runs if st >= ing + 1.0][0]
        out.append(t0 + float(r["alert_ingested"]) - (t0 + float(r["release"])) - (nxt - ing))
    out.sort()
    assert len(out) == 52
    assert round(sum(out) / len(out), 1) == 6.4 and round((out[25] + out[26]) / 2, 1) == 4.5
    assert (round(out[0], 1), round(out[-1], 1)) == (1.5, 16.6)


def test_no_info_column_and_no_bot_ids():
    for p in (ROOT / "data/mta").glob("*-packets.csv"):
        text = p.read_text()
        assert "Info" not in text.splitlines()[0]
    amos = list(csv.DictReader((ROOT / "data/mta/2026-09-10-amos-packets.csv").open()))
    assert {r["http_request"] for r in amos if r["http_request"]} == {"GET /api", "POST /api", "POST /contact"}


def test_port3306_flows_carry_no_fin_or_rst():
    """The softflowd 31% median rests on the two port-3306 episodes: per-flow
    release changes nothing because none of their flows has FIN or RST."""
    from ebd.experiments.cadence_sweep import load_flows
    from ebd.experiments.pipeline_profiles import _fkey, _flow_flags
    from ebd.realdata.episodes import group_into_episodes
    import json
    gt = {json.loads(l)["episode_id"] for l in (ROOT / "data/realdata/groundtruth.jsonl").open() if l.strip()}
    flags = _flow_flags(ROOT)
    eps = [e for e in group_into_episodes(load_flows(ROOT / "data/realdata/exfil_flows.csv"))
           if e.episode_id in gt and e.dst_port == 3306]
    assert len(eps) == 2 and sum(len(e.flows) for e in eps) == 300
    assert all(flags[_fkey(f)] == (0, 0) for e in eps for f in e.flows)
    assert "six long-lived MySQL connections captured in one direction" in TEX


def test_lab_overhead_script():
    out = subprocess.run([sys.executable, str(ROOT / "scripts/lab_overhead.py")], capture_output=True,
                         text=True, check=True).stdout
    assert "mean 6.4 s" in out and "4.3-8.9 s" in out and "13 runs" in out
    assert "$4.3$ to $8.9$\\,s" in TEX
