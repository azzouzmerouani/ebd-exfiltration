"""Provenance index of ebd.experiments.revision_runs: partial re-runs keep history."""

from __future__ import annotations

import json

from ebd.experiments.revision_runs import _update_index


def _run(run_id, parts, commit="c1"):
    return {"run_id": run_id, "started_utc": run_id, "platform": "p", "machine": "m",
            "python": "3.12", "versions": {"nfstream": "x"}, "git_commit": commit,
            "git_dirty": False,
            "parts": [{"name": n, "status": s, "outputs": {}, "error": ""} for n, s in parts]}


def test_legacy_manifest_is_preserved_then_folded_in(tmp_path):
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    index = tmp_path / "revision_runs_manifest.json"
    legacy = _run(None, [("session", "ok"), ("fast-wilson", "failed")])
    legacy.pop("run_id")
    legacy["started_utc"] = "2026-10-10T15:37:01Z"
    index.write_text(json.dumps(legacy))

    _update_index(index, mdir, _run("20261010T170000Z", [("fast-wilson", "ok")], "c2"))

    kept = json.loads((mdir / "revision_runs_20261010T153701Z.json").read_text())
    assert [p["name"] for p in kept["parts"]] == ["session", "fast-wilson"]
    idx = json.loads(index.read_text())
    assert idx["runs"] == ["20261010T153701Z", "20261010T170000Z"]
    assert idx["parts_latest"]["session"]["run"]["git_commit"] == "c1"
    assert idx["parts_latest"]["fast-wilson"]["status"] == "ok"
    assert idx["parts_latest"]["fast-wilson"]["run"]["git_commit"] == "c2"


def test_repeated_writes_of_one_run_are_idempotent(tmp_path):
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    index = tmp_path / "idx.json"
    run = _run("r1", [("a", "ok")])
    _update_index(index, mdir, run)
    run["parts"].append({"name": "b", "status": "ok", "outputs": {}, "error": ""})
    _update_index(index, mdir, run)
    idx = json.loads(index.read_text())
    assert idx["runs"] == ["r1"] and set(idx["parts_latest"]) == {"a", "b"}
