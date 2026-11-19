"""Every experiment of the revised manuscript that needs XGBoost and nfstream.

    bash scripts/run_revision.sh          # syncs the environment, then runs this
    uv run python -m ebd.experiments.revision_runs

Runs each part in turn, keeps going if one fails, and records library
versions, the git commit, the status and wall time of every part, and the
SHA-256 of every file it wrote, so the numbers quoted in the paper can be
traced to the run that produced them. Each invocation writes its own record,
``results/manifests/revision_runs_<UTC start>.json``; the index
``results/revision_runs_manifest.json`` keeps, for every part, the latest
record together with the run it came from, so a partial re-run (``--only``)
never erases the provenance of the parts it did not touch.

Parts
-----
edxgb-wilson     Table 2 under the a-priori Wilson operating-point rule
                 (strict and leave-one-out), xgboost single-threaded.
edxgb-point      Table 2's archived point rule, single-threaded, so the effect
                 of the rule can be separated from the thread count.
multimodel       Sec. 7.8's four detector families under the Wilson rule.
session          Sec. 7.9's session-granularity detector.
fast-wilson      Sec. 9 under the Wilson rule, with held-out AUC (M1) and the
                 10x-asset deviations (M4).
fast-point       Sec. 9 under the archived rule, as a reproduction check.
fast-netem       Sec. 9 on the capture recorded under netem (M5), if present.
escape           Sec. 9.4 under the Wilson rule, with every incremental setting
                 written out so the feasible set can be reported (DA-3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from importlib import import_module
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _version(mod: str) -> str:
    try:
        return str(getattr(import_module(mod), "__version__", "unknown"))
    except Exception as exc:                     # noqa: BLE001 - report, never fail
        return f"unavailable ({type(exc).__name__})"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _parts(root: Path) -> list[tuple[str, str, list[str], list[str]]]:
    r, f, d = root / "results", root / "figures", root / "data"
    train = str(d / "realdata/train_flows.csv")
    gt = str(d / "realdata/groundtruth.jsonl")
    real = ["--train", train, "--groundtruth", gt]
    parts = [
        ("edxgb-wilson-strict", "ebd.experiments.real_edxgb",
         real + ["--rule", "wilson", "--n-jobs", "1"],
         [str(r / "real_edxgb_gap_strict_wilson.csv"), str(r / "real_edxgb_detail_strict_wilson.json")]),
        ("edxgb-wilson-loo", "ebd.experiments.real_edxgb",
         real + ["--rule", "wilson", "--n-jobs", "1", "--loo"],
         [str(r / "real_edxgb_gap_loo_wilson.csv"), str(r / "real_edxgb_detail_loo_wilson.json")]),
        ("edxgb-point-strict", "ebd.experiments.real_edxgb",
         real + ["--rule", "point", "--n-jobs", "1", "--suffix", "_point_1thread"],
         [str(r / "real_edxgb_gap_strict_point_1thread.csv"),
          str(r / "real_edxgb_detail_strict_point_1thread.json")]),
        ("edxgb-point-loo", "ebd.experiments.real_edxgb",
         real + ["--rule", "point", "--n-jobs", "1", "--loo", "--suffix", "_point_1thread"],
         [str(r / "real_edxgb_gap_loo_point_1thread.csv"),
          str(r / "real_edxgb_detail_loo_point_1thread.json")]),
        ("multimodel", "ebd.experiments.real_multimodel",
         real + ["--rule", "wilson"],
         [str(r / f"real_multimodel_{x}_wilson.csv")
          for x in ("summary", "per_episode", "maxscores", "sensitivity", "sweep")]
         + [str(r / "real_multimodel_h1_wilson.txt")]),
        ("session", "ebd.experiments.session_detector",
         real + ["--rule", "wilson"],
         [str(r / "session_detector.csv"), str(r / "session_detector_episodes.csv"),
          str(f / "session_detector.png")]),
        ("fast-wilson", "ebd.experiments.fast_regime",
         ["--rule", "wilson", "--skip-escape",
          "--out", str(r / "fast_regime_wilson.csv"),
          "--meta-out", str(r / "fast_regime_wilson_meta.json"),
          "--figure", str(f / "fast_regime_wilson.png")],
         [str(r / "fast_regime_wilson.csv"), str(r / "fast_regime_wilson_meta.json"),
          str(f / "fast_regime_wilson.png")]),
        ("fast-point", "ebd.experiments.fast_regime",
         ["--rule", "point", "--skip-escape",
          "--out", str(r / "fast_regime_point_check.csv"),
          "--meta-out", str(r / "fast_regime_point_meta.json"),
          "--figure", str(f / "fast_regime_point_check.png")],
         [str(r / "fast_regime_point_check.csv"), str(r / "fast_regime_point_meta.json")]),
    ]
    if (d / "testbed_netem/manifest.json").exists():
        parts.append(
            ("fast-netem", "ebd.experiments.fast_regime",
             ["--testbed", str(d / "testbed_netem"), "--rule", "wilson", "--skip-escape",
              "--exclude-unreconciled",
              "--out", str(r / "fast_regime_netem.csv"),
              "--meta-out", str(r / "fast_regime_netem_meta.json"),
              "--figure", str(f / "fast_regime_netem.png")],
             [str(r / "fast_regime_netem.csv"), str(r / "fast_regime_netem_meta.json"),
              str(f / "fast_regime_netem.png")]))
    parts.append(
        ("escape", "ebd.experiments.escape",
         ["--rule", "wilson", "--out", str(r / "escape_wilson.csv"),
          "--sweep-out", str(r / "escape_sweep.csv"),
          "--figure", str(f / "escape_wilson.png")],
         [str(r / "escape_wilson.csv"), str(r / "escape_sweep.csv"),
          str(f / "escape_wilson.png")]))
    return parts


_RUN_KEYS = ("run_id", "started_utc", "finished_utc", "platform", "machine", "python",
             "versions", "git_commit", "git_dirty")


def _update_index(index_path: Path, mdir: Path, manifest: dict) -> None:
    """Merge one run's part records into the per-part provenance index.

    A manifest written by an earlier version of this module (one run, no
    ``parts_latest``) is first preserved under ``results/manifests/`` and then
    folded in, so nothing it recorded is lost.
    """
    index = {"runs": [], "parts_latest": {}}
    if index_path.exists():
        try:
            old = json.loads(index_path.read_text())
        except Exception:                        # noqa: BLE001 - keep a copy, start fresh
            old = None
            index_path.replace(mdir / "revision_runs_manifest.unreadable.json")
        if old and "parts_latest" in old:
            index = old
        elif old and "parts" in old:
            legacy_id = old.get("run_id") or str(old.get("started_utc", "legacy")).replace(
                "-", "").replace(":", "")
            old.setdefault("run_id", legacy_id)
            legacy = mdir / f"revision_runs_{legacy_id}.json"
            if not legacy.exists():
                legacy.write_text(json.dumps(old, indent=1))
            _merge(index, old)
    _merge(index, manifest)
    index_path.write_text(json.dumps(index, indent=1))


def _merge(index: dict, run: dict) -> None:
    rid = run.get("run_id")
    if rid and rid not in index["runs"]:
        index["runs"].append(rid)
    ctx = {k: run.get(k) for k in _RUN_KEYS}
    for part in run.get("parts", []):
        index["parts_latest"][part["name"]] = {**part, "run": ctx}


def main(argv: list[str] | None = None) -> int:
    root = _root()
    ap = argparse.ArgumentParser(description="Run every experiment the revision needs")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these parts (names as in the module docstring)")
    args = ap.parse_args(argv)
    (root / "results").mkdir(exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)

    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                    capture_output=True, text=True, timeout=20).stdout.strip())
    except Exception:                            # noqa: BLE001
        commit, dirty = "unknown", None

    manifest = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version.split()[0],
        "versions": {m: _version(m) for m in
                     ("numpy", "pandas", "sklearn", "xgboost", "scipy", "matplotlib",
                      "nfstream", "dpkt")},
        "git_commit": commit, "git_dirty": dirty, "parts": [],
    }
    print(json.dumps({k: manifest[k] for k in ("platform", "python", "versions",
                                                "git_commit")}, indent=1))

    run_id = manifest["started_utc"].replace("-", "").replace(":", "")
    manifest["run_id"] = run_id
    manifest["only"] = args.only
    mdir = root / "results" / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    run_path = mdir / f"revision_runs_{run_id}.json"
    index_path = root / "results" / "revision_runs_manifest.json"

    def _write() -> None:
        run_path.write_text(json.dumps(manifest, indent=1))
        _update_index(index_path, mdir, manifest)

    for name, module, argv_part, outputs in _parts(root):
        if args.only and name not in args.only:
            continue
        print("\n" + "#" * 78 + f"\n# {name}: python -m {module} {' '.join(argv_part)}\n"
              + "#" * 78, flush=True)
        t0 = time.time()
        status, error = "ok", ""
        try:
            rc = import_module(module).main(argv_part)
            if rc not in (0, None):
                status = f"exit {rc}"
        except SystemExit as exc:
            status = "ok" if exc.code in (0, None) else f"exit {exc.code}"
        except Exception as exc:                 # noqa: BLE001 - keep going
            status, error = "failed", "".join(traceback.format_exception(exc))[-4000:]
            print(error, flush=True)
        files = {}
        for o in outputs:
            p = Path(o)
            if p.exists():
                files[str(p.relative_to(root))] = _sha256(p)
        manifest["parts"].append({"name": name, "module": module, "argv": argv_part,
                                  "status": status, "seconds": round(time.time() - t0, 1),
                                  "outputs": files, "error": error})
        print(f"# {name}: {status} in {time.time() - t0:.0f}s", flush=True)
        _write()

    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write()
    print(f"\nprovenance: {run_path.relative_to(root)} "
          f"(index: {index_path.relative_to(root)})")
    print("\nsummary:")
    for p in manifest["parts"]:
        print(f"  {p['name']:<22} {p['status']:<8} {p['seconds']:>7.0f}s  "
              f"{len(p['outputs'])} files")
    return 0 if all(p["status"] == "ok" for p in manifest["parts"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
