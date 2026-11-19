#!/usr/bin/env bash
# Runs every experiment of the revised manuscript that needs XGBoost and nfstream
# (see src/ebd/experiments/revision_runs.py for the list), on this machine.
#
#   bash scripts/run_revision.sh
#
# 1. syncs the project environment, with the optional nfstream extra used by the
#    Sec. 9 captures (needs internet; if it fails the run continues and the
#    capture-based parts report themselves as failed);
# 2. runs ebd.experiments.revision_runs;
# 3. leaves a full log in results/logs/ and a provenance record (versions, git
#    commit, timings, SHA-256 of every output) in results/manifests/, indexed
#    per part in results/revision_runs_manifest.json.
#
# Extra arguments go to the driver, e.g. to re-run only some parts:
#
#   bash scripts/run_revision.sh --only fast-wilson fast-point fast-netem escape
set -u
cd "$(dirname "$0")/.."
mkdir -p results/logs figures
LOG="results/logs/revision_runs_$(date -u +%Y%m%dT%H%M%SZ).log"
{
  echo "== revision runs started $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(uname -srm)"
  if command -v uv >/dev/null 2>&1; then
    if uv sync --extra testbed; then
      echo "== environment synced, with the testbed extra (nfstream)"
      uv run --no-sync python -c "import nfstream; print('== nfstream', nfstream.__version__, 'imports')" \
        || echo "!! nfstream is installed but does not import; Sec. 9 parts will fail"
    elif uv sync; then
      echo "!! testbed extra (nfstream) could not be installed; Sec. 9 parts will fail"
    else
      echo "!! uv sync failed; running with the environment as it is"
    fi
    uv run --no-sync python -m ebd.experiments.revision_runs "$@"
  else
    echo "!! uv not found; running with .venv/bin/python as it is"
    .venv/bin/python -m ebd.experiments.revision_runs "$@"
  fi
  echo "== finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$LOG"
