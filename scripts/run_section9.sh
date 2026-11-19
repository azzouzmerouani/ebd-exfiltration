#!/usr/bin/env bash
# Sec. 9 (the D <= T regime, netem, escape) on a machine where the project's own
# nfstream does not import -- in particular an Intel Mac, where nfstream's macOS
# x86_64 wheels for CPython 3.9+ ship without the nDPI library they link against.
#
#   bash scripts/run_section9.sh
#
# 1. builds a separate CPython 3.8 environment (.venv-nfstream) holding only
#    nfstream 6.5.4, whose 3.8 wheel is complete (needs internet once);
# 2. exports the three captures to flow records with it, into
#    data/testbed*/exports/ (scripts/nfstream_export.py);
# 3. runs the Sec. 9 parts of the revision in the project environment, which
#    read those exports instead of calling nfstream (src/ebd/testbed/export.py).
set -u
set -o pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
LOG="results/logs/section9_exports_$(date -u +%Y%m%dT%H%M%SZ).log"
{
  echo "== Sec. 9 exports started $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(uname -srm)"
  if [ ! -x .venv-nfstream/bin/python ]; then
    uv venv --python 3.8 .venv-nfstream || exit 1
  fi
  uv pip install --python .venv-nfstream/bin/python "nfstream==6.5.4" || exit 1
  .venv-nfstream/bin/python -c "import nfstream, platform; print('== nfstream', nfstream.__version__, 'imports under Python', platform.python_version())" \
    || { echo "!! nfstream does not import under Python 3.8 either"; exit 1; }
  .venv-nfstream/bin/python scripts/nfstream_export.py \
      data/testbed data/testbed_escape data/testbed_netem --timeouts 5 15 60 || exit 1
  echo "== exports done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$LOG" || { echo "!! the export step failed; see $LOG"; exit 1; }

bash scripts/run_revision.sh --only fast-wilson fast-point fast-netem escape
