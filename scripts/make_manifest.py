"""Write results/MANIFEST.sha256: the SHA-256 of every result file in results/
(test-only *_SHIMTEST* outputs excluded) and of the real-malware tables in
data/mta/, sorted by path. Run after any rerun.

    python3 scripts/make_manifest.py
"""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
res = root / "results"
lines = []
for p in sorted(list(res.rglob("*")) + list((root / "data/mta").glob("*.csv"))):
    if p.is_file() and "SHIMTEST" not in p.name and p.name != "MANIFEST.sha256":
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}")
(res / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
print(f"{len(lines)} files hashed")
