"""Export the testbed capture to flow records at several active timeouts.

Uses nfstream -- the same exporter DAPT2021 was built with -- so the record
semantics in the testbed and in the real-data section are identical, and the
active timeout is a real exporter setting rather than a model of one.

The record availability instant
-------------------------------
A record covering packets up to ``bidirectional_last_seen_ms`` cannot be emitted
before that instant, whether it was closed by FIN/RST or chopped by the active
timeout. That instant is therefore the earliest a consumer could score it, and
it is what both the floor and the measured EBD use. It is a *lower bound* on the
real availability time -- a collector adds export and transport delay on top --
so every number here is generous to the detector.

Exported records may come from a cache
---------------------------------------
``scripts/nfstream_export.py`` runs nfstream once per capture and timeout and
writes the records to ``<testbed>/exports/nfstream_a<T>_i<I>.jsonl.gz``, with
the exporter's version and the capture's SHA-256 in the first line. When such a
file exists for the requested timeouts, :func:`export` reads it instead of
calling nfstream, after checking that the capture is the one it was made from.
This lets the analysis run where nfstream does not install (nfstream's macOS
x86_64 wheels for CPython 3.9 and later ship without the nDPI library they link
against), with the export made by the same nfstream release in a separate
environment. :func:`export_provenance` reports which source each export came
from.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# Active timeouts to export at. Short by deployment standards on purpose: the
# bound depends on T and D only through their ratio, so short timeouts paired
# with short transfers cover the same ground as production timeouts paired with
# transfers lasting hours (see ebd.testbed.traffic).
DEFAULT_ACTIVE_TIMEOUTS: tuple[float, ...] = (5.0, 15.0, 60.0)

# Fields the analysis needs, plus everything EDXGB's pipeline consumes.
_KEEP = (
    "src_ip", "src_port", "dst_ip", "dst_port", "protocol",
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "bidirectional_duration_ms", "bidirectional_packets", "bidirectional_bytes",
    "src2dst_packets", "src2dst_bytes", "dst2src_packets", "dst2src_bytes",
    "src2dst_duration_ms", "dst2src_duration_ms",
    "bidirectional_min_ps", "bidirectional_mean_ps", "bidirectional_max_ps",
    "bidirectional_min_piat_ms", "bidirectional_mean_piat_ms",
    "bidirectional_max_piat_ms",
)


@dataclass(frozen=True)
class Record:
    """One exported flow record, reduced to what the analysis uses."""

    src_port: int
    dst_port: int
    first_seen: float          # epoch seconds
    last_seen: float           # epoch seconds -- the availability instant
    fields: dict

    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen


_PROVENANCE: list[dict] = []


def cache_path(pcap: Path, active_timeout: float, idle_timeout: float) -> Path:
    """Where ``scripts/nfstream_export.py`` writes this capture's export."""
    return (Path(pcap).parent / "exports"
            / f"nfstream_a{int(active_timeout)}_i{int(idle_timeout)}.jsonl.gz")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _record(d: dict) -> Record:
    return Record(
        src_port=int(d["src_port"]),
        dst_port=int(d["dst_port"]),
        first_seen=float(d["bidirectional_first_seen_ms"]) / 1000.0,
        last_seen=float(d["bidirectional_last_seen_ms"]) / 1000.0,
        fields=d,
    )


def _load_cache(path: Path, pcap: Path, active: float, idle: float) -> list[Record]:
    with gzip.open(path, "rt") as fh:
        lines = fh.read().splitlines()
    meta = json.loads(lines[0])["meta"]
    end = json.loads(lines[-1]).get("end")
    if end is None or end.get("records") != len(lines) - 2:
        raise ValueError(f"{path}: incomplete export (no matching end line)")
    if (int(meta["active_timeout"]), int(meta["idle_timeout"])) != (int(active), int(idle)):
        raise ValueError(f"{path}: made for other timeouts ({meta})")
    digest = _sha256(Path(pcap))
    if meta["pcap_sha256"] != digest:
        raise ValueError(f"{path}: made from a different capture "
                         f"({meta['pcap_sha256'][:12]} != {digest[:12]})")
    recs = []
    for line in lines[1:-1]:
        d = json.loads(line)
        missing = [k for k in _KEEP if k not in d]
        if missing:
            raise ValueError(f"{path}: record lacks {missing}")
        recs.append(_record({k: d[k] for k in _KEEP}))
    _PROVENANCE.append({"pcap": str(pcap), "active_timeout": active, "idle_timeout": idle,
                        "source": "cache", "cache": str(path), "records": len(recs),
                        **{k: meta.get(k) for k in ("nfstream", "python", "platform",
                                                    "created_utc", "pcap_sha256")}})
    return recs


def export_provenance() -> list[dict]:
    """Where every export of this process came from (cache or live nfstream)."""
    return list(_PROVENANCE)


def export(pcap: Path, active_timeout: float,
           idle_timeout: float | None = None, use_cache: bool = True) -> list[Record]:
    """Export ``pcap`` at the given active timeout.

    ``idle_timeout`` defaults to the active timeout, so a flow is never held
    open by idleness longer than the active ceiling -- which keeps the single
    knob in the experiment genuinely single. A cached export made by
    ``scripts/nfstream_export.py`` for the same capture and timeouts is used
    when present (module docstring).
    """
    idle = active_timeout if idle_timeout is None else idle_timeout
    cached = cache_path(pcap, active_timeout, idle)
    if use_cache and cached.exists():
        return _load_cache(cached, Path(pcap), active_timeout, idle)

    import nfstream
    from nfstream import NFStreamer

    streamer = NFStreamer(
        source=str(pcap),
        active_timeout=int(active_timeout),
        idle_timeout=int(idle),
        statistical_analysis=True,
        n_dissections=0,
        accounting_mode=0,          # link-layer bytes, as DAPT2021 was exported
    )

    out: list[Record] = []
    for f in streamer:
        out.append(_record({k: getattr(f, k, 0) for k in _KEEP}))
    _PROVENANCE.append({"pcap": str(pcap), "active_timeout": active_timeout,
                        "idle_timeout": idle, "source": "nfstream",
                        "nfstream": getattr(nfstream, "__version__", "unknown"),
                        "records": len(out)})
    return out
