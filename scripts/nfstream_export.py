#!/usr/bin/env python3
"""Export a testbed capture to nfstream flow records, once, into the cache that
the Sec. 9 drivers read (see ``src/ebd/testbed/export.py``).

    python scripts/nfstream_export.py data/testbed data/testbed_escape \
        data/testbed_netem --timeouts 5 15 60

For each capture and active timeout ``T`` this writes
``<testbed>/exports/nfstream_a<T>_i<T>.jsonl.gz``: a first line with the
exporter's settings, its version, the Python and platform it ran on and the
SHA-256 of the capture; one line per flow record with the fields the analysis
uses; and a last line with the record count. The idle timeout equals the active
timeout, as in ``ebd.testbed.export.export``.

The script needs only nfstream and the standard library, and avoids syntax newer
than Python 3.8. It does not import the ``ebd`` package, so it can run in a
separate environment. That is what makes Sec. 9 runnable on an Intel Mac:
nfstream's macOS x86_64 wheels for CPython 3.9 and later are published without
the nDPI library their engine links against (they fail to import with "symbol
not found in flat namespace"), while the CPython 3.8 wheel of the same release
is complete. ``scripts/run_section9.sh`` sets that environment up.
"""

import argparse
import datetime
import gzip
import hashlib
import json
import platform
import sys
from pathlib import Path

# Must match ebd.testbed.export._KEEP; the loader there checks every field.
KEEP = (
    "src_ip", "src_port", "dst_ip", "dst_port", "protocol",
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "bidirectional_duration_ms", "bidirectional_packets", "bidirectional_bytes",
    "src2dst_packets", "src2dst_bytes", "dst2src_packets", "dst2src_bytes",
    "src2dst_duration_ms", "dst2src_duration_ms",
    "bidirectional_min_ps", "bidirectional_mean_ps", "bidirectional_max_ps",
    "bidirectional_min_piat_ms", "bidirectional_mean_piat_ms",
    "bidirectional_max_piat_ms",
)


def sha256(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export_one(pcap, active, out, digest):
    import nfstream
    from nfstream import NFStreamer

    streamer = NFStreamer(
        source=str(pcap),
        active_timeout=int(active),
        idle_timeout=int(active),
        statistical_analysis=True,
        n_dissections=0,
        accounting_mode=0,          # link-layer bytes, as DAPT2021 was exported
    )
    meta = {
        "pcap": str(pcap), "pcap_sha256": digest,
        "active_timeout": int(active), "idle_timeout": int(active),
        "statistical_analysis": True, "n_dissections": 0, "accounting_mode": 0,
        "nfstream": getattr(nfstream, "__version__", "unknown"),
        "python": platform.python_version(), "platform": platform.platform(),
        "created_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fields": list(KEEP),
    }
    tmp = out.with_name(out.name + ".part")
    n = 0
    with gzip.open(str(tmp), "wt") as fh:
        fh.write(json.dumps({"meta": meta}) + "\n")
        for f in streamer:
            fh.write(json.dumps({k: getattr(f, k, 0) for k in KEEP}) + "\n")
            n += 1
        fh.write(json.dumps({"end": {"records": n}}) + "\n")
    tmp.replace(out)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("testbeds", nargs="+", type=Path,
                    help="testbed directories, each holding capture.pcap")
    ap.add_argument("--timeouts", nargs="+", type=float, default=[5.0, 15.0, 60.0])
    args = ap.parse_args(argv)

    import nfstream
    print("nfstream %s, Python %s, %s" % (getattr(nfstream, "__version__", "?"),
                                          platform.python_version(), platform.platform()))
    for tb in args.testbeds:
        pcap = tb / "capture.pcap"
        if not pcap.exists():
            print("!! %s: no capture.pcap" % tb)
            return 1
        digest = sha256(pcap)
        (tb / "exports").mkdir(exist_ok=True)
        for ta in args.timeouts:
            out = tb / "exports" / ("nfstream_a%d_i%d.jsonl.gz" % (int(ta), int(ta)))
            n = export_one(pcap, ta, out, digest)
            print("%s  T=%3ds  %7d records  -> %s" % (tb, int(ta), n, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
