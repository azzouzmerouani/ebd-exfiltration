"""The fast-regime capture under emulated network conditions (Sec. 9, M5).

The original capture of Sec. 9 runs over the loopback interface: a 65 KB MTU, no
delay and no loss. Record *emission timing* should not depend on any of those,
because an exporter emits a record at teardown or at its active timeout whatever
the path does to the packets -- but that was asserted, not shown. This module
re-records the same transfer plan with ``tc netem`` shaping the loopback of an
unprivileged network namespace: 10 ms per packet (a 20 ms round trip), 0.5%
random loss per packet, a 1500-byte MTU and segmentation offloads disabled, so
that TCP really segments, loses and retransmits.

The capture is produced in chunks because the machine that can run ``netem``
without root only allows short-lived processes. Every chunk uses the same sink
port and the same per-namespace conditions; ``merge`` concatenates the chunks
into one ``capture.pcap`` and one ``manifest.json`` that the Sec. 9 driver reads
exactly as it reads the original capture. Transfers longer than
``--max-duration`` do not fit in one chunk and are left out, and the manifest
says which.

    python -m ebd.testbed.netem_capture plan      --out data/testbed_netem
    python -m ebd.testbed.netem_capture run-chunk --out data/testbed_netem --chunk 0
    ...
    python -m ebd.testbed.netem_capture merge     --out data/testbed_netem

``run-chunk`` must be started inside the shaped namespace, for example::

    unshare -r -n sh -c 'ip link set lo up; ip link set lo mtu 1500;
        ethtool -K lo gso off tso off gro off;
        tc qdisc add dev lo root netem delay 10ms loss 0.5%;
        python -m ebd.testbed.netem_capture run-chunk --out DIR --chunk K'
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import struct
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .run_capture import build_plan
from .traffic import Sink, Transfer, run_transfer

DEFAULT_PORT = 47123
ATTACK_SETTLE = 0.30          # as in run_capture
BENIGN_SETTLE = 0.06


def _estimate_seconds(t: Transfer) -> float:
    settle = ATTACK_SETTLE if t.kind == "attack" else BENIGN_SETTLE
    # handshake + teardown at a 20 ms round trip, plus pacing slack
    return t.duration_s + t.n_connections * (settle + 0.12) + 0.05


def make_plan(seed: int, max_duration: float, chunk_seconds: float, port: int) -> dict:
    full = build_plan(seed)
    kept, excluded = [], []
    for t in full:
        if t.kind == "attack" and t.duration_s > max_duration:
            excluded.append(t.label)
        else:
            kept.append(t)
    chunks: list[list[int]] = [[]]
    used = 0.0
    for i, t in enumerate(kept):
        est = _estimate_seconds(t)
        if chunks[-1] and used + est > chunk_seconds:
            chunks.append([])
            used = 0.0
        chunks[-1].append(i)
        used += est
    return {
        "seed": seed,
        "max_duration_s": max_duration,
        "chunk_seconds": chunk_seconds,
        "sink_port": port,
        "excluded_labels": excluded,
        "n_transfers": len(kept),
        "chunks": chunks,
        "estimated_seconds": [round(sum(_estimate_seconds(kept[i]) for i in c), 1)
                              for c in chunks],
        "transfers": [asdict(t) for t in kept],
    }


def run_chunk(out: Path, k: int, hard_limit_s: float) -> int:
    plan = json.loads((out / "plan.json").read_text())
    port = int(plan["sink_port"])
    idx = plan["chunks"][k]
    pcap = out / f"chunk_{k:02d}.pcap"
    progress = out / f"chunk_{k:02d}.json"

    sink = Sink(port=port).start()
    env = dict(os.environ)
    cap = subprocess.Popen(
        [sys.executable, "-m", "ebd.testbed.afpacket_capture", "--iface", "lo",
         "--port", str(port), "--out", str(pcap), "--snaplen", "96", "--announce"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True,
    )
    if cap.stdout.readline().strip() != "READY":
        cap.kill()
        raise RuntimeError("capture did not start")
    time.sleep(0.8)

    t0 = time.time()
    done: list[dict] = []
    skipped: list[int] = []
    for j in idx:
        spec = dict(plan["transfers"][j])
        t = Transfer(label=spec["label"], asset_bytes=spec["asset_bytes"],
                     duration_s=spec["duration_s"],
                     n_connections=spec["n_connections"], kind=spec["kind"])
        if time.time() - t0 + _estimate_seconds(t) > hard_limit_s:
            skipped.append(j)
            continue
        t.port = port
        run_transfer(t, "127.0.0.1", port,
                     settle=ATTACK_SETTLE if t.kind == "attack" else BENIGN_SETTLE)
        rec = asdict(t)
        rec["plan_index"] = j
        done.append(rec)
        progress.write_text(json.dumps({"chunk": k, "complete": False,
                                        "transfers": done, "skipped": skipped}))
    time.sleep(1.5)
    cap.send_signal(signal.SIGINT)
    cap.wait(timeout=30)
    sink.stop()
    progress.write_text(json.dumps({
        "chunk": k, "complete": True, "wall_seconds": round(time.time() - t0, 2),
        "transfers": done, "skipped": skipped,
    }))
    print(f"chunk {k}: {len(done)} transfers in {time.time() - t0:.0f}s, "
          f"{len(skipped)} deferred; wrote {pcap.name}")
    return 0


def _pcap_records(path: Path):
    data = path.read_bytes()
    if len(data) < 24:
        return data[:0], []
    header, off, recs = data[:24], 24, []
    while off + 16 <= len(data):
        incl = struct.unpack("<I", data[off + 8:off + 12])[0]
        if off + 16 + incl > len(data):
            break                                  # truncated tail
        ts = struct.unpack("<II", data[off:off + 8])
        recs.append((ts[0] + ts[1] / 1e6, data[off:off + 16 + incl]))
        off += 16 + incl
    return header, recs


def merge(out: Path, conditions: str) -> int:
    plan = json.loads((out / "plan.json").read_text())
    header, all_recs, transfers, deferred = None, [], [], []
    last = -1.0
    for k in range(len(plan["chunks"])):
        meta_p = out / f"chunk_{k:02d}.json"
        pcap_p = out / f"chunk_{k:02d}.pcap"
        if not meta_p.exists() or not pcap_p.exists():
            raise SystemExit(f"chunk {k} missing; run it before merging")
        meta = json.loads(meta_p.read_text())
        if not meta.get("complete"):
            raise SystemExit(f"chunk {k} did not complete; re-run it")
        h, recs = _pcap_records(pcap_p)
        header = header or h
        if recs and recs[0][0] < last:
            raise SystemExit(f"chunk {k} starts before chunk {k - 1} ends")
        if recs:
            last = recs[-1][0]
        all_recs.extend(r for _, r in recs)
        transfers.extend(meta["transfers"])
        deferred.extend(plan["transfers"][j]["label"] for j in meta["skipped"])
    with open(out / "capture.pcap", "wb") as fh:
        fh.write(header)
        for r in all_recs:
            fh.write(r)
    manifest = {
        "seed": plan["seed"],
        "sink_port": plan["sink_port"],
        "capture_backend": "afpacket",
        "conditions": conditions,
        "chunked": True,
        "n_chunks": len(plan["chunks"]),
        "max_duration_s": plan["max_duration_s"],
        "excluded_labels": plan["excluded_labels"],
        "deferred_labels": deferred,
        "transfers": transfers,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    n_att = sum(t["kind"] == "attack" for t in transfers)
    print(f"merged {len(plan['chunks'])} chunks: {len(all_recs)} frames, "
          f"{len(transfers)} transfers ({n_att} attack); "
          f"excluded {len(plan['excluded_labels'])}, deferred {len(deferred)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sec. 9 capture under netem, in chunks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--max-duration", type=float, default=128.0)
    p.add_argument("--chunk-seconds", type=float, default=140.0)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    r = sub.add_parser("run-chunk")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--chunk", type=int, required=True)
    r.add_argument("--hard-limit", type=float, default=158.0)
    m = sub.add_parser("merge")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--conditions", default="")
    args = ap.parse_args(argv)

    if args.cmd == "plan":
        args.out.mkdir(parents=True, exist_ok=True)
        plan = make_plan(args.seed, args.max_duration, args.chunk_seconds, args.port)
        (args.out / "plan.json").write_text(json.dumps(plan, indent=1))
        print(f"{plan['n_transfers']} transfers in {len(plan['chunks'])} chunks "
              f"(estimated {sum(plan['estimated_seconds']):.0f}s); excluded "
              f"{plan['excluded_labels']}")
        return 0
    if args.cmd == "run-chunk":
        return run_chunk(args.out, args.chunk, args.hard_limit)
    return merge(args.out, args.conditions)


if __name__ == "__main__":
    raise SystemExit(main())
