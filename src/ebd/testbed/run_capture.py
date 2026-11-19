"""Generate the fast-regime capture: attack episodes plus benign background.

    uv run ebd-capture --out data/testbed

Writes ``capture.pcap`` and ``manifest.json``. The manifest records, for every
episode, the asset size, the intended and achieved duration, the connection
count and the source ports it actually used -- the ports are what let the byte
timeline be attributed to an episode without guessing.

Design of the matrix
--------------------
The bound depends on ``T`` and ``D`` only through ``T/D``, so the sweep is over
the ratio. Durations span 1 s to 256 s and the exporter is run at active
timeouts of 5, 15 and 60 s, giving ``T/D`` from 60 down to 0.02.

Two axes beyond duration:

* **Chunking.** Every duration is run once as a single connection and once split
  across eight sequential connections. This is the case the chunking corollary
  is about, and it has not previously been measured on packets.
* **Asset size.** A subset is repeated at 10x the asset to check that the floor
  depends on the ratio and not on the volume, as the derivation claims.

Benign traffic includes deliberately hard negatives -- short, high-rate bulk
transfers whose volume and instantaneous rate overlap the attacks -- so that the
false-positive burden is not trivially cheap to satisfy.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .traffic import Sink, Transfer, run_transfer

ATTACK_DURATIONS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
ATTACK_ASSET = 2 * 1024 * 1024           # 2 MiB
INVARIANCE_ASSET = 20 * 1024 * 1024      # 20 MiB
INVARIANCE_DURATIONS = (4.0, 32.0, 256.0)
CHUNK_COUNTS = (1, 8)


def build_plan(seed: int = 20260815) -> list[Transfer]:
    rng = random.Random(seed)
    plan: list[Transfer] = []

    for d in ATTACK_DURATIONS:
        for k in CHUNK_COUNTS:
            plan.append(Transfer(
                label=f"attack_d{d:g}_k{k}_a2M",
                asset_bytes=ATTACK_ASSET, duration_s=d, n_connections=k,
                kind="attack",
            ))
    for d in INVARIANCE_DURATIONS:
        plan.append(Transfer(
            label=f"attack_d{d:g}_k1_a20M",
            asset_bytes=INVARIANCE_ASSET, duration_s=d, n_connections=1,
            kind="attack",
        ))

    # Benign background. Three populations, the third deliberately adversarial
    # to the detector: short high-rate bulk that looks like exfiltration on
    # every volume- or rate-based feature and differs only in persistence.
    for i in range(700):                                   # interactive / small
        plan.append(Transfer(
            label=f"benign_small_{i}",
            asset_bytes=rng.randint(2_000, 80_000),
            duration_s=rng.uniform(0.05, 0.8),
            n_connections=1, kind="benign",
        ))
    for i in range(300):                                   # medium downloads
        plan.append(Transfer(
            label=f"benign_med_{i}",
            asset_bytes=rng.randint(100_000, 900_000),
            duration_s=rng.uniform(0.4, 2.0),
            n_connections=rng.choice((1, 1, 2)), kind="benign",
        ))
    for i in range(200):                                   # hard negatives
        plan.append(Transfer(
            label=f"benign_bulk_{i}",
            asset_bytes=rng.randint(1_500_000, 3_000_000),
            duration_s=rng.uniform(0.6, 2.0),
            n_connections=1, kind="benign",
        ))

    rng.shuffle(plan)
    return plan


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fast-regime capture")
    p.add_argument("--out", type=Path, default=Path("data/testbed"))
    p.add_argument("--iface", default="lo")
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--capture-backend", choices=("tcpdump", "afpacket"),
                   default="tcpdump",
                   help="afpacket: ebd.testbed.afpacket_capture, for namespaces "
                        "in which tcpdump cannot drop privileges (Sec. 9, M5)")
    p.add_argument("--conditions", default="",
                   help="free-text network conditions, recorded in the manifest "
                        "(e.g. the netem qdisc, MTU and offload settings)")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    plan = build_plan(args.seed)
    n_att = sum(t.kind == "attack" for t in plan)
    budget = sum(t.duration_s for t in plan)
    print(f"plan: {len(plan)} transfers ({n_att} attack), "
          f"~{budget / 60:.1f} min of transfer time")

    sink = Sink().start()
    pcap = args.out / "capture.pcap"

    # snaplen 96: enough for eth+IP+TCP headers with options. Payload length is
    # taken from ip.len, so truncating the payload costs nothing and keeps the
    # trace small enough to re-export at several timeouts.
    if args.capture_backend == "tcpdump":
        tcpdump = subprocess.Popen(
            ["tcpdump", "-i", args.iface, "-w", str(pcap), "-s", "96", "-U",
             f"tcp port {sink.port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(2.0)
    else:
        tcpdump = subprocess.Popen(
            [sys.executable, "-m", "ebd.testbed.afpacket_capture",
             "--iface", args.iface, "--port", str(sink.port), "--out", str(pcap),
             "--snaplen", "96", "--announce"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        if tcpdump.stdout.readline().strip() != "READY":
            raise RuntimeError("afpacket capture did not start")
        time.sleep(1.0)

    t0 = time.time()
    for i, tr in enumerate(plan, 1):
        tr.port = sink.port
        run_transfer(tr, "127.0.0.1", sink.port,
                     settle=0.30 if tr.kind == "attack" else 0.06)
        if i % 100 == 0 or tr.kind == "attack":
            print(f"  [{i:>4}/{len(plan)}] {tr.label:<28} "
                  f"D={tr.t_end - tr.t_start:7.2f}s  "
                  f"elapsed={time.time() - t0:6.0f}s", flush=True)

    time.sleep(2.5)
    tcpdump.send_signal(signal.SIGINT)
    tcpdump.wait(timeout=30)
    sink.stop()

    manifest = {
        "seed": args.seed,
        "sink_port": sink.port,
        "wall_seconds": time.time() - t0,
        "capture_backend": args.capture_backend,
        "conditions": args.conditions,
        "transfers": [asdict(t) for t in plan],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {pcap} ({pcap.stat().st_size / 1e6:.1f} MB) and manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
