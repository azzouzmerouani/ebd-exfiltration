"""A second capture, built to make the escape claim hard rather than easy.

    uv run ebd-escape-capture --out data/testbed_escape

Why a second capture
--------------------
The first capture (``ebd-capture``) answers the ``D <= T`` question, and its
benign background is a mixture of short sessions -- which is fine for that
question, because the floor is a property of the exporter and does not depend on
the benign mix at all.

It is *not* fine for the escape question. An incremental detector that keys on
persistence separates trivially from a background whose longest session is two
seconds, and a result obtained that way would be an artefact of the generator.
The honest test needs the hard negative for a persistence rule: **benign
sessions that are long-lived and low-rate**, overlapping the attacks in duration
and in instantaneous rate, and differing only in total volume and in sustained
direction asymmetry.

This capture supplies them, and runs them *concurrently* with everything else --
both because that is what a real network looks like, and because generating
hundreds of minute-long sessions sequentially would take hours.

What the background contains
----------------------------
* short interactive sessions (as before);
* short high-rate bulk -- the hard negative for a volume/rate rule;
* **long-lived low-rate sessions**, 20-240 s, at rates that overlap the slow
  attacks -- the hard negative for a persistence rule;
* **long-lived bursty sessions** that idle and then transfer, so "sustained
  volume eventually" is not by itself a discriminator.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

from .traffic import Sink, Transfer, run_transfer

ATTACK_DURATIONS = (2.0, 8.0, 32.0, 128.0, 256.0)
ATTACK_ASSET = 2 * 1024 * 1024


def _background(sink_port: int, stop: threading.Event, out: list,
                lock: threading.Lock, seed: int, n_workers: int = 32) -> list:
    """Continuously generate benign sessions until ``stop`` is set."""

    def worker(wid: int) -> None:
        rng = random.Random(seed + wid * 7919)
        i = 0
        while not stop.is_set():
            roll = rng.random()
            if roll < 0.50:                      # short interactive
                t = Transfer(label=f"benign_small_w{wid}_{i}",
                             asset_bytes=rng.randint(2_000, 80_000),
                             duration_s=rng.uniform(0.05, 0.8),
                             n_connections=1, kind="benign")
            elif roll < 0.70:                    # short high-rate bulk
                t = Transfer(label=f"benign_bulk_w{wid}_{i}",
                             asset_bytes=rng.randint(1_500_000, 3_000_000),
                             duration_s=rng.uniform(0.6, 2.0),
                             n_connections=1, kind="benign")
            elif roll < 0.90:                    # long-lived, low-rate
                d = rng.uniform(20.0, 120.0)
                rate = rng.uniform(1_000.0, 40_000.0)
                t = Transfer(label=f"benign_long_w{wid}_{i}",
                             asset_bytes=int(d * rate),
                             duration_s=d, n_connections=1, kind="benign")
            else:                                # long-lived, bursty
                d = rng.uniform(30.0, 120.0)
                t = Transfer(label=f"benign_burst_w{wid}_{i}",
                             asset_bytes=rng.randint(200_000, 1_200_000),
                             duration_s=d, n_connections=1, kind="benign")
            t.port = sink_port
            try:
                run_transfer(t, "127.0.0.1", sink_port, settle=0.05)
            except OSError:
                continue
            with lock:
                out.append(t)
            i += 1
            time.sleep(rng.uniform(0.02, 0.4))

    threads = [threading.Thread(target=worker, args=(w,), daemon=True)
               for w in range(n_workers)]
    for t in threads:
        t.start()
    return threads


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Escape-question capture")
    p.add_argument("--out", type=Path, default=Path("data/testbed_escape"))
    p.add_argument("--iface", default="lo")
    p.add_argument("--seed", type=int, default=20260816)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    sink = Sink().start()
    pcap = args.out / "capture.pcap"
    tcpdump = subprocess.Popen(
        ["tcpdump", "-i", args.iface, "-w", str(pcap), "-s", "96", "-U",
         f"tcp port {sink.port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2.0)

    benign: list[Transfer] = []
    lock = threading.Lock()
    stop = threading.Event()
    _background(sink.port, stop, benign, lock, args.seed)
    print("background traffic running; starting attacks")

    # Let the background establish some long-lived sessions before the first
    # attack, so the attacks are not the only long flows in the capture.
    time.sleep(45.0)

    attacks: list[Transfer] = []
    t0 = time.time()
    for d in ATTACK_DURATIONS:
        t = Transfer(label=f"attack_d{d:g}_k1_a2M", asset_bytes=ATTACK_ASSET,
                     duration_s=d, n_connections=1, kind="attack",
                     port=sink.port)
        run_transfer(t, "127.0.0.1", sink.port, settle=0.3)
        attacks.append(t)
        with lock:
            n_b = len(benign)
        print(f"  {t.label:<24} D={t.t_end - t.t_start:7.2f}s  "
              f"benign so far={n_b:>4}  elapsed={time.time() - t0:6.0f}s",
              flush=True)
        time.sleep(3.0)

    # Drain: let in-flight long benign sessions finish so they appear complete.
    print("draining background...")
    time.sleep(20.0)
    stop.set()
    time.sleep(45.0)

    tcpdump.send_signal(signal.SIGINT)
    tcpdump.wait(timeout=30)
    sink.stop()

    with lock:
        done = [b for b in benign if b.t_end > 0]
    plan = attacks + done
    longs = [b for b in done if b.t_end - b.t_start >= 20.0]
    print(f"\ncaptured {len(attacks)} attacks and {len(done)} benign sessions, "
          f"of which {len(longs)} last >= 20 s "
          f"(longest {max((b.t_end - b.t_start for b in done), default=0):.0f} s)")

    (args.out / "manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "sink_port": sink.port,
        "transfers": [asdict(t) for t in plan],
    }, indent=1))
    print(f"wrote {pcap} ({pcap.stat().st_size / 1e6:.1f} MB) and manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
