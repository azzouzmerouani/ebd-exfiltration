"""Controlled exfiltration traffic generator: real TCP, real packets.

Purpose
-------
Every public exfiltration dataset we examined is low-and-slow, so the regime the
architectural floor warns about -- ``D <= T``, total loss -- cannot be exercised
on any of them. This module generates it.

Why a short-run testbed is legitimate here
------------------------------------------
The bound of Eq. (3) depends on ``T`` and ``D`` **only through their ratio**.
Nothing in the derivation refers to absolute time. So a testbed that spans
``T/D`` from far below 1 to far above 1 exercises the whole bound, whatever the
absolute seconds are. We therefore run with short exporter timeouts
(``Ta in {5, 15, 60}``s) and transfer durations from 1 s to ~256 s, which covers
``T/D`` over roughly three orders of magnitude in under half an hour of wall
clock, instead of the days a 1800 s timeout would demand.

What is real and what is emulated -- stated plainly, because a reviewer will ask
-------------------------------------------------------------------------------
REAL: the packets (a real TCP connection, real segmentation, real teardown), the
exporter (nfstream, the same tool DAPT2021 was exported with), the flow records,
the features, the classifier, and the byte timeline (reconstructed from the
capture by the same receiver-side, de-duplicated, payload-only discipline used
on the real episodes).

EMULATED: the adversary. There is no malware here; the "exfiltration" is a paced
bulk upload to a listener, and the benign traffic is synthesised from a mixture
model. This testbed therefore establishes an **architectural** result -- that a
record consumer loses everything when the transfer fits inside one emission
interval -- and makes no claim about the behavioural realism of the traffic.
The behavioural claims in this paper rest on the DAPT2021 episodes; this
testbed rests on the packets.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Sink                                                                         #
# --------------------------------------------------------------------------- #

class Sink:
    """A TCP listener that reads and discards. The capture is the measurement."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(128)
        self.host, self.port = self._srv.getsockname()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> "Sink":
        self._thread.start()
        return self

    def _serve(self) -> None:
        self._srv.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._drain, args=(conn,), daemon=True).start()

    @staticmethod
    def _drain(conn: socket.socket) -> None:
        try:
            while conn.recv(65536):
                pass
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def stop(self) -> None:
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Transfer plan                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class Transfer:
    """One episode of the testbed.

    ``n_connections`` > 1 splits the asset across sequential short connections
    -- MITRE T1030 transfer-size limiting -- which is the case the paper's
    chunking corollary is about. With one connection the record is emitted at
    teardown or at the active timeout, whichever comes first; with many, each
    closes and emits early.
    """

    label: str
    asset_bytes: int
    duration_s: float
    n_connections: int = 1
    kind: str = "attack"          # "attack" | "benign"
    port: int = 0
    t_start: float = 0.0
    t_end: float = 0.0
    conn_windows: list[tuple[float, float]] = field(default_factory=list)
    src_ports: list[int] = field(default_factory=list)

    @property
    def rate_bps(self) -> float:
        return self.asset_bytes / self.duration_s if self.duration_s > 0 else 0.0


def _chunk_size(asset: int, duration: float, n_conns: int) -> int:
    """Write size that gives enough pacing opportunities to hit ``duration``.

    Pacing works by sleeping between writes, so the achievable duration is
    bounded by (number of writes) x (mean sleep). A fixed write size therefore
    silently caps long transfers: at 16 KiB a 2 MiB asset is only 128 writes, so
    a 256 s target lands in a fraction of that no matter how long each sleep is.
    Sizing the write from the target instead keeps the cap far above any
    duration in the matrix.
    """
    per_conn = max(asset // max(n_conns, 1), 1)
    per_conn_dur = duration / max(n_conns, 1)
    # aim for a write every ~20 ms, bounded so tiny and huge transfers stay sane
    n_writes = min(max(int(per_conn_dur / 0.02), 8), 8192)
    return int(min(max(per_conn // n_writes, 1460), 262144))


def run_transfer(t: Transfer, host: str, port: int,
                 chunk: int | None = None, settle: float = 0.35) -> Transfer:
    """Execute one transfer, pacing writes so the asset lands over ``duration_s``.

    Pacing is closed-loop against wall clock rather than a fixed sleep, so
    scheduler jitter does not accumulate into a duration error.
    """
    per_conn = t.asset_bytes // t.n_connections
    remainder = t.asset_bytes - per_conn * t.n_connections
    per_conn_dur = t.duration_s / t.n_connections
    if chunk is None:
        chunk = _chunk_size(t.asset_bytes, t.duration_s, t.n_connections)
    payload = b"\x5a" * chunk

    t.t_start = time.time()
    for i in range(t.n_connections):
        target = per_conn + (remainder if i == t.n_connections - 1 else 0)
        conn_start = time.time()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.connect((host, port))
        t.src_ports.append(s.getsockname()[1])
        sent = 0
        while sent < target:
            n = min(chunk, target - sent)
            s.sendall(payload[:n] if n < chunk else payload)
            sent += n
            # closed-loop pacing: sleep until this many bytes *should* have gone.
            # The gap is slept in full -- capping it here is what made long
            # transfers finish early.
            frac = sent / target if target else 1.0
            due = conn_start + frac * per_conn_dur
            gap = due - time.time()
            if gap > 0:
                time.sleep(gap)
        s.shutdown(socket.SHUT_WR)
        s.close()
        t.conn_windows.append((conn_start, time.time()))
        # let the FIN exchange land in the capture before the next connection
        time.sleep(settle)

    t.t_end = time.time()
    return t
