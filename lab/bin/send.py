"""Victim-side exfiltration-like transfer with known timing. Appends one JSON line per transfer.
kinds: tcp_single (one TCP connection, FIN at end), tcp_chunked (n sequential connections),
       udp_single (one source port, query-sized datagrams), udp_rotate (new source port per datagram)."""
import argparse, json, socket, time, os
ap = argparse.ArgumentParser()
ap.add_argument("--id", type=int, required=True); ap.add_argument("--kind", required=True)
ap.add_argument("--bytes", type=int, required=True); ap.add_argument("--duration", type=float, required=True)
ap.add_argument("--chunks", type=int, default=8); ap.add_argument("--log", default="/opt/lab/run/transfers.jsonl")
a = ap.parse_args()
DST = "10.10.2.2"
rec = dict(id=a.id, kind=a.kind, bytes=a.bytes, duration=a.duration)
def paced(sendfn, nbytes, dur, block):
    n = max(1, nbytes // block); dt = dur / n; t0 = time.time(); sent = 0
    for i in range(n):
        sendfn(block); sent += block
        tgt = t0 + (i + 1) * dt; w = tgt - time.time()
        if w > 0: time.sleep(w)
    return sent
rec["t_start"] = time.time()
if a.kind == "tcp_single":
    s = socket.create_connection((DST, 20000 + a.id)); rec["sport"] = [s.getsockname()[1]]
    buf = os.urandom(1024); paced(lambda b: s.sendall(buf[:b]), a.bytes, a.duration, 1024)
    s.shutdown(socket.SHUT_WR); s.close()
elif a.kind == "tcp_chunked":
    rec["sport"] = []; per = a.bytes // a.chunks; dur = a.duration / a.chunks; buf = os.urandom(1024)
    for k in range(a.chunks):
        s = socket.create_connection((DST, 20000 + a.id)); rec["sport"].append(s.getsockname()[1])
        paced(lambda b: s.sendall(buf[:b]), per, dur, 1024); s.shutdown(socket.SHUT_WR); s.close()
elif a.kind in ("udp_single", "udp_rotate"):
    q = os.urandom(200); rec["sport"] = []
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); u.bind(("0.0.0.0", 0)); rec["sport"].append(u.getsockname()[1])
    def snd(b):
        global u
        if a.kind == "udp_rotate":
            u.close(); u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        u.sendto(q[:b], (DST, 30000 + a.id))
    paced(snd, a.bytes, a.duration, 200)
rec["t_end"] = time.time()
with open(a.log, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
