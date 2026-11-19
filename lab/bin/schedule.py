"""Launch the measurement transfers in the victim namespace at randomised start
times (seed 20261028), so that their phases relative to the collector rotation
and the rule schedule are spread. Writes the plan to /opt/lab/run/plan.jsonl."""
import json, random, subprocess, time
random.seed(20261028)
TYPES = [("tcp_short", "tcp_single", 2_000_000, 20), ("tcp_med", "tcp_single", 5_000_000, 120),
         ("tcp_chunk", "tcp_chunked", 5_000_000, 120), ("udp_single", "udp_single", 600_000, 300),
         ("udp_rotate", "udp_rotate", 600_000, 300), ("tcp_long", "tcp_single", 20_000_000, 900)]
plan = [("tcp_vlong", "tcp_single", 20_000_000, 2400), ("tcp_vlong", "tcp_single", 20_000_000, 2400)]
for rep in range(6):
    order = TYPES[:]; random.shuffle(order); plan += order
procs = []; t = time.time()
with open("/opt/lab/run/plan.jsonl", "w") as fh:
    for i, (label, kind, nbytes, dur) in enumerate(plan):
        at = t + (0 if i < 2 else 30 + (i - 2) * 150 + random.uniform(0, 60))
        fh.write(json.dumps(dict(id=i, label=label, kind=kind, bytes=nbytes, duration=dur, planned_start=at)) + "\n")
        fh.flush()
        while time.time() < at:
            time.sleep(0.2)
        procs.append(subprocess.Popen(["ip", "netns", "exec", "victim", "python3", "/opt/lab/bin/send.py",
                                       "--id", str(i), "--kind", kind, "--bytes", str(nbytes),
                                       "--duration", str(dur), "--log", "/opt/lab/run/transfers.jsonl"]))
for p in procs:
    p.wait()
print("all transfers done", time.time())
