"""Analytic self-tests for the EBD-DNS engine. Every floor here has a value that
can be computed by hand, so a wrong implementation fails rather than looks fine."""
import sys, math
from ebd.dns.engine import (Query, Episode, build_episodes, measure_floor, measure_ebd,
                            check_invariants, export_records, pick_threshold, wilson,
                            label_suffix, InvariantViolation, CONSUMERS)

FAILS = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond: FAILS.append(name)

def make(tool, n, rate_hz, payload_chars, tunnel="exf.example.com",
         port_per_query=False, src="10.0.0.5", t0=1000.0, backloaded=False):
    qs = []
    for i in range(n):
        if backloaded:
            # slow drip for 90% of the wall clock carrying 10% of the queries,
            # then a terminal burst — V(t) sits well below the linear diagonal
            span = (n - 1) / rate_hz
            k = max(1, n // 10)
            t = (t0 + span * 0.9 * (i / k)) if i < k else \
                (t0 + span * 0.9 + span * 0.1 * ((i - k) / max(1, n - k - 1)))
        else:
            t = t0 + i / rate_hz
        # distinct payload labels, as a real tunnel emits
        lab = (f"{i:06d}" + "abcdefghijklmnopqrstuvwxyz234567" * 4)[:payload_chars]
        qs.append(Query(ts=t, src=src, dst="8.8.8.8",
                        sport=(40000 + i) if port_per_query else 40000,
                        qname=lab + "." + tunnel, qtype=16))
    return qs

print("\n=== 1. export_records: emission never precedes the traffic it covers (A2) ===")
ts = [0.0, 1.0, 2.0, 50.0, 51.0]
recs = export_records(ts, t_active=1800.0, t_idle=15.0, capture_end=60.0)
check("idle timeout splits the 48 s gap", len(recs) == 2, f"{recs}")
check("first record emits at last+idle = 17.0", abs(recs[0][2] - 17.0) < 1e-9, f"{recs[0]}")
check("every emit >= last packet covered", all(e >= ts[i1] for _, i1, e in recs))
recs_a = export_records([i * 1.0 for i in range(100)], t_active=10.0, t_idle=15.0, capture_end=200.0)
check("active timeout cuts the flow", len(recs_a) > 1, f"{len(recs_a)} records")
check("active cut emits at start+T_a", abs(recs_a[0][2] - 10.0) < 1e-9, f"{recs_a[0]}")

print("\n=== 2. label_suffix infers the tunnel domain, tool-agnostically ===")
check("common suffix found", label_suffix(["aXY.t.evil.io", "bZZ.t.evil.io"]) == "t.evil.io")

print("\n=== 3. Per-query consumer: floor = 1/n (bound is near-vacuous) ===")
eps = build_episodes(make("iodine", 100, 1.0, 40), "iodine", gap_s=60)
check("one episode built", len(eps) == 1, f"{len(eps)}")
ep = eps[0]
fq = measure_floor(ep, "per-query")
check("floor == 1/100", abs(fq["floor_measured"] - 0.01) < 1e-9, f"{fq['floor_measured']:.4f}")

print("\n=== 4. Flow consumer, PORT-PER-QUERY: floor = V(t0+idle)/V(total) ===")
eps = build_episodes(make("dnscat", 100, 1.0, 40, port_per_query=True), "dnscat", gap_s=60)
ep = eps[0]
ff = measure_floor(ep, "flow", t_active=1800.0, t_idle=15.0)
# each query is its own single-packet flow -> first record at t0 + 15 s
# queries at t0..t0+15 inclusive = 16 of 100
check("t_eff == idle timeout (15 s)", abs(ff["t_eff_s"] - 15.0) < 1e-9, f"{ff['t_eff_s']}")
check("floor == 16/100", abs(ff["floor_measured"] - 0.16) < 1e-9, f"{ff['floor_measured']:.4f}")
check("closed form min(1,T/D) == 15/99", abs(ff["floor_closed_form"] - 15/99) < 1e-9,
      f"{ff['floor_closed_form']:.4f}")

print("\n=== 5. Flow consumer, SINGLE PORT: the flow never closes -> TOTAL LOSS ===")
eps = build_episodes(make("iodine", 100, 1.0, 40, port_per_query=False), "iodine", gap_s=60)
ep = eps[0]
ff2 = measure_floor(ep, "flow", t_active=1800.0, t_idle=15.0)
check("floor == 1.0 (whole asset gone first)", abs(ff2["floor_measured"] - 1.0) < 1e-9,
      f"{ff2['floor_measured']:.4f}")
check("exactly one record for the episode", ff2["n_records"] == 1, f"{ff2['n_records']}")

print("\n=== 6. D <= T: total loss, the regime Proposition 1 is about ===")
eps = build_episodes(make("cobaltstrike", 60, 10.0, 40, port_per_query=True), "cs", gap_s=60)
ep = eps[0]      # 60 queries at 10 Hz -> D = 5.9 s, well inside a 15 s idle timeout
f6 = measure_floor(ep, "flow", t_active=1800.0, t_idle=15.0)
check("D < T", ep.duration < 15.0, f"D={ep.duration:.2f}s")
check("floor == 1.0", abs(f6["floor_measured"] - 1.0) < 1e-9, f"{f6['floor_measured']}")
check("closed form == 1.0", abs(f6["floor_closed_form"] - 1.0) < 1e-9)

print("\n=== 7. Pipeline-position ordering: per-query <= flow <= session ===")
eps = build_episodes(make("iodine", 200, 2.0, 40, port_per_query=True), "iodine", gap_s=60)
ep = eps[0]
floors = {c: measure_floor(ep, c, t_active=1800.0, t_idle=15.0, window_s=300.0)
          for c in ("per-query", "flow", "session")}
vals = [floors[c]["floor_measured"] for c in ("per-query", "flow", "session")]
check("monotone non-decreasing", all(a <= b + 1e-12 for a, b in zip(vals, vals[1:])),
      " <= ".join(f"{v:.4f}" for v in vals))
notes = check_invariants(ep, floors)
check("invariants clean", notes == [], str(notes))

print("\n=== 8. I4 is SOFT: a back-loaded transfer may sit below the closed form ===")
eps = build_episodes(make("slowdrip", 100, 1.0, 40, port_per_query=True, backloaded=True),
                     "slowdrip", gap_s=600)
ep = eps[0]
fb = measure_floor(ep, "flow", t_active=1800.0, t_idle=15.0)
notes = check_invariants(ep, {"flow": fb})
check("below-closed-form is REPORTED not raised",
      fb["excess_pp"] < 0 and any("I4" in n for n in notes),
      f"excess {fb['excess_pp']:.2f} pp; notes={len(notes)}")

print("\n=== 9. I6 is HARD: no detector can beat its own floor ===")
eps = build_episodes(make("iodine", 100, 1.0, 40, port_per_query=True), "iodine", gap_s=60)
ep = eps[0]
oracle = lambda e, idxs, emit: 1.0                  # alerts on its first record
m = measure_ebd(ep, "flow", oracle, 0.5, t_active=1800.0, t_idle=15.0)
check("oracle sits exactly ON the floor", m["at_floor"], f"ebd={m['ebd']:.4f} floor={m['floor_measured']:.4f}")
late = lambda e, idxs, emit: 1.0 if emit > e.t0 + 60 else 0.0
m2 = measure_ebd(ep, "flow", late, 0.5, t_active=1800.0, t_idle=15.0)
check("a late detector sits ABOVE the floor", m2["ebd"] > m2["floor_measured"],
      f"ebd={m2['ebd']:.4f} floor={m2['floor_measured']:.4f} gap={m2['gap_pp']:.1f}pp")
blind = lambda e, idxs, emit: 0.0
m3 = measure_ebd(ep, "flow", blind, 0.5, t_active=1800.0, t_idle=15.0)
check("never alerting == total loss", m3["ebd"] == 1.0 and not m3["alerted"])
try:
    bad = dict(m); bad["ebd"] = bad["floor_measured"] - 0.1
    check_invariants(ep, {"flow": bad}); check("I6 raises on a violation", False)
except InvariantViolation:
    check("I6 raises on a violation", True)

print("\n=== 10. Operating point: Wilson-upper rule vs the point estimate ===")
benign = [0.0] * 990 + [0.6] * 10          # 1.0% of 1000 benign units score 0.6
thr_pt = min(t for t in [i/200 for i in range(201)] if sum(1 for s in benign if s >= t)/len(benign) <= 0.01)
burden_pt = sum(1 for s in benign if s >= thr_pt) / len(benign)
thr_w, meta = pick_threshold(benign, budget=0.01)
check("point estimate lands exactly ON the budget", abs(burden_pt - 0.01) < 1e-9,
      f"thr={thr_pt} burden={burden_pt:.4f}")
check("Wilson-upper rule is strictly more conservative", thr_w > thr_pt,
      f"wilson={thr_w} > point={thr_pt}")
check("its train upper bound is inside budget", meta["train_upper"] <= 0.01,
      f"upper={meta['train_upper']:.4f}")
lo, hi = wilson(10, 1000)
check("Wilson(10/1000) upper > 1%", hi > 0.01, f"[{lo*100:.2f}, {hi*100:.2f}]%")

print("\n=== 11. Episode splitting is sensitive to the gap threshold (reported, not hidden) ===")
qs = make("iodine", 60, 1.0, 40, port_per_query=True) + make(
    "iodine", 60, 1.0, 40, port_per_query=True, t0=1000.0 + 400.0)
counts = {g: len(build_episodes(list(qs), "iodine", gap_s=g)) for g in (60, 300, 600)}
check("gap 60 s -> 2 episodes, 600 s -> 1", counts[60] == 2 and counts[600] == 1, str(counts))

print("\n" + "=" * 66)
print(f"{'ALL TESTS PASSED' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
