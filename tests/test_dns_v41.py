"""v4.1, module level: slotted records and compact benign units (the memory refactor).

A measurement run holds millions of Query objects and benign units, so what each one costs decides
whether the run fits in a 16 GB session. These checks pin down that the refactor changed the cost
and nothing else: the same units, in the same order, with the same contents.
(The notebook-level v4.1 checks -- the preallocated feature matrix, the container-limit readers, the
memory plan and its two guards -- need the notebook and live with its generator.)"""
import random, sys
from collections import defaultdict
from ebd.dns.engine import (Query, Episode, benign_units, registered_domain, label_suffix, attribute_payload)

FAILS = []
def check(name, cond, d=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + d) if d else ''}")
    if not cond: FAILS.append(name)

# ---------------------------------------------------------------- A. slots
print("\n=== A. records are slotted (no per-object __dict__) ===")
q = Query(ts=1.0, src="10.0.0.1", dst="10.0.0.2", sport=4000, qname="a.b.example.com", qtype=1)
e = Episode("e", "t", "10.0.0.1", "example.com", [q], "f", {})
if sys.version_info >= (3, 10):
    check("Query has no __dict__", not hasattr(q, "__dict__"))
    check("Episode has no __dict__", not hasattr(e, "__dict__"))
    try:
        q.surprise = 1; ok = False
    except AttributeError:
        ok = True
    check("a stray attribute cannot be attached to a Query", ok)
else:
    print("  (Python < 3.10: slots are unavailable, the engine runs unslotted and is correct, only larger)")
q.payload, q.tunnel = 7, "example.com"
check("fields stay assignable (attribute_payload writes payload and tunnel)", q.payload == 7 and q.tunnel == "example.com")
check("Episode properties work on the slotted class", e.t0 == 1.0 and e.n_queries == 1 and e.asset == 7 and e.n_ports == 1)

# ---------------------------------------------------------------- B. benign_units == the v4 algorithm
print("\n=== B. benign_units: the same units as v4, in the same order, in less memory ===")

def benign_units_v4(queries, gap_s, min_queries=1):                 # the v4 function, verbatim
    by_key = defaultdict(list)
    for q in queries:
        by_key[(q.file, q.src, registered_domain(q.qname))].append(q)
    out = []
    for (fname, src, regdom), qs in by_key.items():
        qs.sort(key=lambda x: x.ts)
        tun = label_suffix([q.qname for q in qs])
        attribute_payload(qs, tun)
        run = []
        for q in qs:
            if run and (q.ts - run[-1].ts) > gap_s:
                if len(run) >= min_queries:
                    out.append(Episode(f"benign|{fname}|{src}|{regdom}|{run[0].ts:.3f}",
                                       "benign", src, regdom, run, "benign", {"file": fname}))
                run = []
            run.append(q)
        if len(run) >= min_queries:
            out.append(Episode(f"benign|{fname}|{src}|{regdom}|{run[0].ts:.3f}",
                               "benign", src, regdom, run, "benign", {"file": fname}))
    return out

def make_queries(seed=11, n=6000):
    r = random.Random(seed); out = []
    files = ["cap_a.pcap", "cap_b.pcap", "cap_c.pcap"]
    doms = ["example.com", "cdn.akamaiedge.net", "tel.vendor.io", "mail.uni.edu.au", "x.co.uk", "lab-tunnel.net"]
    for f in files:                                    # captures are contiguous in the list, as the notebook builds it
        t = 1.4e9 + files.index(f) * 1e5
        for _ in range(n // len(files)):
            t += r.choice([0.01, 0.2, 1.0, 3.0, 400.0, 900.0]) * r.random()      # some gaps exceed the 300 s cut
            lab = "".join(r.choice("abcdef0123456789") for _ in range(r.choice([3, 8, 24, 40])))
            out.append(Query(ts=t, src=f"10.0.{r.randint(0, 1)}.{r.randint(1, 4)}", dst="10.9.9.9",
                             sport=r.randint(1025, 65000), qname=f"{lab}.{r.choice(doms)}",
                             qtype=r.choice([1, 16, 28]), file=f))
    return out

def shape(units):
    return [(u.meta["file"], u.src, u.family, u.tool,
             [(q.ts, q.qname, q.sport, q.qtype, q.payload, q.tunnel) for q in u.queries]) for u in units]

for gap, minq in ((300.0, 1), (300.0, 2), (60.0, 1), (5.0, 3)):
    old = benign_units_v4(make_queries(), gap, minq)
    new = benign_units(make_queries(), gap, minq)
    check(f"gap {gap:g} s, min_queries {minq}: {len(new)} units, identical to v4 unit by unit and in order",
          len(old) == len(new) and shape(old) == shape(new), f"{len(old)} vs {len(new)}")
    check(f"gap {gap:g} s, min_queries {minq}: each unit keeps its v4 registered domain (Episode.tunnel)",
          [u.tunnel for u in old] == [u.tunnel for u in new])

new = benign_units(make_queries(), 300.0, 1)
check("unit ids are unique", len({u.eid for u in new}) == len(new))
check("unit ids are short (v4's carried file name, client, domain and a timestamp)", max(len(u.eid) for u in new) <= 24,
      f"longest {max(len(u.eid) for u in new)} characters")
by_file = defaultdict(list)
for u in new: by_file[u.meta["file"]].append(u)
check("every unit of a capture shares ONE meta dict", all(len({id(u.meta) for u in us}) == 1 for us in by_file.values()))
check("different captures do not share a meta dict", len({id(us[0].meta) for us in by_file.values()}) == len(by_file))
a = [u for u in by_file["cap_a.pcap"] if u.tunnel][0]
b = [u for u in by_file["cap_b.pcap"] if u.tunnel == a.tunnel][0]
check("one domain string is shared across units and captures", a.tunnel is b.tunnel)
check("a unit never mixes captures, clients or registered domains",
      all(len({(q.file, q.src, registered_domain(q.qname)) for q in u.queries}) == 1 for u in new))
check("every query lands in exactly one unit at min_queries = 1",
      sum(len(u.queries) for u in new) == 6000)
check("benign_units([]) is empty", benign_units([], 300.0) == [])

print("\n" + ("ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED: " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
