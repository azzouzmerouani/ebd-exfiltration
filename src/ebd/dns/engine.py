"""
EBD-DNS engine — architectural floor and detector attainment for DNS exfiltration.

Companion to "Exfiltrated Bytes before Detection: Damage-Denominated Evaluation
and the Architectural Floor of Record-Consuming Exfiltration Detectors".

The manuscript measures the floor on n = 7 episodes from one dataset. This engine
measures the same quantity on an independent, packet-level corpus of DNS
exfiltration, and runs consumers at different pipeline positions -- per-query,
flow-record, session -- over IDENTICAL traffic, so only the unit of inference
varies. A streaming accumulator in the style of ibHH (Ozery et al.) adds a fourth,
outside-the-family point: its excess over the per-query floor is STATISTICAL delay
(evidence accumulation), not architectural delay -- the two components the
manuscript's QCD discussion says compose.

Payload unit (A5). EBD is a ratio V(t)/V(total). Encoded QNAME characters are an
exact stand-in for decoded bytes when each query's characters are proportional to
its bytes. Per-query protocol overhead (sequence numbers, session ids) breaks
exact proportionality, so the engine re-measures every floor with an estimated
per-query overhead removed and reports the difference. The unit is measured, not
assumed.
"""

from __future__ import annotations

import bisect
import math
import random
import struct
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Statistics                                                                    #
# --------------------------------------------------------------------------- #

def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion, as fractions."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_upper(k: int, n: int, z: float = 1.959963985) -> float:
    return wilson(k, n, z)[1]


def bootstrap_ci(xs: Sequence[float], n_boot: int = 10_000, seed: int = 20260825,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. Returns (lo, hi)."""
    xs = list(xs)
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return (means[int((alpha / 2) * n_boot)], means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])


def _family_lists(by_family: dict) -> list[list[float]]:
    """The non-empty families as lists of floats, in a fixed order (sorted by key), so that a result never depends on
    the order the dict was built in. A NaN or infinite value is an upstream bug and stops here, naming the family."""
    out = []
    for key in sorted(by_family, key=str):
        v = [float(x) for x in by_family[key]]
        if not v:
            continue
        if not all(math.isfinite(x) for x in v):
            raise ValueError(f"non-finite value in family {key!r}")
        out.append(v)
    return out


def family_bootstrap(by_family: dict, n_boot: int = 10_000, seed: int = 20260825,
                     alpha: float = 0.05) -> tuple[float, float, float]:
    """Two-stage cluster bootstrap of a mean over episodes that come in families.

    Episodes of one tool differ mainly by DNS record type, so n episodes is closer to the number of families than
    to n independent observations. Each draw resamples the families with replacement and then the episodes inside
    each sampled family with replacement. Returns (mean of the family means, lo, hi): the point estimate weights
    every family equally, which is the number to quote when one tool contributes four episodes and another one.

    Coverage (checked by simulation in the v4.2 review): with only 8 families the percentile interval is
    anti-conservative. Its nominal 95% covered the family-level mean 0.86 to 0.94 of the time (about 0.89 for the
    between-family spread of the 7 Oct data), so read it as roughly a 90% interval. `family_t_interval` is the
    alternative with the right coverage at this size. The result does not depend on the order of `by_family`."""
    fams = _family_lists(by_family)
    if len(fams) < 2:
        m = sum(fams[0]) / len(fams[0]) if fams else float("nan")
        return (m, float("nan"), float("nan"))
    point = sum(sum(v) / len(v) for v in fams) / len(fams)
    rng = random.Random(seed)
    k = len(fams)
    draws = []
    for _ in range(n_boot):
        tot = 0.0
        for _f in range(k):
            v = fams[rng.randrange(k)]
            tot += sum(v[rng.randrange(len(v))] for _e in range(len(v))) / len(v)
        draws.append(tot / k)
    draws.sort()
    return (point, draws[int((alpha / 2) * n_boot)], draws[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])


# Student's t and the exact binomial interval, in the standard library (the engine has no dependencies). Both rest on
# the regularized incomplete beta function; the tests compare them with scipy to 1e-9 where scipy is installed.

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz; Numerical Recipes 6.4)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            de = d * c
            h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_quantile(p: float, df: float) -> float:
    """The p-quantile of Student's t distribution with df degrees of freedom."""
    if not 0.0 < p < 1.0 or df <= 0:
        raise ValueError("t_quantile needs 0 < p < 1 and df > 0")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    tail = 1.0 - p                                       # P(T > t) for t > 0 is I_{df/(df+t^2)}(df/2, 1/2) / 2
    lo, hi = 0.0, 1.0
    while 0.5 * _betai(df / 2.0, 0.5, df / (df + hi * hi)) > tail and hi < 1e12:
        lo, hi = hi, hi * 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * _betai(df / 2.0, 0.5, df / (df + mid * mid)) > tail:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _beta_quantile(p: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _betai(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided interval for k successes in n trials, as fractions. Wilson is the interval
    used elsewhere in this notebook; for counts as small as 3 of 8 families the exact interval is the conservative
    companion to quote beside it."""
    if n <= 0:
        return (0.0, 1.0)
    lo = 0.0 if k <= 0 else _beta_quantile(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k >= n else _beta_quantile(1.0 - alpha / 2.0, k + 1, n - k)
    return (lo, hi)


def family_t_interval(by_family: dict, alpha: float = 0.05,
                      clip: tuple | None = None) -> tuple[float, float, float]:
    """Mean of the family means with a Student-t interval over the families (df = families - 1).

    The unit of replication is the family, so this treats the family means as the sample. In the v4.2 review's
    coverage simulation (8 families, nominal 95%) it covered 0.95 to 0.97 where the percentile cluster bootstrap
    covered 0.86 to 0.94, at the price of an interval about a quarter wider. `clip=(lo, hi)` bounds it for
    quantities that cannot leave a range (a floor is a fraction); a bound may be None."""
    means = [sum(v) / len(v) for v in _family_lists(by_family)]
    k = len(means)
    if k == 0:
        return (float("nan"), float("nan"), float("nan"))
    m = sum(means) / k
    if k < 2:
        return (m, float("nan"), float("nan"))
    sd = math.sqrt(sum((x - m) ** 2 for x in means) / (k - 1))
    h = t_quantile(1.0 - alpha / 2.0, k - 1) * sd / math.sqrt(k)
    lo, hi = m - h, m + h
    if clip:
        if clip[0] is not None:
            lo = max(lo, clip[0])
        if clip[1] is not None:
            hi = min(hi, clip[1])
    return (m, lo, hi)


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


# --------------------------------------------------------------------------- #
# Records, domains, episodes                                                    #
# --------------------------------------------------------------------------- #

# A benign corpus holds millions of Query objects and Episodes. __slots__ drops the per-instance dict
# (~50 B each); on Python < 3.10 the classes are simply ordinary dataclasses.
_DC = dict(slots=True) if sys.version_info >= (3, 10) else {}


@dataclass(**_DC)
class Query:
    """One DNS query observed on the wire."""
    ts: float
    src: str
    dst: str
    sport: int
    qname: str
    qtype: int
    payload: int = 0          # encoded payload characters (set by attribute_payload)
    tunnel: str = ""          # inferred tunnel suffix
    file: str = ""            # capture file it came from: episodes never span files


@dataclass(**_DC)
class Episode:
    eid: str
    tool: str
    src: str
    tunnel: str
    queries: list[Query] = field(default_factory=list)
    family: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def t0(self) -> float: return self.queries[0].ts
    @property
    def t1(self) -> float: return self.queries[-1].ts
    @property
    def duration(self) -> float: return self.t1 - self.t0
    @property
    def asset(self) -> int: return sum(q.payload for q in self.queries)
    @property
    def n_queries(self) -> int: return len(self.queries)
    @property
    def n_ports(self) -> int: return len({q.sport for q in self.queries})


# Second-level labels under which registrations sit one level deeper (co.uk, com.au ...)
_SLD = {"co", "com", "net", "org", "ac", "gov", "edu", "or", "ne", "go", "mil", "nic"}


def registered_domain(name: str) -> str:
    """eTLD+1 heuristic. Exact for the tunnel domains that matter here; for benign
    traffic it only needs to group a client's queries by the domain they serve."""
    labels = name.rstrip(".").lower().split(".")
    if len(labels) <= 2:
        return ".".join(labels)
    if len(labels[-1]) == 2 and labels[-2] in _SLD:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def label_suffix(qnames: Iterable[str], min_labels: int = 2) -> str:
    """Longest common label-wise suffix shared by every qname, capped so that at
    least one label is always left of it (a tunnel query must carry a payload label;
    without the cap a run of identical names reports zero payload)."""
    parts = [q.rstrip(".").lower().split(".") for q in qnames if q]
    if not parts:
        return ""
    max_len = max(1, min(len(p) for p in parts) - 1)
    common: list[str] = []
    for i in range(1, max_len + 1):
        seg = parts[0][-i]
        if all(p[-i] == seg for p in parts):
            common.insert(0, seg)
        else:
            break
    return ".".join(common)


def attribute_payload(queries: list[Query], tunnel: str) -> None:
    """Set q.payload to the encoded payload characters carried left of `tunnel`."""
    suffix = "." + tunnel if tunnel else ""
    for q in queries:
        name = q.qname.rstrip(".").lower()
        if tunnel and name.endswith(suffix):
            q.payload = len(name) - len(suffix)
        elif tunnel and name == tunnel:
            q.payload = 0
        else:
            q.payload = len(name)
        q.tunnel = tunnel
        q.payload = max(0, q.payload)


def episode_encoding_cv(ep: Episode) -> float:
    """Coefficient of variation of per-query payload length. Descriptive only:
    variable chunk sizes are harmless to A5; per-query overhead is what matters,
    and `measure_floor` tests that directly."""
    lens = [q.payload for q in ep.queries[:-1] if q.payload > 0]
    if len(lens) < 2:
        return 0.0
    mu = sum(lens) / len(lens)
    if mu == 0:
        return 0.0
    var = sum((x - mu) ** 2 for x in lens) / (len(lens) - 1)
    return math.sqrt(var) / mu


def build_episodes(queries: list[Query], tool: str, gap_s: float,
                   min_queries: int = 20, min_payload: int = 512,
                   dominant_only: bool = True, family: str = "",
                   log: list | None = None) -> list[Episode]:
    """Episodes from one tool's queries.

    1. Scope to the capture file. Lab captures reuse client IPs across files, so
       grouping across files would merge independent tool runs into one episode.
    2. Within a file, group by (client, registered domain).
    3. dominant_only: keep only the registered domain carrying the most payload
       in each file. Every capture is one tool run against one tunnel domain;
       everything else in the file is background DNS and must not become an
       exfiltration episode. If two clients carry that domain (a capture that saw
       both the stub->resolver and resolver->authoritative legs) only the larger
       is kept, so one transfer is never counted twice.
    4. Infer the tunnel suffix, attribute payload, split on silence > gap_s.
    """
    by_file: dict[str, list[Query]] = defaultdict(list)
    for q in queries:
        by_file[q.file].append(q)

    episodes: list[Episode] = []
    for fname, fqs in by_file.items():
        groups: dict[tuple[str, str], list[Query]] = defaultdict(list)
        for q in fqs:
            groups[(q.src, registered_domain(q.qname))].append(q)

        # payload potential per group (characters left of the inferred suffix)
        scored = []
        for key, qs in groups.items():
            qs.sort(key=lambda x: x.ts)
            tun = label_suffix([q.qname for q in qs])
            attribute_payload(qs, tun)
            scored.append((sum(q.payload for q in qs), key, qs, tun))
        scored.sort(key=lambda r: -r[0])

        keep = scored
        if dominant_only and scored:
            top_dom = scored[0][1][1]
            keep = [scored[0]]
            same_dom = [r for r in scored[1:] if r[1][1] == top_dom]
            bg = [r for r in scored[1:] if r[1][1] != top_dom]
            if log is not None:
                tot = sum(r[0] for r in scored) or 1
                log.append({"file": fname, "tool": tool, "tunnel_domain": top_dom,
                            "kept_src": scored[0][1][0],
                            "n_queries_file": len(fqs), "n_groups": len(scored),
                            "n_queries_kept": len(scored[0][2]), "payload_kept": scored[0][0],
                            "payload_share_kept": scored[0][0] / tot,
                            "n_background_domains": len({r[1][1] for r in bg}),
                            "n_duplicate_vantage": len(same_dom),
                            "duplicate_vantage_max_share":
                                max([r[0] / max(1, scored[0][0]) for r in same_dom] or [0.0])})

        for _, (src, regdom), qs, tun in keep:
            run: list[Query] = []
            for q in qs:
                if run and (q.ts - run[-1].ts) > gap_s:
                    episodes.append(Episode("", tool, src, tun, run, family or tool,
                                            {"file": fname}))
                    run = []
                run.append(q)
            if run:
                episodes.append(Episode("", tool, src, tun, run, family or tool,
                                        {"file": fname}))

    kept = []
    for ep in episodes:
        if ep.n_queries >= min_queries and ep.asset >= min_payload and ep.duration > 0:
            ep.eid = f"{tool}|{ep.meta.get('file','')}|{ep.src}->{ep.tunnel}|{ep.t0:.3f}"
            kept.append(ep)
        elif log is not None:
            log.append({"file": ep.meta.get("file", ""), "tool": tool,
                        "rejected_run": True, "n_queries": ep.n_queries,
                        "asset": ep.asset, "duration_s": ep.duration})
    return kept


def benign_units(queries: list[Query], gap_s: float, min_queries: int = 1) -> list[Episode]:
    """Benign traffic cut by EXACTLY the malicious construction -- file, client,
    registered domain, silence -- so a benign unit is built the way a malicious
    one is, and the false-positive burden is measured on comparable units.

    Memory: a measurement run holds millions of these (a unit is cheap only if it is small), so
    ids are short ("benign|<file number>|<unit number>"), every unit of a capture shares ONE meta
    dict (the notebook writes the same per-file value into it), and each distinct domain is ONE
    string. Grouping is per capture file, which bounds the working dictionary to one file."""
    by_file: dict[str, list[Query]] = {}
    for q in queries:
        by_file.setdefault(q.file, []).append(q)
    dom_cache: dict[str, str] = {}
    out: list[Episode] = []
    for fid, (fname, fqs) in enumerate(by_file.items()):
        meta = {"file": fname}                                  # shared by every unit of this capture
        by_key: dict[tuple, list[Query]] = defaultdict(list)
        for q in fqs:
            rd = registered_domain(q.qname)
            by_key[(q.src, dom_cache.setdefault(rd, rd))].append(q)
        for (src, regdom), qs in by_key.items():
            qs.sort(key=lambda x: x.ts)
            tun = label_suffix([q.qname for q in qs])
            attribute_payload(qs, dom_cache.setdefault(tun, tun))
            run: list[Query] = []
            for q in qs:
                if run and (q.ts - run[-1].ts) > gap_s:
                    if len(run) >= min_queries:
                        out.append(Episode(f"benign|{fid}|{len(out)}", "benign", src, regdom, run, "benign", meta))
                    run = []
                run.append(q)
            if len(run) >= min_queries:
                out.append(Episode(f"benign|{fid}|{len(out)}", "benign", src, regdom, run, "benign", meta))
    return out


# --------------------------------------------------------------------------- #
# Record emission — the consumers                                               #
# --------------------------------------------------------------------------- #

def export_records(ts: Sequence[float], t_active: float, t_idle: float,
                   capture_end: float) -> list[tuple[int, int, float]]:
    """nfstream-style flow export over one flow key.

    Returns (i0, i1, emit_ts): packets ts[i0..i1] are summarised by a record that
    becomes available at emit_ts. (A2) is enforced structurally -- emit_ts is never
    earlier than the last packet the record covers.
    """
    out: list[tuple[int, int, float]] = []
    n = len(ts)
    i0 = 0
    while i0 < n:
        start = ts[i0]
        deadline = start + t_active
        i = i0
        while i + 1 < n:
            nxt = ts[i + 1]
            if nxt > deadline or (nxt - ts[i]) > t_idle:
                break
            i += 1
        last = ts[i]
        if i + 1 < n:
            emit = (last + t_idle) if (ts[i + 1] - last) > t_idle else deadline
        else:
            emit = min(last + t_idle, max(capture_end, last))
        out.append((i0, i, max(emit, last)))
        i0 = i + 1
    return out


def records_per_query(ep: Episode, **_) -> list[tuple[list[int], float]]:
    """Per-query consumer: the record IS the query, available as it arrives."""
    return [([i], q.ts) for i, q in enumerate(ep.queries)]


def records_flow(ep: Episode, t_active: float = 1800.0, t_idle: float = 15.0,
                 capture_end: float | None = None, **_) -> list[tuple[list[int], float]]:
    """Flow-record consumer: 5-tuple flows with active and idle timeouts."""
    end = capture_end if capture_end is not None else ep.t1
    by_flow: dict[tuple, list[int]] = defaultdict(list)
    for i, q in enumerate(ep.queries):
        by_flow[(q.src, q.dst, q.sport, 53, 17)].append(i)
    out: list[tuple[list[int], float]] = []
    for idxs in by_flow.values():
        idxs.sort(key=lambda i: ep.queries[i].ts)
        ts = [ep.queries[i].ts for i in idxs]
        for i0, i1, emit in export_records(ts, t_active, t_idle, end):
            out.append((idxs[i0:i1 + 1], emit))
    out.sort(key=lambda r: r[1])
    return out


def records_session(ep: Episode, window_s: float = 300.0, **_) -> list[tuple[list[int], float]]:
    """Session consumer: queries grouped by silence < window_s, one record each.
    Credited with the EARLIEST emission physically possible (its last packet), so
    every session figure is a LOWER BOUND on that family's true floor."""
    groups: list[list[int]] = []
    cur: list[int] = []
    for i, q in enumerate(ep.queries):
        if cur and (q.ts - ep.queries[cur[-1]].ts) > window_s:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return [(g, ep.queries[g[-1]].ts) for g in groups]


CONSUMERS = {"per-query": records_per_query, "flow": records_flow, "session": records_session}


# --------------------------------------------------------------------------- #
# Floors                                                                        #
# --------------------------------------------------------------------------- #

def timeline(ep: Episode) -> tuple[list[float], list[int]]:
    ts, cum, run = [], [], 0
    for q in ep.queries:
        run += q.payload
        ts.append(q.ts)
        cum.append(run)
    return ts, cum


def fraction_lost_by(ep: Episode, alert_ts: float, overhead: int = 0) -> float:
    """V(alert_ts) / V(total). `overhead` removes a fixed per-query cost from every
    query before accounting -- the A5 unit-sensitivity check."""
    total = sum(max(0, q.payload - overhead) for q in ep.queries)
    if total <= 0:
        return float("nan")
    lost = 0
    for q in ep.queries:
        if q.ts <= alert_ts + 1e-12:
            lost += max(0, q.payload - overhead)
        else:
            break
    return lost / total


def overhead_estimate(ep: Episode) -> int:
    """Assumed per-query protocol overhead for the A5 sensitivity check.

    The shortest queries a tunnel sends are polls and acks, so the 5th-percentile
    length is a natural estimate -- but in a fixed-length tunnel the shortest
    query IS a full data chunk, and taking it as overhead would erase the whole
    asset. So the estimate is capped at half the median length: "assume up to half
    of a typical query is overhead". Removing a CONSTANT per-query cost leaves the
    ratio V(t)/V(total) exactly unchanged unless query lengths vary over time, so
    any non-zero delta this produces is attributable to length variation alone.
    """
    lens = sorted(q.payload for q in ep.queries if q.payload > 0)
    if not lens:
        return 0
    p5 = lens[int(0.05 * (len(lens) - 1))]
    med = lens[len(lens) // 2]
    return int(min(p5, 0.5 * med))


def measure_floor(ep: Episode, consumer: str, **kw) -> dict:
    """The architectural floor for one consumer on one episode (oracle, A4)."""
    recs = CONSUMERS[consumer](ep, **kw)
    if not recs:
        return {}
    first_emit = min(r[1] for r in recs)
    t_eff = max(0.0, first_emit - ep.t0)
    d = ep.duration
    measured = fraction_lost_by(ep, first_emit)
    h = overhead_estimate(ep)
    alt = fraction_lost_by(ep, first_emit, overhead=h)
    closed = min(1.0, t_eff / d) if d > 0 else 1.0
    return {
        "episode": ep.eid, "tool": ep.tool, "family": ep.family, "consumer": consumer,
        "file": ep.meta.get("file", ""), "direction": ep.meta.get("direction", ""),
        "rtype": ep.meta.get("rtype", ""),
        "n_queries": ep.n_queries, "n_ports": ep.n_ports,
        "asset_units": ep.asset, "duration_s": d,
        "t_eff_s": t_eff, "n_records": len(recs),
        "floor_measured": measured, "floor_closed_form": closed,
        "excess_pp": (measured - closed) * 100.0,
        "overhead_est": h, "floor_overhead_removed": alt,
        "unit_delta_pp": (alt - measured) * 100.0 if alt == alt else float("nan"),
        "encoding_cv": episode_encoding_cv(ep),
    }


# --------------------------------------------------------------------------- #
# Operating point — the manuscript's a-priori rule (closes reviewer item M2)     #
# --------------------------------------------------------------------------- #

_TINY_F32 = 1.401298464324817e-45        # the smallest positive float32: "alert on any positive score"
_MAX_F32 = 3.4028234663852886e38         # the largest finite float32


def _next_up_f32(x: float) -> float:
    """The smallest float32 strictly above x, as a float. Detector scores come out of float32 models, and a
    threshold held in float64 can round back onto a tie when the comparison is made in float32."""
    if x != x:
        return x
    if x >= _MAX_F32:
        return math.inf
    if x < -_MAX_F32:
        return -_MAX_F32
    if 0.0 <= x < _TINY_F32:                           # zero, and values that would round to zero in float32
        return _TINY_F32
    f = struct.unpack("<f", struct.pack("<f", x))[0]
    if f > x:
        return f
    b = struct.unpack("<i", struct.pack("<f", f))[0] + (1 if f > 0 else -1)
    return struct.unpack("<f", struct.pack("<i", b))[0]


def max_alerts_within(n: int, budget: float) -> int:
    """Largest k whose upper 95% Wilson bound on k/n is inside the budget; -1 if even k = 0 is not."""
    if n <= 0 or wilson_upper(0, n) > budget:
        return -1
    lo, hi = 0, n
    while lo < hi:                                     # wilson_upper is increasing in k
        mid = (lo + hi + 1) // 2
        if wilson_upper(mid, n) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _budget_used(upper: float, budget: float) -> float:
    return upper / budget if budget > 0 else (0.0 if upper == 0 else math.inf)


# What set the threshold (the `limit` of pick_threshold's meta dict):
LIMITS = ("budget", "tie plateau", "zero scores", "all fit", "not certifiable", "over budget")


def pick_threshold(benign_scores: Sequence[float], budget: float,
                   grid: Sequence[float] | None = None) -> tuple[float, dict]:
    """Most aggressive threshold whose UPPER 95% Wilson bound on the training burden sits inside the budget.
    Headroom comes from the number of benign units actually observed. Fixed a priori; never inspects held-out data.

    Scores are probabilities in [0, 1] (NaN counts as the lowest score: a unit that can never alert). A unit alerts
    when its score is >= the threshold.

    v4.2: with no grid the answer is exact to the resolution of the scores. The budget allows at most k benign
    units to alert; the (k+1)-th highest score must stay silent, so the threshold is the smallest observed score
    above it (an observed value, so a comparison in any float width agrees with it; the top of the gap between two
    observed scores). Only when no observed score lies above it is the threshold made up: the next float32 above
    the pivot, or the smallest positive float32 when the pivot is zero. v4.1 searched a grid whose lowest point
    was 0.005, and in the 7 Oct run that point was already inside the budget in 54 of 54 folds: the rule never
    bound. Pass `grid` to get that search back (its unreachable case now returns inf, as the rule does).

    The meta dict says what set the threshold in `limit`, and how much of the budget the training units use in
    `budget_used` (the Wilson upper bound over the budget; above 1 means over):
      "budget"           k_allowed units alert and one more would break the budget: the budget bound exactly;
      "tie plateau"      a group of equal scores straddles the budget's edge and cannot be split, so fewer than
                         k_allowed units alert (an RF with 300 trees gives only 301 distinct scores);
      "zero scores"      the model scores more benign units zero (or NaN) than the budget lets alert; the
                         threshold is 'any positive score' and the model, not the budget, limits sensitivity;
      "all fit"          the budget admits alerting on every benign unit;
      "not certifiable"  fewer than 381 units at a 1% budget: even zero alerts has an upper bound over the
                         budget, so nothing alerts (threshold inf);
      "over budget"      the units that alert anyway exceed the budget: only a score of +inf, which no probability
                         is, can do that. It is reported rather than hidden.
    `budget_bound` is True for the first two, False for the next two and None for the last two."""
    if budget != budget:
        raise ValueError("budget is NaN")
    scores = list(benign_scores)                       # a generator is consumed once
    n_nan = sum(1 for x in scores if x != x)
    s = sorted(-math.inf if x != x else x for x in scores)    # a NaN compares False with every threshold: it never alerts
    n = len(s)
    if n == 0:
        return (math.inf, {"n_benign": 0, "reason": "no benign units", "budget_bound": None,
                           "limit": "not certifiable", "budget_used": math.nan})
    if grid is not None:
        positive = [g for g in grid if g > 0]
        lowest = min(positive) if positive else None   # the most sensitive setting that is not "alert on everything"
        for thr in sorted(grid):
            k = n - bisect.bisect_left(s, thr)
            upper = wilson_upper(k, n)
            if upper <= budget:
                return (thr, {"n_benign": n, "train_burden": k / n, "train_upper": upper, "threshold": thr,
                              "budget_bound": lowest is not None and thr > lowest,
                              "budget_used": _budget_used(upper, budget), "n_nan": n_nan})
        k = n - bisect.bisect_left(s, math.inf)
        return (math.inf, {"n_benign": n, "train_burden": k / n, "train_upper": wilson_upper(k, n),
                           "threshold": math.inf, "reason": "budget unreachable", "budget_bound": None,
                           "limit": "not certifiable", "budget_used": _budget_used(wilson_upper(k, n), budget),
                           "n_nan": n_nan})
    k_max = max_alerts_within(n, budget)
    if k_max < 0:                                      # too few benign units to certify even zero alerts: alert on nothing
        k = n - bisect.bisect_left(s, math.inf)        # (only a score of +inf could still alert)
        upper = wilson_upper(k, n)
        return (math.inf, {"n_benign": n, "train_burden": k / n, "train_upper": upper, "threshold": math.inf,
                           "reason": "budget unreachable", "budget_bound": None, "limit": "not certifiable",
                           "budget_used": _budget_used(upper, budget), "k_allowed": k_max, "n_nan": n_nan})
    if k_max >= n:
        thr, limit = 0.0, "all fit"                    # the budget admits alerting on everything
    else:
        pivot = s[n - k_max - 1]                       # the (k_max+1)-th highest score: it has to stay silent
        if pivot == 0.0 or pivot == -math.inf:         # (-inf: the NaN scores, which can never alert)
            thr, limit = _TINY_F32, "zero scores"      # the model's own zeros, not the budget, limit sensitivity
        else:
            j = bisect.bisect_right(s, pivot)
            thr = s[j] if j < n else _next_up_f32(pivot)
            limit = "budget"
    k = n - bisect.bisect_left(s, thr)
    if limit == "budget" and k < k_max:
        limit = "tie plateau"                          # equal scores at the edge: the budget cannot be spent in full
    upper = wilson_upper(k, n)
    if upper > budget:
        limit = "over budget"
    bound = {"budget": True, "tie plateau": True, "zero scores": False, "all fit": False}.get(limit)
    return (thr, {"n_benign": n, "train_burden": k / n, "train_upper": upper, "threshold": thr,
                  "budget_bound": bound, "limit": limit,
                  "budget_used": _budget_used(upper, budget), "k_allowed": k_max, "n_nan": n_nan})


def limit_summary(metas: Iterable[dict]) -> dict:
    """Counts of what set each fold's threshold (see `pick_threshold`), and how much of the budget was used where
    the budget could be certified. Used by the digest so that 'the budget was binding' is only said when it was."""
    metas = list(metas)
    out = {name: sum(1 for m in metas if m.get("limit") == name) for name in LIMITS}
    used = sorted(m["budget_used"] for m in metas
                  if m.get("limit") != "not certifiable" and m.get("budget_used") is not None
                  and m["budget_used"] == m["budget_used"])
    mid = len(used) // 2
    med = math.nan if not used else (used[mid] if len(used) % 2 else (used[mid - 1] + used[mid]) / 2)
    out.update(folds=len(metas), used_median=med,
               used_min=(used[0] if used else math.nan), used_max=(used[-1] if used else math.nan))
    return out


def alert_time(ep: Episode, consumer: str, score_of, threshold: float, **kw) -> float | None:
    recs = CONSUMERS[consumer](ep, **kw)
    fired = [emit for idxs, emit in recs if score_of(ep, idxs, emit) >= threshold]
    return min(fired) if fired else None


def measure_ebd(ep: Episode, consumer: str, score_of, threshold: float, **kw) -> dict:
    """EBD for a real detector, against the floor that binds it."""
    fl = measure_floor(ep, consumer, **kw)
    t = alert_time(ep, consumer, score_of, threshold, **kw)
    ebd = 1.0 if t is None else fraction_lost_by(ep, t)
    fl.update({"alerted": t is not None, "alert_ts": t, "ebd": ebd,
               "gap_pp": (ebd - fl["floor_measured"]) * 100.0,
               "at_floor": abs(ebd - fl["floor_measured"]) < 1e-9})
    return fl


# --------------------------------------------------------------------------- #
# ibHH-style streaming accumulator (outside the record-consuming family)        #
# --------------------------------------------------------------------------- #

def unique_info_trace(queries: Sequence[Query], window_s: float) -> list[tuple[float, int]]:
    """Exact count of UNIQUE subdomain characters sent to one registered domain
    within a sliding window -- the quantity ibHH estimates with a constant-memory
    sketch. Exact counting here, so this is ibHH-STYLE, not ibHH itself."""
    out: list[tuple[float, int]] = []
    win: deque = deque()
    cnt: Counter = Counter()
    info = 0
    for q in queries:
        rd = registered_domain(q.qname)
        name = q.qname.rstrip(".").lower()
        sub = name[: -len(rd) - 1] if name.endswith("." + rd) else ""
        while win and q.ts - win[0][0] > window_s:
            _, old = win.popleft()
            cnt[old] -= 1
            if cnt[old] == 0:
                info -= len(old)
                del cnt[old]
        if cnt[sub] == 0:
            info += len(sub)
        cnt[sub] += 1
        win.append((q.ts, sub))
        out.append((q.ts, info))
    return out


def ibhh_threshold(benign_streams: Sequence[Sequence[Query]], window_s: float,
                   budget: float) -> tuple[float, dict]:
    """Smallest information threshold whose upper Wilson bound on the fraction of
    TRAINING benign streams that ever reach it sits inside the budget."""
    maxima = sorted(max((v for _, v in unique_info_trace(s, window_s)), default=0)
                    for s in benign_streams)
    n = len(maxima)
    if n == 0:
        return (float("inf"), {"n_benign_streams": 0})
    for thr in sorted(set(maxima)) + [maxima[-1] + 1]:
        k = n - bisect.bisect_left(maxima, thr)
        if wilson_upper(k, n) <= budget:
            return (float(thr), {"n_benign_streams": n, "train_burden": k / n,
                                 "train_upper": wilson_upper(k, n)})
    return (float("inf"), {"n_benign_streams": n})


def ibhh_alert_time(ep: Episode, threshold: float, window_s: float) -> float | None:
    for ts, v in unique_info_trace(ep.queries, window_s):
        if v >= threshold:
            return ts
    return None


# --------------------------------------------------------------------------- #
# Runtime invariants                                                            #
# --------------------------------------------------------------------------- #

class InvariantViolation(AssertionError):
    pass


def check_invariants(ep: Episode, floors: dict[str, dict], strict: bool = True) -> list[str]:
    """HARD (raise): I1 reconciliation, I2 monotone timeline, I3 emission never
    precedes its traffic, I5 pipeline-position ordering, I6 EBD >= floor.
    SOFT (report): I4 measured floor vs the closed form min(1, T/D) -- a
    back-loaded transfer legitimately sits below it; asserting it would hide that."""
    notes: list[str] = []
    if sum(q.payload for q in ep.queries) != ep.asset:
        raise InvariantViolation(f"I1 reconciliation failed on {ep.eid}")
    ts, cum = timeline(ep)
    if any(b < a for a, b in zip(cum, cum[1:])) or any(b < a for a, b in zip(ts, ts[1:])):
        raise InvariantViolation(f"I2 timeline not monotone on {ep.eid}")
    for name, fl in floors.items():
        if not fl:
            continue
        if fl["t_eff_s"] < -1e-9:
            raise InvariantViolation(f"I3 record emitted before its traffic on {ep.eid}/{name}")
        if fl["excess_pp"] < -1e-6:
            notes.append(f"I4 {ep.eid}/{name}: measured floor {fl['excess_pp']:.2f} pp "
                         f"below the closed form (transfer ran behind the constant-rate diagonal)")
        if "ebd" in fl and fl["ebd"] + 1e-9 < fl["floor_measured"]:
            raise InvariantViolation(
                f"I6 EBD {fl['ebd']:.6f} < floor {fl['floor_measured']:.6f} on "
                f"{ep.eid}/{name} -- a detector cannot beat its own floor. "
                f"Proposition 1 is falsified, or there is a bug.")
    order = ["per-query", "flow", "session"]
    have = [floors[k]["floor_measured"] for k in order if floors.get(k)]
    for a, b in zip(have, have[1:]):
        if b + 1e-9 < a:
            msg = f"I5 pipeline-position ordering violated on {ep.eid}: {have}"
            if strict:
                raise InvariantViolation(msg)
            notes.append(msg)
    return notes
