# EBD-DNS — widening the empirical base (v4.2, 8 Oct 2026)

Companion to *Exfiltrated Bytes before Detection*. Measures the architectural floor on
`daumel/dns-tunneling-dataset` (raw PCAP) at three pipeline positions over **identical
traffic** — per-query, flow-record, session — plus an ibHH-style streaming accumulator
outside the record-consuming family.

    notebooks/ebd-dns-floor.ipynb   the Kaggle notebook (self-contained; embeds both modules)
    src/ebd/dns/engine.py           episodes, floors, operating points, ibHH-style, invariants
    src/ebd/dns/pcap_io.py          PCAP/PCAPNG -> DNS queries, pure stdlib, about 180k pkt/s on a mixed synthetic capture
    tests/test_dns_engine.py        28 analytic assertions — every floor computable by hand
    tests/test_dns_pcap.py          21 round-trip assertions across containers and link types
    tests/test_dns_v2.py            23 assertions on the real-capture hazards below
    tests/test_dns_v3.py            16 assertions on link types, pcapng variants, diagnostics
    tests/test_dns_v4.py            28 assertions on malformed-message reasons and the address cache
    tests/test_dns_v41.py           21 assertions on slotted records and compact benign units (memory refactor)
    tests/test_dns_v42.py           108 assertions on the iterative name decoder (and its two RFC 1035 rules), the per-packet guard,
                                    split TCP length prefixes (paired by length, sequence number and question count), the exact
                                    operating point and its labels, and the family-level intervals (bootstrap, Student-t, exact)

Run the tests: `PYTHONPATH=src python3 tests/test_dns_v42.py` (or `uv run pytest tests/`). The seven suites hold 245 assertions.
The notebook-level checks need the notebook and its synthetic corpus and are not part of this package:
47 for v4 (linear-time featuriser, per-source ibHH calibration, active-timeout sweep, benign-read
configuration, per-cell timing hooks), 101 for v4.1 (feature matrix, container-limit and memory-in-use
readers, the memory plan and both of its guards, the floor-table checkpoint, the pre-training check) and
69 for v4.2 (the limit's source, a mirror of the 7 Oct failure with pointer loops and split TCP prefixes inside the
university captures, the exact threshold inside the leave-one-family-out cell, family-level lines, the digest,
a budget that cannot be certified, the in-sample labels).

## v4.2 (8 Oct 2026) — what the 7 Oct measurement run (v4.1) exposed, and what two independent reviews of the first v4.2 build then changed

The first measurement run that completed (7 Oct, `SMOKE = False`, v4.1, real xgboost) gave a complete attack side
(18 episodes, 8 families) and a benign side with four defects. **None touches a floor**: floors use no benign
data, so they are the same in every notebook version, mode and session.

1. **The parser ended whole captures on one bad packet.** `_qname` called itself once per compression pointer
   and restarted its label counter on every call, so a name whose pointer leads back to its own start recursed
   until Python gave up (`RecursionError`); the per-capture handler then ended the capture. 4 of the 13 university
   captures that were read stopped that way (at 6.0, 21.8, 29.4 and 51.8 minutes). v4.2: the decoder is iterative,
   keeps v4.1's rules (backward jumps only, at most 8 labels before a jump, at most 128 in a stretch), adds a cap
   of 128 jumps per name (`MAX_NAME_HOPS`) and, from the review, two RFC 1035 rules: **a pointer into the 12-byte
   header is rejected, and so is a name longer than 255 octets on the wire**. Those two rules are the only places
   where v4.2 reads a name differently from v4.1, and they only reject. A packet that raises anywhere in the
   decoder is counted (`packet_errors`, with up to three samples) and skipped; a capture ends early only if the
   container reader itself fails, and then `error` names the failure. On two sets of 60,000 random messages (labels,
   junk, chains, pointers to earlier names; in the second set pointers may lead anywhere) the names and end offsets
   equal v4.1's apart from those two rules (59,998 and 56,949 identical, 2 and 3,051 rejected by them; 452 and 322 of the messages
   hold a pointer loop, which v4.1 cannot finish and v4.2 rejects). A 5,000-pointer chain is rejected instead of fatal.
2. **The 1% budget never bound.** The operating point was searched on a grid, and its lowest useful value (0.005)
   was already inside the budget in 54 of 54 folds, so every detector ran at the most sensitive setting the grid
   allowed, using a tenth of the budget or less. v4.2 sets the threshold exactly: the budget allows at most *k* of
   the *n* training-benign units to alert (the largest *k* whose upper 95% Wilson bound is inside the budget), and the
   threshold is the lowest *observed* score above the (*k*+1)-th highest one, so that a float32 model score and a
   float64 threshold agree on who alerts. If no observed score lies above that one, the threshold is the next float32
   above it; if it is zero (at most *k* units get a positive score) the threshold is the smallest positive float32.
   `pick_threshold` also returns **what set the threshold** (`limit`): *budget*; *tie plateau* (equal scores let fewer than
   *k* units alert); *zero scores* (the model's own zeros, not the budget, limit the sensitivity); *all fit*;
   *not certifiable* (fewer than 381 units cannot certify a 1% budget: the threshold is infinite and nothing alerts);
   *over budget*. It returns `budget_used` (upper Wilson bound / budget), `budget_bound` (True only for *budget* and
   *tie plateau*) and `k_allowed`. NaN scores never alert and are counted (`n_nan`). `grid=` keeps the old search.
3. **A DNS-over-TCP message whose 2-byte length arrived in a segment of its own was read from the wrong offset.**
   A bare 2-byte segment is remembered per direction (at most 4,096 at a time) together with the sequence number of
   the byte after it. The next segment from the same sender is read from offset 0 only if it starts at exactly that
   sequence number, is exactly as long as the prefix announced and holds one question (`tcp_split_read`); anything
   else is read from offset 2 as before. The first v4.2 build paired on length alone, and the review showed that an
   unrelated 2-byte segment could then turn the next ordinary `[length][message]` segment into an empty name.
   `tcp_prefix_seen` counts the remembered prefixes and `tcp_prefix_dropped` those that found no partner. The 7 Oct
   run's malformed-message samples are consistent with this defect for one connection; v4.2 counts it.
4. **The memory limit's source was not recorded.** The digest said 16.0 GB, which is exactly the Kaggle fallback
   constant (a 16 GiB cgroup would have printed 17.2), so the run could not say whether it had read its limit, and the
   guard planned the university slice from that figure: 13 of 56 captures. v4.1 also threw away any reading above
   32 GB on a `/kaggle` machine, which includes a genuine 30 GiB cgroup limit (32.2 GB). Kaggle's session RAM is quoted
   as 16, 29 or 30 GB depending on the source, so **the 16.0 GB of 7 Oct may have been a real 30 GiB limit that was
   thrown away; this is a hypothesis, not a finding**, and the raw readings printed by the next run will settle it.
   v4.2 prints the limit, where it came from and the raw readings (cgroup v2 `memory.max`, cgroup v1, `MemTotal`,
   `/kaggle`) in the first cell, the parse cell, the digest and `dns_provenance.json`; it discards only readings
   above 40 GB (`KAGGLE_PLAUSIBLE_MAX_BYTES`); a fallback says it is a fallback ("assumed … unconfirmed … may be the
   host's"); `MEMORY_LIMIT_GB` overrides everything.

**Family-level statistics.** The digest gives the floors and the attainment per family: the family-weighted mean (the
mean of the family means), a Student-t interval over the family means, the two-stage cluster bootstrap (resample
families, then episodes inside a family), and the families with total loss as *k*/8 with an exact Clopper–Pearson
interval (3/8 gives [8.5%, 75.5%]). Every interval is for the family-weighted mean itself (the 7 Oct results note had
printed the bootstrap interval of the pooled episode mean beside the family-weighted mean; the project note is
corrected). **The bootstrap is anti-conservative at 8 families.** In a simulation with family sizes 4, 4, 3, 3, 2, 2, 1, 1
its nominal 95% interval covered 0.903 of the time (1,000 repetitions; 0.907 in a 2,000-repetition run; 0.86 to 0.94
across the scenarios tried), against 0.948 (0.950; 0.95 to 0.97) for the t interval, which is about 25% wider. Quote the
t interval, or call the bootstrap a 90% interval.

**Also in the digest.** The operating-point threshold range per consumer, what set it and how much of the budget it
used; folds over budget per source by the Wilson lower bound (a source with no held-out units in the sample is shown
as n/a, never as "nan%"); the **informative** ibHH comparison (episodes whose flow floor is under 100%; the "of which"
count had been wrong when ibHH never alerted); the parser's counters; the malformed share of the payloads sent *to*
port 53 (`malformed_c2s`); and an *in-sample* label for a benign source with a single capture. It no longer prints the
flow/per-query ratio, "first last packets" is fixed, and `dns_provenance.json` is strict JSON (no `NaN` tokens).

**What did not change.** On the synthetic mirror (SMOKE True and False, final build), `dns_floors.csv`,
`dns_cadence_sweep.csv`, `dns_episode_decisions.csv`, `dns_episode_sensitivity.csv` and `dns_malformed_samples.csv`
are byte-identical to v4.1's. The floor-summary, attainment-summary, ibHH and parse-statistics tables differ only by
added columns; `dns_attainment.csv` differs in `threshold` and `fpb_pooled`, and `dns_operating_points.csv` in the
thresholds and the burdens that follow from them, plus the new columns. On a 400,000-packet synthetic capture v4.2
returns exactly the same 320,543 queries as v4.1 (identical hash) at about 175k packets/s against 185k.

**Verification.** 108 package assertions (`tests/test_dns_v42.py`) and 69 notebook-level ones, which include a mirror
of the 7 Oct failure: pointer loops, split TCP prefixes and garbage on port 53 inside the university captures. On it,
v4.1 ends 8 of 12 university captures with `RecursionError` and v4.2 reads all 12 and recovers the 240 split-prefix
messages. The self-test inside the notebook fails if the recursive decoder is put back. The 47 notebook-level checks
of v4 and the 101 of v4.1 still pass. Their changes: the v4.1 suite's mock of the limit reader was adapted to the new
reader and the suite gained the memory-limit checks; in the v4 suite one input (130 one-character labels, 261 octets)
is now rejected as "name longer than 255 octets" where v4.1 said "more than 128 labels". No expectation about a
result was loosened. Two independent reviews of the first v4.2 build found the two name rules and the TCP
pairing rule (items 1 and 3), the 32 GB cut and the unlabelled fallback (item 4), the misleading threshold labels (item 2),
the bootstrap's coverage, and digest wording (the "of which" count, "nan%"); all were fixed and tested
before this build.

**Limits.** v4.2 has not run on real bytes. The expected effect on the benign side: thresholds far below 0.005, the
training burden close to (not above) the budget, and held-out university burdens that can exceed it in some folds,
which are reported and never retuned. How many university captures are read still depends on the memory limit; if
the first line of output says FALLBACK and the session has more RAM (the Kaggle session panel shows it), set
`MEMORY_LIMIT_GB` in the CONFIGURATION cell. Parser limits that remain, as in v4.1, none of them quantified on real
data: two length-prefixed messages in one TCP segment keep only the first and a message split over three segments is
not reassembled (both lose TCP queries); a retransmission repeats a query; and about 0.2% of random 12–200-byte
payloads on port 53 happen to parse as a query (replies are not separated from queries by source port).

## v4.1 (5 Oct 2026) — the first measurement run was killed for memory

The first measurement run (`SMOKE = False`) of the v4 notebook on Kaggle ended in code cell 12, the
leave-one-family-out training cell, with *"Your notebook tried to allocate more memory than is available."*
The v4 notebook had promised "well under 10 GB". That was an **extrapolation** from a synthetic replica at
half the corpus (4.85 GB peak), not a measurement of the full corpus, and it left no margin for a session
whose limit and real per-query cost were unknown. **What the failed session's limit was, and how much it had in
use, is not known**: the lines above the crash were not available. Kaggle's session RAM is quoted as 16, 29 or
30 GB depending on the source and the date, and was not verified, so v4.1 measures the container instead of
trusting a number.

What v4.1 changes. None of it can change a result: every CSV of the v4.1 build is byte-identical to the v4
build's on the synthetic mirror (SMOKE True and False), and the stable provenance fields match.

- **Smaller records.** `Query` and `Episode` are slotted dataclasses (Python 3.10+; older interpreters fall back
  to plain ones). Benign units get short ids, one `meta` dict per capture instead of one per unit, and one string
  per domain; `benign_units` groups per capture file. Per object (Python 3.13): a query 216 → 168 B, a singleton
  unit 646 → 267 B; the same pattern on 3.11 and 3.12.
- **Smaller peaks.** The feature matrix is filled row by row into a preallocated array (v4 built a list of lists
  first), the record lists are freed as soon as the matrices exist, freed memory is handed back to the OS
  (`gc` + `malloc_trim`), and model threads are capped at the cores the container may use (at most 8).
- **A memory plan with two guards.** The parse cell reads the container's memory limit (cgroup v2, then v1,
  then `/proc/meminfo`; on a `/kaggle` machine an unreadable limit, or one above 32 GB, is taken as 16 GB;
  `MEMORY_LIMIT_GB` overrides). After the first university capture it knows what one capture really costs. If the
  rest would not fit in `MEM_FRAC_PARSE` (30%) of the limit, with a 1.5x safety factor, it reads an evenly
  spaced subset (never fewer than 8) and says so in the digest; if memory still passes the share, it stops
  reading. Malicious and top-1M captures are always read in full.
- **A clean stop instead of a kill.** Before the training cell, if memory in use plus 0.5 GB plus 200 B per benign
  query would exceed 95% of the limit, the notebook stops with the reason and the fix.
- **Nothing lost.** The floor tables, parse statistics and malformed-message samples are written as soon as the
  sweep finishes, before any model is trained, so a later kill cannot take them.
- **Visible.** Every phase prints `[memory] <phase>: X GB in use of Y GB`; the same marks go to
  `dns_provenance.json` and the digest.

Measured on synthetic replicas at the real packet counts (xgboost replaced by a gradient-boosting stand-in;
replica *results* are not findings):

| | v4 | v4.1 |
|---|---|---|
| half corpus (28 university files, 3.6M benign queries): peak memory | 4.85 GB | 2.77 GB |
| half corpus: time | 589 s | 647 s |
| full corpus (56 university files, 6.1M benign queries): peak memory | killed at about 6 GB in the sandbox | **4.2 GB** |
| full corpus: time | not measured | **759 s** |
| full corpus, default limit detection (8.4 GB sandbox): guard read 29 of 56 university files | not measured | 629 s, 2.8 GB |

The replica has about twice as many benign units per query as the 4 Oct smoke run implies for the real corpus
(0.75 against about 0.35), which leans conservative, but real captures can still cost more per query than synthetic
ones; the guard measures that on the first real capture. v4.1 has not run on real bytes or with real xgboost.
If it is killed anyway, the `[memory]` lines above the crash and the saved floor tables are what to send back.

## v4 (5 Oct 2026) — what the first real Kaggle run (SMOKE=True, 4 Oct) taught

The smoke run completed on real bytes (8 captures, 0 parse errors, real XGBoost, I6 held on 24
pairs), but it took **31 minutes** where the notebook promised about one.

- **Speed (a defect of mine).** `featurise()` computed a standard deviation with a lambda that
  recomputed the mean for *every element*, so its cost was quadratic in the size of one unit. The
  session and flow units of the long captures hold up to 134k queries (about 4 minutes per call).
  One-pass std gives bit-identical features; cost is now 5–10 µs per query (a 134k-query unit: 1.3 s, was ~4 minutes). A guard cell times
  `featurise()` on the largest unit and **stops the run** if it is ever super-linear again.
- **Benign corpus.** The default (measurement) run reads **all 56** university captures (first
  300k packets each) and all 68 top-1M captures; v3 read 16 of 56. Malicious captures are never
  truncated. Benign train/held-out is split by capture file, per source.
- **ibHH calibration.** v3 calibrated one threshold on all benign streams pooled, which let the
  easy top-1M traffic dilute the budget: pooled burden 0.89% but **1.43% [1.29, 1.59] on the
  university** — over the 1% budget on the one realistic source. v4 takes the **largest of the
  per-source thresholds** (sources with fewer than 400 streams cannot certify a 1% budget and are
  excluded from the maximum). The pooled-rule threshold is kept and scored, so the effect is on
  record.
- **Diagnostics.** The smoke run's university slice had 52,599 port-53 messages rejected by the
  query parser (against 95,842 accepted) and no way to say why. The parser now classifies every
  rejection (`malformed_reason`), keeps up to three samples per reason, and the notebook exports
  them as `dns_malformed_samples.csv` and flags any source above 5%.
- **Flow timer.** The flow floor depends on the exporter's *active timeout*, which vendor defaults
  span 30× (Juniper 60 s, CICFlowMeter 120 s, Cisco 1800 s). A new sweep axis (30–1800 s at
  idle 15 s) reports how the flow floor moves; a transfer shorter than the timer is a total loss.
- **Attainment.** A second count, *within 0.01 pp of the floor*, beside the strict
  `|EBD − floor| < 1e-9` rule: a per-query detector that fires on the 2nd or 3rd query is "not at
  the floor" strictly and indistinguishable from it within the A5 payload-unit uncertainty.
  The strict rule stays the primary count.
- **Provenance.** Per-cell wall-clock and peak memory (Jupyter only) land in
  `dns_provenance.json` and in the digest, so a long run can be explained from its outputs.
- **Memory.** Address text is cached (one string per host instead of one per query). A synthetic
  replica at half the corpus ran in about 10 minutes with a 4.9 GB peak, and the notebook said
  "well under 10 GB" for the full run. **The run did not fit on Kaggle: see v4.1 above.**

**Verification limits.** The real captures cannot be fetched into the development sandbox
(Kaggle's storage host is blocked), and real xgboost cannot be installed there, so the runtime
and memory figures come from synthetic replicas at real packet counts with a gradient-boosting
stand-in. Replica *results* are not findings. No Jupyter kernel was available either: the timing
hooks were exercised against a stand-in for IPython's event API.

## v3 (4 Oct 2026) — the first Kaggle runs stopped with "No episodes built"

The v2 notebook only *printed* a warning when discovery found no captures, then failed three
cells later with a bare assert. That reproduces exactly with no dataset attached (importing an
`.ipynb` into Kaggle does not bring its inputs). It would also follow from captures in a link
type the parser could not read. The corpus comes from three labs (Chen et al.: cobaltstrike,
dns2tcp, dnscat2, iodine, ozymandns; Gao et al.: tcp-over-dns, dnspot, top-1M benign; the
dataset author: dnsexfiltrator ×2; Singh et al.: university benign), so capture set-ups differ.
v3:

- **Discovery** follows symlinked mounts and **stops** with attach instructions if no
  malicious capture is found.
- **Parser** reads Linux cooked v2 (276, `tcpdump -i any` on libpcap ≥ 1.10), raw IPv4/IPv6
  (228/229), NFLOG (239), FCS-flagged link-type fields, pcapng obsolete Packet Blocks, and
  IPv6 extension headers. An unknown link type is **counted and reported, never guessed**.
- **Parse cell** shows link types per tool, explains every capture that yields no DNS
  queries (link type, no IP, no port-53 traffic, malformed), and stops if none yields any.
- **Episode cell** prints a per-capture verdict table *before* any stop, and replaces the
  bare assert with the reasons.

## Corpus decisions (from the dataset's full file listing)

| | Used | Why |
|---|---|---|
| malicious `_up` | 15 captures | upstream = exfiltration; payload rides in QNAMEs |
| malicious, no direction | dnspot, tcp-over-dns ×2 | included, flagged, reported separately |
| malicious `_down` | 13 captures — **excluded** | data flows *into* the network: wrong denominator |
| benign top-1M | 68 captures | Gao et al.'s lookups of popular domains (names with a subdomain) |
| benign university | 56 hourly 2016 captures, first 300k packets each (all of each with `FULL`) | real campus DNS: a realistic false-positive baseline |

`dnsexfiltrator` and `dnsexfiltrator_modified` are one family in the leave-one-out split.
Expect 18 episodes across 8 families — tens, not hundreds. Four of the five Chen et al. captures
are 50,000-packet prefixes, so for those tools the "transfer" is whatever the first 25,000
queries carried (53–177 s): a flow-record total loss there is a statement about a transfer of
that size and speed, not about the real exfiltration's length.

## Real-capture hazards the engine guards against

- **Client IPs reused across captures** → episodes are scoped to their capture file.
- **Background DNS inside tool captures** → only the dominant registered domain per file is kept.
- **Captures that saw both legs** (stub→resolver and resolver→authoritative) → counted once.
- **Raw non-DNS bytes on port 53** (iodine raw mode) → surfaced as a per-tool malformed share.
- **Payload unit (A5)** → every floor re-measured with up to half of each query treated as
  overhead; the difference is reported, not assumed away.
- **Look-ahead in features** → features split names against the registered domain only.
- **Truncated exfiltration captures** → malicious captures are never truncated.
- **Plotting failure** → export runs before figures; each figure is isolated.

## Invariants

Hard: I1 reconciliation · I2 monotone timeline · I3 emission never precedes its traffic ·
I5 pipeline-position ordering · **I6 EBD ≥ floor** (the falsification test).
Soft: I4 measured floor vs `min(1, T/D)` — a transfer that runs behind the constant-rate
diagonal legitimately sits below it.
