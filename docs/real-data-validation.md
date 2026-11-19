# Real-data validation of the architectural EBD floor

**Dataset:** DAPT2021 / "Unraveled" (Arizona State University)
**Detector:** EDXGB (Cai et al.) — feature pipeline + XGBoost predictor
**Reproduce:** `uv run ebd-real` and `uv run ebd-edxgb` (see below)

---

## 1. Data source and how it was obtained

DAPT2021 is distributed from a **public S3 bucket with no credentials**:

```bash
aws --no-sign-request --region us-east-1 s3 ls s3://dapt2021/
```

| prefix | contents |
|---|---|
| `processed/network-flows/` | 173 labeled flow CSVs (~3.7 GB), **nfstream** schema |
| `raw/network-captures/` | **176 PCAP files (~477 GB)**, unlabeled |

**Trap worth recording.** The project's GitLab mirror
(`gitlab.thothlab.org/dapt2021/apt-dataset`) publishes only the *processed*
half. Checking it alone leads to the conclusion that no PCAP exists — which is
exactly what stalled the first attempt at this experiment. The S3 bucket is the
authoritative distribution.

**Why not SCVIC-APT-2021.** SCVIC is the other dataset EDXGB targets, but IEEE
DataPort does not distribute its PCAP, so byte-level ground truth cannot be
reconstructed from it at all. DAPT2021 is the same research lineage
(DAPT2020 → DAPT2021/Unraveled) and EDXGB's released source names it
explicitly — `extracted_by_NFStream = ['Unraveled']` — so **no feature had to be
invented to run the detector on it.**

## 2. Episode audit

Scanning all 173 labeled CSVs (0 files missing) found **7,522 flow rows** labeled
`Stage = Data Exfiltration`:

| Activity | rows | MITRE |
|---|---|---|
| Data Transfer Size Limits | 6,988 | T1030 |
| Exfiltration over C2 channel | 446 | T1041 |
| Unsecured Credentials | 88 | T1552 |

Flows are **not** incidents. Grouping by `(src_ip, dst_ip, dst_port)` — excluding
source port, so parallel connections of one transfer stay together — and
splitting on silence longer than a flow timeout yields **465 nominal episodes, of
which only 7 carry ≥ 0.1 MB**. The other 458 are trivial.

Grouping is threshold-sensitive (457 episodes at a 300 s gap, 25 at 600 s), which
is reported rather than hidden; `episodes.sensitivity()` produces that table.

**Scope limit, stated up front.** Total exfiltrated across the entire six-week
campaign is **8.94 MB of application-layer payload**, the largest single episode
being 2.4 MB. Every episode is low-and-slow. This dataset can validate the floor
formula and measure a detector against it; **it cannot exercise the blind spot**,
which requires `D ≤ T`.

## 3. Ground truth from PCAP, and its reconciliation

Byte timelines were reconstructed by streaming the raw captures and accumulating
**de-duplicated application-layer payload** in the exfiltration direction. One
vantage point per episode (the host capture its labeled flows came from), so the
same bytes are never counted twice.

The captures are **not clean**: records occur whose declared length overruns the
next valid record header, with timestamp discontinuities across the damage. The
reader resyncs and **counts every byte it skips** — measured loss was
0.0000 %–0.0098 % per episode.

### Reconciliation is exact once the quantities match

Comparing our payload figure directly against the dataset's `src2dst_bytes`
showed a spurious **46.5 %** divergence on a *perfectly* reconstructed episode.
The two count different things:

| quantity | value |
|---|---|
| packets in direction | **19,912** — vs exporter's `src2dst_packets` **19,912** ✓ |
| IP bytes | 2.072 MB |
| + 14 B × 19,912 (Ethernet header) | 0.279 MB |
| = link-layer total | **2.351 MB** — vs exporter's `src2dst_bytes` **2.351 MB** ✓ |
| **payload — the asset** | **1.257 MB** (53 % of what crossed the wire) |

`src2dst_bytes` is a **link-layer** count over **every** packet including pure
ACKs; the asset is **de-duplicated application-layer payload**. Compared like for
like, the reconstruction is **identical to what the exporter saw**.

**EBD is denominated in payload, not wire traffic** — framing is not stolen data.

All **7/7 episodes admitted**, reconciliation **0.0 %–0.2 %**, four exact.

## 4. Measuring T instead of assuming it

`EBD_floor` is only as trustworthy as `T`, the parameter the literature does not
report. Two distinct quantities, both measured:

**Active timeout = 1800 s.** Of 7,522 exfiltration flows the longest is
**1798.25 s** and none exceeds 1800 s — pinning nfstream's default. (The ~60 s
mass in the duration distribution is the *attacker's* beacon interval, not the
exporter's: flows do exceed 60 s.)

**Effective T** is what actually bounds a detector:

> `T_eff = min(active_timeout, time until the episode's earliest-ENDING flow closes)`

A record is emitted when its flow **ends**, and a later, shorter connection can
end before a long-running first one. Using the earliest-*starting* flow instead
overstated one episode's floor by 7× (14.02 % → 1.99 %).

**A consequence worth stating in the paper:** chunking a transfer into many short
connections **shortens** the architectural blind spot — each connection emits its
record sooner. T1030-style chunking evades size-based detection but hands a
flow-record consumer earlier evidence. Against *this* detector class the
attacker's best move is one long, fast connection, not many small ones.

## 5. Result: EDXGB against the floor

Trained on EDXGB's own feature pipeline (PTR/BTR exactly as in the released
`EnvPreparing.py`) with its XGBoost predictor, split by **whole episode**.

### Leakage: verified zero

The decisive question for a result of this shape is whether any scored episode
was in the training data. It was not, and the check is mechanical:

```
held-out flow rows      5,688      (all 7 episodes, held out SIMULTANEOUSLY)
training rows         175,657      (exfil flows in training: 1,834)
accounting        5,688 + 1,834 = 7,522 = every exfiltration flow  ✓
rows both SCORED and TRAINED ON:  0
```

The default split holds out **all seven episodes at once**, which is *stricter*
than leave-one-out: under LOO a model still sees the other six substantial
episodes, whereas here it sees none of them — its only exfiltration training
data is 1,834 flows from the 458 trivial episodes.

Both splits are reported, because they answer different questions. The strict
split asks whether the floor is reachable by a detector that has never seen an
episode like this one; LOO asks how the detector does with representative
training data, and is the fairer test of the detector itself.

| | strict (all 7 out) | leave-one-out |
|---|---|---|
| episodes at floor | **4 / 7** | **5 / 7** |
| mean gap | +22.52 pp | **+7.62 pp** |

Three of the strict-split failures were largely a *training-data* artifact: the
37-hour episode goes from 89.22 % to **0.09 %** once the model has seen
comparable exfiltration. That is worth knowing before concluding anything from a
large gap.

### "At floor" is a stated cutoff, not an eyeballed line

**At floor := gap ≤ 1.00 percentage point of the episode's asset.**

The count is insensitive to where that line sits: under LOO the largest
at-floor gap is **0.07 pp** and the smallest large gap is **45.59 pp** — a
**610×** separation with nothing in between. Any cutoff from ~0.1 pp to ~45 pp
gives the same 5/7.

Both splits, side by side — the floor is a property of the episode and does not
change between them:

| V_e (MB) | D | floor | LOO actual | LOO gap | strict actual | strict gap |
|---|---|---|---|---|---|---|
| 2.413 | 0.4 h | 0.28 % | 0.28 % | **+0.00 %** | 0.36 % | **+0.08 %** |
| 0.226 | 3.6 h | 1.99 % | 1.99 % | **+0.00 %** | 1.99 % | **+0.00 %** |
| 0.231 | 3.6 h | 1.03 % | 1.03 % | **+0.00 %** | 1.03 % | **+0.00 %** |
| 1.257 | 22.2 h | 0.08 % | 0.15 % | **+0.07 %** | 0.37 % | **+0.30 %** |
| 2.135 | 37.4 h | 0.04 % | 0.09 % | **+0.04 %** | 89.22 % | +89.18 % |
| 2.253 | 21.5 h | 0.42 % | 46.01 % | +45.59 % | 46.01 % | +45.59 % |
| 0.425 | 7.4 h | 1.50 % | 100.00 % | +98.50 % | 100.00 % | +98.50 % |
| | | | | **5 / 7 at floor** | | **4 / 7 at floor** |

**5 of 7 episodes land essentially on the floor** under leave-one-out, 4 of 7
under the stricter split. Only one episode (2.135 MB / 37 h) differs between
them, and it differs enormously — 89.18 pp to 0.04 pp — which is precisely the
training-data effect described above, not a property of the detector's
architecture.

### The 46 pp outlier is the most instructive row in the table

That episode is **not** a detection failure in any conventional sense:

```
flows 1293   flagged 1289   missed 4   flow recall 99.69 %
   ends + 25.3 s   score 0.370   MISSED
   ends + 33.0 s   score 0.370   MISSED
   ends + 85.4 s   score 0.462   MISSED
   ends + 93.0 s   score 0.621   FLAGGED   <- earliest possible alert
```

The detector misses exactly the three **earliest-ending** records, and the
episode front-loads: 46 % of its asset moves in the first 100 seconds. So EBD
goes from a floor of 0.42 % to 46.01 % — a **109× penalty for missing 0.31 % of
flows**.

A paper reporting 99.69 % recall would call this a success. Nearly half the
asset was gone. This is the argument for EBD, measured on real traffic with a
real detector rather than constructed to make the point.

### What each direction supports

**The at-floor result is conclusive.** A real classifier attains its architectural
ceiling where the architecture permits. This direction cannot be dismissed with
"a stronger model would have done better" — a stronger model could only *also*
match the floor. It confirms the floor is attainable, not merely theoretical.

**The three large gaps are NOT conclusive.** The deep DNN/CNN extractors were
omitted, so weak reproduction and a genuine detector deficit are not separable
here. They are reported as inconclusive, and they indicate where installing the
full ensemble would actually change what the result means.

## 6. Limitations

1. **n = 7 episodes, 8.94 MB total.** Percentages are of a few megabytes; this
   characterises a beacon, not a heist.
2. **Drip regime only.** Floors are 0.04 %–1.99 %. The `D ≤ T` blind spot is
   untouched, because no public dataset we found contains fast bulk exfiltration.
3. **Architectural simplification.** XGBoost predictor without the deep
   extractors — see §5 for exactly which conclusions survive it.
4. **Our figures will not reproduce EDXGB's published numbers**, by design:
   episode-level splitting is stricter than a flow-level split, since flows from
   one incident are strongly correlated.
5. **One vantage point per episode.** For traffic leaving the network the gateway
   would be the truer egress observation point; the host capture is an upper
   bound on it.

## 7. Reproducing

```bash
uv run ebd-real  --groundtruth data/realdata/groundtruth.jsonl
uv run ebd-edxgb --train data/realdata/train_flows.csv \
                 --groundtruth data/realdata/groundtruth.jsonl
```

Reconstructed ground truth is committed under `data/realdata/`, so the analysis
reruns without re-reading 477 GB of capture.

## 8. Errors found and corrected during this work

Recorded because each was caught by a check rather than by a reader, and the
checks are the reusable part:

| error | how it surfaced | fix |
|---|---|---|
| Payload compared against link-layer bytes | 46.5 % divergence on a perfect reconstruction | compare like for like; EBD stays in payload |
| Episode observed at two vantage points | divergence of **exactly** 0.5001× | reconcile only against flows from the capture actually read |
| Single time-span swept in unrelated traffic | +16.7 % over-count | match the union of per-flow windows |
| Floor as duration-fraction vs EBD as byte-fraction | **negative** gaps | read the floor off the reconstructed timeline |
| `T_eff` from earliest-*starting* flow | **negative** gap persisted | earliest-*ending* flow; records emit at flow close |
| Guessed end byte-offset truncated episodes | identical byte totals across distinct episodes | bound the read by time, not offset |

The gap's sign is now asserted at runtime, so the last two cannot regress
silently.
