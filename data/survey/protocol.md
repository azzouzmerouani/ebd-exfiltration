# Coded survey: what do network exfiltration detectors consume?

## Why this exists

The architectural floor applies to detectors whose unit of inference is a
*completed record*. The paper claims that this characterises the dominant
family in the network exfiltration detection literature. Until now that claim
was argued architecturally rather than coded, and was flagged as an open item in
three places in the draft. This is the coding.

## Question

**Primary (load-bearing).** For each paper, what is the *unit of inference* —
the object the detector assigns a score to?

| code | meaning | record consumer? |
|---|---|---|
| `packet` | scores individual packets or packet-level fields as they arrive | no |
| `payload_dpi` | scores reassembled payload / deep packet inspection | no |
| `flow_record` | scores a completed flow record (NetFlow / IPFIX / CICFlowMeter / nfstream / Zeek conn) | **yes** |
| `session_aggregate` | scores an aggregate over several completed flows or a fixed window | **yes** |
| `host_timeseries` | scores a per-host aggregate accumulated over hours to weeks | **yes** |
| `host_endpoint` | host telemetry (API calls, syscalls, hardware counters) — out of scope | n/a |

A paper is a **record consumer** if it cannot produce a score before some
record covering the traffic has been emitted, i.e. codes `flow_record`,
`session_aggregate` or `host_timeseries`.

**Secondary (screening level, NOT load-bearing).** Does the paper state the
telemetry cadence — the exporter's active/idle timeout or aggregation window —
for the data it evaluates on?

## Sampling frame

Papers were retrieved through Consensus (Semantic Scholar / Scopus / PubMed /
arXiv) using the query set below, and supplemented with the works already cited
in this paper and the two detectors examined in depth.

```
network data exfiltration detection using flow records machine learning
data exfiltration detection CICFlowMeter features classifier evaluation network traffic
advanced persistent threat dataset raw packet captures labelled exfiltration benchmark
evaluation methodology pitfalls machine learning network intrusion detection experimental bias
flow export active timeout NetFlow IPFIX aggregation granularity impact on anomaly detection
```

**Inclusion.** Network-based detection of data exfiltration or of intrusions
whose evaluation includes an exfiltration class, with an empirical evaluation,
2013–2026.

**Exclusion.** Host-only detection (API-call, syscall, hardware-counter), pure
surveys (recorded separately, not coded as detectors), and papers whose unit of
inference could not be determined.

## Honest limitations, stated because they bound what may be concluded

1. **This is a keyword-sampled frame, not a systematic review.** It follows no
   PRISMA protocol and makes no completeness claim. It is sufficient to test
   whether record consumption is *common*, and insufficient to estimate its
   exact prevalence in the field.
2. **Coding is from abstracts and methods sections.** The primary variable
   (unit of inference) is stated explicitly by essentially every paper, so
   abstract-level coding is reliable for it. The secondary variable (is the
   cadence reported) is **not** reliably determinable from an abstract: a value
   buried in a full-text methods paragraph would be coded `no` here. The
   secondary column is therefore reported as a screening observation that
   motivates the recommendation, and no percentage is claimed from it.
3. **Single coder, no inter-rater agreement.** A second coder would be required
   before treating the counts as a measurement rather than an indication.

## Columns

`key, year, venue, unit_of_inference, record_consumer, telemetry_tool,
cadence_stated, notes`

`cadence_stated` values: `yes` (an active timeout, idle timeout or aggregation
window is given), `partial` (granularity is discussed as a variable but no
single deployment value is pinned), `no` (not found at screening level),
`n_a` (not a record consumer, so no cadence applies).

---

## Frame extension (revision 5)

The original frame under-sampled the DNS covert channel, which is both the most
heavily published single exfiltration vector and — as it turns out —
architecturally atypical. One query was added:

```
DNS tunneling data exfiltration detection machine learning real-time
```

Eight papers were added from it. One retrieved paper was **excluded**: Ahmed et
al., IFIP/IEEE IM 2019, is the conference precursor of the same system as Ahmed
et al., IEEE TNSM 2020, and coding both would double-count one detector.

**This changed the headline number, and the change is reported rather than the
more favourable original.** Record consumers fall from 27/31 (87.1%) to 28/39
(71.8%). The mechanism is visible in the coding: of the eight DNS papers added,
seven score individual queries and one consumes flow records.

The extension is deliberately *adverse* — it targets a subfield known in advance
to be per-query, so it can only push the record-consumer fraction down. We
report it in that spirit: a frame extension chosen to favour the paper's claim
would not be evidence. A reader should treat 71.8% as the better estimate and
87.1% as an artefact of a frame that happened to miss a subfield.

---

## Second coding (revision 6, 9 October 2026)

**Why.** The panel asked for a second coder with a reported agreement statistic
(roadmap item D2), because the first coder is also the party the result favours.

**Who, and what that is.** The second coding was made by an LLM-based coder
(Claude, Anthropic), run as three separate sessions of 13 papers each. The first
coding was also made with the assistance of a large language model (disclosed
by the author on 20 October 2026), so the two codings are not independent: they
differ in the information given, the level of reading and the masking, not in
the kind of coder. Their agreement is a consistency check between two assisted
codings, not inter-rater reliability. No human coder was used (author decision,
20 October 2026), and the paper reports RQ1 descriptively.

**Blinding.** Two steps, so that no first-coding information reached the coder:

1. *Identification* (`sample_bibliography.csv`). A separate agent, which did see
   the first coder's notes, resolved each citation key to its title, authors,
   venue, volume/pages and DOI, and nothing else. DOIs were checked against
   Crossref, the publisher, DataCite or the DOI handle service; the one that could
   not be checked (Silva et al. 2022) is marked.
2. *Coding* (`second_coding.csv`). The coders received opaque ids (`S01`-`S39`,
   shuffled with seed 20261009), the bibliographic record, links to the abstract
   and any open full text, and the codebook above verbatim (unit of inference,
   telemetry tool, cadence stated) plus the inclusion rule. They were told not to
   use any published classification of the papers and not to read any local file.
   They did not see the citation keys (some keys name the coded unit, e.g.
   `...dpi`), the first coder's notes, the hypothesis, or the counts.

Each row records what the coder read (`source_level`: abstract only, or abstract
and full text), a verbatim quote supporting the unit code with its URL, and a
confidence. The coder read the full text of 22 of the 39 papers.

**Agreement** (`ebd-survey-agreement`, `results/survey_agreement.csv`). Computed
over the 34 papers for which the second coder could determine the unit; for 5 it
could not (abstract-only access): `cai2025edxgb`, `alsaraireh2022apt`,
`lambion2020dnst`, `bykov2024dnstunnels`, and the survey `elmahdaouy2026contextual`.

| | agreement | Cohen's kappa | 95% bootstrap interval |
|---|---|---|---|
| record consumer (yes/no) | 32/34 | 0.85 | [0.60, 1.00] |
| unit of inference (5 codes) | 26/34 | 0.64 | [0.42, 0.83] |
| cadence stated (24 papers both call record consumers) | | 0.27 | [-0.04, 0.59] |

Six of the eight unit disagreements stay inside the record-consumer class
(`flow_record` against `session_aggregate` or `host_timeseries`): the second
coder, reading full texts, more often found that the scored object is a windowed
or per-host aggregate of flow records. The two that cross the class boundary are
both DNS papers whose unit is ambiguous in the text (`dangelo2022dnsimages`:
per-query features vs per-flow wording; `acikgozoglu2024dnscomparison`: a
stateless per-query and a stateful per-domain-window classifier side by side).
Neither moves the DNS count: 2 record consumers under either coding.

The cadence variable disagrees much more, and in one direction: reading full
texts, the second coder found a timeout or aggregation window in 9 of the 24
papers both coders call record consumers (and 2 more partial), where the
screening-level first coding found 2 (and 2 partial). This is the limitation
stated above ("a value buried in a full-text methods paragraph would be coded
`no` here") made measurable. The paper now quotes the full-text figure.

**A protocol deviation the second coding exposed.** Applying the written
inclusion rule strictly ("network-based detection of data exfiltration or of
intrusions whose evaluation includes an exfiltration class"), the second coder
would exclude 17 papers as general intrusion- or anomaly-detection studies whose
evaluation data contain no exfiltration class (CIC-IDS2017's "Infiltration" is
not exfiltration), and one as a pure survey. The first coding had included them:
they were retrieved by the frame's evaluation-methodology and flow-export queries
and are where the record pipeline (timeouts, feature sets, cross-dataset
evaluation) is studied, but they do not meet the rule as written, and the survey
falls under an explicit exclusion.

We do not re-write the rule after the fact. Instead `frame.csv` records each
paper's **scope** from the second coder's inclusion decision (`exfiltration`,
`general_nids`, `survey`; the four on-topic papers it could not code are
`exfiltration` by its own notes) and its **channel** (DNS or other), and the
paper reports the record-consumer share per stratum under both codings
(`results/survey_strata.csv`):

| stratum | n | first coding | second coding |
|---|---|---|---|
| exfiltration detectors, DNS channel | 11 | 2/11 | 2/9 determined |
| exfiltration detectors, other channels | 10 | 8/10 | 6/8 determined |
| general intrusion detection (no exfiltration class) | 17 | 17/17 | 17/17 |
| all, survey excluded | 38 | 27/38 | 25/34 determined |

The survey is excluded from every count from revision 6 on (n = 38); the first
coding's row for it is kept in `coded.csv` unchanged, so the correction is
visible rather than silent.
