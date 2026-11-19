# exfil-ebd

**Damage-denominated evaluation for ransomware data-exfiltration detection.**

The exfiltration-detection literature reports F1 / PR-AUC — *whether* a detector
eventually decides correctly. For exfiltration that is the wrong target. Encrypted
data has a recovery path (backups); exfiltrated data does not — once bytes leave,
the loss never closes. A detector with F1 = 0.95 that fires after the whole asset
has left is worthless; one with F1 = 0.78 that fires after 4 % has left prevented
the incident. F1 ranks the first one higher.

This project measures the thing that actually matters:

> **EBD — Exfiltrated Bytes before Detection.** For an attack episode and a
> detector that alerts at some window, EBD is the receiver-side bytes already
> landed by that window. Never firing costs the whole asset. Lower is better.

EBD alone is gamed by a detector that alerts on everything, so it is always
reported **paired with a false-positive burden (FPB)** and detectors are compared
along the resulting EBD–FPB trade-off — the damage-denominated analogue of an ROC.

### Pre-registered hypothesis (H1)

> The F1 ranking of detectors and the EBD ranking **disagree**. Operationally:
> Kendall's τ between the two rankings **< 0.5** counts as divergence (H1
> supported); **τ ≥ 0.5** falsifies it.

If the rankings diverge, the field has been optimising the wrong objective.

---

## Status

Pilot scaffold on **synthetic** data. The full pipeline runs end-to-end —
episodes → detectors → EBD/FPB/F1 → H1 verdict → figures — and the metric core is
unit-tested. **No lab captures exist yet**, so nothing here is evidence for the
paper's empirical claim; it validates the *instrument*, not the result.

```
src/ebd/
  episode.py        Episode model + receiver-side reconciliation (<1% tolerance)
  scenarios.py      36-cell adversary matrix + rclone command emission (REFERENCE ONLY)
  synth.py          synthetic episode generator (pilot data)
  metrics.py        compute_ebd, ebd_fpb_curve, episode F1, compare_rankings (H1)
  detectors/        BaseDetector + baselines (volume, ewma, cumulative, cusum)
                    + WindowClassifier (trained, persistence-aware base)
  triggers/ects.py  ECTS optimal-stopping trigger: byte / time / byte-floored delay cost
  harness.py        train → sweep → rank → H1, as one run_benchmark()
  d5.py             the byte-vs-time control: FPB match, per-rate EBD, paired bootstrap
  experiments/      synthetic (H1 pilot), d5_control (D5), stability (seed-spread)
tests/              metric, detector, generator, and D5 unit tests (24)
```

## Run it

```bash
uv run ebd-synth                       # H1 pilot: F1-vs-EBD ranking + figures
uv run ebd-d5                          # D5 control: byte-cost vs time-cost, per rate
uv run ebd-d5 --variant byte-floored   # D5 with the decoupled fix
uv run ebd-stability --seeds 20        # seed-spread of the H1 tau
uv run pytest -q                       # tests (24)
```

Outputs land in `results/` and `figures/` (both gitignored, regenerated on each run).

## The six pilot detectors

| detector | idea | expected character |
|---|---|---|
| `volume` | bytes in the current window | early on bursts, blind to drips |
| `ewma` | smoothed rate over recent windows | catches sustained transfers |
| `cumulative` | total bytes so far | catches everything **late** — F1-good, EBD-bad |
| `cusum` | CUSUM on rate vs a benign baseline | sustained-drift, different trade-off |
| `ects-time[ewma]` | ECTS stop rule, **fixed** per-window delay cost | rate-blind — the **D5 control** |
| `ects-byte[ewma]` | ECTS stop rule, **byte** delay cost (∝ current rate) | fires fast on fast transfers — the **contribution** |

The `ects-byte` / `ects-time` pair share one base scorer and one calibration and
differ **only** in the delay-cost denomination. That is the controlled comparison
(**experiment D5**) the whole paper rests on: if the byte-denominated trigger does
not beat the time-denominated one on EBD, the contribution collapses to "applied
ECTS to exfiltration". Run it early, not last.

These six are deliberately naive **stand-ins** to exercise the ranking machinery.
They are replaced by the real published detectors once lab captures exist; all
satisfy the same `BaseDetector` streaming contract.

## What the current pilot shows — and does not

On the synthetic dataset (sustained attacks vs short benign "bulk" bursts):

- **H1 is FALSIFIED on this synthetic pilot — and the seed-stability check is what
  caught it.** A single early seed gave τ = +0.09 (apparent divergence), but that
  was a *superseded* 6-detector configuration. With the current suite (which
  includes a competent trained detector), the F1 and EBD rankings **agree**: over
  20 seeds τ = **+0.72 ± 0.06**, range [+0.60, +0.92], **0/20 below the 0.5
  divergence bar** (`figures/h1_stability_full.png`). And it is not just the
  strong classifier masking it: on the **clustered-F1 suite** (naive detectors
  only — the regime H1 is actually *about*, where EBD would have to be the
  separator) divergence is still only a minority event — τ = +0.74 ± 0.18 over 15
  seeds, **2/15 (13 %)** below the bar. The earlier τ = +0.09 was one of those
  minority seeds. **Caveat:** synthetic data with stand-in detectors does not
  settle H1 for real published detectors — but it shows divergence is *not
  automatic*; the paper must earn it empirically rather than assume it. Run
  `uv run ebd-stability` and `uv run ebd-stability --suite clustered`.
- **The trained `classifier` base discriminates** (F1 ≈ 0.99, ~99 % detection at
  < 4 % FPB), so the D5 comparison is meaningful rather than washed out.
- **D5 verdict: byte-denomination does NOT beat time-cost — in either form we
  could construct, and we know why.** On a discriminating base:
  - **Naive myopic byte-cost** (`ects-byte`) is dominated by `ects-time`: paired
    nEBD saving **−0.34, 95 % CI [−0.44, −0.25]**. It wins only on **burst** and
    loses on fast/throttled/drip (`figures/d5_per_rate_byte.png`).
  - **Decoupled "byte-floored" fix** — rate may only *advance* the stop, never
    push the detection bar above time-cost's — comes out **bit-identical to
    byte** and loses just the same.
  - **Mechanism (verified at the threshold-setting stage).** The false-positive
    budget is spent almost entirely on *high-rate benign bursts*, and on those
    byte and byte-floored behave identically (both use the low, rate-driven
    cutoff). That forces the same high global operating threshold (T ≈ 0.59 vs
    time's 0.27), which makes low-and-slow attacks unreachable no matter how the
    cutoff is floored on the slow side — so byte detects 62 % where time detects
    99 %. The rate-coupling's damage is done *upstream* of any per-rate cutoff.
- **The theoretical kernel.** When the base `p` is near-sufficient, no rate
  re-weighting of the *detection* threshold can beat the optimal `p`-threshold at
  matched FPB; and the only fully-decoupled detector (a pure `p`-threshold) *is*
  time-cost. Byte-denomination could help only if instantaneous rate carried
  damage-relevant information **orthogonal** to `p` — which it does not, once the
  base already uses rate as a feature. This is a falsification **with a proof
  obligation and its empirical demonstration**, not a null result.

## Real-data validation (`uv run ebd-real`, `uv run ebd-edxgb`)

The floor is no longer only analytic. On **DAPT2021/Unraveled** — a public
capture that ships labelled flows *and* 477 GB of raw PCAP, and one of the two
datasets EDXGB was actually built for — seven real exfiltration episodes were
reconstructed from packets, reconciled against the dataset's own byte counts
(**0.0–0.2 %**, four exact), and used to measure EDXGB against its own floor.

**Leakage is verified zero**: all seven episodes are held out *simultaneously*
(stricter than leave-one-out — the model's only exfiltration training data is
1,834 flows from the 458 *trivial* episodes), and 0 rows are both scored and
trained on. Both splits are reported.

**5 of 7 episodes land essentially on the floor** under leave-one-out (4 of 7
under the stricter split), where *at floor* is a stated cutoff — gap ≤ 1.00
percentage point of the asset — and the count is insensitive to it: the largest
at-floor gap is 0.07 pp and the smallest large gap is 45.59 pp, a **610×**
separation with nothing between. That is the conclusive direction: a real
classifier reaches its architectural ceiling where the architecture permits, and
no stronger model could beat a floor.

**The one instructive outlier**: an episode with **99.69 % flow recall**
(1,289/1,293 flagged) still loses **46 %** of its asset, because the three
*earliest-ending* records are among the four it misses and the episode
front-loads 46 % of its bytes into its first 100 seconds — a 109× EBD penalty
for missing 0.31 % of flows. A paper reporting 99.69 % recall would call that a
success. This is the case for EBD, measured rather than constructed.

Scope, stated plainly: DAPT2021's exfiltration totals **8.94 MB over six weeks**
and is entirely low-and-slow, so this validates the formula and the measurement
method but **cannot exercise the `D ≤ T` blind spot**. Full write-up, including
the six errors caught by reconciliation along the way:
[docs/real-data-validation.md](docs/real-data-validation.md).

## The cadence result (`uv run ebd-cadence`)

The incumbent detectors are **batch per-flow classifiers** — EDXGB scores
CICFlowMeter rows, NEDS scores completed per-protocol session aggregates. Neither
has a notion of a score at time *t*: their first opportunity to say anything is
when the telemetry pipeline **emits a record**. That bounds how early they can
possibly be, regardless of model quality:

> **EBD_floor / asset = min(1, T / D)** — for emission interval `T` and transfer
> duration `D`. If **D ≤ T the floor is 100 %**: the transfer finishes inside one
> interval, so the first record exists only after every byte is gone.

The floor is computed against an **oracle** flow detector (correct on its first
record, zero false positives), so real detectors can only do worse — the claim
does not depend on anyone's ML. Verified against the closed form: max deviation
**0.008** for steady streams, rising to **0.16** for `many-small` (idle gaps make
`min(1,T/D)` an approximation, not an identity).

**Fixed 500 MB asset, oracle floor vs this project's online detector:**

| rate | D | floor @120 s (CICFlowMeter) | floor @1800 s (IOS XR) | online (this work) |
|---|---|---|---|---|
| burst | 17 s | **100 %** | **100 %** | 74.2 % |
| fast | 50 s | **100 %** | **100 %** | **19.9 %** |
| throttled | 167 s | 72.0 % | **100 %** | 8.9 % |
| drip | 500 s | 24.2 % | **100 %** | 4.7 % |

Two things to read off it:

1. **The blind spot is rate-inverted.** The *faster* the theft, the shorter `D`,
   the closer the floor to total loss. A 500 MB transfer at 10 MB/s is **entirely
   gone** before any flow-based detector can look — while the slow drip, the
   "stealthy" case, is the only one they can catch early. The architecture is
   blindest exactly where damage accrues fastest.
2. **It is a choice, not a necessity.** Same episodes, scored continuously: 19.9 %
   vs 100 % on the `fast` cell. But note the honest limit — on a 17 s burst the
   online detector still loses 74 %, because 17 seconds is genuinely little time
   to accumulate evidence. Online scoring is not magic; it is just not
   *architecturally* blind.

Export cadence spans a **30×** range in practice — CICFlowMeter 120 s, Cisco IOS
default 60 s, Cisco IOS XR default **1800 s** — and the detection literature does
not report it. At 1800 s every scenario here is a total loss. **Incumbent
earliness is an unexamined artifact of a telemetry parameter, not a property of
the detection logic.**

### Why the bound is conservative where it looks weakest

A flow record spans `T` seconds, so its volume is `rate × T`. A real detector must
separate *that slice* from benign bulk — and the corpus says where it can't:

| rate | slice @120 s | benign bulk flows ≥ slice |
|---|---|---|
| burst | 3600 MB | 0.0 % |
| fast | 1200 MB | 0.0 % |
| throttled | 360 MB | 0.0 % |
| **drip** | **120 MB** | **21.7 %** ← slice hides *inside* the benign range |

So the oracle is tight exactly where the floor is already catastrophic (burst/fast:
separable, but 100 % gone anyway), and loosest on the **drip — the one cell that
looks survivable at 24 %**. There, the slice overlaps benign bulk, which forces the
threshold up: the same overlap that defeated the byte-cost trigger in D5. **"Real
detectors can only do worse" bites hardest precisely where incumbents look least
blind.** The "your oracle is unrealistically generous" objection therefore argues
*for* the bound, not against it — and it gives a real-data experiment a concrete
job: measure the oracle-to-real gap per rate.

### The thesis (adversary-side, and it needs no ranking)

`EBD_floor = min(1, T/D)` with **the adversary choosing D** inverts the low-and-slow
folklore:

> Flow-based exfiltration detectors are architecturally blind **in proportion to
> transfer speed**, which makes **speed the adversary's dominant strategy** against
> them. Against this detector class, slow is not stealthy — it is the *only* regime
> in which the detector has a chance. Online scoring removes that free win (`fast`:
> 100 % → 19.9 %), and EBD is what makes the blind spot visible.

*Scope:* this is a claim about the **flow-record-consuming ML class**, not about all
defenses — going fast trips volume/DLP and egress controls that sit elsewhere in the
stack. Stated at that scope it depends on neither H1 nor any ranking result.

*Scope:* n = 2 detectors inspected (EDXGB source, NEDS paper). Whether "batch
per-flow" characterises the field is the open question — the next check is
whether an SLR of the exfiltration literature codes a batch-vs-online axis at all.

**Where this leaves the pilot.** On synthetic data with stand-in detectors, *both*
empirical bets fail: byte-cost does not beat time-cost (D5), and the F1/EBD
rankings agree rather than diverge (H1). What survives is the **EBD/FPB metric
itself** — defined, implemented and validated — which reports the decision-relevant
quantity (how much data left before the alarm) that F1 never does, *whether or not*
it re-ranks detectors. Both negatives are synthetic and do not settle the real-data
question; they do say the original "F1 optimises the wrong thing" punch is **not
automatic** and must be demonstrated on real published detectors, not assumed.

Reproduce: `uv run ebd-d5 --variant byte` and `uv run ebd-d5 --variant byte-floored`
(per-rate breakdown + paired bootstrap), `uv run ebd-stability` (H1 seed-spread).

> ⚠️ Every synthetic number is a methodology check. The empirical claim is settled
> only on lab captures. The `scenarios.py` module *prints* rclone commands for the
> isolated lab and **never executes them**; running exfiltration tooling is gated
> on network isolation (project failure points C3/C4).

## Exporting on machines where nfstream does not import

nfstream's macOS x86-64 wheels for CPython 3.9 and later ship without nDPI, so
`import nfstream` fails there. `scripts/run_section9.sh` exports the controlled
captures with nfstream 6.5.4 under CPython 3.8 into cached record files that
carry the SHA-256 of their capture; cached and live exports give identical
results. Single-threaded XGBoost reproduces the archived four-thread Table
`tab:gap` exactly.

## Round-4 runs (revision 10)

`scripts/run_round4.sh` needs XGBoost and, for its last step, HTTP access to
the public DAPT2021 bucket. It supersedes `scripts/run_round3.sh` and adds:
the multimodel table refreshed (hashes, admissible flag), a threshold grid
extended below 0.05 (`--grid extended`), out-of-fold calibration, ten benign
splits, a port-only model and a non-learnt peer rule, a destination-disjoint
split (`--peer-disjoint`), and `scripts/check_end_reason.py` (TCP end of the
first connections of episode 443 #3). The sandbox analyses `ebd-floor-all`,
`ebd-response` and `ebd-time-vs-bytes` need no XGBoost.

## Commands behind each result in the paper

Model-free analyses (minutes, no XGBoost needed):

| Result | Command | Output |
|---|---|---|
| Synthetic pilot (Supplementary Material, S1) | `uv run ebd-synth`; `uv run ebd-d5`; `uv run ebd-stability --seeds 20`; `uv run ebd-stability --seeds 15 --suite clustered` | `results/` (synthetic files) |
| Floor vs rate (Sec. 3) | `uv run ebd-cadence` | `results/cadence_*` |
| DAPT2021 floors, Table semantics | `uv run ebd-real`; `uv run ebd-semantics` | `results/emission_semantics*.csv` |
| Floor vs active timeout, Table sweep | `uv run ebd-sweep` | `results/cadence_sweep.csv` |
| Flow vs session floors | `uv run ebd-granularity` | `results/granularity*` |
| All 465 nominal episodes | `uv run ebd-floor-all` | `results/real_floor_all.csv` |
| Response latency, Table response | `uv run ebd-response` | `results/response_sensitivity.csv` |
| Time vs bytes, Table timebytes | `uv run ebd-time-vs-bytes --protocol loo` (and `strict`) | `results/time_vs_bytes_*.csv` |
| Episode-gap sensitivity, Table gapsens | `python -m ebd.experiments.gap_sensitivity` | `results/gap_sensitivity.csv` |
| Composed pipelines, Table profiles | `python -m ebd.experiments.pipeline_profiles` | `results/pipeline_profiles.csv` |
| Composed pipelines at episode gap 300 s (Section 5.2) | `python -m ebd.experiments.pipeline_profiles_gap` | `results/pipeline_profiles_gap.csv` |
| Lab pipeline delays, Table labdelays | see `lab/README.md`; `python3 lab/bin/analyse.py` on the lab host | `results/lab_pipeline/delays.csv` |
| Composed pipelines with measured delays | `python -m ebd.experiments.pipeline_profiles_measured` | `results/pipeline_profiles_measured.csv` |
| Real-malware episodes, Table mta | `python -m ebd.experiments.mta_episodes` (packet tables exported by hand from Malware-Traffic-Analysis.net captures, `data/mta/`) | `data/mta/mta_episodes.csv` |
| Composed pipelines, low rule overhead (4 s) | `python -m ebd.experiments.pipeline_profiles_measured --low-overhead` | `results/pipeline_profiles_measured_ovh4.csv` |
| Rule-period sweep with measured delays, Table period; softflowd decomposition | `python -m ebd.experiments.pipeline_sweep` | `results/pipeline_sweep.csv` |
| Fig. front_loaded_episode | `PYTHONPATH=src python scripts/plot_front_loaded.py` | `paper/figs/front_loaded_episode.png` |
| Lab overhead, direct estimate with run-resampled interval | `python scripts/lab_overhead.py` | printed (6.4 s; 4.3–8.9 s) |
| Leave-one-out at own operating points, Table attain | `bash scripts/run_round7.sh` | `results/round7/` |
| DNS behind a resolver | `python -m ebd.experiments.dns_resolver_leg` | `results/dns_v42/dns_resolver_leg.csv` |
| Coded sample, Table survey | `uv run ebd-survey`; `uv run ebd-survey-agreement` | `results/survey_*` |
| First DNS corpus | `notebooks/ebd-dns-floor.ipynb` (Kaggle) | `results/dns_v42/` |
| Second DNS corpus (CIC-Bell-DNS-EXF-2021) | `python3 scripts/cic_dns_floor.py --root <pcaps> --dedup --split-init --out results/cic_dns_split` | `results/cic_dns_split/` |

Detector experiments (XGBoost; hours, single-threaded):

| Result | Command |
|---|---|
| EDXGB vs floor (Table gap), four families, session detector, controlled captures, escape | `bash scripts/run_revision.sh` (and `bash scripts/run_section9.sh` where nfstream does not import) |
| Single runs: port-only, lookup rule, benign negatives, out-of-fold calibration (Table r12-robust, single rows) | `bash scripts/run_round5.sh` |
| Resamplings and destination-disjoint split, out of fold; four-family table twice | `bash scripts/run_round6.sh` |
| Leave-one-out models at their own operating points; logistic regression at tol 1e-10 | `bash scripts/run_round7.sh` |
| Logistic-regression reproducibility check | `results/lr_tol_check/` (two runs, identical hashes) |
| Fig. ebd-fpb | `python3 scripts/plot_ebd_fpb_curve.py` |

`python3 scripts/make_manifest.py` writes `results/MANIFEST.sha256`, the hash of every result file.
# ebd-exfiltration
