# Lab measurement of telemetry-pipeline delays (28 October 2026)

This lab measures, on real open-source software, the delays that the paper's
composed-pipeline profiles had assumed. The goal is to know when the first
record covering an exfiltration transfer becomes available at each stage of the
pipeline.

## Topology

Everything runs on one Linux host (Ubuntu 24.04, kernel 6.18). The traffic path
uses network namespaces:

- `victim` namespace: 10.10.1.2.
- Router: the root namespace, with interface `vh` on the victim side.
- `attacker` namespace: 10.10.2.2.

The following processes watch `vh`:

| Process | Version | Configuration |
|---|---|---|
| softflowd, instance E1 | 1.1.0, Ubuntu package | Package defaults: TCP 3600 s, TCP after FIN 300 s, UDP 300 s, expiry check every 60 s. Sends NetFlow v9 to nfcapd on :9995. |
| softflowd, instance E2 | 1.1.0, Ubuntu package | Configured like Cisco Flexible NetFlow: inactive 15 s, active (maxlife) 1800 s, TCP after FIN/RST 1 s, expiry check every 1 s. Sends to nfcapd on :9996. |
| nfcapd | nfdump 1.7.3 | Rotates its files every 300 s. Forwards every packet to Filebeat (:2055 for E1, :2056 for E2). |
| Zeek | 9.0.1, default settings | JSON `conn.log`. |
| Filebeat | 8.15.3 | `netflow` inputs, plus a `filestream` input on `conn.log` (checked every 1 s). |
| Elasticsearch | 8.15.3 | Single node. An ingest pipeline sets `event.ingested`. |
| Kibana | 8.15.3 | Two Security detection rules, see below. |

**Detection rules.** Both are query rules on any record from the victim to the
exfiltration peer, i.e. a lookup rule that fires on the first record.

- Interval 5 min, look-back 1 min (`from: now-360s`).
- `lab-exfil-peer-default` uses the default time field.
- `lab-exfil-peer-ingested` uses `timestamp_override: event.ingested`.

Rule definitions are in `config/rule_*.json`.

**Immediate mode for softflowd.** softflowd opens its capture with a 0 ms
timeout. On Linux, libpcap then delivers packets only when a ring block fills,
which on a quiet lab link never happens. `shim/pcapimm.c` is an LD_PRELOAD shim
that puts the handle into immediate mode, so packets are delivered as they
arrive, as they effectively are on a busy link. `bin/start_softflowd.sh`
applies it.

## Transfers

`bin/schedule.py` launches 38 transfers (seed 20261028). Start times are
randomised by up to 60 s within 150 s slots, so the transfers fall at different
phases of the collector rotation and of the rule schedule. The sender is
`bin/send.py` and the attacker-side sink is `bin/sink.py`.

| Transfer | Repeats | Size | Duration |
|---|---|---|---|
| One TCP connection | 6 | 2 MB | 20 s |
| One TCP connection | 6 | 5 MB | 120 s |
| One TCP connection | 6 | 20 MB | 900 s |
| One TCP connection | 2 | 20 MB | 2400 s |
| 8 sequential TCP connections | 6 | 5 MB | 120 s |
| UDP, query-sized datagrams from one source port | 6 | 600 KB | 300 s |
| UDP, a new source port for each datagram | 6 | 600 KB | 300 s |

All TCP connections close with FIN.

## Timestamps

`bin/analyse.py` takes ground truth from the router capture (`vh.pcap`). For
every transfer and pipeline it then finds the first record covering the
transfer and records:

- **release:** for NetFlow, the time Filebeat received the export packet
  (`event.created`); for Zeek, the time Filebeat read the `conn.log` line.
- **file:** for the NetFlow paths, the time nfcapd closed the 300 s file that
  holds the record.
- **ingest:** `event.ingested`.
- **alert:** the first alert of each rule.

Alert times are derived from the measured rule-run times (`rule_runs.jsonl`,
taken from the Kibana event log). The alert is placed at the end of the first
run that starts at least 1 s after ingestion (one refresh interval). For an
event-time rule, the record must also fall inside that run's look-back.

The derivation is needed because the lookup rule matched thousands of records
from the port-rotating transfers, so several runs hit the per-run alert cap of
1000, and the capped records were alerted one run later. The cap is an
artefact of this rule design, not of the pipeline. Of the 82 alerts that were
actually observed, 52 fall within 25 s of the derived time; the other 30 are
exactly one run later because of the cap. The remaining 32 of the 114
first records (38 transfers × 3 pipelines) were never alerted: a capped record
is alerted in the next run only if it still falls inside that run's
look-back. The paper reports both the derived and the observed outcomes.

The rule overhead is also read directly from the event log: scheduling delay
1.3 s (median, 0.1–4.2 s) and run time 0.3–0.6 s while the rule matched few
records, up to 22 s under the port-rotating load. The composed profiles use
25 s (mean wait minus mean phase) and 4 s (unloaded rule) as high and low
values (`python -m ebd.experiments.pipeline_profiles_measured [--low-overhead]`).

Measured on the 52 alerts that match their derivation, the overhead from
release to alert, minus the wait for the next scheduled run, is 6.4 s on
average (median 4.5 s, range 1.5–16.6 s): Kibana created the alerts a median
of 1.0 s after the start of the run, not at its end (`python scripts/lab_overhead.py`;
run-resampled 95% interval 4.3–8.9 s). Revisions 18–19 use 6.4 s
as the overhead and 25 s as a conservative bound
(`python -m ebd.experiments.pipeline_sweep`). Note that the rules here set
`max_signals` to 1,000; the default is 100.

## Outputs

| File | Content |
|---|---|
| `results/lab_pipeline/delays.csv` | One row per transfer and pipeline. |
| `results/lab_pipeline/{transfers,plan,rule_runs}.jsonl` | Transfer log, launch plan and rule-run times. |
| `results/lab_pipeline/versions.txt` | Software versions. |
| `results/lab_pipeline/vh.pcap.gz`, `export.pcap.gz` | Router capture and NetFlow export capture. |
| `results/pipeline_profiles_measured.csv` | Composed-pipeline floors recomputed with the measured delays (`python -m ebd.experiments.pipeline_profiles_measured`). |

Paths in `bin/*.py` are those of the lab host (`/opt/lab`).
