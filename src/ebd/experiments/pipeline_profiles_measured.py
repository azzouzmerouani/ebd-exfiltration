"""Composed-pipeline floors with the delays measured in the lab (lab/README.md).

    python -m ebd.experiments.pipeline_profiles_measured

Same composition as ``pipeline_profiles`` (release, collection, ingestion and a
scheduled rule, averaged over the phase of every clock-driven stage), with the
assumed delays replaced by the medians measured on the lab pipeline
(results/lab_pipeline/delays.csv):

* release after flow end, softflowd configured like Cisco Flexible NetFlow
  (inactive 15 s, active 1800 s, FIN/RST 1 s): 1.6 s for FIN-closed TCP flows,
  15.7 s for UDP and for flows without FIN;
* release after connection end, Zeek 9.0 defaults: 7.8 s for TCP, 64.9 s for
  UDP; TCP without FIN after the 5-min inactivity timeout (configured, not
  measured);
* release after flow end, softflowd defaults: 314.7 s for FIN-closed TCP,
  330.2 s for UDP; TCP without FIN after the 3600 s idle timeout (configured,
  not measured);
* ingestion (Filebeat to Elasticsearch, 1 s) plus rule overhead (refresh,
  scheduling delay and run time): the measured mean wait of the 5-min rule
  minus its mean phase, 25 s in total, against 60 s assumed before;
* collector rotation (300 s) and rule period (300 s) are configuration values,
  and their measured waits match a uniform phase.

With ``--low-overhead`` the 25 s overhead is replaced by that of an unloaded
rule (4 s) and the output is results/pipeline_profiles_measured_ovh4.csv.

Cloud flow-log profiles are not measured and are not repeated here.
Writes results/pipeline_profiles_measured.csv.
"""
from __future__ import annotations

from .pipeline_profiles import main as _main

FIXED = 25.0  # ingestion + rule overhead, measured
PROFILES_MEASURED = {
    "netflow_onprem": dict(fin=1.6, nofin=15.7, window=300.0, fixed=FIXED, period=300.0,
                           anchor="flow_end", udp=15.7, maxlife=1800.0),
    "zeek_conn": dict(fin=7.8, nofin=300.0, window=0.0, fixed=FIXED, period=300.0,
                      anchor="flow_end", udp=64.9, maxlife=float("inf")),
    "softflowd_default": dict(fin=314.7, nofin=3600.0, window=300.0, fixed=FIXED, period=300.0,
                              anchor="flow_end", udp=330.2, maxlife=604800.0),
    "dns_query_log": dict(fin=0.0, nofin=0.0, window=0.0, fixed=FIXED, period=300.0,
                          anchor="first_packet", udp=0.0),
}

LOW_OVERHEAD = 4.0  # unloaded rule: ingestion 1 s + refresh 1 s + scheduling 1.3 s + run 0.3-0.6 s
PROFILES_MEASURED_LOW = {k: dict(v, fixed=LOW_OVERHEAD) for k, v in PROFILES_MEASURED.items()}

if __name__ == "__main__":
    import sys
    if "--low-overhead" in sys.argv:
        raise SystemExit(_main(PROFILES_MEASURED_LOW, "pipeline_profiles_measured_ovh4.csv"))
    raise SystemExit(_main(PROFILES_MEASURED, "pipeline_profiles_measured.csv"))
