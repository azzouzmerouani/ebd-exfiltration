"""Where the real data comes from, and how to get it.

Recorded here rather than in a notebook so the access path is reproducible
without re-discovering it — the first attempt at this cost a day because the
obvious mirror of this dataset silently ships only half of it.

DAPT2021 / "Unraveled" (Arizona State University)
-------------------------------------------------
A semi-synthetic APT capture over roughly six weeks of a small emulated
enterprise, labeled per-flow with ``Stage`` (kill-chain phase), ``Activity``
(MITRE technique), ``DefenderResponse`` and ``Signature``.

**Access: a public S3 bucket, no credentials, no login.**

    https://dapt2021.s3.amazonaws.com/

    aws --no-sign-request --region us-east-1 s3 ls s3://dapt2021/

Two prefixes, and the difference is the entire reason this dataset was chosen:

* ``processed/network-flows/`` — 173 labeled CSVs (~3.7 GB), one per capture
  day and host, in **nfstream** schema. This is what an incumbent detector
  consumes.
* ``raw/network-captures/``   — 176 **PCAP** files (~477 GB). Unlabeled, but
  the only possible source of byte-level ground truth.

**Trap worth documenting:** the project's GitLab mirror
(``gitlab.thothlab.org/dapt2021/apt-dataset``) publishes only the *processed*
half. Checking it alone leads to the false conclusion that no PCAP exists —
which is exactly the conclusion that blocked the first attempt at this
experiment. The S3 bucket is the authoritative distribution, and its README
states plainly: "the processed data is labeled while the raw data is not."

Why this dataset and not SCVIC-APT-2021
---------------------------------------
SCVIC-APT-2021 is the dataset EDXGB reports on, but IEEE DataPort distributes
only the derived CSV — its PCAP is not downloadable — so EBD ground truth
cannot be reconstructed from it at all. DAPT2021 is the same lineage (DAPT2020
-> DAPT2021/Unraveled, same lab) and, crucially, EDXGB already has a native
code path for its nfstream column names, so no feature has to be invented.

Schema note that matters for the floor
--------------------------------------
The flow records are produced by **nfstream**, whose defaults are
``idle_timeout=120s`` and ``active_timeout=1800s``. The active timeout is the
record-emission interval ``T`` in ``EBD_floor = min(1, T/D)``. We do not assume
the defaults were used — :mod:`ebd.realdata.cadence_probe` measures ``T``
empirically from the flow duration distribution instead.
"""

from __future__ import annotations

BUCKET_URL = "https://dapt2021.s3.amazonaws.com/"
FLOWS_PREFIX = "processed/network-flows/"
PCAP_PREFIX = "raw/network-captures/"

# nfstream defaults, stated for reference only — the actual value used by this
# dataset is measured, never assumed. See cadence_probe.
NFSTREAM_DEFAULT_IDLE_TIMEOUT_S = 120.0
NFSTREAM_DEFAULT_ACTIVE_TIMEOUT_S = 1800.0

# The label the exfiltration episodes carry in the Stage column.
EXFIL_STAGE = "Data Exfiltration"

# Activities observed under that stage, with their MITRE technique.
EXFIL_ACTIVITIES = {
    "Data Transfer Size Limits": "T1030",       # chunked bulk transfer
    "Exfiltration over C2 channel": "T1041",
    "Unsecured Credentials": "T1552",
}
