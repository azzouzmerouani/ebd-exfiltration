# DNS second corpus: outputs of the v4.2 measurement run

These files are the outputs of `notebooks/ebd-dns-floor.ipynb` v4.2, run on Kaggle
on 9 October 2026 (started 10:19:35, finished 10:32:28 notebook clock; 12.8 min in
cells; peak memory 4.1 GB; memory limit 32.2 GB read from cgroup v2) with
`SMOKE = False`, `FULL = False`, model XGBoost, seed 20260825, against the Kaggle
dataset `daumel/dns-tunneling-dataset` (version 8, CC BY 4.0).

They were returned as `ebd_dns_results3.zip`
(SHA-256 `60ad4194e6d086258c7d6cd9177287a5d9311e36586c3279915efdea8997a595`).
`dns_malformed_samples.csv` is left out of the repository: it holds raw bytes of
rejected campus DNS payloads and nothing in the paper depends on it.

Section 8 of the manuscript quotes only these files. `dns_digest.txt` is the
human-readable summary the notebook prints; what not to quote from it (the pooled
"0 of 24 folds over budget" line, the ibHH comparison, the bootstrap interval for
the flow mean) is listed in `docs/ebd-dns.md`.
