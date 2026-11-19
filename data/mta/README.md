# Real-malware packet tables (not included in this release)

The seven packet tables behind Table "The seven sandboxed stealer infections" were exported by hand
from captures published by Malware-Traffic-Analysis.net. They are withheld until the publisher
confirms that releasing header-derived tables is within its terms of use.

To rebuild them:
1. Download each capture from `www.malware-traffic-analysis.net/<year>/<month>/<day>/`
   (posts 2023-10-11 Lumma, 2023-11-22 AgentTesla, 2024-01-12 StealC, 2024-09-16 Snake SMTP,
   2024-09-17 Snake FTP, 2024-10-23 RedLine, 2026-09-10 AMOS).
2. In Wireshark, filter on the exfiltration peer (`ip.addr == <peer>`) and export a CSV with the columns
   frame number, time, source, destination, protocol, length, TCP stream index, TCP segment length
   and TCP flags (for AMOS add the HTTP method and path). Remove the Info column.
3. Save them here as `<date>-<family>-packets.csv` and run `python -m ebd.experiments.mta_episodes`,
   which writes `mta_episodes.csv`.

`tests/test_mta.py` is skipped while this folder holds only this README.
