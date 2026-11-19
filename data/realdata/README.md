# DAPT2021 flow records

- `groundtruth.jsonl`: reconstructed byte-level delivery timelines of the seven exfiltration episodes.
- `exfil_flows.csv`: the 7,522 flow records labelled as exfiltration.
- `train_flows.csv.gz`: all other flow records used for training and the benign split.
  Decompress before running the detector experiments: `gunzip -k data/realdata/train_flows.csv.gz`.

All three are derived from the public DAPT2021 / Unraveled dataset (Myneni et al.); the raw captures are
not redistributed and must be obtained from the dataset authors.
