"""Counterfactual flow floor of the first DNS corpus at a resolver-leg vantage
(rev12 review, R2-2 / R3-2).

    python -m ebd.experiments.dns_resolver_leg

At the client edge, the single-port tunnels of the first corpus form one flow
per transfer, which ends only when the transfer does. Behind an enterprise
recursive resolver, the exporter at the egress sees the resolver's queries to
the authoritative server, and a resolver draws a fresh source port per query
(RFC 5452). Each query is then its own flow, released by the idle timeout I.
The first record therefore exists about I seconds after the first query, and
at a constant query rate the floor is min(1, I/D). The packet captures of the
first corpus are not in this repository, so this is the closed form, not a
measurement. On the port-rotating tools of the second corpus the measured floor
exceeds this closed form by 0.03 percentage points on average
(results/cic_dns/), and on the port-rotating families of the first corpus it
lies 0.13 points below it on average.

Writes ``results/dns_v42/dns_resolver_leg.csv``.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

IDLE = 15.0


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    src = root / "results/dns_v42/dns_floors.csv"
    out = root / "results/dns_v42/dns_resolver_leg.csv"
    rows = []
    for r in csv.DictReader(src.open()):
        if r["consumer"] != "flow":
            continue
        d = float(r["duration_s"])
        rows.append(dict(episode=r["episode"], family=r["family"], n_ports=int(r["n_ports"]),
                         duration_s=round(d, 3),
                         floor_client_edge=float(r["floor_measured"]),
                         floor_resolver_leg_closed_form=min(1.0, IDLE / d)))
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fam = defaultdict(list)
    for r in rows:
        fam[r["family"]].append(r)
    print(f"wrote {out}")
    means = []
    for f, rs in fam.items():
        ce = 100 * sum(r["floor_client_edge"] for r in rs) / len(rs)
        rl = 100 * sum(r["floor_resolver_leg_closed_form"] for r in rs) / len(rs)
        means.append((ce, rl))
        print(f"{f:<16} n={len(rs)}  client edge {ce:7.2f}%   resolver leg {rl:6.2f}%")
    print(f"mean of families: client edge {sum(m[0] for m in means)/len(means):.1f}%  "
          f"resolver leg {sum(m[1] for m in means)/len(means):.2f}%")
    single = [r for r in rows if r["n_ports"] == 1 and r["floor_client_edge"] >= 0.999]
    print(f"client-edge total losses: {len(single)}; at resolver leg: "
          f"{sum(r['floor_resolver_leg_closed_form'] >= 0.999 for r in single)}; "
          f"their resolver-leg range {100*min(r['floor_resolver_leg_closed_form'] for r in single):.1f}"
          f"-{100*max(r['floor_resolver_leg_closed_form'] for r in single):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
