"""EDXGB's feature pipeline and predictor, reproduced for the EBD measurement.

What this is, and what it is not
--------------------------------
EDXGB (Cai et al.) is a stacked model: four DNN and four CNN feature extractors
feeding an XGBoost predictor. **This reproduces the feature pipeline exactly and
the XGBoost predictor exactly, and omits the deep extractors.**

That simplification is deliberate and it shapes how the result may be read:

* If this reaches the architectural floor on an episode, the conclusion is
  sound — a real classifier attains its ceiling when the architecture permits,
  and a *good* result cannot be dismissed with "a stronger model would have done
  better", because a stronger model could only match the floor too.
* If it detects poorly, the result is **inconclusive**: weak reproduction and a
  genuine detector gap are not separable without the full ensemble.

The asymmetry is why the cheap version is worth running before investing in the
deep front-end. The stacked extractors earn their keep on noisy, ambiguous
traffic; here the question is only whether a real classifier reaches a bound,
against already-labelled exfiltration.

Feature construction
--------------------
Taken from ``ExfilEnv/EnvPreparing.py`` in the released source, nfstream branch
(the repo names ``Unraveled`` explicitly in ``extracted_by_NFStream``):

    PTR_I = dst2src_packets / bidirectional_duration_ms
    PTR_O = src2dst_packets / bidirectional_duration_ms
    BTR_I = dst2src_bytes   / bidirectional_duration_ms
    BTR_O = src2dst_bytes   / bidirectional_duration_ms

These four, plus the numeric nfstream statistics, are the model's input.

Identifier columns (IP and MAC addresses, flow id) are **excluded**. The
exfiltration in this capture goes to one destination, so a model given
``dst_ip`` learns that address rather than any behaviour, and would post a
near-perfect score that says nothing about detecting exfiltration. Ports and
protocol are retained, as is standard for flow-based NIDS features.

When a record becomes available
-------------------------------
A flow record cannot be scored before it exists, and it exists when its flow
ends. Each row's alert time is therefore ``bidirectional_last_seen_ms`` — not
the flow's start. This is the same fact the architectural floor rests on, and it
is what makes a per-flow classifier's EBD measurable at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Columns that identify *who*, not *how* — excluded so the model cannot simply
# memorise the exfiltration endpoint.
IDENTIFIER_COLUMNS = {
    "id", "expiration_id", "src_ip", "dst_ip", "src_mac", "dst_mac",
    "src_oui", "dst_oui", "vlan_id", "tunnel_id", "ip_version",
    "requested_server_name", "client_fingerprint", "server_fingerprint",
    "user_agent", "content_type", "application_name",
    "application_category_name", "application_is_guessed",
    "source_file",
}
LABEL_COLUMNS = {"Activity", "Stage", "DefenderResponse", "Signature"}

# Timestamp columns: absolute epochs would let the model memorise *when* the
# attack happened, which is leakage of exactly the same kind as dst_ip.
TIME_COLUMNS = {
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
}

DERIVED = ["PTR_I", "PTR_O", "BTR_I", "BTR_O"]


def derive_rates(row: dict) -> dict[str, float]:
    """EDXGB's PTR/BTR features, guarding the zero-duration case."""
    dur = float(row.get("bidirectional_duration_ms") or 0.0)
    if dur <= 0:
        return {k: 0.0 for k in DERIVED}
    return {
        "PTR_I": float(row.get("dst2src_packets") or 0) / dur,
        "PTR_O": float(row.get("src2dst_packets") or 0) / dur,
        "BTR_I": float(row.get("dst2src_bytes") or 0) / dur,
        "BTR_O": float(row.get("src2dst_bytes") or 0) / dur,
    }


def feature_columns(header: list[str]) -> list[str]:
    """Numeric behavioural columns, in a stable order."""
    return [
        c for c in header
        if c not in IDENTIFIER_COLUMNS
        and c not in LABEL_COLUMNS
        and c not in TIME_COLUMNS
    ]


def build_matrix(rows: list[dict], columns: list[str], rates: bool = True) -> np.ndarray:
    """Feature matrix over ``columns`` plus (unless ``rates=False``) the four derived rates."""
    out = np.zeros((len(rows), len(columns) + (len(DERIVED) if rates else 0)), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, c in enumerate(columns):
            v = r.get(c, "")
            try:
                out[i, j] = float(v) if v not in ("", None) else 0.0
            except (TypeError, ValueError):
                out[i, j] = 0.0
        if not rates:
            continue
        d = derive_rates(r)
        for k, name in enumerate(DERIVED):
            out[i, len(columns) + k] = d[name]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class TrainedDetector:
    model: object
    columns: list[str]
    rates: bool = True

    def score(self, rows: list[dict]) -> np.ndarray:
        X = build_matrix(rows, self.columns, self.rates)
        return self.model.predict_proba(X)[:, 1]


def train(
    train_rows: list[dict],
    labels: np.ndarray,
    columns: list[str],
    seed: int = 20260815,
    n_jobs: int = 4,
    rates: bool = True,
) -> TrainedDetector:
    """Train the XGBoost predictor on EDXGB's feature pipeline.

    Hyperparameters follow the released code (learning rate 0.01, max depth 10,
    30 estimators). ``scale_pos_weight`` handles the ~100:1 class imbalance,
    standing in for the resampling their pipeline applies via imbalanced-learn.
    ``n_jobs=1`` makes reruns bit-identical on one machine; the archived Table 2
    run used the default of four threads.
    """
    from xgboost import XGBClassifier

    X = build_matrix(train_rows, columns, rates)
    y = np.asarray(labels, dtype=int)
    pos = max(int(y.sum()), 1)
    neg = max(len(y) - pos, 1)
    model = XGBClassifier(
        learning_rate=0.01,
        max_depth=10,
        n_estimators=30,
        scale_pos_weight=neg / pos,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=n_jobs,
        tree_method="hist",
    )
    model.fit(X, y)
    return TrainedDetector(model=model, columns=columns, rates=rates)


def first_alert_time(
    rows: list[dict], scores: np.ndarray, threshold: float
) -> float | None:
    """Earliest time a scored flow record both exists and is called exfiltration.

    A record is available only once its flow has ended, so the candidate alert
    time is ``bidirectional_last_seen_ms``. Returns ``None`` if nothing in the
    episode is ever flagged.
    """
    best: float | None = None
    for r, s in zip(rows, scores):
        if s < threshold:
            continue
        try:
            t = float(r["bidirectional_last_seen_ms"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            continue
        if best is None or t < best:
            best = t
    return best
