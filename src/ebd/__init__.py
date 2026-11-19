"""ebd — damage-denominated evaluation for ransomware exfiltration detection.

The package answers one question the exfiltration-detection literature does not:
not *whether* a detector eventually fires, but *how much data has already left*
when it does. That quantity is **EBD — Exfiltrated Bytes before Detection**,
reported against a false-positive burden (FPB) and used to re-rank detectors so
the ranking can be compared with the conventional F1 ranking (hypothesis H1).

Public surface
--------------
* :class:`~ebd.episode.Episode` — the unit of evaluation (a receiver-side byte
  timeline with reconciled ground truth).
* :mod:`ebd.metrics` — ``compute_ebd``, ``ebd_fpb_curve``, ``compare_rankings``.
* :mod:`ebd.detectors` / :mod:`ebd.triggers` — the online detector suite.
* :mod:`ebd.synth` — the synthetic pilot dataset generator.
* :mod:`ebd.harness` — ``run_benchmark`` and its report.
"""

from .episode import Episode, ReconciliationError, ScenarioCell

__all__ = ["Episode", "ScenarioCell", "ReconciliationError", "__version__"]

__version__ = "0.1.0"
