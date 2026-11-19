"""Detectors that plug into the EBD harness."""

from .base import BaseDetector
from .baselines import CumulativeBytes, EWMARate, VolumeThreshold
from .change import CusumRate
from .classifier import WindowClassifier

__all__ = [
    "BaseDetector",
    "VolumeThreshold",
    "EWMARate",
    "CumulativeBytes",
    "CusumRate",
    "WindowClassifier",
]
