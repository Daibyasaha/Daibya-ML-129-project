"""Evaluation and benchmarking module for dermatological classification models."""

from .metrics import compute_comprehensive_metrics, evaluate_model
from .plots import save_evaluation_artifacts

__all__ = [
    "compute_comprehensive_metrics",
    "evaluate_model",
    "save_evaluation_artifacts",
]
