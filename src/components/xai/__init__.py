"""
FraudSentinel AI - Explainable AI Layer.

This package provides deterministic, auditable explanations for:
    - Transaction-level fraud risk
    - Temporal fraud-spike risk
    - Fusion-level unified risk
    - Financial exposure
    - Recommended response context
"""

from .config import XAIConfig
from .pipeline import run_xai_pipeline

__all__ = [
    "XAIConfig",
    "run_xai_pipeline",
]

