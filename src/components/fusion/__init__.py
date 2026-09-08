"""
FraudSentinel AI
Fusion Layer

Combines:
    Phase 1 transaction-level fraud intelligence
    +
    Phase 2 temporal/spike intelligence

into a unified transaction-level risk decision.
"""

from .config import FusionConfig
from .pipeline import run_fusion_pipeline

__all__ = [
    "FusionConfig",
    "run_fusion_pipeline",
]

