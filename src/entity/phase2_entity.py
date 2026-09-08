"""
Central Phase-2 domain entity module.

This module provides a stable import surface for the core Phase-2
domain objects.

The module intentionally contains domain entities only.
No pandas/DataFrame or detector logic belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .phase1_risk import Phase1RiskRecord
from .merchant_window import MerchantRiskWindow
from .spike_evidence import SpikeEvidence
from .spike_state import SpikeState
from .spike_event import SpikeEvent


@dataclass(frozen=True, slots=True)
class VerificationDecision:
    """
    Result of Phase-2 spike verification for a merchant/time window.
    """

    candidate: bool
    verified: bool
    critical: bool
    exceptional: bool
    state: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, bool):
            raise TypeError("candidate must be bool")

        if not isinstance(self.verified, bool):
            raise TypeError("verified must be bool")

        if not isinstance(self.critical, bool):
            raise TypeError("critical must be bool")

        if not isinstance(self.exceptional, bool):
            raise TypeError("exceptional must be bool")

        if not self.state:
            raise ValueError("state cannot be empty")

        if not self.reason:
            raise ValueError("reason cannot be empty")

        # Logical consistency checks.
        if self.critical and not self.verified:
            raise ValueError(
                "critical decision must also be verified"
            )

        if self.verified and not self.candidate:
            raise ValueError(
                "verified decision must also be a candidate"
            )


__all__ = [
    "Phase1RiskRecord",
    "MerchantRiskWindow",
    "SpikeEvidence",
    "SpikeState",
    "SpikeEvent",
    "VerificationDecision",
]