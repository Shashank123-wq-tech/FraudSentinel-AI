"""Domain entity for Phase-2 evidence and scoring state."""

from __future__ import annotations

from dataclasses import dataclass


def _bounded(value: float, field: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return value


@dataclass(frozen=True, slots=True)
class SpikeEvidence:
    intensity: float
    persistence: float
    acceleration: float
    breadth: float
    financial: float
    corroboration: float
    materiality: float

    def __post_init__(self) -> None:
        for field in (
            "intensity",
            "persistence",
            "acceleration",
            "breadth",
            "financial",
            "corroboration",
            "materiality",
        ):
            _bounded(getattr(self, field), field)

    @property
    def tas(self) -> float:
        return 100.0 * (
            0.30 * self.intensity
            + 0.30 * self.persistence
            + 0.10 * self.acceleration
            + 0.10 * self.breadth
            + 0.20 * self.financial
        )

    @property
    def verification_score(self) -> float:
        value = 1.0 - (
            (1.0 - self.persistence)
            * (1.0 - self.corroboration)
        )
        return max(0.0, min(1.0, value))

    @property
    def fraud_attack_score(self) -> float:
        return max(0.0, min(100.0, self.tas * self.verification_score * self.materiality))
