"""Domain entity for a merchant-level fraud-spike event."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .spike_state import SpikeState


@dataclass(frozen=True, slots=True)
class SpikeEvent:
    event_id: str
    merchant_id: str
    attack_start_time: datetime
    attack_end_time: datetime
    detection_time: datetime
    detection_state: SpikeState
    peak_state: SpikeState
    peak_tas: float
    peak_fraud_attack_score: float
    transaction_count: int
    total_amount: float
    expected_fraud_count: float
    expected_fraud_amount: float
    baseline_expected_fraud_amount: float
    unique_cards: int
    new_cards: int
    window_count: int
    event_duration_minutes: float
    single_transaction_event: bool
    exceptional_event: bool
    explanations: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.event_id or not self.merchant_id:
            raise ValueError("event_id and merchant_id are required")
        if self.attack_end_time < self.attack_start_time:
            raise ValueError("attack_end_time cannot precede attack_start_time")
        if self.detection_time < self.attack_start_time:
            raise ValueError("detection_time cannot precede attack_start_time")
        if not 0.0 <= self.peak_tas <= 100.0:
            raise ValueError("peak_tas must be in [0, 100]")
        if not 0.0 <= self.peak_fraud_attack_score <= 100.0:
            raise ValueError("peak_fraud_attack_score must be in [0, 100]")
        if self.transaction_count < 0 or self.window_count < 0:
            raise ValueError("counts cannot be negative")
