"""Domain entity for a merchant temporal risk window."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class MerchantRiskWindow:
    merchant_id: str
    window_start: datetime
    window_minutes: int
    transaction_count: int
    total_amount: float
    expected_fraud_count: float
    expected_fraud_amount: float
    unique_cards: int
    new_cards: int

    def __post_init__(self) -> None:
        if self.window_minutes <= 0:
            raise ValueError("window_minutes must be positive")
        if self.transaction_count < 0:
            raise ValueError("transaction_count cannot be negative")
        if self.unique_cards < 0 or self.new_cards < 0:
            raise ValueError("card counts cannot be negative")
        if self.new_cards > self.unique_cards and self.unique_cards > 0:
            raise ValueError("new_cards cannot exceed unique_cards")
        if self.total_amount < 0:
            raise ValueError("total_amount cannot be negative")
        if self.expected_fraud_count < 0 or self.expected_fraud_amount < 0:
            raise ValueError("expected fraud values cannot be negative")
