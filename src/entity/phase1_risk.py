"""Domain entity representing a validated Phase-1 transaction risk record."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Phase1RiskRecord:
    transaction_id: str
    timestamp: datetime
    merchant_id: str
    card_id: str
    amount: float
    fraud_probability: float
    risk_score: float
    risk_band: str
    predicted_fraud: int
    model_version: str
    threshold: float

    def __post_init__(self) -> None:
        if not self.transaction_id:
            raise ValueError("transaction_id cannot be empty")
        if not self.merchant_id:
            raise ValueError("merchant_id cannot be empty")
        if not 0.0 <= self.fraud_probability <= 1.0:
            raise ValueError("fraud_probability must be in [0, 1]")
        if self.amount < 0:
            raise ValueError("amount cannot be negative")
        if self.risk_score < 0:
            raise ValueError("risk_score cannot be negative")