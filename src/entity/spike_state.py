"""State definitions for the merchant fraud-spike lifecycle."""

from __future__ import annotations

from enum import Enum


class SpikeState(str, Enum):
    NORMAL = "NORMAL"
    ANOMALY = "ANOMALY"
    CANDIDATE = "CANDIDATE"
    VERIFIED = "VERIFIED"
    CRITICAL = "CRITICAL"
    DECAYING = "DECAYING"
    RESOLVED = "RESOLVED"
