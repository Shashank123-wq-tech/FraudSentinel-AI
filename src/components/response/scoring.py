import numpy as np
import pandas as pd

from .config import (
    SPIKE_CRITICAL,
    SPIKE_VERIFIED,
    SPIKE_EARLY,
    UNIFIED_RISK_HIGH,
    UNIFIED_RISK_MEDIUM,
)


def _clip(value, low=0.0, high=100.0):

    return float(
        np.clip(
            value,
            low,
            high,
        )
    )


def calculate_threat_score(row: pd.Series) -> float:

    risk = float(
        row.get(
            "unified_risk_score",
            0.0,
        )
    )

    spike = str(
        row.get(
            "spike_state",
            "NORMAL",
        )
    )

    peak_spike = float(
        row.get(
            "peak_spike_score",
            0.0,
        )
    )

    fraud_probability = float(
        row.get(
            "fraud_probability",
            0.0,
        )
    )

    score = (
        0.55 * risk
        + 0.20 * peak_spike
        + 25.0 * fraud_probability
    )

    if spike == SPIKE_CRITICAL:
        score += 20.0

    elif spike == SPIKE_VERIFIED:
        score += 10.0

    elif spike == SPIKE_EARLY:
        score += 5.0

    return _clip(score)


def calculate_financial_urgency(
    projected_30m: float,
    projected_60m: float,
) -> float:

    score_30 = min(
        projected_30m / 5000.0,
        1.0,
    )

    score_60 = min(
        projected_60m / 8000.0,
        1.0,
    )

    return _clip(
        100.0
        * (
            0.60 * score_30
            + 0.40 * score_60
        )
    )


def calculate_response_score(
    threat_score: float,
    financial_urgency: float,
    confidence: float,
) -> float:

    confidence = np.clip(
        confidence,
        0.0,
        1.0,
    )

    base = (
        0.60 * threat_score
        + 0.40 * financial_urgency
    )

    # Confidence modifies certainty, but does not
    # completely suppress a critical threat.
    adjusted = (
        0.80 * base
        + 0.20 * base * confidence
    )

    return _clip(adjusted)


def classify_risk_level(
    unified_risk_score: float,
) -> str:

    if unified_risk_score >= UNIFIED_RISK_HIGH:
        return "HIGH"

    if unified_risk_score >= UNIFIED_RISK_MEDIUM:
        return "MEDIUM"

    return "LOW"


def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:

    result = df.copy()

    result["threat_score"] = result.apply(
        calculate_threat_score,
        axis=1,
    )

    result["financial_urgency"] = result.apply(
        lambda row: calculate_financial_urgency(
            float(row["projected_loss_30m"]),
            float(row["projected_loss_60m"]),
        ),
        axis=1,
    )

    result["response_score"] = result.apply(
        lambda row: calculate_response_score(
            float(row["threat_score"]),
            float(row["financial_urgency"]),
            float(row["fusion_confidence"]),
        ),
        axis=1,
    )

    result["risk_level"] = result[
        "unified_risk_score"
    ].apply(
        classify_risk_level
    )

    return result
