from typing import Dict

import pandas as pd

from .config import XAIConfig


def _safe_float(
    value,
    default: float = 0.0,
) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_text(
    value,
    default: str = "UNKNOWN",
) -> str:
    if value is None or pd.isna(value):
        return default

    text = str(value).strip()

    return text if text else default


def build_unified_explanation(
    row: pd.Series,
    transaction_explanation: Dict,
    spike_explanation: Dict,
    fusion_explanation: Dict,
    config: XAIConfig,
) -> Dict:
    """
    Build final human-readable explanation.

    The explanation is generated entirely from existing
    model/detector/fusion outputs.
    """

    transaction_id = _safe_text(
        row.get("transaction_id")
    )

    merchant_id = _safe_text(
        row.get("merchant_id")
    )

    timestamp = _safe_text(
        row.get("timestamp")
    )

    risk_band = _safe_text(
        row.get("risk_band")
    )

    state = _safe_text(
        row.get("spike_state"),
        "NORMAL",
    )

    unified_risk = _safe_float(
        row.get("unified_risk_score")
    )

    fraud_probability = _safe_float(
        row.get("fraud_probability")
    )

    amount = _safe_float(
        row.get("amount")
    )

    expected_exposure = _safe_float(
        row.get("expected_fraud_exposure")
    )

    response_action = _safe_text(
        row.get("response_action"),
        "MONITOR",
    )

    response_priority = _safe_text(
        row.get("response_priority"),
        "LOW",
    )

    event_id = _safe_text(
        row.get("event_id"),
        "",
    )

    event_active = bool(
        row.get(
            "event_active",
            False,
        )
    )

    # --------------------------------------------------------
    # Main narrative
    # --------------------------------------------------------

    opening = (
        f"Transaction {transaction_id} for merchant {merchant_id} "
        f"has a unified risk score of {unified_risk:.2f} "
        f"and is classified as {risk_band} risk."
    )

    transaction_text = (
        transaction_explanation[
            "transaction_explanation"
        ]
    )

    spike_text = (
        spike_explanation[
            "spike_explanation"
        ]
    )

    fusion_text = (
        fusion_explanation[
            "fusion_explanation"
        ]
    )

    financial_text = (
        f"The transaction amount is "
        f"{amount:.2f}, with transaction-level expected loss "
        f"of {fusion_explanation['fusion_transaction_expected_loss']:.2f} "
        f"and combined expected fraud exposure of "
        f"{expected_exposure:.2f}."
    )

    if event_active:
        event_text = (
            f"The transaction is associated with active event "
            f"{event_id}."
        )
    else:
        event_text = (
            "The transaction is not currently associated with "
            "an active fraud event."
        )

    response_text = (
        f"Recommended response: {response_action} "
        f"with priority {response_priority}."
    )

    full_explanation = " ".join(
        [
            opening,
            transaction_text,
            spike_text,
            fusion_text,
            financial_text,
            event_text,
            response_text,
        ]
    )

    # --------------------------------------------------------
    # Structured explanation
    # --------------------------------------------------------

    return {
        "xai_version": config.explanation_version,

        "xai_transaction_id": transaction_id,
        "xai_merchant_id": merchant_id,
        "xai_timestamp": timestamp,

        "xai_risk_band": risk_band,
        "xai_spike_state": state,
        "xai_unified_risk_score": unified_risk,
        "xai_fraud_probability": fraud_probability,

        "xai_primary_reason": fusion_explanation[
            "fusion_primary_reason"
        ],

        "xai_transaction_reason": transaction_explanation[
            "transaction_primary_reason"
        ],

        "xai_spike_reason": spike_explanation[
            "spike_primary_reason"
        ],

        "xai_fusion_reason": fusion_explanation[
            "fusion_primary_reason"
        ],

        "xai_transaction_explanation": transaction_text,
        "xai_spike_explanation": spike_text,
        "xai_fusion_explanation": fusion_text,

        "xai_financial_explanation": financial_text,
        "xai_event_explanation": event_text,
        "xai_response_explanation": response_text,

        "xai_full_explanation": full_explanation,

        "xai_event_id": event_id,
        "xai_event_active": event_active,

        "xai_expected_exposure": expected_exposure,
        "xai_response_action": response_action,
        "xai_response_priority": response_priority,
    }

