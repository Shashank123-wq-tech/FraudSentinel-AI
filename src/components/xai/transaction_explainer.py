from typing import List

import numpy as np
import pandas as pd

from .config import XAIConfig


def _safe_numeric(
    series: pd.Series,
    default: float = 0.0,
) -> pd.Series:
    """
    Safely convert a Series to numeric.
    """
    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(default)


def explain_transactions(
    df: pd.DataFrame,
    config: XAIConfig,
) -> pd.DataFrame:
    """
    Vectorized transaction-level XAI.

    IMPORTANT:
    This implementation deliberately avoids iterrows().
    It is designed for million-row FraudSentinel datasets.
    """

    result = pd.DataFrame(
        index=df.index
    )

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    fraud_probability = _safe_numeric(
        df["fraud_probability"]
    )

    risk_score = _safe_numeric(
        df["risk_score"]
    )

    phase1_score = _safe_numeric(
        df["phase1_score"]
    )

    amount = _safe_numeric(
        df["amount"]
    )

    risk_band = (
        df["risk_band"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    predicted_fraud = (
        df["predicted_fraud"]
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # Signal masks
    # --------------------------------------------------------

    very_high_probability = (
        fraud_probability
        >= config.very_high_probability_threshold
    )

    high_probability = (
        fraud_probability
        >= config.high_probability_threshold
    )

    critical_phase1 = (
        phase1_score
        >= config.critical_risk_score_threshold
    )

    high_phase1 = (
        phase1_score
        >= config.high_risk_score_threshold
    )

    critical_band = risk_band.str.upper().isin(
        [
            "CRITICAL",
            "VERY_HIGH",
        ]
    )

    high_band = risk_band.str.upper().isin(
        [
            "HIGH",
        ]
    )

    # --------------------------------------------------------
    # Primary reason
    # --------------------------------------------------------

    primary_reason = np.select(
        [
            very_high_probability,
            high_probability,
            critical_phase1,
            high_phase1,
            critical_band,
            high_band,
            predicted_fraud,
        ],
        [
            "Very high transaction-level fraud probability.",
            "High transaction-level fraud probability.",
            "Phase-1 transaction risk score is in the critical range.",
            "Phase-1 transaction risk score is elevated.",
            "Transaction is classified in a critical risk band.",
            "Transaction is classified in a high risk band.",
            "The Phase-1 fraud classifier marked the transaction as fraudulent.",
        ],
        default="Transaction-level risk signals are not strongly elevated.",
    )

    result[
        "transaction_explanation_version"
    ] = config.explanation_version

    result[
        "transaction_primary_reason"
    ] = primary_reason

    # --------------------------------------------------------
    # Individual signal columns
    # --------------------------------------------------------

    result[
        "transaction_very_high_probability"
    ] = very_high_probability

    result[
        "transaction_high_probability"
    ] = high_probability

    result[
        "transaction_critical_phase1"
    ] = critical_phase1

    result[
        "transaction_high_phase1"
    ] = high_phase1

    result[
        "transaction_predicted_fraud"
    ] = predicted_fraud

    # --------------------------------------------------------
    # Signal count
    # --------------------------------------------------------

    signal_count = (
        very_high_probability.astype(int)
        + high_probability.astype(int)
        + critical_phase1.astype(int)
        + high_phase1.astype(int)
        + predicted_fraud.astype(int)
    )

    result[
        "transaction_signal_count"
    ] = signal_count

    # --------------------------------------------------------
    # Compact signal labels
    # --------------------------------------------------------

    signal_labels = pd.Series(
        "",
        index=df.index,
        dtype="object",
    )

    conditions = [
        (
            very_high_probability,
            "very_high_fraud_probability",
        ),
        (
            high_probability & ~very_high_probability,
            "high_fraud_probability",
        ),
        (
            critical_phase1,
            "critical_phase1_score",
        ),
        (
            high_phase1 & ~critical_phase1,
            "high_phase1_score",
        ),
        (
            predicted_fraud,
            "predicted_fraud",
        ),
    ]

    labels = []

    for mask, label in conditions:
        labels.append(
            pd.Series(
                np.where(mask, label, ""),
                index=df.index,
            )
        )

    signal_frame = pd.concat(
        labels,
        axis=1,
    )

    result[
        "transaction_signals"
    ] = signal_frame.apply(
        lambda row: "|".join(
            value
            for value in row
            if value
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    result[
        "transaction_explanation"
    ] = (
        result["transaction_primary_reason"]
        + " Fraud probability is "
        + fraud_probability.map(
            lambda x: f"{x:.2%}"
        )
        + " and Phase-1 score is "
        + phase1_score.map(
            lambda x: f"{x:.2f}"
        )
        + "."
    )

    # --------------------------------------------------------
    # Reasons
    # --------------------------------------------------------

    reason_columns = [
        very_high_probability,
        high_probability & ~very_high_probability,
        critical_phase1,
        high_phase1 & ~critical_phase1,
        critical_band,
        high_band,
        predicted_fraud,
    ]

    reason_texts = [
        "Very high fraud probability",
        "High fraud probability",
        "Critical Phase-1 score",
        "Elevated Phase-1 score",
        "Critical risk band",
        "High risk band",
        "Predicted fraud",
    ]

    reason_frame = pd.DataFrame(
        {
            text: mask
            for text, mask in zip(
                reason_texts,
                reason_columns,
            )
        },
        index=df.index,
    )

    result[
        "transaction_reasons"
    ] = reason_frame.apply(
        lambda row: " | ".join(
            row.index[row].tolist()
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Numeric context
    # --------------------------------------------------------

    result[
        "transaction_fraud_probability"
    ] = fraud_probability

    result[
        "transaction_phase1_score"
    ] = phase1_score

    result[
        "transaction_risk_score"
    ] = risk_score

    result[
        "transaction_risk_band"
    ] = risk_band

    result[
        "transaction_amount"
    ] = amount

    return result
