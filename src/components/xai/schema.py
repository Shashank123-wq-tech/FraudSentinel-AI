from typing import List

import pandas as pd


# ============================================================
# REQUIRED IDENTIFICATION COLUMNS
# ============================================================

IDENTITY_COLUMNS: List[str] = [
    "transaction_id",
    "timestamp",
    "merchant_id",
    "card_id",
]


# ============================================================
# PHASE 1 COLUMNS
# ============================================================

PHASE1_COLUMNS: List[str] = [
    "amount",
    "fraud_probability",
    "risk_score",
    "risk_band",
    "predicted_fraud",
    "phase1_score",
    "phase1_primary_signal",
    "phase1_model_version",
]


# ============================================================
# PHASE 2 COLUMNS
# ============================================================

PHASE2_COLUMNS: List[str] = [
    "window_start",
    "window_end",
    "phase2_score",
    "spike_state",
    "phase2_confidence",
    "phase2_state_multiplier",
    "statistical_evidence",
    "financial_evidence",
    "temporal_evidence",
    "breadth_score",
    "coordination_score",
    "tas",
    "fas",
    "materiality",
    "expected_fraud_count",
    "expected_fraud_amount",
    "risk_rate",
    "transaction_count",
    "total_amount",
    "unique_cards",
    "high_risk_count",
    "new_cards",
    "new_card_rate",
    "history_count",
    "baseline_ready",
    "ewma_value",
    "ewma_residual",
    "cusum_value",
    "cusum_signal",
    "acceleration",
    "phase2_primary_signal",
    "temporal_spike_signal",
]


# ============================================================
# EVENT COLUMNS
# ============================================================

EVENT_COLUMNS: List[str] = [
    "event_id",
    "event_active",
    "max_state",
    "peak_spike_score",
    "duration_minutes",
    "windows",
    "transaction_count_event",
    "total_amount_event",
    "expected_fraud_count_event",
    "expected_fraud_amount_event",
    "unique_cards_event",
    "max_coordination_score",
    "max_tas",
    "max_fas",
]


# ============================================================
# FUSION COLUMNS
# ============================================================

FUSION_COLUMNS: List[str] = [
    "base_fused_score",
    "confidence_adjusted_score",
    "fusion_confidence",
    "unified_risk_score",
    "transaction_expected_loss",
    "phase2_expected_fraud_amount",
    "expected_fraud_exposure",
    "response_action",
    "response_priority",
    "alert_flag",
    "early_warning_signal",
    "verified_spike_signal",
    "critical_spike_signal",
    "active_event_signal",
    "fusion_model_version",
    "model_version",
    "threshold",
]


# ============================================================
# TECHNICAL / GOVERNANCE COLUMNS
# ============================================================

TECHNICAL_COLUMNS: List[str] = [
    "_transaction_order",
    "_phase2_window_matched",
    "_phase2_window_age_minutes",
]


# ============================================================
# COMPLETE CONTRACT
# ============================================================

REQUIRED_COLUMNS: List[str] = (
    IDENTITY_COLUMNS
    + PHASE1_COLUMNS
    + PHASE2_COLUMNS
    + EVENT_COLUMNS
    + FUSION_COLUMNS
    + TECHNICAL_COLUMNS
)


def validate_schema(
    df: pd.DataFrame,
    strict: bool = False,
) -> None:
    """
    Validate the Fusion -> XAI data contract.

    Parameters
    ----------
    df:
        Fusion output dataframe.

    strict:
        If True, every listed column must exist.
        If False, only core columns are mandatory.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Expected pandas.DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Fusion input is empty."
        )

    if strict:
        required = REQUIRED_COLUMNS
    else:
        required = (
            IDENTITY_COLUMNS
            + [
                "amount",
                "fraud_probability",
                "risk_score",
                "phase1_score",
                "phase2_score",
                "spike_state",
                "unified_risk_score",
                "fusion_confidence",
            ]
        )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required Fusion columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )


def validate_numeric_columns(
    df: pd.DataFrame,
) -> None:
    """
    Verify that critical numerical fields can be converted
    to numeric values.
    """

    numeric_columns = [
        "amount",
        "fraud_probability",
        "risk_score",
        "phase1_score",
        "phase2_score",
        "phase2_confidence",
        "statistical_evidence",
        "financial_evidence",
        "temporal_evidence",
        "breadth_score",
        "coordination_score",
        "tas",
        "fas",
        "materiality",
        "expected_fraud_count",
        "expected_fraud_amount",
        "transaction_count",
        "total_amount",
        "unique_cards",
        "transaction_expected_loss",
        "phase2_expected_fraud_amount",
        "expected_fraud_exposure",
        "base_fused_score",
        "confidence_adjusted_score",
        "fusion_confidence",
        "unified_risk_score",
    ]

    existing = [
        c for c in numeric_columns
        if c in df.columns
    ]

    for column in existing:
        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if converted.isna().all():
            raise ValueError(
                f"Column '{column}' contains no valid numeric values."
            )

