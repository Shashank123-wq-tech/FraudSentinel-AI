import numpy as np
import pandas as pd

from .config import XAIConfig


def _safe_numeric(
    series: pd.Series,
    default: float = 0.0,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(default)


def explain_fusion(
    df: pd.DataFrame,
    config: XAIConfig,
) -> pd.DataFrame:
    """
    Vectorized Fusion explanation.

    Designed for million-row datasets.
    """

    result = pd.DataFrame(
        index=df.index
    )

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

    phase1_score = _safe_numeric(
        df["phase1_score"]
    )

    phase2_score = _safe_numeric(
        df["phase2_score"]
    )

    base_fused = _safe_numeric(
        df["base_fused_score"]
    )

    confidence_adjusted = _safe_numeric(
        df["confidence_adjusted_score"]
    )

    fusion_confidence = _safe_numeric(
        df["fusion_confidence"]
    )

    unified_risk = _safe_numeric(
        df["unified_risk_score"]
    )

    expected_loss = _safe_numeric(
        df["transaction_expected_loss"]
    )

    phase2_amount = _safe_numeric(
        df["phase2_expected_fraud_amount"]
    )

    exposure = _safe_numeric(
        df["expected_fraud_exposure"]
    )

    # --------------------------------------------------------
    # State / flags
    # --------------------------------------------------------

    state = (
        df["spike_state"]
        .fillna("NORMAL")
        .astype(str)
        .str.upper()
    )

    critical_state = (
        state == "CRITICAL_ACTIVE_SPIKE"
    )

    verified_state = (
        state == "VERIFIED_FRAUD_SPIKE"
    )

    early_warning_state = (
        state == "EARLY_WARNING"
    )

    event_active = (
        df["event_active"]
        .fillna(False)
        .astype(bool)
    )

    alert_flag = (
        df["alert_flag"]
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # Score conditions
    # --------------------------------------------------------

    phase1_critical = (
        phase1_score
        >= config.critical_risk_score_threshold
    )

    phase1_high = (
        phase1_score
        >= config.high_risk_score_threshold
    )

    phase2_critical = (
        phase2_score
        >= config.critical_spike_score_threshold
    )

    phase2_high = (
        phase2_score
        >= config.high_spike_score_threshold
    )

    # --------------------------------------------------------
    # Primary reason
    # --------------------------------------------------------

    primary_reason = np.select(
        [
            critical_state,
            phase2_critical,
            phase1_critical,
            verified_state,
            phase2_high,
            phase1_high,
            early_warning_state,
            event_active,
            exposure > 0,
        ],
        [
            "The transaction is associated with a critical active spike.",
            "Temporal fraud-spike risk is critical.",
            "Transaction-level fraud risk is critical.",
            "The transaction is associated with a verified fraud spike.",
            "Temporal fraud-spike risk is elevated.",
            "Transaction-level fraud risk is elevated.",
            "The transaction is associated with an early-warning spike.",
            "The transaction is part of an active fraud event.",
            "The transaction contributes to measurable expected fraud exposure.",
        ],
        default="Unified risk is driven primarily by transaction and temporal signals.",
    )

    # --------------------------------------------------------
    # Main fields
    # --------------------------------------------------------

    result[
        "fusion_explanation_version"
    ] = config.explanation_version

    result[
        "fusion_primary_reason"
    ] = primary_reason

    result[
        "fusion_phase1_score"
    ] = phase1_score

    result[
        "fusion_phase2_score"
    ] = phase2_score

    result[
        "fusion_base_score"
    ] = base_fused

    result[
        "fusion_confidence_adjusted_score"
    ] = confidence_adjusted

    result[
        "fusion_confidence"
    ] = fusion_confidence

    result[
        "fusion_unified_risk"
    ] = unified_risk

    result[
        "fusion_transaction_expected_loss"
    ] = expected_loss

    result[
        "fusion_phase2_expected_amount"
    ] = phase2_amount

    result[
        "fusion_expected_exposure"
    ] = exposure

    result[
        "fusion_spike_state"
    ] = state

    result[
        "fusion_event_active"
    ] = event_active

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    result[
        "fusion_response_action"
    ] = (
        df["response_action"]
        .fillna("MONITOR")
        .astype(str)
    )

    result[
        "fusion_response_priority"
    ] = (
        df["response_priority"]
        .fillna("LOW")
        .astype(str)
    )

    result[
        "fusion_alert"
    ] = alert_flag

    # --------------------------------------------------------
    # Signal columns
    # --------------------------------------------------------

    result[
        "fusion_phase1_critical"
    ] = phase1_critical

    result[
        "fusion_phase2_critical"
    ] = phase2_critical

    result[
        "fusion_active_event"
    ] = event_active

    result[
        "fusion_alert_triggered"
    ] = alert_flag

    # --------------------------------------------------------
    # Signal count
    # --------------------------------------------------------

    result[
        "fusion_signal_count"
    ] = (
        phase1_critical.astype(int)
        + phase1_high.astype(int)
        + phase2_critical.astype(int)
        + phase2_high.astype(int)
        + critical_state.astype(int)
        + verified_state.astype(int)
        + early_warning_state.astype(int)
        + event_active.astype(int)
        + (exposure > 0).astype(int)
        + alert_flag.astype(int)
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    result[
        "fusion_explanation"
    ] = (
        result["fusion_primary_reason"]
        + " Phase-1 score is "
        + phase1_score.map(
            lambda x: f"{x:.2f}"
        )
        + ", Phase-2 score is "
        + phase2_score.map(
            lambda x: f"{x:.2f}"
        )
        + ", and final unified risk is "
        + unified_risk.map(
            lambda x: f"{x:.2f}"
        )
        + "."
    )

    # --------------------------------------------------------
    # Reasons
    # --------------------------------------------------------

    reason_map = {
        "Phase-1 critical": phase1_critical,
        "Phase-1 elevated": phase1_high,
        "Phase-2 critical": phase2_critical,
        "Phase-2 elevated": phase2_high,
        "Critical spike": critical_state,
        "Verified spike": verified_state,
        "Early warning": early_warning_state,
        "Active event": event_active,
        "Financial exposure": exposure > 0,
        "Alert triggered": alert_flag,
    }

    reason_frame = pd.DataFrame(
        reason_map,
        index=df.index,
    )

    result[
        "fusion_reasons"
    ] = reason_frame.apply(
        lambda row: " | ".join(
            row.index[row].tolist()
        ),
        axis=1,
    )

    return result
