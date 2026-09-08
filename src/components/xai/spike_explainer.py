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


def explain_spikes(
    df: pd.DataFrame,
    config: XAIConfig,
) -> pd.DataFrame:
    """
    Vectorized Phase-2 temporal/spike explanation.

    Designed for million-row datasets.
    """

    result = pd.DataFrame(
        index=df.index
    )

    # --------------------------------------------------------
    # Numeric inputs
    # --------------------------------------------------------

    phase2_score = _safe_numeric(
        df["phase2_score"]
    )

    phase2_confidence = _safe_numeric(
        df["phase2_confidence"]
    )

    statistical = _safe_numeric(
        df["statistical_evidence"]
    )

    financial = _safe_numeric(
        df["financial_evidence"]
    )

    temporal = _safe_numeric(
        df["temporal_evidence"]
    )

    breadth = _safe_numeric(
        df["breadth_score"]
    )

    coordination = _safe_numeric(
        df["coordination_score"]
    )

    tas = _safe_numeric(
        df["tas"]
    )

    fas = _safe_numeric(
        df["fas"]
    )

    materiality = _safe_numeric(
        df["materiality"]
    )

    expected_fraud_amount = _safe_numeric(
        df["expected_fraud_amount"]
    )

    transaction_count = _safe_numeric(
        df["transaction_count"]
    )

    unique_cards = _safe_numeric(
        df["unique_cards"]
    )

    # Optional columns
    new_cards = _safe_numeric(
        df["new_cards"]
        if "new_cards" in df.columns
        else pd.Series(0, index=df.index)
    )

    new_card_rate = _safe_numeric(
        df["new_card_rate"]
        if "new_card_rate" in df.columns
        else pd.Series(0, index=df.index)
    )

    ewma_residual = _safe_numeric(
        df["ewma_residual"]
        if "ewma_residual" in df.columns
        else pd.Series(0, index=df.index)
    )

    cusum_signal = _safe_numeric(
        df["cusum_signal"]
        if "cusum_signal" in df.columns
        else pd.Series(0, index=df.index)
    )

    acceleration = _safe_numeric(
        df["acceleration"]
        if "acceleration" in df.columns
        else pd.Series(0, index=df.index)
    )

    # --------------------------------------------------------
    # State
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

    # --------------------------------------------------------
    # Evidence masks
    # --------------------------------------------------------

    strong_statistical = (
        statistical
        >= config.strong_evidence_threshold
    )

    high_statistical = (
        statistical
        >= config.high_evidence_threshold
    )

    strong_financial = (
        financial
        >= config.strong_evidence_threshold
    )

    high_financial = (
        financial
        >= config.high_evidence_threshold
    )

    strong_temporal = (
        temporal
        >= config.strong_evidence_threshold
    )

    high_temporal = (
        temporal
        >= config.high_evidence_threshold
    )

    high_breadth = (
        breadth
        >= config.high_evidence_threshold
    )

    high_coordination = (
        coordination
        >= config.high_evidence_threshold
    )

    material_spike = (
        materiality
        >= config.materiality_threshold
    )

    cusum_active = (
        cusum_signal > 0
    )

    positive_acceleration = (
        acceleration > 0
    )

    new_cards_present = (
        new_cards > 0
    )

    high_new_card_rate = (
        new_card_rate >= 0.50
    )

    # --------------------------------------------------------
    # Primary reason
    # --------------------------------------------------------

    primary_reason = np.select(
        [
            critical_state,
            verified_state,
            early_warning_state,
            strong_statistical,
            strong_financial,
            strong_temporal,
            high_statistical,
            high_financial,
            high_temporal,
            material_spike,
        ],
        [
            "The merchant is in a critical active fraud-spike state.",
            "The merchant is in a verified fraud-spike state.",
            "The merchant shows an early warning of abnormal fraud activity.",
            "Statistical evidence shows a strong deviation from the merchant baseline.",
            "Expected fraud amount is strongly elevated.",
            "Temporal dynamics indicate persistent or accelerating abnormal activity.",
            "Statistical evidence indicates elevated deviation from baseline.",
            "Financial evidence indicates elevated expected fraud impact.",
            "Temporal evidence indicates abnormal activity over time.",
            "The spike has material financial significance.",
        ],
        default="No strong temporal-spike driver was identified.",
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result[
        "spike_explanation_version"
    ] = config.explanation_version

    result[
        "spike_primary_reason"
    ] = primary_reason

    result[
        "spike_state_explained"
    ] = state

    result[
        "spike_is_active"
    ] = (
        critical_state
        | verified_state
        | early_warning_state
    )

    result[
        "spike_score_explained"
    ] = phase2_score

    result[
        "spike_confidence_explained"
    ] = phase2_confidence

    result[
        "spike_statistical_evidence"
    ] = statistical

    result[
        "spike_financial_evidence"
    ] = financial

    result[
        "spike_temporal_evidence"
    ] = temporal

    result[
        "spike_breadth"
    ] = breadth

    result[
        "spike_coordination"
    ] = coordination

    result[
        "spike_tas"
    ] = tas

    result[
        "spike_fas"
    ] = fas

    result[
        "spike_materiality"
    ] = materiality

    result[
        "spike_expected_fraud_amount"
    ] = expected_fraud_amount

    result[
        "spike_transaction_count"
    ] = transaction_count

    result[
        "spike_unique_cards"
    ] = unique_cards

    # --------------------------------------------------------
    # Individual signal columns
    # --------------------------------------------------------

    result[
        "spike_strong_statistical"
    ] = strong_statistical

    result[
        "spike_strong_financial"
    ] = strong_financial

    result[
        "spike_strong_temporal"
    ] = strong_temporal

    result[
        "spike_high_breadth"
    ] = high_breadth

    result[
        "spike_high_coordination"
    ] = high_coordination

    result[
        "spike_cusum_active"
    ] = cusum_active

    result[
        "spike_positive_acceleration"
    ] = positive_acceleration

    result[
        "spike_new_cards_present"
    ] = new_cards_present

    result[
        "spike_high_new_card_rate"
    ] = high_new_card_rate

    # --------------------------------------------------------
    # Signal count
    # --------------------------------------------------------

    signal_count = (
        critical_state.astype(int)
        + verified_state.astype(int)
        + early_warning_state.astype(int)
        + strong_statistical.astype(int)
        + strong_financial.astype(int)
        + strong_temporal.astype(int)
        + high_breadth.astype(int)
        + high_coordination.astype(int)
        + cusum_active.astype(int)
        + positive_acceleration.astype(int)
        + material_spike.astype(int)
        + new_cards_present.astype(int)
        + high_new_card_rate.astype(int)
    )

    result[
        "spike_signal_count"
    ] = signal_count

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    result[
        "spike_explanation"
    ] = (
        result["spike_primary_reason"]
        + " Phase-2 spike score is "
        + phase2_score.map(
            lambda x: f"{x:.2f}"
        )
        + ", with statistical evidence "
        + statistical.map(
            lambda x: f"{x:.2f}"
        )
        + ", temporal evidence "
        + temporal.map(
            lambda x: f"{x:.2f}"
        )
        + ", and financial evidence "
        + financial.map(
            lambda x: f"{x:.2f}"
        )
        + "."
    )

    # --------------------------------------------------------
    # Reasons
    # --------------------------------------------------------

    reason_map = {
        "Critical spike state": critical_state,
        "Verified fraud spike": verified_state,
        "Early warning": early_warning_state,
        "Strong statistical evidence": strong_statistical,
        "Strong financial evidence": strong_financial,
        "Strong temporal evidence": strong_temporal,
        "Broad activity": high_breadth,
        "Elevated coordination": high_coordination,
        "CUSUM active": cusum_active,
        "Positive acceleration": positive_acceleration,
        "Material spike": material_spike,
        "New cards present": new_cards_present,
        "High new-card rate": high_new_card_rate,
    }

    reason_frame = pd.DataFrame(
        reason_map,
        index=df.index,
    )

    result[
        "spike_reasons"
    ] = reason_frame.apply(
        lambda row: " | ".join(
            row.index[row].tolist()
        ),
        axis=1,
    )

    return result
