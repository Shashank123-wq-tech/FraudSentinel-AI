import numpy as np
import pandas as pd


# =============================================================
# SAFE DIVISION
# =============================================================

def safe_divide(
    numerator,
    denominator,
    default=0.0,
):

    numerator = np.asarray(
        numerator,
        dtype=float,
    )

    denominator = np.asarray(
        denominator,
        dtype=float,
    )

    result = np.divide(
        numerator,
        denominator,
        out=np.full_like(
            numerator,
            float(default),
            dtype=float,
        ),
        where=np.abs(denominator) > 1e-12,
    )

    return result


# =============================================================
# EVENT FEATURE ENGINEERING
# =============================================================

def build_event_features(
    events: pd.DataFrame,
    config,
):

    df = events.copy()

    # ---------------------------------------------------------
    # BASIC CLEANING
    # ---------------------------------------------------------

    df["duration_minutes"] = (
        df["duration_minutes"]
        .fillna(0)
        .clip(lower=0)
    )

    df["transaction_count"] = (
        df["transaction_count"]
        .fillna(0)
        .clip(lower=0)
    )

    df["total_amount"] = (
        df["total_amount"]
        .fillna(0)
        .clip(lower=0)
    )

    df["expected_fraud_count"] = (
        df["expected_fraud_count"]
        .fillna(0)
        .clip(lower=0)
    )

    df["expected_fraud_amount"] = (
        df["expected_fraud_amount"]
        .fillna(0)
        .clip(lower=0)
    )

    df["unique_cards"] = (
        df["unique_cards"]
        .fillna(0)
        .clip(lower=0)
    )

    # ---------------------------------------------------------
    # EXPECTED FRAUD RATE
    # ---------------------------------------------------------

    df["expected_fraud_rate"] = safe_divide(
        df["expected_fraud_count"],
        df["transaction_count"],
    )

    # ---------------------------------------------------------
    # EXPECTED LOSS RATE
    # ---------------------------------------------------------

    df["expected_loss_rate"] = safe_divide(
        df["expected_fraud_amount"],
        df["total_amount"],
    )

    # ---------------------------------------------------------
    # LOSS VELOCITY
    # ---------------------------------------------------------

    duration_hours = (
        df["duration_minutes"]
        .clip(lower=1)
        / 60.0
    )

    df["loss_velocity_per_hour"] = (
        df["expected_fraud_amount"]
        / duration_hours
    )

    # ---------------------------------------------------------
    # FRAUD TRANSACTION VELOCITY
    # ---------------------------------------------------------

    df["fraud_txn_velocity_per_hour"] = (
        df["expected_fraud_count"]
        / duration_hours
    )

    # ---------------------------------------------------------
    # TRANSACTION VELOCITY
    # ---------------------------------------------------------

    df["transaction_velocity_per_hour"] = (
        df["transaction_count"]
        / duration_hours
    )

    # ---------------------------------------------------------
    # TRANSACTION AMOUNT VELOCITY
    # ---------------------------------------------------------

    df["transaction_amount_velocity_per_hour"] = (
        df["total_amount"]
        / duration_hours
    )

    # ---------------------------------------------------------
    # AVERAGE TRANSACTION AMOUNT
    # ---------------------------------------------------------

    df["average_transaction_amount"] = safe_divide(
        df["total_amount"],
        df["transaction_count"],
    )

    # ---------------------------------------------------------
    # AVERAGE EXPECTED FRAUD AMOUNT
    # ---------------------------------------------------------

    df["average_expected_fraud_amount"] = safe_divide(
        df["expected_fraud_amount"],
        df["expected_fraud_count"],
    )

    df["average_expected_fraud_amount"] = (
        df["average_expected_fraud_amount"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    # ---------------------------------------------------------
    # COORDINATION
    # ---------------------------------------------------------

    df["coordination_score"] = (
        df["max_coordination_score"]
        .fillna(0)
        .clip(
            lower=0,
            upper=1,
        )
    )

    # ---------------------------------------------------------
    # SPIKE INTENSITY
    # ---------------------------------------------------------

    df["spike_intensity"] = (
        df["max_fas"]
        .fillna(0)
        .clip(
            lower=0,
            upper=100,
        )
        / 100.0
    )

    # ---------------------------------------------------------
    # CURRENT EXPECTED LOSS
    # ---------------------------------------------------------

    df["current_expected_loss"] = (
        df["expected_fraud_amount"]
        .clip(lower=0)
    )

    return df


# =============================================================
# GROWTH ESTIMATION
# =============================================================

def estimate_growth_rate(
    df: pd.DataFrame,
):

    result = df.copy()

    intensity = (
        result["spike_intensity"]
        .fillna(0)
        .clip(0, 1)
    )

    coordination = (
        result["coordination_score"]
        .fillna(0)
        .clip(0, 1)
    )

    fraud_rate = (
        result["expected_fraud_rate"]
        .fillna(0)
        .clip(0, 1)
    )

    growth_signal = (
        0.50 * intensity
        + 0.30 * coordination
        + 0.20 * fraud_rate
    )

    result["growth_rate_per_hour"] = (
        0.10 * growth_signal
    )

    return result


# =============================================================
# FORECAST CONFIDENCE
# =============================================================

def calculate_confidence(
    df: pd.DataFrame,
    config,
):

    history = (
        df["windows"]
        .fillna(0)
        .clip(lower=0)
    )

    history_confidence = (
        history
        / float(
            config.min_history_windows
        )
    ).clip(
        0,
        1,
    )

    duration_confidence = (
        df["duration_minutes"]
        .fillna(0)
        / 60.0
    ).clip(
        0,
        1,
    )

    event_quality = (
        0.70 * history_confidence
        + 0.30 * duration_confidence
    )

    return event_quality.clip(
        0,
        1,
    )
