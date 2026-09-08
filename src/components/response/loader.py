"""
FraudSentinel AI
Response Recommendation Layer

Input loaders for:
    1. Fusion
    2. Impact Forecast
    3. Impact Scenarios
"""

from pathlib import Path

import pandas as pd

from .config import (
    FUSION_PATH,
    IMPACT_FORECAST_PATH,
    IMPACT_SCENARIO_PATH,
)


# ============================================================
# VALIDATION HELPER
# ============================================================

def _validate_columns(
    df: pd.DataFrame,
    required: list[str],
    name: str,
) -> None:

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{name} is missing required columns: "
            f"{missing}"
        )


# ============================================================
# FUSION
# ============================================================

def load_fusion() -> pd.DataFrame:
    """
    Load the transaction-level Fusion artifact.
    """

    if not Path(FUSION_PATH).exists():

        raise FileNotFoundError(
            f"Fusion artifact not found:\n"
            f"{FUSION_PATH}"
        )

    df = pd.read_csv(
        FUSION_PATH,
        low_memory=True,
    )

    required = [
        "transaction_id",
        "timestamp",
        "merchant_id",
        "fraud_probability",
        "unified_risk_score",
        "spike_state",
        "event_id",
    ]

    _validate_columns(
        df,
        required,
        "Fusion artifact",
    )

    return df


# ============================================================
# IMPACT FORECAST
# ============================================================

def load_impact_forecast() -> pd.DataFrame:
    """
    Load the existing Impact Forecast artifact.

    IMPORTANT:
    We intentionally do not require lower_bound or upper_bound
    here because Response Recommendation does not need those
    fields to select a response.

    The loader accepts the actual forecast structure generated
    by the current Impact Forecasting component.
    """

    if not Path(IMPACT_FORECAST_PATH).exists():

        raise FileNotFoundError(
            f"Impact forecast artifact not found:\n"
            f"{IMPACT_FORECAST_PATH}"
        )

    df = pd.read_csv(
        IMPACT_FORECAST_PATH,
        low_memory=True,
    )

    # --------------------------------------------------------
    # Minimum information actually required by Response
    # --------------------------------------------------------

    required = [
        "event_id",
        "projected_loss",
    ]

    _validate_columns(
        df,
        required,
        "Impact forecast",
    )

    # --------------------------------------------------------
    # Optional columns
    #
    # We do NOT fail if these are absent.
    # --------------------------------------------------------

    if "forecast_confidence" not in df.columns:

        df["forecast_confidence"] = 0.0

    if "horizon_minutes" not in df.columns:

        # If the existing Impact artifact has a single
        # aggregate forecast, treat it as a 60-minute
        # planning horizon.
        df["horizon_minutes"] = 60

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    df["horizon_minutes"] = pd.to_numeric(
        df["horizon_minutes"],
        errors="coerce",
    )

    df["projected_loss"] = pd.to_numeric(
        df["projected_loss"],
        errors="coerce",
    )

    df["forecast_confidence"] = pd.to_numeric(
        df["forecast_confidence"],
        errors="coerce",
    )

    df["horizon_minutes"] = (
        df["horizon_minutes"]
        .fillna(60)
        .astype(int)
    )

    df["projected_loss"] = (
        df["projected_loss"]
        .fillna(0.0)
    )

    df["forecast_confidence"] = (
        df["forecast_confidence"]
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    return df


# ============================================================
# IMPACT SCENARIOS
# ============================================================

def load_impact_scenarios() -> pd.DataFrame:
    """
    Load Impact scenario artifact.
    """

    if not Path(IMPACT_SCENARIO_PATH).exists():

        raise FileNotFoundError(
            f"Impact scenario artifact not found:\n"
            f"{IMPACT_SCENARIO_PATH}"
        )

    df = pd.read_csv(
        IMPACT_SCENARIO_PATH,
        low_memory=True,
    )

    required = [
        "event_id",
        "scenario",
        "projected_loss",
        "loss_prevented",
    ]

    _validate_columns(
        df,
        required,
        "Impact scenario",
    )

    # --------------------------------------------------------
    # Optional horizon
    # --------------------------------------------------------

    if "horizon_minutes" not in df.columns:

        df["horizon_minutes"] = 60

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    df["horizon_minutes"] = pd.to_numeric(
        df["horizon_minutes"],
        errors="coerce",
    )

    df["projected_loss"] = pd.to_numeric(
        df["projected_loss"],
        errors="coerce",
    )

    df["loss_prevented"] = pd.to_numeric(
        df["loss_prevented"],
        errors="coerce",
    )

    df["horizon_minutes"] = (
        df["horizon_minutes"]
        .fillna(60)
        .astype(int)
    )

    df["projected_loss"] = (
        df["projected_loss"]
        .fillna(0.0)
    )

    df["loss_prevented"] = (
        df["loss_prevented"]
        .fillna(0.0)
    )

    return df
