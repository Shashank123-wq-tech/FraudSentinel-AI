import numpy as np
import pandas as pd


def safe_sum(
    df: pd.DataFrame,
    column: str,
) -> float:

    if column not in df.columns:
        return 0.0

    return float(
        pd.to_numeric(
            df[column],
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .sum()
    )


def safe_mean(
    df: pd.DataFrame,
    column: str,
) -> float:

    if column not in df.columns:
        return 0.0

    values = (
        pd.to_numeric(
            df[column],
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if values.empty:
        return 0.0

    return float(values.mean())


def calculate_response_metrics(
    response: pd.DataFrame,
) -> dict:

    if response.empty:
        return {
            "events": 0,
            "prevented": 0.0,
            "no_action_loss": 0.0,
            "action_loss": 0.0,
            "confidence": 0.0,
        }

    return {
        "events": len(response),

        "prevented": safe_sum(
            response,
            "estimated_loss_prevented",
        ),

        "no_action_loss": safe_sum(
            response,
            "no_action_loss_60m",
        ),

        "action_loss": safe_sum(
            response,
            "expected_loss_if_action",
        ),

        "confidence": safe_mean(
            response,
            "recommendation_confidence",
        ),
    }


def calculate_impact_metrics(
    forecast: pd.DataFrame,
) -> dict:

    if forecast.empty:
        return {
            "current_loss": 0.0,
            "forecast_15": 0.0,
            "forecast_30": 0.0,
            "forecast_60": 0.0,
        }

    if "horizon_minutes" not in forecast.columns:
        return {
            "current_loss": safe_sum(
                forecast,
                "current_expected_loss",
            ),
            "forecast_15": 0.0,
            "forecast_30": 0.0,
            "forecast_60": safe_sum(
                forecast,
                "projected_loss",
            ),
        }

    forecast = forecast.copy()

    forecast["horizon_minutes"] = pd.to_numeric(
        forecast["horizon_minutes"],
        errors="coerce",
    )

    forecast["projected_loss"] = pd.to_numeric(
        forecast["projected_loss"],
        errors="coerce",
    ).fillna(0)

    return {
        "current_loss": safe_sum(
            forecast,
            "current_expected_loss",
        ),

        "forecast_15": forecast.loc[
            forecast["horizon_minutes"] == 15,
            "projected_loss",
        ].sum(),

        "forecast_30": forecast.loc[
            forecast["horizon_minutes"] == 30,
            "projected_loss",
        ].sum(),

        "forecast_60": forecast.loc[
            forecast["horizon_minutes"] == 60,
            "projected_loss",
        ].sum(),
    }