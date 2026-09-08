import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (
    build_event_features,
    estimate_growth_rate,
    calculate_confidence,
)

from .forecaster import (
    forecast_loss,
)

from .scenarios import (
    build_scenarios,
)


# =============================================================
# JSON CLEANING
# =============================================================

def _clean_for_json(
    value,
):

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):

        if not np.isfinite(value):
            return None

        return float(value)

    if isinstance(
        value,
        float,
    ):

        if not np.isfinite(value):
            return None

    return value


# =============================================================
# BUILD FORECAST
# =============================================================

def build_impact_forecast(
    events: pd.DataFrame,
    config,
):

    # ---------------------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------------------

    features = build_event_features(
        events,
        config,
    )

    # ---------------------------------------------------------
    # GROWTH
    # ---------------------------------------------------------

    features = estimate_growth_rate(
        features,
    )

    # ---------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------

    features["forecast_confidence"] = (
        calculate_confidence(
            features,
            config,
        )
    )

    # ---------------------------------------------------------
    # FORECAST
    # ---------------------------------------------------------

    forecast = forecast_loss(
        features,
        config,
    )

    return (
        features,
        forecast,
    )


# =============================================================
# BUILD SCENARIOS
# =============================================================

def build_impact_scenarios(
    forecast: pd.DataFrame,
    config,
):

    return build_scenarios(
        forecast,
        config,
    )


# =============================================================
# BUILD SUMMARY
# =============================================================

def build_summary(
    features: pd.DataFrame,
    forecast: pd.DataFrame,
    scenarios: pd.DataFrame,
    config,
):

    summary = {}

    summary["project"] = (
        "FraudSentinel AI"
    )

    summary["component"] = (
        "Loss / Impact Forecasting"
    )

    summary["model_version"] = (
        config.model_version
    )

    summary["events_processed"] = int(
        features[
            "event_id"
        ].nunique()
    )

    summary["forecast_rows"] = int(
        len(forecast)
    )

    summary["scenario_rows"] = int(
        len(scenarios)
    )

    # ---------------------------------------------------------
    # CURRENT EXPOSURE
    # ---------------------------------------------------------

    if len(features):

        summary[
            "current_expected_loss"
        ] = float(
            features[
                "current_expected_loss"
            ].sum()
        )

        summary[
            "current_expected_fraud_count"
        ] = float(
            features[
                "expected_fraud_count"
            ].sum()
        )

        summary[
            "mean_forecast_confidence"
        ] = float(
            features[
                "forecast_confidence"
            ].mean()
        )

        summary[
            "max_spike_score"
        ] = float(
            features[
                "max_fas"
            ].max()
        )

    else:

        summary[
            "current_expected_loss"
        ] = 0.0

        summary[
            "current_expected_fraud_count"
        ] = 0.0

        summary[
            "mean_forecast_confidence"
        ] = 0.0

        summary[
            "max_spike_score"
        ] = 0.0

    # ---------------------------------------------------------
    # HORIZONS
    # ---------------------------------------------------------

    horizon_summary = {}

    for horizon in (
        config.horizons_minutes
    ):

        subset = forecast[
            forecast[
                "forecast_horizon_minutes"
            ] == horizon
        ]

        if len(subset):

            horizon_summary[
                str(horizon)
            ] = {

                "total_projected_loss":
                    float(
                        subset[
                            "projected_loss"
                        ].sum()
                    ),

                "lower_bound":
                    float(
                        subset[
                            "projected_loss_lower"
                        ].sum()
                    ),

                "upper_bound":
                    float(
                        subset[
                            "projected_loss_upper"
                        ].sum()
                    ),

                "projected_fraud_transactions":
                    float(
                        subset[
                            "projected_fraud_transactions"
                        ].sum()
                    ),

                "projected_transactions":
                    float(
                        subset[
                            "projected_transactions"
                        ].sum()
                    ),
            }

    summary[
        "horizons"
    ] = horizon_summary

    # ---------------------------------------------------------
    # SCENARIOS
    # ---------------------------------------------------------

    scenario_summary = {}

    for scenario in (
        scenarios[
            "scenario"
        ]
        .dropna()
        .unique()
    ):

        subset = scenarios[
            scenarios[
                "scenario"
            ] == scenario
        ]

        scenario_summary[
            str(scenario)
        ] = {

            "projected_loss":
                float(
                    subset[
                        "projected_loss"
                    ].sum()
                ),

            "loss_prevented":
                float(
                    subset[
                        "loss_prevented"
                    ].sum()
                ),
        }

    summary[
        "scenarios"
    ] = scenario_summary

    return summary


# =============================================================
# SAVE SUMMARY
# =============================================================

def save_summary(
    summary: dict,
    path: str,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            default=_clean_for_json,
        )
