"""
FraudSentinel AI
Response Recommendation Engine

Combines:

    Fusion Intelligence
        +
    Impact / Loss Forecasting
        +
    Response Scoring
        +
    Recommendation Confidence
        +
    Response Policy

into final event-level response recommendations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# SAFE NUMERIC HELPERS
# ============================================================

def _numeric_series(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    """
    Return a numeric Series safely.

    Handles:
        - missing columns
        - NaN
        - None
        - invalid strings
        - infinite values
    """

    if column not in df.columns:

        return pd.Series(
            default,
            index=df.index,
            dtype=float,
        )

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return values.fillna(
        default
    ).astype(float)


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Safe vectorized division.
    """

    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    ).fillna(0.0)

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    ).fillna(0.0)

    result = pd.Series(
        0.0,
        index=numerator.index,
        dtype=float,
    )

    mask = denominator != 0

    result.loc[mask] = (
        numerator.loc[mask]
        / denominator.loc[mask]
    )

    return result.replace(
        [np.inf, -np.inf],
        0.0,
    ).fillna(0.0)


# ============================================================
# IMPACT FEATURE CONSTRUCTION
# ============================================================

def build_impact_features(
    forecast: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one event-level Impact feature table.

    Forecast input:
        event_id
        projected_loss
        forecast_confidence
        horizon_minutes

    Scenario input:
        event_id
        scenario
        projected_loss
        loss_prevented
    """

    if forecast is None:

        raise ValueError(
            "Impact forecast cannot be None."
        )

    if scenarios is None:

        raise ValueError(
            "Impact scenarios cannot be None."
        )

    if not isinstance(
        forecast,
        pd.DataFrame,
    ):

        raise TypeError(
            "forecast must be a pandas DataFrame."
        )

    if not isinstance(
        scenarios,
        pd.DataFrame,
    ):

        raise TypeError(
            "scenarios must be a pandas DataFrame."
        )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    if "event_id" not in forecast.columns:

        raise ValueError(
            "Impact forecast is missing required column: "
            "'event_id'"
        )

    if "event_id" not in scenarios.columns:

        raise ValueError(
            "Impact scenarios are missing required column: "
            "'event_id'"
        )

    forecast = forecast.copy()
    scenarios = scenarios.copy()

    # ========================================================
    # EVENT IDS
    # ========================================================

    forecast["event_id"] = (
        forecast["event_id"]
        .astype(str)
        .str.strip()
    )

    scenarios["event_id"] = (
        scenarios["event_id"]
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # HORIZON
    # ========================================================

    if "horizon_minutes" in forecast.columns:

        forecast["horizon_minutes"] = pd.to_numeric(
            forecast["horizon_minutes"],
            errors="coerce",
        )

    else:

        forecast["horizon_minutes"] = 60.0

    forecast["horizon_minutes"] = (
        forecast["horizon_minutes"]
        .fillna(60.0)
        .astype(float)
    )

    # ========================================================
    # PROJECTED LOSS
    # ========================================================

    forecast["projected_loss"] = _numeric_series(
        forecast,
        "projected_loss",
        0.0,
    )

    # ========================================================
    # FORECAST CONFIDENCE
    # ========================================================

    forecast["forecast_confidence"] = (
        _numeric_series(
            forecast,
            "forecast_confidence",
            0.0,
        )
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    # ========================================================
    # OPTIONAL BOUNDS
    # ========================================================

    if "lower_bound" in forecast.columns:

        forecast["lower_bound"] = _numeric_series(
            forecast,
            "lower_bound",
            0.0,
        )

    else:

        forecast["lower_bound"] = (
            forecast["projected_loss"]
            * 0.70
        )

    if "upper_bound" in forecast.columns:

        forecast["upper_bound"] = _numeric_series(
            forecast,
            "upper_bound",
            0.0,
        )

    else:

        forecast["upper_bound"] = (
            forecast["projected_loss"]
            * 1.30
        )

    # ========================================================
    # FORECAST EVENT SUMMARY
    # ========================================================

    forecast_summary = (
        forecast
        .groupby(
            "event_id",
            as_index=False,
        )
        .agg(
            forecast_confidence=(
                "forecast_confidence",
                "mean",
            ),
            forecast_min_loss=(
                "projected_loss",
                "min",
            ),
            forecast_max_loss=(
                "projected_loss",
                "max",
            ),
            forecast_mean_loss=(
                "projected_loss",
                "mean",
            ),
        )
    )

    # ========================================================
    # HORIZON PIVOT
    # ========================================================

    horizon_pivot = (
        forecast
        .pivot_table(
            index="event_id",
            columns="horizon_minutes",
            values="projected_loss",
            aggfunc="first",
        )
        .reset_index()
    )

    # ========================================================
    # STANDARDIZE HORIZON COLUMN NAMES
    # ========================================================

    renamed_columns = []

    for column in horizon_pivot.columns:

        if column == "event_id":

            renamed_columns.append(
                "event_id"
            )

        else:

            try:

                horizon_value = int(
                    float(column)
                )

                renamed_columns.append(
                    f"projected_loss_{horizon_value}m"
                )

            except (
                TypeError,
                ValueError,
            ):

                renamed_columns.append(
                    str(column)
                )

    horizon_pivot.columns = renamed_columns

    # ========================================================
    # ENSURE STANDARD HORIZONS
    # ========================================================

    for horizon in [15, 30, 60]:

        column = (
            f"projected_loss_{horizon}m"
        )

        if column not in horizon_pivot.columns:

            horizon_pivot[column] = np.nan

    # ========================================================
    # CORRECT HORIZON FALLBACK
    # ========================================================
    #
    # IMPORTANT:
    # Do NOT use:
    #
    #     fillna(numpy_array)
    #
    # because pandas requires a scalar, dict or Series.
    #
    # Instead create an event-indexed fallback Series.
    # ========================================================

    forecast_mean_by_event = (
        forecast_summary
        .set_index("event_id")[
            "forecast_mean_loss"
        ]
    )

    # --------------------------------------------------------
    # 15-MINUTE
    # --------------------------------------------------------

    fallback_15 = (
        horizon_pivot["event_id"]
        .map(
            forecast_mean_by_event
        )
    )

    horizon_pivot[
        "projected_loss_15m"
    ] = (
        horizon_pivot[
            "projected_loss_15m"
        ]
        .fillna(
            fallback_15
        )
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # 30-MINUTE
    # --------------------------------------------------------

    horizon_pivot[
        "projected_loss_30m"
    ] = (
        horizon_pivot[
            "projected_loss_30m"
        ]
        .fillna(
            horizon_pivot[
                "projected_loss_60m"
            ]
        )
        .fillna(
            horizon_pivot[
                "projected_loss_15m"
            ]
        )
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # 60-MINUTE
    # --------------------------------------------------------

    horizon_pivot[
        "projected_loss_60m"
    ] = (
        horizon_pivot[
            "projected_loss_60m"
        ]
        .fillna(
            horizon_pivot[
                "projected_loss_30m"
            ]
        )
        .fillna(
            horizon_pivot[
                "projected_loss_15m"
            ]
        )
        .fillna(0.0)
    )

    # ========================================================
    # SCENARIO VALIDATION
    # ========================================================

    if "scenario" not in scenarios.columns:

        raise ValueError(
            "Impact scenarios are missing required column: "
            "'scenario'"
        )

    scenarios["scenario"] = (
        scenarios["scenario"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    scenarios["projected_loss"] = (
        _numeric_series(
            scenarios,
            "projected_loss",
            0.0,
        )
    )

    scenarios["loss_prevented"] = (
        _numeric_series(
            scenarios,
            "loss_prevented",
            0.0,
        )
    )

    # ========================================================
    # SCENARIO LOSS PIVOT
    # ========================================================

    scenario_loss = (
        scenarios
        .pivot_table(
            index="event_id",
            columns="scenario",
            values="projected_loss",
            aggfunc="sum",
        )
        .reset_index()
    )

    renamed = []

    for column in scenario_loss.columns:

        if column == "event_id":

            renamed.append(
                "event_id"
            )

        else:

            renamed.append(
                "scenario_loss_"
                + str(column).lower()
            )

    scenario_loss.columns = renamed

    # ========================================================
    # SCENARIO PREVENTED LOSS PIVOT
    # ========================================================

    scenario_prevented = (
        scenarios
        .pivot_table(
            index="event_id",
            columns="scenario",
            values="loss_prevented",
            aggfunc="sum",
        )
        .reset_index()
    )

    renamed = []

    for column in scenario_prevented.columns:

        if column == "event_id":

            renamed.append(
                "event_id"
            )

        else:

            renamed.append(
                "scenario_prevented_"
                + str(column).lower()
            )

    scenario_prevented.columns = renamed

    # ========================================================
    # MERGE FORECAST FEATURES
    # ========================================================

    impact = forecast_summary.merge(
        horizon_pivot,
        on="event_id",
        how="outer",
    )

    impact = impact.merge(
        scenario_loss,
        on="event_id",
        how="left",
    )

    impact = impact.merge(
        scenario_prevented,
        on="event_id",
        how="left",
    )

    # ========================================================
    # STANDARD SCENARIO OUTPUTS
    # ========================================================

    scenario_mapping = {

        "scenario_loss_no_action":
            "no_action_loss_60m",

        "scenario_loss_moderate_intervention":
            "moderate_loss_60m",

        "scenario_loss_aggressive_intervention":
            "aggressive_loss_60m",

        "scenario_prevented_no_action":
            "no_action_prevented_60m",

        "scenario_prevented_moderate_intervention":
            "moderate_prevented_60m",

        "scenario_prevented_aggressive_intervention":
            "aggressive_prevented_60m",
    }

    for source, target in scenario_mapping.items():

        if source in impact.columns:

            impact[target] = _numeric_series(
                impact,
                source,
                0.0,
            )

        else:

            impact[target] = 0.0

    # ========================================================
    # NUMERIC CLEANUP
    # ========================================================

    numeric_columns = [

        "forecast_confidence",

        "forecast_min_loss",

        "forecast_max_loss",

        "forecast_mean_loss",

        "projected_loss_15m",

        "projected_loss_30m",

        "projected_loss_60m",

        "no_action_loss_60m",

        "moderate_loss_60m",

        "aggressive_loss_60m",

        "no_action_prevented_60m",

        "moderate_prevented_60m",

        "aggressive_prevented_60m",
    ]

    for column in numeric_columns:

        if column in impact.columns:

            impact[column] = _numeric_series(
                impact,
                column,
                0.0,
            )

    # ========================================================
    # REMOVE DUPLICATE EVENTS
    # ========================================================

    impact = (
        impact
        .drop_duplicates(
            subset=["event_id"],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return impact


# ============================================================
# MERGE FUSION + IMPACT
# ============================================================

def merge_impact_into_fusion(
    fusion: pd.DataFrame,
    impact: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge transaction-level Fusion intelligence with
    event-level Impact intelligence.

    One recommendation is generated per event.
    """

    if fusion is None:

        raise ValueError(
            "Fusion data cannot be None."
        )

    if impact is None:

        raise ValueError(
            "Impact data cannot be None."
        )

    if not isinstance(
        fusion,
        pd.DataFrame,
    ):

        raise TypeError(
            "fusion must be a pandas DataFrame."
        )

    if not isinstance(
        impact,
        pd.DataFrame,
    ):

        raise TypeError(
            "impact must be a pandas DataFrame."
        )

    # ========================================================
    # REQUIRED EVENT IDS
    # ========================================================

    if "event_id" not in fusion.columns:

        raise ValueError(
            "Fusion is missing required column: "
            "'event_id'"
        )

    if "event_id" not in impact.columns:

        raise ValueError(
            "Impact is missing required column: "
            "'event_id'"
        )

    fusion = fusion.copy()
    impact = impact.copy()

    # ========================================================
    # NORMALIZE IDS
    # ========================================================

    fusion["event_id"] = (
        fusion["event_id"]
        .astype(str)
        .str.strip()
    )

    impact["event_id"] = (
        impact["event_id"]
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # REMOVE INVALID EVENT IDS
    # ========================================================

    invalid_event_ids = {
        "",
        "nan",
        "none",
        "nat",
    }

    fusion = fusion[
        ~fusion["event_id"]
        .str.lower()
        .isin(invalid_event_ids)
    ].copy()

    impact = impact[
        ~impact["event_id"]
        .str.lower()
        .isin(invalid_event_ids)
    ].copy()

    if fusion.empty:

        return pd.DataFrame()

    if impact.empty:

        return pd.DataFrame()

    # ========================================================
    # SELECT REPRESENTATIVE FUSION ROW PER EVENT
    # ========================================================

    if "unified_risk_score" in fusion.columns:

        fusion["_event_sort_score"] = (
            _numeric_series(
                fusion,
                "unified_risk_score",
                0.0,
            )
        )

    else:

        fusion["_event_sort_score"] = 0.0

    event_fusion = (
        fusion
        .sort_values(
            [
                "event_id",
                "_event_sort_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=["event_id"],
            keep="first",
        )
        .drop(
            columns=[
                "_event_sort_score",
            ],
            errors="ignore",
        )
    )

    # ========================================================
    # MERGE
    # ========================================================

    result = event_fusion.merge(
        impact,
        on="event_id",
        how="inner",
        suffixes=(
            "",
            "_impact",
        ),
    )

    return result.reset_index(
        drop=True
    )


# ============================================================
# RECOMMENDATION CONFIDENCE
# ============================================================

def calculate_recommendation_confidence(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate recommendation confidence.

    Formula:

        55% Fusion confidence
        45% Forecast confidence

    Critical events receive +0.10.

    Final value is clipped to [0, 1].

    Fully vectorized implementation.
    """

    if df is None:

        raise ValueError(
            "Recommendation data cannot be None."
        )

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "Recommendation confidence expects "
            "a pandas DataFrame."
        )

    result = df.copy()

    # ========================================================
    # FUSION CONFIDENCE
    # ========================================================

    fusion_confidence = (
        _numeric_series(
            result,
            "fusion_confidence",
            0.0,
        )
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    # ========================================================
    # FORECAST CONFIDENCE
    # ========================================================

    forecast_confidence = (
        _numeric_series(
            result,
            "forecast_confidence",
            0.0,
        )
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    # ========================================================
    # BASE CONFIDENCE
    # ========================================================

    recommendation_confidence = (
        0.55 * fusion_confidence
        +
        0.45 * forecast_confidence
    )

    # ========================================================
    # CRITICAL EVENT BONUS
    # ========================================================

    if "spike_state" in result.columns:

        critical_mask = (
            result["spike_state"]
            .astype(str)
            .eq(
                "CRITICAL_ACTIVE_SPIKE"
            )
        )

        recommendation_confidence = (
            recommendation_confidence
            +
            0.10
            *
            critical_mask.astype(float)
        )

    # ========================================================
    # CLIP
    # ========================================================

    result[
        "recommendation_confidence"
    ] = (
        recommendation_confidence
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    return result


# ============================================================
# FINALIZE RECOMMENDATIONS
# ============================================================

def finalize_recommendations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the expected economic outcome of the
    selected response action.

    Mapping:

        NO_ACTION
            → no-action loss

        MONITOR
            → no-action loss

        RESTRICT
            → moderate intervention loss

        CONTAIN
            → aggressive intervention loss
    """

    if df is None:

        raise ValueError(
            "Recommendation data cannot be None."
        )

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "Finalization expects a pandas DataFrame."
        )

    result = df.copy()

    # ========================================================
    # SCENARIO LOSSES
    # ========================================================

    no_action_loss = _numeric_series(
        result,
        "no_action_loss_60m",
        0.0,
    )

    moderate_loss = _numeric_series(
        result,
        "moderate_loss_60m",
        0.0,
    )

    aggressive_loss = _numeric_series(
        result,
        "aggressive_loss_60m",
        0.0,
    )

    # ========================================================
    # PREVENTED LOSS
    # ========================================================

    moderate_prevented = _numeric_series(
        result,
        "moderate_prevented_60m",
        0.0,
    )

    aggressive_prevented = _numeric_series(
        result,
        "aggressive_prevented_60m",
        0.0,
    )

    # ========================================================
    # RESPONSE ACTION
    # ========================================================

    if "response_action" in result.columns:

        action = (
            result["response_action"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

    else:

        action = pd.Series(
            "NO_ACTION",
            index=result.index,
        )

    # ========================================================
    # INITIALIZE OUTPUT
    # ========================================================

    expected_loss = pd.Series(
        0.0,
        index=result.index,
        dtype=float,
    )

    estimated_loss_prevented = pd.Series(
        0.0,
        index=result.index,
        dtype=float,
    )

    # ========================================================
    # NO ACTION
    # ========================================================

    no_action_mask = (
        action == "NO_ACTION"
    )

    expected_loss.loc[
        no_action_mask
    ] = no_action_loss.loc[
        no_action_mask
    ]

    # ========================================================
    # MONITOR
    # ========================================================

    monitor_mask = (
        action == "MONITOR"
    )

    expected_loss.loc[
        monitor_mask
    ] = no_action_loss.loc[
        monitor_mask
    ]

    # ========================================================
    # RESTRICT
    # ========================================================

    restrict_mask = (
        action == "RESTRICT"
    )

    expected_loss.loc[
        restrict_mask
    ] = moderate_loss.loc[
        restrict_mask
    ]

    estimated_loss_prevented.loc[
        restrict_mask
    ] = moderate_prevented.loc[
        restrict_mask
    ]

    # ========================================================
    # CONTAIN
    # ========================================================

    contain_mask = (
        action == "CONTAIN"
    )

    expected_loss.loc[
        contain_mask
    ] = aggressive_loss.loc[
        contain_mask
    ]

    estimated_loss_prevented.loc[
        contain_mask
    ] = aggressive_prevented.loc[
        contain_mask
    ]

    # ========================================================
    # UNKNOWN ACTION
    # ========================================================

    known_action_mask = (
        no_action_mask
        |
        monitor_mask
        |
        restrict_mask
        |
        contain_mask
    )

    unknown_mask = ~known_action_mask

    expected_loss.loc[
        unknown_mask
    ] = no_action_loss.loc[
        unknown_mask
    ]

    estimated_loss_prevented.loc[
        unknown_mask
    ] = 0.0

    # ========================================================
    # WRITE OUTPUT COLUMNS
    # ========================================================

    result[
        "no_action_loss_60m"
    ] = no_action_loss

    result[
        "moderate_loss_60m"
    ] = moderate_loss

    result[
        "aggressive_loss_60m"
    ] = aggressive_loss

    result[
        "moderate_loss_prevented_60m"
    ] = moderate_prevented

    result[
        "aggressive_loss_prevented_60m"
    ] = aggressive_prevented

    result[
        "expected_loss_if_action"
    ] = (
        expected_loss
        .clip(
            lower=0.0
        )
    )

    result[
        "estimated_loss_prevented"
    ] = (
        estimated_loss_prevented
        .clip(
            lower=0.0
        )
    )

    # ========================================================
    # ACTION EFFECTIVENESS
    # ========================================================

    result[
        "action_effectiveness"
    ] = (
        _safe_divide(
            result[
                "estimated_loss_prevented"
            ],
            result[
                "no_action_loss_60m"
            ],
        )
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    return result
