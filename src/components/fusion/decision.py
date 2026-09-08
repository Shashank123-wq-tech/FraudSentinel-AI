from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FusionConfig


def assign_risk_band(
    score: pd.Series,
    config: FusionConfig,
) -> pd.Series:
    """
    Assign unified risk band.
    """

    conditions = [
        score < config.low_threshold,
        score < config.medium_threshold,
        score < config.high_threshold,
        score < config.very_high_threshold,
        score >= config.very_high_threshold,
    ]

    choices = [
        "LOW",
        "MEDIUM",
        "HIGH",
        "VERY_HIGH",
        "CRITICAL",
    ]

    return pd.Series(
        np.select(
            conditions,
            choices,
            default="LOW",
        ),
        index=score.index,
    )


def assign_response_action(
    score: pd.Series,
    phase2_state: pd.Series,
    config: FusionConfig,
) -> pd.Series:
    """
    Map unified risk to defensive response.

    Phase 2 state is allowed to strengthen the response because
    coordinated temporal activity is an important contextual signal.
    """

    score = pd.to_numeric(
        score,
        errors="coerce",
    ).fillna(0.0)

    state = (
        phase2_state
        .astype("string")
        .str.strip()
        .str.upper()
    )

    action = pd.Series(
        "ALLOW_MONITOR",
        index=score.index,
        dtype="string",
    )

    action.loc[
        score >= config.monitor_threshold
    ] = "MONITOR"

    action.loc[
        score >= config.investigate_threshold
    ] = "INVESTIGATE"

    action.loc[
        score >= config.enhanced_control_threshold
    ] = "ENHANCED_CONTROLS"

    action.loc[
        score >= config.immediate_response_threshold
    ] = "IMMEDIATE_RESPONSE"

    # Phase 2 state escalation.
    action.loc[
        state == "VERIFIED_FRAUD_SPIKE"
    ] = action.loc[
        state == "VERIFIED_FRAUD_SPIKE"
    ].where(
        score >= config.investigate_threshold,
        "INVESTIGATE",
    )

    action.loc[
        state == "CRITICAL_ACTIVE_SPIKE"
    ] = "IMMEDIATE_RESPONSE"

    return action


def assign_priority(
    score: pd.Series,
    phase2_state: pd.Series,
    config: FusionConfig,
) -> pd.Series:
    """
    Assign operational response priority.

    P0 = immediate response
    P1 = enhanced controls
    P2 = investigation
    P3 = monitoring
    P4 = allow/monitor
    """

    score = pd.to_numeric(
        score,
        errors="coerce",
    ).fillna(0.0)

    state = (
        phase2_state
        .astype("string")
        .str.strip()
        .str.upper()
    )

    priority = pd.Series(
        "P4",
        index=score.index,
        dtype="string",
    )

    priority.loc[
        score >= config.monitor_threshold
    ] = "P3"

    priority.loc[
        score >= config.investigate_threshold
    ] = "P2"

    priority.loc[
        score >= config.enhanced_control_threshold
    ] = "P1"

    priority.loc[
        score >= config.immediate_response_threshold
    ] = "P0"

    priority.loc[
        state == "CRITICAL_ACTIVE_SPIKE"
    ] = "P0"

    priority.loc[
        (
            state == "VERIFIED_FRAUD_SPIKE"
        ) & (
            priority == "P4"
        )
    ] = "P2"

    return priority


def assign_alert_flag(
    score: pd.Series,
    phase2_state: pd.Series,
    config: FusionConfig,
) -> pd.Series:
    """
    Whether the transaction should generate an operational alert.
    """

    state = (
        phase2_state
        .astype("string")
        .str.strip()
        .str.upper()
    )

    return (
        (score >= config.investigate_threshold)
        | state.isin(
            [
                "VERIFIED_FRAUD_SPIKE",
                "CRITICAL_ACTIVE_SPIKE",
            ]
        )
    )

