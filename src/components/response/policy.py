"""
FraudSentinel AI
Response Recommendation Policy Engine

This module converts response intelligence into:

    1. Response Priority
    2. Response Action
    3. Recommended Control

Policy hierarchy:

    CRITICAL_ACTIVE_SPIKE
            ↓
        CONTAIN

    VERIFIED_FRAUD_SPIKE
            ↓
        RESTRICT

    EARLY_WARNING
            ↓
         MONITOR

    NORMAL
            ↓
       NO_ACTION

The policy also considers:

    - response_score
    - unified_risk_score
    - projected 30-minute loss
    - projected 60-minute loss
    - recommendation confidence
"""


import pandas as pd


from .config import (
    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------
    ACTION_CONTAIN,
    ACTION_RESTRICT,
    ACTION_MONITOR,
    ACTION_NO_ACTION,

    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------
    CONTROL_CRITICAL,
    CONTROL_MODERATE,
    CONTROL_MONITOR,
    CONTROL_NONE,

    # --------------------------------------------------------
    # Priorities
    # --------------------------------------------------------
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    PRIORITY_LOW,

    # --------------------------------------------------------
    # Spike states
    # --------------------------------------------------------
    SPIKE_CRITICAL,
    SPIKE_VERIFIED,
    SPIKE_EARLY,
)


# ============================================================
# SAFE NUMERIC HELPER
# ============================================================

def _safe_float(
    value,
    default: float = 0.0,
) -> float:
    """
    Safely convert a value to float.

    Handles:

        - None
        - NaN
        - strings
        - invalid numeric values

    Parameters
    ----------
    value:
        Input value.

    default:
        Value returned when conversion fails.

    Returns
    -------
    float
    """

    try:

        value = float(value)

        if pd.isna(value):

            return default

        return value

    except (
        TypeError,
        ValueError,
    ):

        return default


# ============================================================
# RESPONSE PRIORITY
# ============================================================

def determine_priority(
    row: pd.Series,
) -> str:
    """
    Determine response priority.

    Priority hierarchy:

        CRITICAL
        HIGH
        MEDIUM
        LOW

    Critical spike always receives critical priority.

    Verified spikes become HIGH when response
    score or unified risk is sufficiently high.

    Early warnings or moderately elevated risk
    receive MEDIUM priority.

    Everything else receives LOW.
    """

    state = str(
        row.get(
            "spike_state",
            "",
        )
    )

    response_score = _safe_float(
        row.get(
            "response_score",
            0.0,
        )
    )

    unified_risk = _safe_float(
        row.get(
            "unified_risk_score",
            0.0,
        )
    )

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    if state == SPIKE_CRITICAL:

        return PRIORITY_CRITICAL

    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    if (
        state == SPIKE_VERIFIED
        and (
            response_score >= 60.0
            or unified_risk >= 70.0
        )
    ):

        return PRIORITY_HIGH

    # --------------------------------------------------------
    # MEDIUM
    # --------------------------------------------------------

    if (
        state == SPIKE_EARLY
        or response_score >= 45.0
        or unified_risk >= 45.0
    ):

        return PRIORITY_MEDIUM

    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    return PRIORITY_LOW


# ============================================================
# RESPONSE ACTION
# ============================================================

def determine_action(
    row: pd.Series,
) -> str:
    """
    Determine recommended response action.

    Decision hierarchy:

        CRITICAL
            → CONTAIN

        VERIFIED + financially significant
            → RESTRICT

        EARLY WARNING or financially significant
            → MONITOR

        LOW confidence
            → MONITOR

        Otherwise
            → NO_ACTION
    """

    state = str(
        row.get(
            "spike_state",
            "",
        )
    )

    priority = str(
        row.get(
            "response_priority",
            PRIORITY_LOW,
        )
    )

    projected_30 = _safe_float(
        row.get(
            "projected_loss_30m",
            0.0,
        )
    )

    projected_60 = _safe_float(
        row.get(
            "projected_loss_60m",
            0.0,
        )
    )

    recommendation_confidence = _safe_float(
        row.get(
            "recommendation_confidence",
            0.0,
        )
    )

    # --------------------------------------------------------
    # CRITICAL SPIKE
    # --------------------------------------------------------
    # Immediate containment takes precedence over
    # financial thresholds and confidence.

    if state == SPIKE_CRITICAL:

        return ACTION_CONTAIN

    # --------------------------------------------------------
    # VERIFIED FRAUD SPIKE
    # --------------------------------------------------------
    # Financially significant verified events receive
    # restrictive intervention.

    if (
        state == SPIKE_VERIFIED
        and (
            projected_30 >= 1000.0
            or projected_60 >= 2000.0
            or priority == PRIORITY_HIGH
        )
    ):

        return ACTION_RESTRICT

    # --------------------------------------------------------
    # EARLY WARNING / FINANCIAL RISK
    # --------------------------------------------------------
    # Monitor emerging events or events with meaningful
    # projected losses.

    if (
        state == SPIKE_EARLY
        or projected_30 >= 1000.0
        or projected_60 >= 2000.0
    ):

        return ACTION_MONITOR

    # --------------------------------------------------------
    # LOW CONFIDENCE
    # --------------------------------------------------------
    # Low-confidence recommendations should not trigger
    # aggressive intervention.

    if recommendation_confidence < 0.20:

        return ACTION_MONITOR

    # --------------------------------------------------------
    # NO ACTION
    # --------------------------------------------------------

    return ACTION_NO_ACTION


# ============================================================
# RECOMMENDED CONTROL
# ============================================================

def determine_control(
    action: str,
) -> str:
    """
    Map response action to recommended defensive control.

    CONTAIN
        → CRITICAL control

    RESTRICT
        → MODERATE control

    MONITOR
        → MONITOR control

    NO_ACTION
        → NONE
    """

    if action == ACTION_CONTAIN:

        return CONTROL_CRITICAL

    if action == ACTION_RESTRICT:

        return CONTROL_MODERATE

    if action == ACTION_MONITOR:

        return CONTROL_MONITOR

    return CONTROL_NONE


# ============================================================
# POLICY ENGINE
# ============================================================

def apply_policy(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the complete response policy.

    Input
    -----
    DataFrame containing at least:

        spike_state
        response_score
        unified_risk_score
        projected_loss_30m
        projected_loss_60m
        recommendation_confidence

    Output
    ------
    DataFrame containing:

        response_priority
        response_action
        recommended_control
    """

    if df is None:

        raise ValueError(
            "Response policy received None instead of DataFrame."
        )

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "Response policy expects a pandas DataFrame."
        )

    result = df.copy()

    # ========================================================
    # REQUIRED BASE COLUMNS
    # ========================================================

    required_columns = [
        "spike_state",
        "response_score",
        "unified_risk_score",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:

        raise ValueError(
            "Response policy is missing required columns: "
            f"{missing_columns}"
        )

    # ========================================================
    # OPTIONAL POLICY COLUMNS
    # ========================================================
    #
    # These are intentionally created with safe defaults.
    #
    # This prevents policy execution from crashing when an
    # earlier artifact does not contain a particular optional
    # forecast/confidence field.
    #
    # ========================================================

    optional_defaults = {

        "projected_loss_30m": 0.0,

        "projected_loss_60m": 0.0,

        "recommendation_confidence": 0.0,
    }

    for column, default in optional_defaults.items():

        if column not in result.columns:

            result[column] = default

    # ========================================================
    # NORMALIZE NUMERIC POLICY INPUTS
    # ========================================================

    numeric_columns = [
        "response_score",
        "unified_risk_score",
        "projected_loss_30m",
        "projected_loss_60m",
        "recommendation_confidence",
    ]

    for column in numeric_columns:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).fillna(0.0)

    # ========================================================
    # RESPONSE PRIORITY
    # ========================================================

    result["response_priority"] = result.apply(
        determine_priority,
        axis=1,
    )

    # ========================================================
    # RESPONSE ACTION
    # ========================================================

    result["response_action"] = result.apply(
        determine_action,
        axis=1,
    )

    # ========================================================
    # RECOMMENDED CONTROL
    # ========================================================

    result["recommended_control"] = (
        result["response_action"]
        .apply(
            determine_control
        )
    )

    # ========================================================
    # POLICY OUTPUT VALIDATION
    # ========================================================

    output_columns = [
        "response_priority",
        "response_action",
        "recommended_control",
    ]

    missing_outputs = [
        column
        for column in output_columns
        if column not in result.columns
    ]

    if missing_outputs:

        raise RuntimeError(
            "Response policy failed to generate required "
            f"output columns: {missing_outputs}"
        )

    return result

