from typing import List


# =============================================================
# PHASE 2 EVENT SCHEMA
# =============================================================

EVENT_REQUIRED_COLUMNS: List[str] = [

    "event_id",
    "merchant_id",

    "start_time",
    "end_time",

    "duration_minutes",
    "windows",

    "transaction_count",
    "total_amount",

    "expected_fraud_count",
    "expected_fraud_amount",

    "unique_cards",

    "max_coordination_score",
    "max_tas",
    "max_fas",
]


# =============================================================
# PHASE 2 WINDOW SCHEMA
# =============================================================

WINDOW_REQUIRED_COLUMNS: List[str] = [

    "merchant_id",
    "window_start",

    "transaction_count",
    "total_amount",

    "expected_fraud_count",
    "expected_fraud_amount",

    "spike_score",
    "spike_state",
]


# =============================================================
# FUSION SCHEMA
# =============================================================

FUSION_REQUIRED_COLUMNS: List[str] = [

    "transaction_id",
    "timestamp",

    "merchant_id",

    "amount",
    "fraud_probability",

    "unified_risk_score",

    "transaction_expected_loss",
]


# =============================================================
# VALIDATION
# =============================================================

def validate_columns(
    df,
    required_columns,
    name: str,
):

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{name} is missing required columns: "
            f"{missing}"
        )

    return True

