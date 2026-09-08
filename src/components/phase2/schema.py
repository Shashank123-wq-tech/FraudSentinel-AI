from typing import List

import pandas as pd


REQUIRED_PHASE1_COLUMNS: List[str] = [
    "transaction_id",
    "timestamp",
    "merchant_id",
    "card_id",
    "amount",
    "fraud_probability",
    "risk_score",
    "risk_band",
    "predicted_fraud",
    "model_version",
    "threshold",
]


NUMERIC_COLUMNS: List[str] = [
    "amount",
    "fraud_probability",
    "risk_score",
    "threshold",
]


def validate_phase1_output(df: pd.DataFrame) -> None:
    """
    Validate the Phase-1 -> Phase-2 contract.

    Raises:
        ValueError: if the contract is violated.
    """

    missing = [
        col
        for col in REQUIRED_PHASE1_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing Phase-1 output columns: {missing}"
        )

    # ------------------------------------------------------------
    # Transaction ID
    # ------------------------------------------------------------

    if df["transaction_id"].isna().any():
        raise ValueError(
            "transaction_id contains null values."
        )

    if df["transaction_id"].duplicated().any():
        duplicates = int(
            df["transaction_id"].duplicated().sum()
        )

        raise ValueError(
            f"Found {duplicates} duplicate transaction_id values."
        )

    # ------------------------------------------------------------
    # Merchant / card
    # ------------------------------------------------------------

    if df["merchant_id"].isna().any():
        raise ValueError(
            "merchant_id contains null values."
        )

    if df["card_id"].isna().any():
        raise ValueError(
            "card_id contains null values."
        )

    # ------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------

    if df["timestamp"].isna().any():
        raise ValueError(
            "timestamp contains invalid/null values."
        )

    # ------------------------------------------------------------
    # Amount
    # ------------------------------------------------------------

    if df["amount"].isna().any():
        raise ValueError(
            "amount contains null values."
        )

    if (df["amount"] < 0).any():
        raise ValueError(
            "amount cannot be negative."
        )

    # ------------------------------------------------------------
    # Fraud probability
    # ------------------------------------------------------------

    if df["fraud_probability"].isna().any():
        raise ValueError(
            "fraud_probability contains null values."
        )

    if (
        (df["fraud_probability"] < 0)
        | (df["fraud_probability"] > 1)
    ).any():

        raise ValueError(
            "fraud_probability must be within [0, 1]."
        )

    # ------------------------------------------------------------
    # Risk score
    # ------------------------------------------------------------

    if df["risk_score"].isna().any():
        raise ValueError(
            "risk_score contains null values."
        )

    if (
        (df["risk_score"] < 0)
        | (df["risk_score"] > 100)
    ).any():

        raise ValueError(
            "risk_score must be within [0, 100]."
        )

    # ------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------

    if df["threshold"].isna().any():
        raise ValueError(
            "threshold contains null values."
        )

    if (
        (df["threshold"] < 0)
        | (df["threshold"] > 1)
    ).any():

        raise ValueError(
            "threshold must be within [0, 1]."
        )


def prepare_phase1_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the Phase-1 output before Phase-2 processing.
    """

    x = df.copy()

    # UTC is mandatory for consistent temporal aggregation.
    x["timestamp"] = pd.to_datetime(
        x["timestamp"],
        utc=True,
        errors="coerce",
    )

    for column in NUMERIC_COLUMNS:
        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        )

    x["predicted_fraud"] = (
        pd.to_numeric(
            x["predicted_fraud"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    validate_phase1_output(x)

    return (
        x
        .sort_values(
            ["timestamp", "transaction_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )