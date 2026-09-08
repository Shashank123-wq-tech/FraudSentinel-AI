from __future__ import annotations

from typing import Iterable

import pandas as pd


PHASE1_REQUIRED_COLUMNS = {
    "transaction_id",
    "timestamp",
    "merchant_id",
    "amount",
    "fraud_probability",
}

PHASE2_WINDOW_REQUIRED_COLUMNS = {
    "merchant_id",
    "window_start",
    "window_end",
    "spike_score",
    "spike_state",
}

PHASE2_EVENT_REQUIRED_COLUMNS = {
    "event_id",
    "merchant_id",
    "start_time",
    "end_time",
    "peak_spike_score",
    "max_state",
}


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    """
    Validate that a dataframe contains all required columns.
    """

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing)}"
        )


def validate_phase1(df: pd.DataFrame) -> None:
    """
    Validate Phase 1 transaction-risk artifact.
    """

    validate_required_columns(
        df,
        PHASE1_REQUIRED_COLUMNS,
        "Phase 1 transaction_risk.csv",
    )

    if df["transaction_id"].duplicated().any():
        duplicated = int(
            df["transaction_id"].duplicated().sum()
        )

        raise ValueError(
            f"Phase 1 contains {duplicated} duplicated "
            "transaction_id values."
        )

    if (df["amount"] < 0).any():
        raise ValueError(
            "Phase 1 contains negative transaction amounts."
        )

    probability = pd.to_numeric(
        df["fraud_probability"],
        errors="coerce",
    )

    if probability.isna().any():
        raise ValueError(
            "Phase 1 fraud_probability contains invalid values."
        )

    if ((probability < 0) | (probability > 1)).any():
        raise ValueError(
            "Phase 1 fraud_probability must be in [0, 1]."
        )


def validate_phase2_windows(df: pd.DataFrame) -> None:
    """
    Validate Phase 2 merchant-window artifact.
    """

    validate_required_columns(
        df,
        PHASE2_WINDOW_REQUIRED_COLUMNS,
        "Phase 2 phase2_windows.csv",
    )

    if df["merchant_id"].isna().any():
        raise ValueError(
            "Phase 2 windows contain missing merchant_id."
        )

    if df["window_start"].isna().any():
        raise ValueError(
            "Phase 2 windows contain missing window_start."
        )

    if df["window_end"].isna().any():
        raise ValueError(
            "Phase 2 windows contain missing window_end."
        )

    score = pd.to_numeric(
        df["spike_score"],
        errors="coerce",
    )

    if score.isna().any():
        raise ValueError(
            "Phase 2 spike_score contains invalid values."
        )

    if ((score < 0) | (score > 100)).any():
        raise ValueError(
            "Phase 2 spike_score must be in [0, 100]."
        )

    if (df["window_end"] < df["window_start"]).any():
        raise ValueError(
            "Phase 2 contains window_end earlier than window_start."
        )


def validate_phase2_events(df: pd.DataFrame) -> None:
    """
    Validate Phase 2 event artifact.
    """

    validate_required_columns(
        df,
        PHASE2_EVENT_REQUIRED_COLUMNS,
        "Phase 2 phase2_events.csv",
    )

    if df["event_id"].duplicated().any():
        duplicated = int(
            df["event_id"].duplicated().sum()
        )

        raise ValueError(
            f"Phase 2 contains {duplicated} duplicated event_id values."
        )

    if (df["end_time"] < df["start_time"]).any():
        raise ValueError(
            "Phase 2 events contain end_time earlier than start_time."
        )

