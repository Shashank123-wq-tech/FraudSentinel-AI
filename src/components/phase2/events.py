# src/components/phase2/events.py

from __future__ import annotations

import numpy as np
import pandas as pd


EVENT_ELIGIBLE_STATES = {
    "VERIFIED_FRAUD_SPIKE",
    "CRITICAL_ACTIVE_SPIKE",
}


def _safe_sum(group: pd.DataFrame, column: str) -> float:

    if column not in group.columns:
        return 0.0

    return float(
        pd.to_numeric(
            group[column],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )


def _safe_max(group: pd.DataFrame, column: str) -> float:

    if column not in group.columns:
        return 0.0

    values = pd.to_numeric(
        group[column],
        errors="coerce",
    )

    if values.notna().sum() == 0:
        return 0.0

    return float(values.max())


def _safe_unique(group: pd.DataFrame, column: str) -> int:

    if column not in group.columns:
        return 0

    return int(
        group[column]
        .dropna()
        .nunique()
    )


def build_fraud_spike_events(
    df: pd.DataFrame,
    max_gap_minutes: float = 15.0,
) -> pd.DataFrame:

    required_columns = [
        "merchant_id",
        "window_start",
        "window_end",
        "spike_score",
        "spike_state",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns for event construction: {missing}"
        )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "merchant_id",
                "start_time",
                "end_time",
                "duration_minutes",
                "peak_spike_score",
                "max_state",
                "windows",
            ]
        )

    x = df.copy()

    x["window_start"] = pd.to_datetime(
        x["window_start"],
        utc=True,
        errors="coerce",
    )

    x["window_end"] = pd.to_datetime(
        x["window_end"],
        utc=True,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Only operationally meaningful states become
    # fraud-spike events.
    #
    # Candidate windows remain monitoring signals.
    # --------------------------------------------------------

    x = x[
        x["spike_state"].isin(
            EVENT_ELIGIBLE_STATES
        )
    ].copy()

    if x.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "merchant_id",
                "start_time",
                "end_time",
                "duration_minutes",
                "peak_spike_score",
                "max_state",
                "windows",
            ]
        )

    x = x.sort_values(
        ["merchant_id", "window_start"],
        kind="mergesort",
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Build contiguous temporal groups.
    #
    # A new event starts when:
    #
    # 1. merchant changes
    # 2. gap exceeds max_gap_minutes
    #
    # This is strictly past/current temporal logic.
    # --------------------------------------------------------

    previous_merchant = x["merchant_id"].shift(1)

    previous_start = (
        x["window_start"]
        .shift(1)
    )

    gap_minutes = (
        x["window_start"]
        - previous_start
    ).dt.total_seconds().div(60.0)

    new_event = (
        x["merchant_id"]
        != previous_merchant
    ) | (
        gap_minutes > max_gap_minutes
    ) | (
        gap_minutes.isna()
    )

    x["_event_group"] = new_event.cumsum()

    # --------------------------------------------------------
    # Aggregate events
    # --------------------------------------------------------

    records = []

    for event_group, group in x.groupby(
        "_event_group",
        sort=True,
    ):

        group = group.sort_values(
            "window_start"
        )

        merchant_id = group["merchant_id"].iloc[0]

        start_time = group["window_start"].min()

        end_time = group["window_end"].max()

        duration_minutes = (
            end_time - start_time
        ).total_seconds() / 60.0

        peak_spike_score = _safe_max(
            group,
            "spike_score",
        )

        # Severity ordering
        if (
            "CRITICAL_ACTIVE_SPIKE"
            in set(group["spike_state"])
        ):
            max_state = "CRITICAL_ACTIVE_SPIKE"
        else:
            max_state = "VERIFIED_FRAUD_SPIKE"

        record = {
            "event_id": f"FSE-{int(event_group):08d}",
            "merchant_id": merchant_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration_minutes": duration_minutes,
            "peak_spike_score": peak_spike_score,
            "max_state": max_state,
            "windows": int(len(group)),
        }

        # ----------------------------------------------------
        # Preserve useful aggregate evidence.
        # ----------------------------------------------------

        record["transaction_count"] = int(
            _safe_sum(
                group,
                "transaction_count",
            )
        )

        record["total_amount"] = _safe_sum(
            group,
            "total_amount",
        )

        record["expected_fraud_count"] = _safe_sum(
            group,
            "expected_fraud_count",
        )

        record["expected_fraud_amount"] = _safe_sum(
            group,
            "expected_fraud_amount",
        )

        record["unique_cards"] = _safe_unique(
            group,
            "card_id",
        )

        record["max_coordination_score"] = _safe_max(
            group,
            "coordination_score",
        )

        record["max_tas"] = _safe_max(
            group,
            "tas",
        )

        record["max_fas"] = _safe_max(
            group,
            "fas",
        )

        records.append(record)

    events = pd.DataFrame(records)

    if not events.empty:
        events = events.sort_values(
            ["start_time", "merchant_id"],
            kind="mergesort",
        ).reset_index(drop=True)

    return events