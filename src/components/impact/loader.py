from pathlib import Path

import pandas as pd

from .schema import (
    EVENT_REQUIRED_COLUMNS,
    WINDOW_REQUIRED_COLUMNS,
    FUSION_REQUIRED_COLUMNS,
    validate_columns,
)


# =============================================================
# DATETIME NORMALIZATION
# =============================================================

def _normalize_datetime(
    df,
    columns,
):

    for column in columns:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
                utc=True,
            )

    return df


# =============================================================
# NUMERIC NORMALIZATION
# =============================================================

def _normalize_numeric(
    df,
    columns,
):

    for column in columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


# =============================================================
# LOAD PHASE 2 EVENTS
# =============================================================

def load_phase2_events(
    path: str,
):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            "Phase 2 event artifact not found: "
            f"{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=True,
    )

    validate_columns(
        df,
        EVENT_REQUIRED_COLUMNS,
        "Phase 2 events",
    )

    df = _normalize_datetime(
        df,
        [
            "start_time",
            "end_time",
        ],
    )

    df = _normalize_numeric(
        df,
        [
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
        ],
    )

    return df


# =============================================================
# LOAD PHASE 2 WINDOWS
# =============================================================

def load_phase2_windows(
    path: str,
):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            "Phase 2 window artifact not found: "
            f"{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=True,
    )

    validate_columns(
        df,
        WINDOW_REQUIRED_COLUMNS,
        "Phase 2 windows",
    )

    df = _normalize_datetime(
        df,
        [
            "window_start",
            "window_end",
        ],
    )

    df = _normalize_numeric(
        df,
        [
            "transaction_count",
            "total_amount",
            "expected_fraud_count",
            "expected_fraud_amount",
            "spike_score",
        ],
    )

    return df


# =============================================================
# LOAD FUSION OUTPUT IN CHUNKS
# =============================================================

def load_fusion_chunks(
    path: str,
    chunksize: int = 100_000,
):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            "Fusion artifact not found: "
            f"{path}"
        )

    for chunk in pd.read_csv(
        path,
        chunksize=chunksize,
        low_memory=True,
    ):

        validate_columns(
            chunk,
            FUSION_REQUIRED_COLUMNS,
            "Fusion output",
        )

        chunk = _normalize_datetime(
            chunk,
            ["timestamp"],
        )

        chunk = _normalize_numeric(
            chunk,
            [
                "amount",
                "fraud_probability",
                "unified_risk_score",
                "transaction_expected_loss",
            ],
        )

        yield chunk
