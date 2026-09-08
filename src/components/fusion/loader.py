from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import (
    validate_phase1,
    validate_phase2_events,
    validate_phase2_windows,
)


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"Expected file but found something else: {path}"
        )


def load_phase1(path: Path) -> pd.DataFrame:
    """
    Load Phase 1 transaction risk artifact.
    """

    _require_file(path)

    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Phase 1 contains invalid timestamp values."
        )

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce",
    )

    df["fraud_probability"] = pd.to_numeric(
        df["fraud_probability"],
        errors="coerce",
    )

    validate_phase1(df)

    return df


def load_phase2_windows(path: Path) -> pd.DataFrame:
    """
    Load Phase 2 temporal-window artifact.
    """

    _require_file(path)

    df = pd.read_csv(path)

    df["window_start"] = pd.to_datetime(
        df["window_start"],
        utc=True,
        errors="coerce",
    )

    df["window_end"] = pd.to_datetime(
        df["window_end"],
        utc=True,
        errors="coerce",
    )

    if df["window_start"].isna().any():
        raise ValueError(
            "Phase 2 windows contain invalid window_start values."
        )

    if df["window_end"].isna().any():
        raise ValueError(
            "Phase 2 windows contain invalid window_end values."
        )

    validate_phase2_windows(df)

    return df


def load_phase2_events(path: Path) -> pd.DataFrame:
    """
    Load Phase 2 event artifact.
    """

    _require_file(path)

    df = pd.read_csv(path)

    df["start_time"] = pd.to_datetime(
        df["start_time"],
        utc=True,
        errors="coerce",
    )

    df["end_time"] = pd.to_datetime(
        df["end_time"],
        utc=True,
        errors="coerce",
    )

    if df["start_time"].isna().any():
        raise ValueError(
            "Phase 2 events contain invalid start_time values."
        )

    if df["end_time"].isna().any():
        raise ValueError(
            "Phase 2 events contain invalid end_time values."
        )

    validate_phase2_events(df)

    return df

