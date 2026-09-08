from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st

from .config import (
    FUSION,
    FUSION_CHUNK_SIZE,
    PHASE2_WINDOWS,
    PHASE2_EVENTS,
    IMPACT_FORECAST,
    IMPACT_SCENARIOS,
    RESPONSE_RECOMMENDATIONS,
    RESPONSE_ACTIONS,
    XAI_TRANSACTION,
    XAI_SPIKE,
    XAI_UNIFIED,
)


# ============================================================
# GENERIC HELPERS
# ============================================================

def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def numeric(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")

    return (
        pd.to_numeric(df[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
    )


def clean_timestamp(
    df: pd.DataFrame,
    column: str = "timestamp",
) -> pd.DataFrame:

    if column in df.columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    return df


# ============================================================
# SMALL ARTIFACT LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv(path_string: str) -> pd.DataFrame:

    path = Path(path_string)

    if not file_exists(path):
        return pd.DataFrame()

    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=False)
def load_phase2_windows() -> pd.DataFrame:
    return load_csv(str(PHASE2_WINDOWS))


@st.cache_data(show_spinner=False)
def load_phase2_events() -> pd.DataFrame:
    return load_csv(str(PHASE2_EVENTS))


@st.cache_data(show_spinner=False)
def load_impact_forecast() -> pd.DataFrame:
    return load_csv(str(IMPACT_FORECAST))


@st.cache_data(show_spinner=False)
def load_impact_scenarios() -> pd.DataFrame:
    return load_csv(str(IMPACT_SCENARIOS))


@st.cache_data(show_spinner=False)
def load_response_recommendations() -> pd.DataFrame:
    return load_csv(str(RESPONSE_RECOMMENDATIONS))


@st.cache_data(show_spinner=False)
def load_response_actions() -> pd.DataFrame:
    return load_csv(str(RESPONSE_ACTIONS))


# ============================================================
# XAI LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_xai_unified() -> pd.DataFrame:
    return load_csv(str(XAI_UNIFIED))


@st.cache_data(show_spinner=False)
def load_xai_spike() -> pd.DataFrame:
    return load_csv(str(XAI_SPIKE))


# ============================================================
# FUSION CHUNK ITERATOR
# ============================================================

def iter_fusion(
    usecols: Optional[Iterable[str]] = None,
):

    if not file_exists(FUSION):
        return

    for chunk in pd.read_csv(
        FUSION,
        usecols=usecols,
        chunksize=FUSION_CHUNK_SIZE,
        low_memory=True,
    ):
        yield chunk


# ============================================================
# FUSION SUMMARY
# ============================================================

@st.cache_data(show_spinner=False)
def build_fusion_summary() -> dict:

    summary = {
        "transactions": 0,
        "amount": 0.0,
        "expected_fraud_amount": 0.0,
        "alerts": 0,
        "high_risk": 0,
        "critical": 0,
        "fraud_probability_sum": 0.0,
        "unified_risk_sum": 0.0,
    }

    usecols = [
        "amount",
        "expected_fraud_amount",
        "alert_flag",
        "risk_band",
        "fraud_probability",
        "unified_risk_score",
    ]

    for chunk in iter_fusion(usecols):

        n = len(chunk)

        summary["transactions"] += n

        summary["amount"] += numeric(
            chunk,
            "amount",
        ).sum()

        summary["expected_fraud_amount"] += numeric(
            chunk,
            "expected_fraud_amount",
        ).sum()

        summary["fraud_probability_sum"] += numeric(
            chunk,
            "fraud_probability",
        ).sum()

        summary["unified_risk_sum"] += numeric(
            chunk,
            "unified_risk_score",
        ).sum()

        if "alert_flag" in chunk.columns:
            summary["alerts"] += (
                chunk["alert_flag"]
                .astype(bool)
                .sum()
            )

        if "risk_band" in chunk.columns:
            summary["high_risk"] += (
                chunk["risk_band"]
                .astype(str)
                .isin(["HIGH", "CRITICAL"])
                .sum()
            )

            summary["critical"] += (
                chunk["risk_band"]
                .astype(str)
                .eq("CRITICAL")
                .sum()
            )

    if summary["transactions"] > 0:

        summary["mean_fraud_probability"] = (
            summary["fraud_probability_sum"]
            / summary["transactions"]
        )

        summary["mean_unified_risk"] = (
            summary["unified_risk_sum"]
            / summary["transactions"]
        )

    else:
        summary["mean_fraud_probability"] = 0.0
        summary["mean_unified_risk"] = 0.0

    return summary


# ============================================================
# TOP TRANSACTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def load_top_transactions(
    limit: int = 100,
) -> pd.DataFrame:

    frames = []

    usecols = [
        "transaction_id",
        "timestamp",
        "merchant_id",
        "card_id",
        "amount",
        "fraud_probability",
        "risk_score",
        "risk_band",
        "phase1_score",
        "phase2_score",
        "spike_state",
        "unified_risk_score",
        "expected_fraud_amount",
        "event_id",
        "response_action",
        "response_priority",
        "alert_flag",
    ]

    for chunk in iter_fusion(usecols):

        if "unified_risk_score" not in chunk.columns:
            continue

        chunk["unified_risk_score"] = numeric(
            chunk,
            "unified_risk_score",
        )

        frames.append(
            chunk.nlargest(
                min(limit, len(chunk)),
                "unified_risk_score",
            )
        )

    if not frames:
        return pd.DataFrame()

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = result.nlargest(
        limit,
        "unified_risk_score",
    )

    return clean_timestamp(result)


# ============================================================
# FUSION TIME SERIES
# ============================================================

@st.cache_data(show_spinner=False)
def build_risk_timeseries(
    freq: str = "15min",
) -> pd.DataFrame:

    frames = []

    usecols = [
        "timestamp",
        "fraud_probability",
        "phase1_score",
        "phase2_score",
        "tas",
        "fas",
        "unified_risk_score",
        "expected_fraud_amount",
        "amount",
    ]

    for chunk in iter_fusion(usecols):

        chunk = clean_timestamp(chunk)

        chunk = chunk.dropna(
            subset=["timestamp"]
        )

        if chunk.empty:
            continue

        for col in [
            "fraud_probability",
            "phase1_score",
            "phase2_score",
            "tas",
            "fas",
            "unified_risk_score",
            "expected_fraud_amount",
            "amount",
        ]:
            if col in chunk.columns:
                chunk[col] = numeric(
                    chunk,
                    col,
                )

        grouped = (
            chunk
            .set_index("timestamp")
            .resample(freq)
            .agg(
                transaction_count=(
                    "amount",
                    "size",
                ),
                total_amount=(
                    "amount",
                    "sum",
                ),
                expected_fraud_amount=(
                    "expected_fraud_amount",
                    "sum",
                ),
                avg_fraud_probability=(
                    "fraud_probability",
                    "mean",
                ),
                avg_phase1_score=(
                    "phase1_score",
                    "mean",
                ),
                avg_phase2_score=(
                    "phase2_score",
                    "mean",
                ),
                avg_tas=(
                    "tas",
                    "mean",
                ),
                avg_fas=(
                    "fas",
                    "mean",
                ),
                avg_unified_risk=(
                    "unified_risk_score",
                    "mean",
                ),
                max_unified_risk=(
                    "unified_risk_score",
                    "max",
                ),
            )
            .reset_index()
        )

        frames.append(grouped)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    numeric_cols = [
        c
        for c in result.columns
        if c != "timestamp"
    ]

    result = (
        result
        .groupby("timestamp", as_index=False)[numeric_cols]
        .sum()
    )

    for col in [
        "avg_fraud_probability",
        "avg_phase1_score",
        "avg_phase2_score",
        "avg_tas",
        "avg_fas",
        "avg_unified_risk",
        "max_unified_risk",
    ]:
        if col in result.columns:
            result[col] = (
                result[col]
                / result["transaction_count"].replace(
                    0,
                    np.nan,
                )
            )

    return result.sort_values("timestamp")


# ============================================================
# MERCHANT RISK RANKING
# ============================================================

@st.cache_data(show_spinner=False)
def build_merchant_ranking(
    limit: int = 20,
) -> pd.DataFrame:

    frames = []

    usecols = [
        "merchant_id",
        "amount",
        "fraud_probability",
        "expected_fraud_amount",
        "unified_risk_score",
        "event_id",
    ]

    for chunk in iter_fusion(usecols):

        if "merchant_id" not in chunk.columns:
            continue

        chunk["amount"] = numeric(
            chunk,
            "amount",
        )

        chunk["fraud_probability"] = numeric(
            chunk,
            "fraud_probability",
        )

        chunk["expected_fraud_amount"] = numeric(
            chunk,
            "expected_fraud_amount",
        )

        chunk["unified_risk_score"] = numeric(
            chunk,
            "unified_risk_score",
        )

        grouped = (
            chunk
            .groupby("merchant_id")
            .agg(
                transactions=(
                    "merchant_id",
                    "size",
                ),
                total_amount=(
                    "amount",
                    "sum",
                ),
                expected_fraud_amount=(
                    "expected_fraud_amount",
                    "sum",
                ),
                avg_fraud_probability=(
                    "fraud_probability",
                    "mean",
                ),
                avg_unified_risk=(
                    "unified_risk_score",
                    "mean",
                ),
                max_unified_risk=(
                    "unified_risk_score",
                    "max",
                ),
                events=(
                    "event_id",
                    "nunique",
                ),
            )
            .reset_index()
        )

        frames.append(grouped)

    if not frames:
        return pd.DataFrame()

    result = (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .groupby("merchant_id", as_index=False)
        .agg(
            transactions=("transactions", "sum"),
            total_amount=("total_amount", "sum"),
            expected_fraud_amount=(
                "expected_fraud_amount",
                "sum",
            ),
            avg_fraud_probability=(
                "avg_fraud_probability",
                "mean",
            ),
            avg_unified_risk=(
                "avg_unified_risk",
                "mean",
            ),
            max_unified_risk=(
                "max_unified_risk",
                "max",
            ),
            events=("events", "sum"),
        )
    )

    return result.nlargest(
        limit,
        "avg_unified_risk",
    )


# ============================================================
# SPIKE DISTRIBUTION
# ============================================================

@st.cache_data(show_spinner=False)
def build_spike_distribution() -> pd.DataFrame:

    events = load_phase2_events()

    if events.empty:
        return pd.DataFrame()

    if "max_fas" in events.columns:
        events["max_fas"] = numeric(
            events,
            "max_fas",
        )

    if "duration_minutes" in events.columns:
        events["duration_minutes"] = numeric(
            events,
            "duration_minutes",
        )

    return events


# ============================================================
# ARTIFACT STATUS
# ============================================================

def artifact_status() -> dict:

    paths = {
        "Phase 2 Windows": PHASE2_WINDOWS,
        "Phase 2 Events": PHASE2_EVENTS,
        "Fusion": FUSION,
        "XAI": XAI_UNIFIED,
        "Impact Forecast": IMPACT_FORECAST,
        "Impact Scenarios": IMPACT_SCENARIOS,
        "Response Recommendations": RESPONSE_RECOMMENDATIONS,
        "Response Actions": RESPONSE_ACTIONS,
    }

    return {
        name: file_exists(path)
        for name, path in paths.items()
    }