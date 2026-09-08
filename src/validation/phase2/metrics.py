from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from scipy.stats import spearmanr


def _safe_auc(y_true, score):
    """
    Return ROC-AUC when both classes are present.
    """

    y_true = np.asarray(y_true)
    score = np.asarray(score)

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y_true,
            score,
        )
    )


def _safe_ap(y_true, score):
    """
    Return Average Precision when both classes are present.
    """

    y_true = np.asarray(y_true)
    score = np.asarray(score)

    if y_true.sum() == 0:
        return np.nan

    return float(
        average_precision_score(
            y_true,
            score,
        )
    )


def build_score_deciles(
    df: pd.DataFrame,
    score_col: str = "spike_score",
    label_col: str = "fraud_positive",
    fraud_amount_col: str = "actual_fraud_amount",
    bins: int = 10,
) -> pd.DataFrame:
    """
    Rank windows into score deciles and measure actual fraud
    concentration in each decile.
    """

    x = df[
        [
            score_col,
            label_col,
            fraud_amount_col,
        ]
    ].copy()

    x = x.dropna(
        subset=[score_col]
    )

    if x.empty:
        return pd.DataFrame()

    x["score_decile"] = pd.qcut(
        x[score_col],
        q=bins,
        labels=False,
        duplicates="drop",
    )

    result = (
        x.groupby(
            "score_decile",
            observed=True,
        )
        .agg(
            windows=(score_col, "size"),
            mean_score=(score_col, "mean"),
            median_score=(score_col, "median"),
            fraud_positive_windows=(
                label_col,
                "sum",
            ),
            fraud_window_rate=(
                label_col,
                "mean",
            ),
            fraud_amount=(
                fraud_amount_col,
                "sum",
            ),
        )
        .reset_index()
    )

    result["fraud_amount_per_window"] = (
        result["fraud_amount"]
        / result["windows"]
    )

    result["score_decile"] = (
        result["score_decile"] + 1
    )

    return result


def evaluate_top_k(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    fraud_amount_col: str,
) -> pd.DataFrame:
    """
    Evaluate top 1%, 5%, 10%, and 20% score-ranked windows.
    """

    x = df[
        [
            score_col,
            label_col,
            fraud_amount_col,
        ]
    ].dropna(
        subset=[score_col]
    )

    x = x.sort_values(
        score_col,
        ascending=False,
    )

    total_fraud_windows = (
        x[label_col].sum()
    )

    total_fraud_amount = (
        x[fraud_amount_col].sum()
    )

    rows = []

    for pct in [0.01, 0.05, 0.10, 0.20]:

        n = max(
            1,
            int(len(x) * pct),
        )

        top = x.head(n)

        detected_fraud_windows = (
            top[label_col].sum()
        )

        captured_fraud_amount = (
            top[fraud_amount_col].sum()
        )

        rows.append(
            {
                "top_fraction": pct,
                "windows_selected": n,
                "fraud_windows_captured": int(
                    detected_fraud_windows
                ),
                "fraud_window_recall": (
                    detected_fraud_windows
                    / total_fraud_windows
                    if total_fraud_windows > 0
                    else np.nan
                ),
                "fraud_amount_captured": (
                    captured_fraud_amount
                ),
                "fraud_amount_capture_rate": (
                    captured_fraud_amount
                    / total_fraud_amount
                    if total_fraud_amount > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def evaluate_state_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate actual fraud concentration by Phase 2 state.
    """

    required = [
        "spike_state",
        "actual_fraud_count",
        "actual_fraud_amount",
        "fraud_positive",
        "coordinated_fraud_positive",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    result = (
        df.groupby(
            "spike_state",
            observed=True,
        )
        .agg(
            windows=(
                "spike_state",
                "size",
            ),
            fraud_positive_windows=(
                "fraud_positive",
                "sum",
            ),
            coordinated_fraud_windows=(
                "coordinated_fraud_positive",
                "sum",
            ),
            actual_fraud_count=(
                "actual_fraud_count",
                "sum",
            ),
            actual_fraud_amount=(
                "actual_fraud_amount",
                "sum",
            ),
        )
        .reset_index()
    )

    result["fraud_window_rate"] = (
        result["fraud_positive_windows"]
        / result["windows"]
    )

    result["coordinated_window_rate"] = (
        result["coordinated_fraud_windows"]
        / result["windows"]
    )

    result["fraud_amount_per_window"] = (
        result["actual_fraud_amount"]
        / result["windows"]
    )

    return result


def evaluate_phase2(
    df: pd.DataFrame,
) -> dict:
    """
    Complete offline Phase 2 validation.

    Main question:

        Does a higher spike_score correspond
        to more actual fraud?
    """

    required = [
        "spike_score",
        "tas",
        "fas",
        "spike_state",
        "fraud_positive",
        "coordinated_fraud_positive",
        "actual_fraud_count",
        "actual_fraud_amount",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Validation dataframe is missing: {missing}"
        )

    x = df.dropna(
        subset=["spike_score"]
    ).copy()

    result = {}

    # ----------------------------------------------------------
    # BASIC DATASET METRICS
    # ----------------------------------------------------------

    result["windows"] = int(len(x))

    result["fraud_positive_windows"] = int(
        x["fraud_positive"].sum()
    )

    result["coordinated_fraud_windows"] = int(
        x["coordinated_fraud_positive"].sum()
    )

    result["actual_fraud_transactions"] = int(
        x["actual_fraud_count"].sum()
    )

    result["actual_fraud_amount"] = float(
        x["actual_fraud_amount"].sum()
    )

    # ----------------------------------------------------------
    # SCORE DISCRIMINATION
    # ----------------------------------------------------------

    result["spike_score_roc_auc"] = _safe_auc(
        x["fraud_positive"],
        x["spike_score"],
    )

    result["spike_score_pr_auc"] = _safe_ap(
        x["fraud_positive"],
        x["spike_score"],
    )

    result[
        "spike_score_coordinated_roc_auc"
    ] = _safe_auc(
        x["coordinated_fraud_positive"],
        x["spike_score"],
    )

    result[
        "spike_score_coordinated_pr_auc"
    ] = _safe_ap(
        x["coordinated_fraud_positive"],
        x["spike_score"],
    )

    # ----------------------------------------------------------
    # SPEARMAN MONOTONICITY
    # ----------------------------------------------------------

    if (
        x["actual_fraud_amount"].nunique()
        > 1
    ):
        corr = spearmanr(
            x["spike_score"],
            x["actual_fraud_amount"],
            nan_policy="omit",
        )

        result[
            "spike_score_fraud_amount_spearman"
        ] = float(corr.statistic)

    else:
        result[
            "spike_score_fraud_amount_spearman"
        ] = np.nan

    # ----------------------------------------------------------
    # SCORE DISTRIBUTION
    # ----------------------------------------------------------

    result["score_mean"] = float(
        x["spike_score"].mean()
    )

    result["score_median"] = float(
        x["spike_score"].median()
    )

    result["score_p90"] = float(
        x["spike_score"].quantile(0.90)
    )

    result["score_p95"] = float(
        x["spike_score"].quantile(0.95)
    )

    result["score_p99"] = float(
        x["spike_score"].quantile(0.99)
    )

    # ----------------------------------------------------------
    # TOP-K
    # ----------------------------------------------------------

    top_k = evaluate_top_k(
        x,
        score_col="spike_score",
        label_col="fraud_positive",
        fraud_amount_col="actual_fraud_amount",
    )

    result["top_k"] = (
        top_k.to_dict(orient="records")
    )

    # ----------------------------------------------------------
    # STATE METRICS
    # ----------------------------------------------------------

    state_metrics = evaluate_state_metrics(
        x
    )

    result["state_metrics"] = (
        state_metrics.to_dict(
            orient="records"
        )
    )

    return result