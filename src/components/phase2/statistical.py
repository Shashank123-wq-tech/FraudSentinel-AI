from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_numeric(
    series: pd.Series,
    default: float = 0.0,
) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
    )


def _safe_unit_interval(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )


def _anomaly_from_baseline(
    current: pd.Series,
    baseline: pd.Series,
    scale: pd.Series,
) -> pd.Series:
    """
    One-sided standardized excess.

    Only increases above the historical baseline are treated as evidence
    of a positive fraud spike.

        z = max((current - baseline) / scale, 0)

    Converted to [0,1] using:

        anomaly = 1 - exp(-z / 3)

    This prevents extreme sparse-window values from exploding the score.
    """

    current = _safe_numeric(current)
    baseline = _safe_numeric(baseline)
    scale = _safe_numeric(scale, default=1e-8).clip(lower=1e-8)

    z = ((current - baseline) / scale).clip(lower=0.0)

    anomaly = 1.0 - np.exp(-z / 3.0)

    return pd.Series(
        anomaly,
        index=current.index,
    ).clip(0.0, 1.0)


def add_statistical_signals(
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add statistical anomaly signals using strictly past-only baselines.

    Output columns preserved for the existing Phase 2 architecture:

        risk_rate_anomaly
        fraud_count_anomaly
        fraud_amount_anomaly
        volume_anomaly
        statistical_evidence
        financial_evidence

    Cold-start policy:
        history_confidence = 0
            -> statistical evidence = 0

    Therefore the first merchant window is not declared anomalous merely
    because no baseline exists.
    """

    if windows.empty:
        return windows.copy()

    x = windows.copy()

    required = [
        "risk_rate",
        "expected_fraud_count",
        "expected_fraud_amount",
        "transaction_count",
        "risk_rate_baseline",
        "expected_fraud_count_baseline",
        "expected_fraud_amount_baseline",
        "transaction_count_baseline",
        "risk_rate_scale",
        "expected_fraud_count_scale",
        "expected_fraud_amount_scale",
        "transaction_count_scale",
        "history_confidence",
    ]

    missing = [c for c in required if c not in x.columns]

    if missing:
        raise ValueError(
            "Missing required columns for statistical signals: "
            f"{missing}"
        )

    # ------------------------------------------------------------------
    # Numeric safety
    # ------------------------------------------------------------------

    for col in [
        "risk_rate",
        "expected_fraud_count",
        "expected_fraud_amount",
        "transaction_count",
        "risk_rate_baseline",
        "expected_fraud_count_baseline",
        "expected_fraud_amount_baseline",
        "transaction_count_baseline",
        "risk_rate_scale",
        "expected_fraud_count_scale",
        "expected_fraud_amount_scale",
        "transaction_count_scale",
    ]:
        x[col] = _safe_numeric(x[col])

    x["history_confidence"] = _safe_unit_interval(
        x["history_confidence"]
    )

    # ------------------------------------------------------------------
    # Individual statistical anomalies
    # ------------------------------------------------------------------

    x["risk_rate_anomaly"] = _anomaly_from_baseline(
        x["risk_rate"],
        x["risk_rate_baseline"],
        x["risk_rate_scale"],
    )

    x["fraud_count_anomaly"] = _anomaly_from_baseline(
        x["expected_fraud_count"],
        x["expected_fraud_count_baseline"],
        x["expected_fraud_count_scale"],
    )

    x["fraud_amount_anomaly"] = _anomaly_from_baseline(
        x["expected_fraud_amount"],
        x["expected_fraud_amount_baseline"],
        x["expected_fraud_amount_scale"],
    )

    x["volume_anomaly"] = _anomaly_from_baseline(
        x["transaction_count"],
        x["transaction_count_baseline"],
        x["transaction_count_scale"],
    )

    # ------------------------------------------------------------------
    # Historical-confidence gating
    # ------------------------------------------------------------------
    #
    # Without historical evidence we cannot claim that the current value
    # is statistically abnormal.
    #
    # This is the key cold-start fix.
    # ------------------------------------------------------------------

    confidence = x["history_confidence"]

    x["risk_rate_anomaly"] *= confidence
    x["fraud_count_anomaly"] *= confidence
    x["fraud_amount_anomaly"] *= confidence
    x["volume_anomaly"] *= confidence

    # ------------------------------------------------------------------
    # Statistical evidence
    # ------------------------------------------------------------------

    x["statistical_evidence"] = (
        0.35 * x["risk_rate_anomaly"]
        + 0.30 * x["fraud_count_anomaly"]
        + 0.20 * x["fraud_amount_anomaly"]
        + 0.15 * x["volume_anomaly"]
    ).clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # Financial evidence
    # ------------------------------------------------------------------

    x["financial_evidence"] = (
        0.70 * x["fraud_amount_anomaly"]
        + 0.30 * x["fraud_count_anomaly"]
    ).clip(0.0, 1.0)

    # Explicitly force cold-start windows to zero statistical evidence.
    if "baseline_ready" in x.columns:
        cold_start = ~x["baseline_ready"].astype(bool)

        x.loc[cold_start, "statistical_evidence"] = 0.0
        x.loc[cold_start, "financial_evidence"] = 0.0

    # Final numerical safety.
    signal_columns = [
        "risk_rate_anomaly",
        "fraud_count_anomaly",
        "fraud_amount_anomaly",
        "volume_anomaly",
        "statistical_evidence",
        "financial_evidence",
    ]

    for col in signal_columns:
        x[col] = (
            pd.to_numeric(x[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

    return x