from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-8


def _safe_numeric(
    series: pd.Series,
    default: float = 0.0,
) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
    )


def _safe_unit(
    series: pd.Series,
    default: float = 0.0,
) -> pd.Series:
    return (
        _safe_numeric(series, default=default)
        .clip(0.0, 1.0)
    )


def add_phase2_scores(
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Phase 2 TAS and FAS scores.

    TAS
    ---
        Temporal-Statistical Anomaly Score

        TAS =
            0.50 * statistical_evidence
          + 0.30 * temporal_evidence
          + 0.20 * breadth_score

    FAS
    ---
        Fraud Attack Severity

        FAS =
            0.60 * intensity
          + 0.20 * materiality
          + 0.20 * coordination

    spike_score = FAS

    Cold-start policy
    -----------------
    A merchant with no historical baseline receives:

        statistical_evidence = 0
        temporal_evidence = 0
        materiality = 0

    but retains the breadth/coordination information.

    Most importantly, every output is finite.
    """

    if windows.empty:
        return windows.copy()

    x = windows.copy()

    required = [
        "statistical_evidence",
        "temporal_evidence",
        "breadth_score",
        "expected_fraud_amount",
        "expected_fraud_amount_baseline",
        "coordination_score",
    ]

    missing = [c for c in required if c not in x.columns]

    if missing:
        raise ValueError(
            "Missing required columns for Phase 2 scoring: "
            f"{missing}"
        )

    # ------------------------------------------------------------------
    # Sanitize inputs
    # ------------------------------------------------------------------

    x["statistical_evidence"] = _safe_unit(
        x["statistical_evidence"]
    )

    x["temporal_evidence"] = _safe_unit(
        x["temporal_evidence"]
    )

    x["breadth_score"] = _safe_unit(
        x["breadth_score"]
    )

    x["coordination_score"] = _safe_unit(
        x["coordination_score"]
    )

    x["expected_fraud_amount"] = _safe_numeric(
        x["expected_fraud_amount"]
    ).clip(lower=0.0)

    x["expected_fraud_amount_baseline"] = _safe_numeric(
        x["expected_fraud_amount_baseline"]
    ).clip(lower=0.0)

    # ------------------------------------------------------------------
    # TAS
    # ------------------------------------------------------------------

    tas_raw = (
        0.50 * x["statistical_evidence"]
        + 0.30 * x["temporal_evidence"]
        + 0.20 * x["breadth_score"]
    )

    x["tas"] = (
        100.0 * tas_raw
    ).clip(0.0, 100.0)

    # ------------------------------------------------------------------
    # Materiality
    # ------------------------------------------------------------------

    baseline_amount = (
        x["expected_fraud_amount_baseline"]
        .clip(lower=0.0)
    )

    current_amount = (
        x["expected_fraud_amount"]
        .clip(lower=0.0)
    )

    # If baseline is exactly zero but current expected fraud amount is
    # positive, avoid an infinite ratio. Use a conservative normalized
    # excess instead.
    denominator = baseline_amount.copy()

    ratio = pd.Series(
        np.zeros(len(x)),
        index=x.index,
        dtype=float,
    )

    nonzero_baseline = denominator > EPSILON

    ratio.loc[nonzero_baseline] = (
        current_amount.loc[nonzero_baseline]
        / denominator.loc[nonzero_baseline]
    )

    zero_baseline = ~nonzero_baseline

    ratio.loc[zero_baseline] = (
        current_amount.loc[zero_baseline]
        .clip(upper=1.0)
    )

    ratio = (
        ratio
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0, upper=100.0)
    )

    x["materiality"] = (
        1.0 - np.exp(-ratio)
    ).clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # Cold-start materiality
    # ------------------------------------------------------------------
    #
    # If there is no history, current amount cannot legitimately be
    # called "above baseline". Therefore suppress materiality.
    # ------------------------------------------------------------------

    if "baseline_ready" in x.columns:
        cold_start = ~x["baseline_ready"].astype(bool)

        x.loc[cold_start, "materiality"] = 0.0

    # ------------------------------------------------------------------
    # Coordination
    # ------------------------------------------------------------------

    x["coordination"] = (
        x["coordination_score"]
        .clip(0.0, 1.0)
    )

    # ------------------------------------------------------------------
    # Intensity
    # ------------------------------------------------------------------

    intensity = (
        x["tas"] / 100.0
    ).clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # FAS
    # ------------------------------------------------------------------

    fas_raw = (
        0.60 * intensity
        + 0.20 * x["materiality"]
        + 0.20 * x["coordination"]
    )

    x["fas"] = (
        100.0 * fas_raw
    ).clip(0.0, 100.0)

    x["spike_score"] = x["fas"]

    # ------------------------------------------------------------------
    # Explicit cold-start policy
    # ------------------------------------------------------------------
    #
    # Cold-start windows should not receive statistical/financial
    # evidence, but their score remains finite and conservative.
    #
    # We retain breadth/coordination contribution so that the system
    # does not discard potentially useful structural information.
    # ------------------------------------------------------------------

    if "baseline_ready" in x.columns:
        cold_start = ~x["baseline_ready"].astype(bool)

        x.loc[cold_start, "statistical_evidence"] = 0.0
        x.loc[cold_start, "temporal_evidence"] = 0.0
        x.loc[cold_start, "materiality"] = 0.0

        # Recompute TAS/FAS after enforcing cold-start policy.
        x.loc[cold_start, "tas"] = (
            100.0
            * (
                0.20
                * x.loc[cold_start, "breadth_score"]
            )
        ).clip(0.0, 100.0)

        cold_intensity = (
            x.loc[cold_start, "tas"] / 100.0
        ).clip(0.0, 1.0)

        x.loc[cold_start, "fas"] = (
            100.0
            * (
                0.60 * cold_intensity
                + 0.20 * x.loc[cold_start, "coordination"]
            )
        ).clip(0.0, 100.0)

        x.loc[cold_start, "spike_score"] = (
            x.loc[cold_start, "fas"]
        )

    # ------------------------------------------------------------------
    # Absolute final safety check
    # ------------------------------------------------------------------

    score_columns = [
        "statistical_evidence",
        "temporal_evidence",
        "breadth_score",
        "materiality",
        "coordination",
        "tas",
        "fas",
        "spike_score",
    ]

    for col in score_columns:
        x[col] = (
            pd.to_numeric(
                x[col],
                errors="coerce",
            )
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    # Bounds.
    for col in [
        "statistical_evidence",
        "temporal_evidence",
        "breadth_score",
        "materiality",
        "coordination",
    ]:
        x[col] = x[col].clip(0.0, 1.0)

    for col in [
        "tas",
        "fas",
        "spike_score",
    ]:
        x[col] = x[col].clip(0.0, 100.0)

    # Hard invariant:
    # Phase 2 must never emit invalid spike scores.
    if not np.isfinite(
        x["spike_score"].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Phase 2 scoring produced non-finite spike_score values."
        )

    return x