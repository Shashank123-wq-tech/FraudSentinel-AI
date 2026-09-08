from __future__ import annotations

import numpy as np
import pandas as pd


# Maximum time gap for considering two merchant windows temporally connected.
DEFAULT_MAX_GAP_MINUTES = 30.0

# EWMA smoothing factor.
DEFAULT_ALPHA = 0.30

# CUSUM reference level.
DEFAULT_CUSUM_K = 0.50

# CUSUM decision threshold.
DEFAULT_CUSUM_H = 3.00


def _safe_numeric(
    series: pd.Series,
    default: float = 0.0,
) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
    )


def _safe_unit(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )


def add_temporal_signals(
    windows: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    cusum_k: float = DEFAULT_CUSUM_K,
    cusum_h: float = DEFAULT_CUSUM_H,
    max_gap_minutes: float = DEFAULT_MAX_GAP_MINUTES,
) -> pd.DataFrame:
    """
    Add temporal fraud-spike signals.

    Signals:

        ewma_value
        ewma_residual
        cusum_value
        cusum_signal
        acceleration
        temporal_evidence

    Design requirements:

    1. Current observations do not enter their own historical baseline.
    2. Temporal state is maintained independently per merchant.
    3. Large time gaps break temporal continuity.
    4. First merchant window receives zero temporal evidence.
    5. All outputs are finite.
    """

    if windows.empty:
        return windows.copy()

    x = windows.copy()

    required = [
        "merchant_id",
        "window_start",
        "risk_rate",
    ]

    missing = [c for c in required if c not in x.columns]

    if missing:
        raise ValueError(
            "Missing required columns for temporal signals: "
            f"{missing}"
        )

    x["window_start"] = pd.to_datetime(
        x["window_start"],
        errors="coerce",
        utc=True,
    )

    if x["window_start"].isna().any():
        raise ValueError(
            "window_start contains invalid timestamps."
        )

    x["risk_rate"] = _safe_numeric(x["risk_rate"])

    # Preserve caller row ordering.
    x["_phase2_temporal_order"] = np.arange(len(x))

    x = x.sort_values(
        ["merchant_id", "window_start"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Historical confidence is optional for compatibility.
    if "history_confidence" in x.columns:
        x["history_confidence"] = _safe_unit(
            x["history_confidence"]
        )
    else:
        x["history_confidence"] = 1.0

    # ------------------------------------------------------------------
    # Previous observation and time gap
    # ------------------------------------------------------------------

    grouped = x.groupby(
        "merchant_id",
        sort=False,
        group_keys=False,
    )

    x["previous_risk_rate"] = grouped["risk_rate"].shift(1)

    x["previous_window_start"] = grouped["window_start"].shift(1)

    x["minutes_since_previous_window"] = (
        (
            x["window_start"]
            - x["previous_window_start"]
        )
        .dt.total_seconds()
        .div(60.0)
    )

    x["minutes_since_previous_window"] = (
        x["minutes_since_previous_window"]
        .replace([np.inf, -np.inf], np.nan)
    )

    has_previous = x["previous_risk_rate"].notna()

    connected = (
        has_previous
        & x["minutes_since_previous_window"].notna()
        & (
            x["minutes_since_previous_window"]
            <= float(max_gap_minutes)
        )
        & (
            x["minutes_since_previous_window"] >= 0
        )
    )

    x["temporal_connected"] = connected.astype(int)

    # ------------------------------------------------------------------
    # EWMA
    # ------------------------------------------------------------------

    ewma_values = np.zeros(len(x), dtype=float)
    ewma_residuals = np.zeros(len(x), dtype=float)

    # ------------------------------------------------------------------
    # CUSUM
    # ------------------------------------------------------------------

    cusum_values = np.zeros(len(x), dtype=float)
    cusum_signals = np.zeros(len(x), dtype=float)

    # ------------------------------------------------------------------
    # Acceleration
    # ------------------------------------------------------------------

    acceleration_values = np.zeros(len(x), dtype=float)

    for _, indices in x.groupby(
        "merchant_id",
        sort=False,
    ).groups.items():

        idx = np.asarray(indices)

        previous_ewma = None
        previous_risk = None
        previous_acceleration = 0.0
        cusum = 0.0

        for position in idx:

            current = float(x.at[position, "risk_rate"])

            is_connected = bool(
                x.at[position, "temporal_connected"]
            )

            # ----------------------------------------------------------
            # First observation or temporal gap
            # ----------------------------------------------------------

            if (
                previous_ewma is None
                or not is_connected
            ):
                previous_ewma = current
                previous_risk = current
                cusum = 0.0

                ewma_values[position] = current
                ewma_residuals[position] = 0.0
                cusum_values[position] = 0.0
                cusum_signals[position] = 0.0
                acceleration_values[position] = 0.0

                continue

            # ----------------------------------------------------------
            # EWMA residual
            # ----------------------------------------------------------

            residual = current - previous_ewma

            ewma_values[position] = (
                alpha * current
                + (1.0 - alpha) * previous_ewma
            )

            ewma_residuals[position] = residual

            # ----------------------------------------------------------
            # Acceleration
            # ----------------------------------------------------------

            first_difference = current - previous_risk

            acceleration = (
                first_difference
                - previous_acceleration
            )

            acceleration_values[position] = acceleration

            previous_acceleration = first_difference

            # ----------------------------------------------------------
            # Positive CUSUM
            # ----------------------------------------------------------

            # Normalize residual relative to a local reference.
            #
            # risk_rate is bounded, so use a small numerical scale to
            # prevent huge values for sparse windows.
            local_scale = max(
                abs(previous_ewma) * 0.10,
                1e-6,
            )

            z = residual / local_scale

            positive_increment = z - float(cusum_k)

            cusum = max(
                0.0,
                cusum + positive_increment,
            )

            # Cap for numerical stability.
            cusum = min(cusum, 50.0)

            cusum_values[position] = cusum

            cusum_signals[position] = (
                1.0
                if cusum >= float(cusum_h)
                else min(
                    cusum / float(cusum_h),
                    1.0,
                )
            )

            previous_ewma = (
                alpha * current
                + (1.0 - alpha) * previous_ewma
            )

            previous_risk = current

    x["ewma_value"] = ewma_values
    x["ewma_residual"] = ewma_residuals
    x["cusum_value"] = cusum_values
    x["cusum_signal"] = cusum_signals
    x["acceleration"] = acceleration_values

    # ------------------------------------------------------------------
    # Convert EWMA residual and acceleration to bounded evidence.
    # ------------------------------------------------------------------

    residual_scale = (
        x["ewma_value"].abs() * 0.10
    ).clip(lower=1e-6)

    positive_residual = (
        x["ewma_residual"] / residual_scale
    ).clip(lower=0.0)

    residual_evidence = (
        1.0 - np.exp(
            -positive_residual / 3.0
        )
    )

    positive_acceleration = (
        x["acceleration"] / residual_scale
    ).clip(lower=0.0)

    acceleration_evidence = (
        1.0 - np.exp(
            -positive_acceleration / 3.0
        )
    )

    cusum_evidence = _safe_unit(
        x["cusum_signal"]
    )

    # ------------------------------------------------------------------
    # Temporal evidence
    # ------------------------------------------------------------------

    temporal_evidence = (
        0.45 * residual_evidence
        + 0.35 * cusum_evidence
        + 0.20 * acceleration_evidence
    )

    # No temporal evidence when windows are not connected.
    temporal_evidence = temporal_evidence.where(
        x["temporal_connected"].eq(1),
        0.0,
    )

    # Do not create strong temporal evidence before enough history exists.
    temporal_evidence *= x["history_confidence"]

    x["temporal_evidence"] = (
        temporal_evidence
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    # ------------------------------------------------------------------
    # Explicit first-window handling
    # ------------------------------------------------------------------

    first_window = ~has_previous

    x.loc[first_window, "ewma_residual"] = 0.0
    x.loc[first_window, "cusum_value"] = 0.0
    x.loc[first_window, "cusum_signal"] = 0.0
    x.loc[first_window, "acceleration"] = 0.0
    x.loc[first_window, "temporal_evidence"] = 0.0

    # ------------------------------------------------------------------
    # Final numerical safety
    # ------------------------------------------------------------------

    for col in [
        "ewma_value",
        "ewma_residual",
        "cusum_value",
        "cusum_signal",
        "acceleration",
        "temporal_evidence",
    ]:
        x[col] = (
            pd.to_numeric(x[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    x["cusum_signal"] = x["cusum_signal"].clip(0.0, 1.0)
    x["temporal_evidence"] = x["temporal_evidence"].clip(0.0, 1.0)

    # Restore original ordering.
    x = (
        x.sort_values(
            "_phase2_temporal_order",
            kind="mergesort",
        )
        .drop(columns="_phase2_temporal_order")
        .reset_index(drop=True)
    )

    return x