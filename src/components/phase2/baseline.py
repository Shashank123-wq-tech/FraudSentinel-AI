from __future__ import annotations

import numpy as np
import pandas as pd


# Metrics for which merchant-level historical baselines are maintained.
BASELINE_METRICS = [
    "risk_rate",
    "expected_fraud_count",
    "expected_fraud_amount",
    "transaction_count",
    "total_amount",
]


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    """
    Convert a Series to numeric and replace invalid values.
    """
    return pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(default)


def _robust_scale(
    q1: float,
    q3: float,
    center: float,
    scale_floor: float,
) -> float:
    """
    Robust scale estimate using the normal-distribution-consistent
    IQR scaling:

        robust_scale = (Q3 - Q1) / 1.349

    A deterministic fallback is used when historical dispersion is zero.
    """
    scale = (q3 - q1) / 1.349

    if not np.isfinite(scale) or scale <= 0:
        scale = max(abs(center) * 0.05, scale_floor)

    return float(max(scale, scale_floor))


def add_past_only_baselines(
    windows: pd.DataFrame,
    history_windows: int = 96,
    min_history: int = 24,
    scale_floor: float = 1e-8,
) -> pd.DataFrame:
    """
    Add strictly past-only merchant baselines.

    Important:
    - The current window is NEVER included in its own baseline.
    - Baselines are calculated independently for each merchant.
    - The first window of a merchant receives a finite conservative fallback.
    - Cold-start windows are marked through baseline_ready=False.
    - history_confidence increases gradually as history accumulates.

    Parameters
    ----------
    windows:
        Merchant-window DataFrame.

    history_windows:
        Maximum number of historical windows used.

    min_history:
        Number of historical windows required for a fully trusted baseline.

    scale_floor:
        Minimum numerical scale.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with baseline columns appended.
    """

    if windows.empty:
        return windows.copy()

    required = {
        "merchant_id",
        "window_start",
        *BASELINE_METRICS,
    }

    missing = required - set(windows.columns)

    if missing:
        raise ValueError(
            "Missing required columns for baseline construction: "
            f"{sorted(missing)}"
        )

    x = windows.copy()

    x["window_start"] = pd.to_datetime(
        x["window_start"],
        errors="coerce",
        utc=True,
    )

    if x["window_start"].isna().any():
        raise ValueError("window_start contains invalid timestamps.")

    # Preserve original ordering so downstream row alignment is unaffected.
    x["_phase2_original_order"] = np.arange(len(x))

    x = x.sort_values(
        ["merchant_id", "window_start"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Clean numerical inputs.
    for metric in BASELINE_METRICS:
        x[metric] = _safe_numeric(x[metric])

    grouped = x.groupby(
        "merchant_id",
        sort=False,
        group_keys=False,
    )

    # Number of strictly previous merchant windows.
    x["history_count"] = grouped.cumcount()

    x["history_confidence"] = (
        x["history_count"] / float(max(min_history, 1))
    ).clip(0.0, 1.0)

    x["baseline_ready"] = (
        x["history_count"] >= int(min_history)
    )

    for metric in BASELINE_METRICS:

        # Strictly past observations.
        historical = (
            grouped[metric]
            .transform(
                lambda s: s.shift(1).rolling(
                    window=history_windows,
                    min_periods=1,
                ).median()
            )
        )

        q1 = (
            grouped[metric]
            .transform(
                lambda s: s.shift(1).rolling(
                    window=history_windows,
                    min_periods=1,
                ).quantile(0.25)
            )
        )

        q3 = (
            grouped[metric]
            .transform(
                lambda s: s.shift(1).rolling(
                    window=history_windows,
                    min_periods=1,
                ).quantile(0.75)
            )
        )

        # ------------------------------------------------------------------
        # Cold-start fallback
        # ------------------------------------------------------------------
        #
        # If no historical observation exists, use the CURRENT value only
        # as a neutral reference.
        #
        # This does NOT create a detection signal because statistical.py
        # explicitly gates evidence using history_confidence.
        #
        baseline = historical.copy()

        no_history = x["history_count"].eq(0)

        baseline.loc[no_history] = x.loc[no_history, metric]

        baseline = baseline.replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(x[metric])

        baseline = baseline.clip(lower=0.0)

        x[f"{metric}_baseline"] = baseline

        # ------------------------------------------------------------------
        # Robust historical scale
        # ------------------------------------------------------------------

        robust_scale = ((q3 - q1) / 1.349)

        fallback_scale = np.maximum(
            baseline.abs() * 0.05,
            scale_floor,
        )

        robust_scale = robust_scale.where(
            np.isfinite(robust_scale) & (robust_scale > 0),
            fallback_scale,
        )

        # If even that is invalid, use the floor.
        robust_scale = robust_scale.replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(scale_floor)

        robust_scale = robust_scale.clip(
            lower=scale_floor
        )

        x[f"{metric}_scale"] = robust_scale

        # Historical quantiles are useful diagnostically. For cold-start
        # windows they are set to the finite fallback baseline.
        x[f"{metric}_q1"] = (
            q1.replace([np.inf, -np.inf], np.nan)
            .fillna(baseline)
            .clip(lower=0.0)
        )

        x[f"{metric}_q3"] = (
            q3.replace([np.inf, -np.inf], np.nan)
            .fillna(baseline)
            .clip(lower=0.0)
        )

    # ----------------------------------------------------------------------
    # Merchant-local elapsed time information
    # ----------------------------------------------------------------------

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
        .fillna(np.nan)
    )

    # Restore original order.
    x = (
        x.sort_values(
            "_phase2_original_order",
            kind="mergesort",
        )
        .drop(columns="_phase2_original_order")
        .reset_index(drop=True)
    )

    return x


def build_merchant_baseline(
    windows: pd.DataFrame,
    history_windows: int = 96,
    min_history: int = 24,
    scale_floor: float = 1e-8,
) -> pd.DataFrame:
    """
    Backward-compatible public wrapper.
    """
    return add_past_only_baselines(
        windows=windows,
        history_windows=history_windows,
        min_history=min_history,
        scale_floor=scale_floor,
    )