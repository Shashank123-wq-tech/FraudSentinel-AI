
# ============================================================
# FraudSentinel AI
# Phase 2 — Temporal Holdout Validation
#
# File:
#     src/validation/phase2/temporal_holdout.py
#
# Purpose:
#     Validate Phase-2 fraud-spike intelligence using a strict
#     chronological holdout.
#
# Design:
#
#     Phase-2 Windows
#           |
#           v
#     Ground-truth alignment
#           |
#           v
#     Unique timestamp split
#           |
#           +----------------------+
#           |                      |
#           v                      v
#     Calibration period      Holdout period
#           |                      |
#           v                      |
#     Threshold calibration        |
#           |                      |
#           v                      |
#     Frozen thresholds           |
#           |                      |
#           +----------+-----------+
#                      |
#                      v
#              Phase-2 detector
#                      |
#                      v
#                Holdout metrics
#
# IMPORTANT:
#
#   1. No labels from holdout are used for calibration.
#   2. The temporal split is performed on UNIQUE timestamps.
#   3. The same timestamp cannot exist in both partitions.
#   4. The detector must preserve every input row.
#   5. ROC-AUC / PR-AUC are calculated from continuous
#      spike_score, not operational states.
#   6. State metrics use ALL fraud-positive holdout windows
#      as the denominator.
#
# ============================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.components.phase2.detector import classify_spike_state
from src.components.phase2.events import build_fraud_spike_events


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_CALIBRATION_FRACTION = 0.70
DEFAULT_EVENT_GAP_MINUTES = 15.0

STATE_ORDER = [
    "NORMAL",
    "CANDIDATE_SPIKE",
    "VERIFIED_FRAUD_SPIKE",
    "CRITICAL_ACTIVE_SPIKE",
]


# ============================================================
# GENERIC HELPERS
# ============================================================

def _json_safe(value: Any) -> Any:
    """
    Convert numpy / pandas objects into JSON-safe values.
    """

    if isinstance(
        value,
        (
            np.integer,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float16,
            np.float32,
            np.float64,
        ),
    ):
        if not np.isfinite(value):
            return None

        return float(value)

    if isinstance(
        value,
        (
            pd.Timestamp,
            np.datetime64,
        ),
    ):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            _json_safe(v)
            for v in value
        ]

    if isinstance(value, tuple):
        return [
            _json_safe(v)
            for v in value
        ]

    return value


def _save_json(
    data: dict,
    output_path: Path,
) -> None:
    """
    Save JSON result.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            _json_safe(data),
            file,
            indent=2,
        )


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert value to finite float.
    """

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (
        TypeError,
        ValueError,
    ):
        pass

    return default


def _safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    Safe scalar division.
    """

    numerator = _safe_float(
        numerator,
        0.0,
    )

    denominator = _safe_float(
        denominator,
        0.0,
    )

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


# ============================================================
# PHASE-2 WINDOW LOADING
# ============================================================

def _load_phase2_windows(
    phase2_windows_path: str | Path,
) -> pd.DataFrame:
    """
    Load Phase-2 merchant windows.

    Required:
        merchant_id
        window_start
        spike_score
    """

    path = Path(
        phase2_windows_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Phase-2 windows file not found: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"Phase-2 windows file is empty: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "merchant_id",
        "window_start",
        "spike_score",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Phase-2 windows are missing required "
            f"columns: {missing}"
        )

    df["window_start"] = pd.to_datetime(
        df["window_start"],
        utc=True,
        errors="coerce",
    )

    invalid_time = int(
        df["window_start"].isna().sum()
    )

    if invalid_time:
        raise ValueError(
            "Phase-2 windows contain "
            f"{invalid_time:,} invalid window_start values."
        )

    df["spike_score"] = pd.to_numeric(
        df["spike_score"],
        errors="coerce",
    )

    invalid_score = int(
        df["spike_score"].isna().sum()
    )

    if invalid_score:
        raise ValueError(
            "Phase-2 windows contain "
            f"{invalid_score:,} invalid spike_score values."
        )

    df["spike_score"] = (
        df["spike_score"]
        .clip(0.0, 100.0)
    )

    if df["merchant_id"].isna().any():
        raise ValueError(
            "Phase-2 windows contain null merchant_id values."
        )

    df["_validation_row_id"] = np.arange(
        len(df),
        dtype=np.int64,
    )

    return df


# ============================================================
# GROUND TRUTH LOADING
# ============================================================

def _load_ground_truth(
    ground_truth_path: str | Path,
) -> pd.DataFrame:
    """
    Load raw labeled fraud transactions.

    Supported timestamp names:
        timestamp
        trans_date_trans_time

    Supported merchant names:
        merchant_id
        merchant

    Supported amount names:
        amount
        amt

    Required:
        is_fraud
    """

    path = Path(
        ground_truth_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"Ground-truth file is empty: {path}"
        )

    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        timestamp_column = "timestamp"
    elif "trans_date_trans_time" in df.columns:
        timestamp_column = "trans_date_trans_time"
    else:
        raise ValueError(
            "Ground truth must contain either "
            "'timestamp' or 'trans_date_trans_time'."
        )

    if "merchant_id" in df.columns:
        merchant_column = "merchant_id"
    elif "merchant" in df.columns:
        merchant_column = "merchant"
    else:
        raise ValueError(
            "Ground truth must contain either "
            "'merchant_id' or 'merchant'."
        )

    if "amount" in df.columns:
        amount_column = "amount"
    elif "amt" in df.columns:
        amount_column = "amt"
    else:
        raise ValueError(
            "Ground truth must contain either "
            "'amount' or 'amt'."
        )

    if "is_fraud" not in df.columns:
        raise ValueError(
            "Ground truth must contain 'is_fraud'."
        )

    df = df.rename(
        columns={
            timestamp_column: "timestamp",
            merchant_column: "merchant_id",
            amount_column: "amount",
        }
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df["merchant_id"] = (
        df["merchant_id"]
        .astype(str)
    )

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce",
    ).fillna(0.0)

    fraud_numeric = pd.to_numeric(
        df["is_fraud"],
        errors="coerce",
    ).fillna(0)

    df["is_fraud"] = (
        fraud_numeric
        .astype(int)
        .clip(0, 1)
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "merchant_id",
        ]
    ).copy()

    return df[
        [
            "timestamp",
            "merchant_id",
            "amount",
            "is_fraud",
        ]
    ]


# ============================================================
# GROUND-TRUTH ALIGNMENT
# ============================================================

def _build_ground_truth_windows(
    ground_truth: pd.DataFrame,
    window_minutes: int = 15,
) -> pd.DataFrame:
    """
    Convert transaction-level labels into merchant-window labels.
    """

    if ground_truth.empty:
        raise ValueError(
            "Ground truth is empty."
        )

    x = ground_truth.copy()

    x["window_start"] = (
        x["timestamp"]
        .dt.floor(
            f"{window_minutes}min"
        )
    )

    x["fraud_amount_component"] = (
        x["amount"]
        .where(
            x["is_fraud"] == 1,
            0.0,
        )
    )

    grouped = (
        x.groupby(
            [
                "merchant_id",
                "window_start",
            ],
            sort=False,
            observed=True,
        )
        .agg(
            fraud_count=(
                "is_fraud",
                "sum",
            ),
            transaction_count=(
                "is_fraud",
                "size",
            ),
            fraud_amount=(
                "fraud_amount_component",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped["fraud_positive"] = (
        grouped["fraud_count"] > 0
    ).astype(int)

    return grouped[
        [
            "merchant_id",
            "window_start",
            "fraud_positive",
            "fraud_count",
            "fraud_amount",
            "transaction_count",
        ]
    ]


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def _temporal_split(
    df: pd.DataFrame,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
):
    """
    Strict chronological split using UNIQUE timestamps.

    Multiple merchants can share the same window_start.
    Therefore the split is performed on unique timestamps,
    not row positions.
    """

    if df.empty:
        raise ValueError(
            "Cannot split an empty dataframe."
        )

    if not (
        0.0 < calibration_fraction < 1.0
    ):
        raise ValueError(
            "calibration_fraction must be between 0 and 1."
        )

    x = df.copy()

    x["window_start"] = pd.to_datetime(
        x["window_start"],
        utc=True,
        errors="coerce",
    )

    if x["window_start"].isna().any():
        raise ValueError(
            "Temporal split contains invalid window_start values."
        )

    unique_times = (
        x["window_start"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(unique_times) < 2:
        raise ValueError(
            "At least two unique timestamps are required."
        )

    split_position = int(
        len(unique_times)
        * calibration_fraction
    )

    split_position = max(
        1,
        min(
            split_position,
            len(unique_times) - 1,
        ),
    )

    split_time = unique_times.iloc[
        split_position
    ]

    calibration = x[
        x["window_start"] < split_time
    ].copy()

    holdout = x[
        x["window_start"] >= split_time
    ].copy()

    if calibration.empty:
        raise RuntimeError(
            "Calibration partition is empty."
        )

    if holdout.empty:
        raise RuntimeError(
            "Holdout partition is empty."
        )

    calibration_max = (
        calibration["window_start"].max()
    )

    holdout_min = (
        holdout["window_start"].min()
    )

    if calibration_max >= holdout_min:
        raise RuntimeError(
            "Temporal leakage detected."
        )

    calibration_times = set(
        calibration["window_start"].unique()
    )

    holdout_times = set(
        holdout["window_start"].unique()
    )

    overlap = (
        calibration_times
        & holdout_times
    )

    if overlap:
        raise RuntimeError(
            "Calibration and holdout contain "
            f"{len(overlap):,} overlapping timestamps."
        )

    return (
        calibration,
        holdout,
        split_time,
    )


# ============================================================
# GROUND-TRUTH MERGE
# ============================================================

def _attach_ground_truth(
    phase2_windows: pd.DataFrame,
    ground_truth_windows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join ground truth onto every Phase-2 window.
    """

    x = phase2_windows.copy()
    y = ground_truth_windows.copy()

    x["merchant_id"] = (
        x["merchant_id"]
        .astype(str)
    )

    y["merchant_id"] = (
        y["merchant_id"]
        .astype(str)
    )

    x["window_start"] = pd.to_datetime(
        x["window_start"],
        utc=True,
    )

    y["window_start"] = pd.to_datetime(
        y["window_start"],
        utc=True,
    )

    duplicate_gt_keys = (
        y.duplicated(
            [
                "merchant_id",
                "window_start",
            ],
            keep=False,
        )
    )

    if duplicate_gt_keys.any():
        raise RuntimeError(
            "Ground truth contains duplicate "
            "merchant/window keys."
        )

    merged = x.merge(
        y,
        how="left",
        on=[
            "merchant_id",
            "window_start",
        ],
        validate="one_to_one",
        suffixes=(
            "",
            "_ground_truth",
        ),
    )

    merged["fraud_positive"] = (
        pd.to_numeric(
            merged["fraud_positive"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .clip(0, 1)
    )

    merged["fraud_count"] = (
        pd.to_numeric(
            merged["fraud_count"],
            errors="coerce",
        )
        .fillna(0)
    )

    merged["fraud_amount"] = (
        pd.to_numeric(
            merged["fraud_amount"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    merged["ground_truth_transaction_count"] = (
        pd.to_numeric(
            merged["transaction_count"],
            errors="coerce",
        )
        .fillna(0)
    )

    return merged


# ============================================================
# METRICS
# ============================================================

def _calculate_ranking_metrics(
    y_true: pd.Series,
    score: pd.Series,
) -> dict:
    """
    Calculate ROC-AUC and PR-AUC.
    """

    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )

    y = pd.to_numeric(
        y_true,
        errors="coerce",
    ).fillna(0).astype(int)

    s = pd.to_numeric(
        score,
        errors="coerce",
    )

    valid = s.notna()

    y = y.loc[valid]
    s = s.loc[valid]

    if len(y) == 0:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "n": 0,
            "positive": 0,
            "negative": 0,
        }

    positive = int(
        (y == 1).sum()
    )

    negative = int(
        (y == 0).sum()
    )

    if positive == 0 or negative == 0:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "n": int(len(y)),
            "positive": positive,
            "negative": negative,
        }

    return {
        "roc_auc": float(
            roc_auc_score(
                y,
                s,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y,
                s,
            )
        ),
        "n": int(len(y)),
        "positive": positive,
        "negative": negative,
    }


def _state_metrics(
    df: pd.DataFrame,
) -> list[dict]:
    """
    Calculate state-level metrics.

    Fraud recall denominator = ALL fraud-positive
    holdout windows.
    """

    if df.empty:
        return []

    if "spike_state" not in df.columns:
        raise ValueError(
            "spike_state is required."
        )

    y_true = (
        pd.to_numeric(
            df["fraud_positive"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    total_fraud_windows = int(
        (y_true == 1).sum()
    )

    total_fraud_amount = float(
        pd.to_numeric(
            df["fraud_amount"],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    total_windows = len(df)

    results = []

    for state in STATE_ORDER:

        mask = (
            df["spike_state"]
            == state
        )

        state_windows = int(
            mask.sum()
        )

        state_fraud_windows = int(
            (
                mask
                & (y_true == 1)
            ).sum()
        )

        state_fraud_amount = float(
            pd.to_numeric(
                df.loc[
                    mask,
                    "fraud_amount",
                ],
                errors="coerce",
            )
            .fillna(0.0)
            .sum()
        )

        results.append(
            {
                "spike_state": state,
                "windows": state_windows,
                "fraud_positive_windows": (
                    state_fraud_windows
                ),
                "fraud_window_recall": (
                    _safe_divide(
                        state_fraud_windows,
                        total_fraud_windows,
                    )
                ),
                "state_precision": (
                    _safe_divide(
                        state_fraud_windows,
                        state_windows,
                    )
                ),
                "alert_rate": (
                    _safe_divide(
                        state_windows,
                        total_windows,
                    )
                ),
                "fraud_amount": (
                    state_fraud_amount
                ),
                "fraud_amount_capture_rate": (
                    _safe_divide(
                        state_fraud_amount,
                        total_fraud_amount,
                    )
                ),
            }
        )

    return results


def _top_k_metrics(
    df: pd.DataFrame,
) -> list[dict]:
    """
    Evaluate continuous spike-score ranking.
    """

    if df.empty:
        return []

    x = df.copy()

    x["spike_score"] = pd.to_numeric(
        x["spike_score"],
        errors="coerce",
    )

    x["fraud_positive"] = pd.to_numeric(
        x["fraud_positive"],
        errors="coerce",
    ).fillna(0).astype(int)

    x["fraud_amount"] = pd.to_numeric(
        x["fraud_amount"],
        errors="coerce",
    ).fillna(0.0)

    x = x.dropna(
        subset=["spike_score"]
    )

    x = x.sort_values(
        "spike_score",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    total_fraud_windows = int(
        x["fraud_positive"].sum()
    )

    total_fraud_amount = float(
        x["fraud_amount"].sum()
    )

    results = []

    for fraction in (
        0.01,
        0.05,
        0.10,
        0.20,
    ):

        k = max(
            1,
            int(
                np.ceil(
                    len(x)
                    * fraction
                )
            ),
        )

        top = x.iloc[:k]

        captured_fraud_windows = int(
            top["fraud_positive"].sum()
        )

        captured_fraud_amount = float(
            top["fraud_amount"].sum()
        )

        results.append(
            {
                "top_fraction": fraction,
                "windows_selected": k,
                "fraud_windows_captured": (
                    captured_fraud_windows
                ),
                "fraud_window_recall": (
                    _safe_divide(
                        captured_fraud_windows,
                        total_fraud_windows,
                    )
                ),
                "fraud_amount_captured": (
                    captured_fraud_amount
                ),
                "fraud_amount_capture_rate": (
                    _safe_divide(
                        captured_fraud_amount,
                        total_fraud_amount,
                    )
                ),
            }
        )

    return results


def _amount_spearman(
    df: pd.DataFrame,
) -> float | None:
    """
    Spearman correlation between spike score and actual
    fraud amount.
    """

    if len(df) < 2:
        return None

    score = pd.to_numeric(
        df["spike_score"],
        errors="coerce",
    )

    amount = pd.to_numeric(
        df["fraud_amount"],
        errors="coerce",
    )

    valid = (
        score.notna()
        & amount.notna()
    )

    if valid.sum() < 2:
        return None

    correlation = (
        score.loc[valid]
        .corr(
            amount.loc[valid],
            method="spearman",
        )
    )

    if pd.isna(correlation):
        return None

    return float(correlation)


# ============================================================
# CALIBRATION
# ============================================================

def _calibrate_thresholds(
    calibration: pd.DataFrame,
) -> dict:
    """
    Calibrate operational state thresholds using ONLY
    the calibration period.
    """

    from src.components.phase2.calibration import (
        calibrate_phase2_states,
    )

    required = [
        "spike_score",
        "fraud_positive",
        "fraud_amount",
    ]

    missing = [
        column
        for column in required
        if column not in calibration.columns
    ]

    if missing:
        raise ValueError(
            "Calibration dataframe is missing: "
            f"{missing}"
        )

    return calibrate_phase2_states(
        calibration[
            required
        ].copy()
    )


# ============================================================
# DETECTOR APPLICATION
# ============================================================

def _apply_detector(
    df: pd.DataFrame,
    calibration: dict,
) -> pd.DataFrame:
    """
    Apply frozen calibration thresholds.

    Detector output must preserve the exact input
    row population and validation IDs.
    """

    x = df.copy()

    if "_validation_row_id" not in x.columns:
        x["_validation_row_id"] = np.arange(
            len(x),
            dtype=np.int64,
        )

    input_ids = set(
        x["_validation_row_id"]
    )

    candidate_threshold = float(
        calibration[
            "candidate_score_threshold"
        ]
    )

    verified_threshold = float(
        calibration[
            "verified_score_threshold"
        ]
    )

    critical_threshold = float(
        calibration[
            "critical_score_threshold"
        ]
    )

    evaluated = classify_spike_state(
        x,
        candidate_score_threshold=(
            candidate_threshold
        ),
        verified_score_threshold=(
            verified_threshold
        ),
        critical_score_threshold=(
            critical_threshold
        ),
    )

    if len(evaluated) != len(x):
        raise RuntimeError(
            "Phase-2 detector changed the number "
            "of validation rows. "
            f"Input={len(x):,}, "
            f"Output={len(evaluated):,}."
        )

    if "_validation_row_id" not in evaluated.columns:
        raise RuntimeError(
            "Phase-2 detector removed "
            "_validation_row_id."
        )

    output_ids = set(
        evaluated["_validation_row_id"]
    )

    if input_ids != output_ids:

        missing_ids = (
            input_ids
            - output_ids
        )

        extra_ids = (
            output_ids
            - input_ids
        )

        raise RuntimeError(
            "Phase-2 detector changed the validation "
            "row population. "
            f"Missing={len(missing_ids):,}, "
            f"Extra={len(extra_ids):,}."
        )

    return evaluated


# ============================================================
# EVENT VALIDATION
# ============================================================

def _build_holdout_events(
    holdout_evaluated: pd.DataFrame,
    max_gap_minutes: float = DEFAULT_EVENT_GAP_MINUTES,
) -> pd.DataFrame:
    """
    Build operational fraud-spike events from holdout states.
    """

    if holdout_evaluated.empty:
        return pd.DataFrame()

    return build_fraud_spike_events(
        holdout_evaluated.copy(),
        max_gap_minutes=max_gap_minutes,
    )


# ============================================================
# MAIN VALIDATION
# ============================================================

def run_temporal_holdout_validation(
    phase2_windows_path: str | Path,
    ground_truth_path: str | Path,
    output_dir: str | Path,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    window_minutes: int = 15,
    max_event_gap_minutes: float = DEFAULT_EVENT_GAP_MINUTES,
) -> dict:
    """
    Run complete Phase-2 temporal holdout validation.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 72)
    print("PHASE 2 TEMPORAL HOLDOUT VALIDATION")
    print("=" * 72)

    # ========================================================
    # STEP 1 — LOAD PHASE-2 WINDOWS
    # ========================================================

    phase2_windows = _load_phase2_windows(
        phase2_windows_path
    )

    print()
    print("Phase-2 windows:")
    print(
        f"  Rows             : "
        f"{len(phase2_windows):,}"
    )
    print(
        f"  Merchants        : "
        f"{phase2_windows['merchant_id'].nunique():,}"
    )
    print(
        f"  Time start       : "
        f"{phase2_windows['window_start'].min()}"
    )
    print(
        f"  Time end         : "
        f"{phase2_windows['window_start'].max()}"
    )

    # ========================================================
    # STEP 2 — LOAD GROUND TRUTH
    # ========================================================

    ground_truth = _load_ground_truth(
        ground_truth_path
    )

    print()
    print("Ground truth:")
    print(
        f"  Transactions     : "
        f"{len(ground_truth):,}"
    )
    print(
        f"  Fraud txns       : "
        f"{int(ground_truth['is_fraud'].sum()):,}"
    )

    # ========================================================
    # STEP 3 — BUILD GROUND-TRUTH WINDOWS
    # ========================================================

    ground_truth_windows = (
        _build_ground_truth_windows(
            ground_truth,
            window_minutes=window_minutes,
        )
    )

    print(
        f"  Fraud windows    : "
        f"{int(ground_truth_windows['fraud_positive'].sum()):,}"
    )

    # ========================================================
    # STEP 4 — ATTACH LABELS
    # ========================================================

    evaluated_base = _attach_ground_truth(
        phase2_windows,
        ground_truth_windows,
    )

    if len(evaluated_base) != len(
        phase2_windows
    ):
        raise RuntimeError(
            "Ground-truth merge changed Phase-2 "
            "window population. "
            f"Before={len(phase2_windows):,}, "
            f"After={len(evaluated_base):,}."
        )

    # ========================================================
    # STEP 5 — TEMPORAL SPLIT
    # ========================================================

    (
        calibration,
        holdout,
        split_time,
    ) = _temporal_split(
        evaluated_base,
        calibration_fraction=(
            calibration_fraction
        ),
    )

    print()
    print("Temporal split:")
    print(
        f"  Split timestamp  : "
        f"{split_time}"
    )
    print(
        f"  Calibration      : "
        f"{len(calibration):,}"
    )
    print(
        f"  Holdout          : "
        f"{len(holdout):,}"
    )

    # ========================================================
    # STEP 6 — CALIBRATE ONLY ON CALIBRATION
    # ========================================================

    print()
    print("Calibrating Phase-2 state thresholds...")

    calibration_result = _calibrate_thresholds(
        calibration
    )

    candidate_threshold = float(
        calibration_result[
            "candidate_score_threshold"
        ]
    )

    verified_threshold = float(
        calibration_result[
            "verified_score_threshold"
        ]
    )

    critical_threshold = float(
        calibration_result[
            "critical_score_threshold"
        ]
    )

    print(
        f"  Candidate threshold : "
        f"{candidate_threshold:.6f}"
    )
    print(
        f"  Verified threshold  : "
        f"{verified_threshold:.6f}"
    )
    print(
        f"  Critical threshold  : "
        f"{critical_threshold:.6f}"
    )

    # ========================================================
    # STEP 7 — FREEZE THRESHOLDS
    # ========================================================

    frozen_calibration = {
        "candidate_score_threshold": (
            candidate_threshold
        ),
        "verified_score_threshold": (
            verified_threshold
        ),
        "critical_score_threshold": (
            critical_threshold
        ),
    }

    # ========================================================
    # STEP 8 — APPLY DETECTOR TO COMPLETE DATASET
    # ========================================================

    evaluated = _apply_detector(
        evaluated_base,
        frozen_calibration,
    )

    # ========================================================
    # STEP 9 — SELECT HOLDOUT USING SAME BOUNDARY
    # ========================================================

    holdout_evaluated = evaluated[
        evaluated["window_start"]
        >= split_time
    ].copy()

    if len(holdout_evaluated) != len(
        holdout
    ):
        raise RuntimeError(
            "Holdout row mismatch after applying "
            "the Phase-2 detector. "
            f"Expected {len(holdout):,}, "
            f"got {len(holdout_evaluated):,}."
        )

    expected_holdout_ids = set(
        holdout["_validation_row_id"]
    )

    actual_holdout_ids = set(
        holdout_evaluated[
            "_validation_row_id"
        ]
    )

    if expected_holdout_ids != actual_holdout_ids:

        missing = (
            expected_holdout_ids
            - actual_holdout_ids
        )

        extra = (
            actual_holdout_ids
            - expected_holdout_ids
        )

        raise RuntimeError(
            "Holdout row identity mismatch. "
            f"Missing={len(missing):,}, "
            f"Extra={len(extra):,}."
        )

    # ========================================================
    # STEP 10 — SCORE PERFORMANCE
    # ========================================================

    ranking_metrics = _calculate_ranking_metrics(
        holdout_evaluated["fraud_positive"],
        holdout_evaluated["spike_score"],
    )

    amount_spearman = _amount_spearman(
        holdout_evaluated
    )

    top_k = _top_k_metrics(
        holdout_evaluated
    )

    # ========================================================
    # STEP 11 — STATE PERFORMANCE
    # ========================================================

    state_metrics = _state_metrics(
        holdout_evaluated
    )

    # ========================================================
    # STEP 12 — EVENT PERFORMANCE
    # ========================================================

    holdout_events = _build_holdout_events(
        holdout_evaluated,
        max_gap_minutes=max_event_gap_minutes,
    )

    # ========================================================
    # STEP 13 — SAVE HOLDOUT WINDOWS
    # ========================================================

    holdout_windows_path = (
        output_dir
        / "phase2_temporal_holdout_windows.csv"
    )

    holdout_evaluated.to_csv(
        holdout_windows_path,
        index=False,
    )

    # ========================================================
    # STEP 14 — SAVE EVENTS
    # ========================================================

    holdout_events_path = (
        output_dir
        / "phase2_temporal_holdout_events.csv"
    )

    holdout_events.to_csv(
        holdout_events_path,
        index=False,
    )

    # ========================================================
    # STEP 15 — SAVE CALIBRATION
    # ========================================================

    calibration_path = (
        output_dir
        / "phase2_state_calibration.json"
    )

    calibration_payload = {
        "calibration_fraction": (
            calibration_fraction
        ),
        "calibration_windows": (
            len(calibration)
        ),
        "calibration_start": str(
            calibration["window_start"].min()
        ),
        "calibration_end": str(
            calibration["window_start"].max()
        ),
        "split_time": str(
            split_time
        ),
        "thresholds": frozen_calibration,
        "calibration_result": calibration_result,
    }

    _save_json(
        calibration_payload,
        calibration_path,
    )

    # ========================================================
    # STEP 16 — BUILD FINAL RESULT
    # ========================================================

    result = {
        "validation": "phase2_temporal_holdout",
        "configuration": {
            "phase2_windows_path": str(
                phase2_windows_path
            ),
            "ground_truth_path": str(
                ground_truth_path
            ),
            "output_dir": str(
                output_dir
            ),
            "calibration_fraction": (
                calibration_fraction
            ),
            "window_minutes": (
                window_minutes
            ),
            "max_event_gap_minutes": (
                max_event_gap_minutes
            ),
        },
        "dataset": {
            "phase2_windows": (
                len(phase2_windows)
            ),
            "ground_truth_transactions": (
                len(ground_truth)
            ),
            "ground_truth_fraud_transactions": (
                int(
                    ground_truth[
                        "is_fraud"
                    ].sum()
                )
            ),
            "ground_truth_fraud_windows": (
                int(
                    ground_truth_windows[
                        "fraud_positive"
                    ].sum()
                )
            ),
        },
        "temporal_split": {
            "calibration_windows": (
                len(calibration)
            ),
            "holdout_windows": (
                len(holdout)
            ),
            "calibration_fraction_actual": (
                _safe_divide(
                    len(calibration),
                    len(evaluated_base),
                )
            ),
            "holdout_fraction_actual": (
                _safe_divide(
                    len(holdout),
                    len(evaluated_base),
                )
            ),
            "calibration_start": str(
                calibration[
                    "window_start"
                ].min()
            ),
            "calibration_end": str(
                calibration[
                    "window_start"
                ].max()
            ),
            "holdout_start": str(
                holdout[
                    "window_start"
                ].min()
            ),
            "holdout_end": str(
                holdout[
                    "window_start"
                ].max()
            ),
            "split_time": str(
                split_time
            ),
        },
        "calibration": {
            "candidate_score_threshold": (
                candidate_threshold
            ),
            "verified_score_threshold": (
                verified_threshold
            ),
            "critical_score_threshold": (
                critical_threshold
            ),
            "details": calibration_result,
        },
        "holdout_score_performance": {
            "roc_auc": (
                ranking_metrics["roc_auc"]
            ),
            "pr_auc": (
                ranking_metrics["pr_auc"]
            ),
            "n": (
                ranking_metrics["n"]
            ),
            "fraud_positive_windows": (
                ranking_metrics["positive"]
            ),
            "non_fraud_windows": (
                ranking_metrics["negative"]
            ),
            "fraud_amount_spearman": (
                amount_spearman
            ),
        },
        "top_k": top_k,
        "state_metrics": state_metrics,
        "events": {
            "event_count": int(
                len(holdout_events)
            ),
            "events_path": str(
                holdout_events_path
            ),
        },
        "artifacts": {
            "holdout_windows": str(
                holdout_windows_path
            ),
            "holdout_events": str(
                holdout_events_path
            ),
            "calibration": str(
                calibration_path
            ),
        },
    }

    # ========================================================
    # STEP 17 — SAVE FINAL JSON
    # ========================================================

    validation_json_path = (
        output_dir
        / "phase2_temporal_holdout_validation.json"
    )

    _save_json(
        result,
        validation_json_path,
    )

    # ========================================================
    # FINAL CONSOLE REPORT
    # ========================================================

    print()
    print("=" * 72)
    print("PHASE 2 TEMPORAL HOLDOUT RESULT")
    print("=" * 72)

    print()
    print("Calibration:")
    print(
        f"  Windows           : "
        f"{len(calibration):,}"
    )

    print()
    print("Holdout:")
    print(
        f"  Windows           : "
        f"{len(holdout):,}"
    )

    print(
        f"  Fraud windows     : "
        f"{ranking_metrics['positive']:,}"
    )

    print()
    print("Score Performance:")

    roc_auc = ranking_metrics[
        "roc_auc"
    ]

    pr_auc = ranking_metrics[
        "pr_auc"
    ]

    if roc_auc is None:
        print(
            "  ROC-AUC           : N/A"
        )
    else:
        print(
            f"  ROC-AUC           : "
            f"{roc_auc:.6f}"
        )

    if pr_auc is None:
        print(
            "  PR-AUC            : N/A"
        )
    else:
        print(
            f"  PR-AUC            : "
            f"{pr_auc:.6f}"
        )

    if amount_spearman is None:
        print(
            "  Amount Spearman   : N/A"
        )
    else:
        print(
            f"  Amount Spearman   : "
            f"{amount_spearman:.6f}"
        )

    print()
    print(
        f"Holdout events      : "
        f"{len(holdout_events):,}"
    )

    print()
    print("State Metrics:")

    for metrics in state_metrics:

        state = metrics[
            "spike_state"
        ]

        print()
        print(state)

        print(
            f"  Windows           : "
            f"{metrics['windows']:,}"
        )

        print(
            f"  Fraud windows     : "
            f"{metrics['fraud_positive_windows']:,}"
        )

        print(
            f"  Fraud recall      : "
            f"{metrics['fraud_window_recall']:.6f}"
        )

        print(
            f"  Precision         : "
            f"{metrics['state_precision']:.6f}"
        )

        print(
            f"  Alert rate        : "
            f"{metrics['alert_rate']:.6f}"
        )

        print(
            f"  Fraud amount cap. : "
            f"{metrics['fraud_amount_capture_rate']:.6f}"
        )

    print()
    print(
        f"Validation JSON     : "
        f"{validation_json_path}"
    )

    print(
        f"Holdout windows CSV : "
        f"{holdout_windows_path}"
    )

    print(
        f"Holdout events CSV  : "
        f"{holdout_events_path}"
    )

    print(
        f"Calibration JSON    : "
        f"{calibration_path}"
    )

    print("=" * 72)

    return result

