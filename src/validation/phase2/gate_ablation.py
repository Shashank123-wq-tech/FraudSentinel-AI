
"""
Phase 2 Gate Ablation Diagnostic
================================

Purpose
-------
Diagnose why VERIFIED_FRAUD_SPIKE and CRITICAL_ACTIVE_SPIKE states
are sparse or unreachable.

This module DOES NOT modify the Phase-2 detector.

It evaluates the detector's current operating gates independently
and cumulatively on the temporal holdout dataset.

The diagnostic answers:

1. How many windows pass the score threshold?
2. How many remain after statistical evidence?
3. How many remain after temporal evidence?
4. How many remain after activity requirements?
5. How many remain after persistence?
6. How many remain after coordination?
7. Which gate causes the largest reduction?
8. How much fraud recall / fraud amount capture is lost at each gate?

The module is intentionally diagnostic-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------

DEFAULT_CANDIDATE_SCORE_THRESHOLD = 57.150510779704376
DEFAULT_VERIFIED_SCORE_THRESHOLD = 57.150510779704376
DEFAULT_CRITICAL_SCORE_THRESHOLD = 75.15031235576491

DEFAULT_VERIFIED_PERSISTENCE_WINDOWS = 2
DEFAULT_CRITICAL_PERSISTENCE_WINDOWS = 2

DEFAULT_VERIFIED_PERSISTENCE_MINUTES = 30.0
DEFAULT_CRITICAL_PERSISTENCE_MINUTES = 45.0

DEFAULT_VERIFIED_MIN_STATISTICAL_EVIDENCE = 0.35
DEFAULT_VERIFIED_MIN_TEMPORAL_EVIDENCE = 0.20

DEFAULT_CRITICAL_MIN_STATISTICAL_EVIDENCE = 0.55
DEFAULT_CRITICAL_MIN_TEMPORAL_EVIDENCE = 0.30

DEFAULT_CRITICAL_MIN_UNIQUE_CARDS = 2
DEFAULT_CRITICAL_MIN_COORDINATION_SCORE = 0.45

DEFAULT_VERIFIED_MIN_TRANSACTIONS = 2
DEFAULT_CRITICAL_MIN_TRANSACTIONS = 2


# ---------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------

BASE_REQUIRED_COLUMNS = [
    "merchant_id",
    "window_start",
    "spike_score",
    "statistical_evidence",
    "temporal_evidence",
    "transaction_count",
    "unique_cards",
    "coordination_score",
]

GROUND_TRUTH_COLUMNS = [
    "fraud_positive_window",
    "actual_fraud_count",
    "actual_fraud_amount",
]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _ensure_datetime(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Ensure a dataframe timestamp column is UTC-aware."""
    x = df.copy()

    x[column] = pd.to_datetime(
        x[column],
        utc=True,
        errors="coerce",
    )

    if x[column].isna().any():
        bad = int(x[column].isna().sum())
        raise ValueError(
            f"Column '{column}' contains {bad} invalid timestamps."
        )

    return x


def _require_columns(
    df: pd.DataFrame,
    columns: List[str],
    name: str = "DataFrame",
) -> None:
    """Validate required columns."""
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


def _ensure_numeric(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """Convert diagnostic columns to numeric."""
    x = df.copy()

    for col in columns:
        x[col] = pd.to_numeric(
            x[col],
            errors="coerce",
        )

    return x


def _finite_or_zero(
    series: pd.Series,
) -> pd.Series:
    """Return finite numeric values, replacing invalid values with zero."""
    x = pd.to_numeric(series, errors="coerce")

    return x.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)


def _safe_rate(
    numerator: float,
    denominator: float,
) -> float:
    """Safe ratio."""
    if denominator <= 0:
        return 0.0

    return float(numerator / denominator)


def _safe_sum(
    series: pd.Series,
) -> float:
    """Safe numeric sum."""
    x = pd.to_numeric(series, errors="coerce")

    return float(
        x.replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(0.0).sum()
    )


# ---------------------------------------------------------------------
# Ground-truth preparation
# ---------------------------------------------------------------------

def prepare_ground_truth(
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize ground-truth columns.

    The temporal holdout validator should already have attached:

        fraud_positive_window
        actual_fraud_count
        actual_fraud_amount

    If only actual_fraud_count / actual_fraud_amount are available,
    fraud_positive_window is derived.
    """

    x = windows.copy()

    if "actual_fraud_count" not in x.columns:
        if "fraud_count" in x.columns:
            x["actual_fraud_count"] = x["fraud_count"]
        else:
            x["actual_fraud_count"] = 0.0

    if "actual_fraud_amount" not in x.columns:
        if "fraud_amount" in x.columns:
            x["actual_fraud_amount"] = x["fraud_amount"]
        else:
            x["actual_fraud_amount"] = 0.0

    x["actual_fraud_count"] = _finite_or_zero(
        x["actual_fraud_count"]
    )

    x["actual_fraud_amount"] = _finite_or_zero(
        x["actual_fraud_amount"]
    )

    if "fraud_positive_window" not in x.columns:
        x["fraud_positive_window"] = (
            x["actual_fraud_count"] > 0
        ).astype(int)
    else:
        x["fraud_positive_window"] = (
            pd.to_numeric(
                x["fraud_positive_window"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    return x


# ---------------------------------------------------------------------
# Persistence calculation
# ---------------------------------------------------------------------

def calculate_persistence(
    windows: pd.DataFrame,
    max_gap_minutes: float = 30.0,
) -> pd.DataFrame:
    """
    Calculate merchant-local persistence.

    Persistence is calculated only from previous qualifying windows.

    A gap larger than max_gap_minutes breaks the sequence.

    Outputs
    -------
    persistence_windows
        Number of consecutive previous active windows, including the
        current window.

    persistence_minutes
        Elapsed time between the first and current active window in
        the persistence chain.

    previous_window_start
        Previous merchant-local window timestamp.

    gap_minutes
        Time gap from previous merchant window.
    """

    x = windows.copy()

    _require_columns(
        x,
        ["merchant_id", "window_start"],
        "windows",
    )

    x = _ensure_datetime(x, "window_start")

    x["_original_order"] = np.arange(len(x))

    x = x.sort_values(
        ["merchant_id", "window_start"],
        kind="mergesort",
    ).reset_index(drop=True)

    previous = (
        x.groupby("merchant_id", sort=False)["window_start"]
        .shift(1)
    )

    x["previous_window_start"] = previous

    gap_minutes = (
        x["window_start"] - previous
    ).dt.total_seconds() / 60.0

    x["gap_minutes"] = gap_minutes

    contiguous = (
        previous.notna()
        & gap_minutes.notna()
        & (gap_minutes <= max_gap_minutes)
        & (gap_minutes >= 0)
    )

    # Every observed merchant window begins a new local sequence.
    sequence_id = (
        (~contiguous)
        .groupby(x["merchant_id"])
        .cumsum()
    )

    x["_sequence_id"] = sequence_id

    x["persistence_windows"] = (
        x.groupby(
            ["merchant_id", "_sequence_id"],
            sort=False,
        )
        .cumcount()
        + 1
    )

    sequence_start = (
        x.groupby(
            ["merchant_id", "_sequence_id"],
            sort=False,
        )["window_start"]
        .transform("first")
    )

    x["persistence_minutes"] = (
        x["window_start"] - sequence_start
    ).dt.total_seconds() / 60.0

    # The first window in a sequence has 0 elapsed persistence minutes.
    x["persistence_minutes"] = (
        x["persistence_minutes"]
        .clip(lower=0)
        .fillna(0.0)
    )

    x["persistence_windows"] = (
        pd.to_numeric(
            x["persistence_windows"],
            errors="coerce",
        )
        .fillna(1)
        .astype(int)
    )

    x = (
        x.sort_values("_original_order", kind="mergesort")
        .drop(
            columns=[
                "_original_order",
                "_sequence_id",
            ]
        )
        .reset_index(drop=True)
    )

    return x


# ---------------------------------------------------------------------
# Individual gate construction
# ---------------------------------------------------------------------

def build_verified_gates(
    windows: pd.DataFrame,
    verified_score_threshold: float,
    verified_min_statistical_evidence: float,
    verified_min_temporal_evidence: float,
    verified_persistence_windows: int,
    verified_persistence_minutes: float,
    verified_min_transactions: int,
) -> pd.DataFrame:
    """
    Build individual VERIFIED gates.

    Each gate is kept separately so that ablation can identify
    the bottleneck.
    """

    x = windows.copy()

    score = _finite_or_zero(x["spike_score"])
    stat = _finite_or_zero(x["statistical_evidence"])
    temporal = _finite_or_zero(x["temporal_evidence"])
    transactions = _finite_or_zero(x["transaction_count"])

    persistence_windows = _finite_or_zero(
        x["persistence_windows"]
    )

    persistence_minutes = _finite_or_zero(
        x["persistence_minutes"]
    )

    x["verified_gate_score"] = (
        score >= verified_score_threshold
    )

    x["verified_gate_statistical"] = (
        stat >= verified_min_statistical_evidence
    )

    x["verified_gate_temporal"] = (
        temporal >= verified_min_temporal_evidence
    )

    x["verified_gate_activity"] = (
        transactions >= verified_min_transactions
    )

    x["verified_gate_persistence_windows"] = (
        persistence_windows >= verified_persistence_windows
    )

    x["verified_gate_persistence_minutes"] = (
        persistence_minutes >= verified_persistence_minutes
    )

    x["verified_gate_persistence"] = (
        x["verified_gate_persistence_windows"]
        & x["verified_gate_persistence_minutes"]
    )

    x["verified_all_gates"] = (
        x["verified_gate_score"]
        & x["verified_gate_statistical"]
        & x["verified_gate_temporal"]
        & x["verified_gate_activity"]
        & x["verified_gate_persistence"]
    )

    return x


def build_critical_gates(
    windows: pd.DataFrame,
    critical_score_threshold: float,
    critical_min_statistical_evidence: float,
    critical_min_temporal_evidence: float,
    critical_persistence_windows: int,
    critical_persistence_minutes: float,
    critical_min_transactions: int,
    critical_min_unique_cards: int,
    critical_min_coordination_score: float,
) -> pd.DataFrame:
    """
    Build individual CRITICAL gates.
    """

    x = windows.copy()

    score = _finite_or_zero(x["spike_score"])
    stat = _finite_or_zero(x["statistical_evidence"])
    temporal = _finite_or_zero(x["temporal_evidence"])

    transactions = _finite_or_zero(
        x["transaction_count"]
    )

    unique_cards = _finite_or_zero(
        x["unique_cards"]
    )

    coordination = _finite_or_zero(
        x["coordination_score"]
    )

    persistence_windows = _finite_or_zero(
        x["persistence_windows"]
    )

    persistence_minutes = _finite_or_zero(
        x["persistence_minutes"]
    )

    x["critical_gate_score"] = (
        score >= critical_score_threshold
    )

    x["critical_gate_statistical"] = (
        stat >= critical_min_statistical_evidence
    )

    x["critical_gate_temporal"] = (
        temporal >= critical_min_temporal_evidence
    )

    x["critical_gate_activity"] = (
        transactions >= critical_min_transactions
    )

    x["critical_gate_unique_cards"] = (
        unique_cards >= critical_min_unique_cards
    )

    x["critical_gate_coordination"] = (
        coordination >= critical_min_coordination_score
    )

    x["critical_gate_persistence_windows"] = (
        persistence_windows >= critical_persistence_windows
    )

    x["critical_gate_persistence_minutes"] = (
        persistence_minutes >= critical_persistence_minutes
    )

    x["critical_gate_persistence"] = (
        x["critical_gate_persistence_windows"]
        & x["critical_gate_persistence_minutes"]
    )

    x["critical_all_gates"] = (
        x["critical_gate_score"]
        & x["critical_gate_statistical"]
        & x["critical_gate_temporal"]
        & x["critical_gate_activity"]
        & x["critical_gate_unique_cards"]
        & x["critical_gate_coordination"]
        & x["critical_gate_persistence"]
    )

    return x


# ---------------------------------------------------------------------
# Gate sequence evaluation
# ---------------------------------------------------------------------

@dataclass
class GateResult:
    """Result for one cumulative gate stage."""

    state: str
    step: int
    gate: str
    windows_selected: int
    alert_rate: float
    fraud_positive_windows: int
    fraud_window_recall: float
    actual_fraud_count: float
    actual_fraud_amount: float
    fraud_amount_capture_rate: float
    precision: float
    previous_windows: Optional[int] = None
    windows_removed: Optional[int] = None
    reduction_rate: Optional[float] = None


def _evaluate_mask(
    df: pd.DataFrame,
    mask: pd.Series,
    state: str,
    step: int,
    gate: str,
    previous_windows: Optional[int],
) -> GateResult:
    """Evaluate one cumulative gate mask."""

    mask = mask.fillna(False).astype(bool)

    selected = df.loc[mask]

    total_windows = len(df)

    windows_selected = int(len(selected))

    fraud_positive_total = int(
        df["fraud_positive_window"].sum()
    )

    fraud_positive_selected = int(
        selected["fraud_positive_window"].sum()
    )

    actual_fraud_count = _safe_sum(
        selected["actual_fraud_count"]
    )

    actual_fraud_amount = _safe_sum(
        selected["actual_fraud_amount"]
    )

    total_fraud_amount = _safe_sum(
        df["actual_fraud_amount"]
    )

    recall = _safe_rate(
        fraud_positive_selected,
        fraud_positive_total,
    )

    precision = _safe_rate(
        fraud_positive_selected,
        windows_selected,
    )

    amount_capture = _safe_rate(
        actual_fraud_amount,
        total_fraud_amount,
    )

    alert_rate = _safe_rate(
        windows_selected,
        total_windows,
    )

    windows_removed = None
    reduction_rate = None

    if previous_windows is not None:
        windows_removed = (
            previous_windows - windows_selected
        )

        reduction_rate = _safe_rate(
            windows_removed,
            previous_windows,
        )

    return GateResult(
        state=state,
        step=step,
        gate=gate,
        windows_selected=windows_selected,
        alert_rate=alert_rate,
        fraud_positive_windows=fraud_positive_selected,
        fraud_window_recall=recall,
        actual_fraud_count=actual_fraud_count,
        actual_fraud_amount=actual_fraud_amount,
        fraud_amount_capture_rate=amount_capture,
        precision=precision,
        previous_windows=previous_windows,
        windows_removed=windows_removed,
        reduction_rate=reduction_rate,
    )


def _results_to_dataframe(
    results: List[GateResult],
) -> pd.DataFrame:
    """Convert gate results into a dataframe."""

    return pd.DataFrame(
        [
            {
                "state": r.state,
                "step": r.step,
                "gate": r.gate,
                "windows_selected": r.windows_selected,
                "alert_rate": r.alert_rate,
                "fraud_positive_windows": r.fraud_positive_windows,
                "fraud_window_recall": r.fraud_window_recall,
                "actual_fraud_count": r.actual_fraud_count,
                "actual_fraud_amount": r.actual_fraud_amount,
                "fraud_amount_capture_rate": r.fraud_amount_capture_rate,
                "precision": r.precision,
                "previous_windows": r.previous_windows,
                "windows_removed": r.windows_removed,
                "reduction_rate": r.reduction_rate,
            }
            for r in results
        ]
    )


# ---------------------------------------------------------------------
# Verified ablation
# ---------------------------------------------------------------------

def run_verified_ablation(
    windows: pd.DataFrame,
    verified_score_threshold: float = DEFAULT_VERIFIED_SCORE_THRESHOLD,
    verified_min_statistical_evidence: float = (
        DEFAULT_VERIFIED_MIN_STATISTICAL_EVIDENCE
    ),
    verified_min_temporal_evidence: float = (
        DEFAULT_VERIFIED_MIN_TEMPORAL_EVIDENCE
    ),
    verified_persistence_windows: int = (
        DEFAULT_VERIFIED_PERSISTENCE_WINDOWS
    ),
    verified_persistence_minutes: float = (
        DEFAULT_VERIFIED_PERSISTENCE_MINUTES
    ),
    verified_min_transactions: int = (
        DEFAULT_VERIFIED_MIN_TRANSACTIONS
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run cumulative VERIFIED gate ablation.

    Returns
    -------
    diagnostic_windows
        Input dataframe with individual gate columns.

    results
        One row per cumulative gate.
    """

    x = prepare_ground_truth(windows)

    x = calculate_persistence(x)

    x = build_verified_gates(
        x,
        verified_score_threshold=verified_score_threshold,
        verified_min_statistical_evidence=(
            verified_min_statistical_evidence
        ),
        verified_min_temporal_evidence=(
            verified_min_temporal_evidence
        ),
        verified_persistence_windows=(
            verified_persistence_windows
        ),
        verified_persistence_minutes=(
            verified_persistence_minutes
        ),
        verified_min_transactions=(
            verified_min_transactions
        ),
    )

    masks = [
        (
            "Score threshold",
            x["verified_gate_score"],
        ),
        (
            "Statistical evidence",
            x["verified_gate_statistical"],
        ),
        (
            "Temporal evidence",
            x["verified_gate_temporal"],
        ),
        (
            "Activity",
            x["verified_gate_activity"],
        ),
        (
            "Persistence windows",
            x["verified_gate_persistence_windows"],
        ),
        (
            "Persistence minutes",
            x["verified_gate_persistence_minutes"],
        ),
    ]

    cumulative = pd.Series(
        True,
        index=x.index,
    )

    results: List[GateResult] = []

    previous_windows = None

    for step, (name, mask) in enumerate(
        masks,
        start=1,
    ):
        cumulative = cumulative & mask

        result = _evaluate_mask(
            x,
            cumulative,
            state="VERIFIED",
            step=step,
            gate=name,
            previous_windows=previous_windows,
        )

        results.append(result)

        previous_windows = result.windows_selected

    return (
        x,
        _results_to_dataframe(results),
    )


# ---------------------------------------------------------------------
# Critical ablation
# ---------------------------------------------------------------------

def run_critical_ablation(
    windows: pd.DataFrame,
    critical_score_threshold: float = DEFAULT_CRITICAL_SCORE_THRESHOLD,
    critical_min_statistical_evidence: float = (
        DEFAULT_CRITICAL_MIN_STATISTICAL_EVIDENCE
    ),
    critical_min_temporal_evidence: float = (
        DEFAULT_CRITICAL_MIN_TEMPORAL_EVIDENCE
    ),
    critical_persistence_windows: int = (
        DEFAULT_CRITICAL_PERSISTENCE_WINDOWS
    ),
    critical_persistence_minutes: float = (
        DEFAULT_CRITICAL_PERSISTENCE_MINUTES
    ),
    critical_min_transactions: int = (
        DEFAULT_CRITICAL_MIN_TRANSACTIONS
    ),
    critical_min_unique_cards: int = (
        DEFAULT_CRITICAL_MIN_UNIQUE_CARDS
    ),
    critical_min_coordination_score: float = (
        DEFAULT_CRITICAL_MIN_COORDINATION_SCORE
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run cumulative CRITICAL gate ablation.

    Returns
    -------
    diagnostic_windows
        Input dataframe with individual gate columns.

    results
        One row per cumulative gate.
    """

    x = prepare_ground_truth(windows)

    x = calculate_persistence(x)

    x = build_critical_gates(
        x,
        critical_score_threshold=critical_score_threshold,
        critical_min_statistical_evidence=(
            critical_min_statistical_evidence
        ),
        critical_min_temporal_evidence=(
            critical_min_temporal_evidence
        ),
        critical_persistence_windows=(
            critical_persistence_windows
        ),
        critical_persistence_minutes=(
            critical_persistence_minutes
        ),
        critical_min_transactions=(
            critical_min_transactions
        ),
        critical_min_unique_cards=(
            critical_min_unique_cards
        ),
        critical_min_coordination_score=(
            critical_min_coordination_score
        ),
    )

    masks = [
        (
            "Score threshold",
            x["critical_gate_score"],
        ),
        (
            "Statistical evidence",
            x["critical_gate_statistical"],
        ),
        (
            "Temporal evidence",
            x["critical_gate_temporal"],
        ),
        (
            "Activity",
            x["critical_gate_activity"],
        ),
        (
            "Unique cards",
            x["critical_gate_unique_cards"],
        ),
        (
            "Coordination",
            x["critical_gate_coordination"],
        ),
        (
            "Persistence windows",
            x["critical_gate_persistence_windows"],
        ),
        (
            "Persistence minutes",
            x["critical_gate_persistence_minutes"],
        ),
    ]

    cumulative = pd.Series(
        True,
        index=x.index,
    )

    results: List[GateResult] = []

    previous_windows = None

    for step, (name, mask) in enumerate(
        masks,
        start=1,
    ):
        cumulative = cumulative & mask

        result = _evaluate_mask(
            x,
            cumulative,
            state="CRITICAL",
            step=step,
            gate=name,
            previous_windows=previous_windows,
        )

        results.append(result)

        previous_windows = result.windows_selected

    return (
        x,
        _results_to_dataframe(results),
    )


# ---------------------------------------------------------------------
# Independent gate analysis
# ---------------------------------------------------------------------

def run_independent_gate_analysis(
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate every gate independently.

    This is different from cumulative ablation.

    It answers:

        "If this gate were the ONLY requirement,
         how many windows would pass?"

    This is useful for finding gates that are intrinsically
    extremely restrictive.
    """

    x = prepare_ground_truth(windows)

    x = calculate_persistence(x)

    gate_columns = [
        "verified_gate_score",
        "verified_gate_statistical",
        "verified_gate_temporal",
        "verified_gate_activity",
        "verified_gate_persistence_windows",
        "verified_gate_persistence_minutes",
        "verified_gate_persistence",
        "critical_gate_score",
        "critical_gate_statistical",
        "critical_gate_temporal",
        "critical_gate_activity",
        "critical_gate_unique_cards",
        "critical_gate_coordination",
        "critical_gate_persistence_windows",
        "critical_gate_persistence_minutes",
        "critical_gate_persistence",
    ]

    # Build both gate sets using defaults.
    x = build_verified_gates(
        x,
        verified_score_threshold=DEFAULT_VERIFIED_SCORE_THRESHOLD,
        verified_min_statistical_evidence=(
            DEFAULT_VERIFIED_MIN_STATISTICAL_EVIDENCE
        ),
        verified_min_temporal_evidence=(
            DEFAULT_VERIFIED_MIN_TEMPORAL_EVIDENCE
        ),
        verified_persistence_windows=(
            DEFAULT_VERIFIED_PERSISTENCE_WINDOWS
        ),
        verified_persistence_minutes=(
            DEFAULT_VERIFIED_PERSISTENCE_MINUTES
        ),
        verified_min_transactions=(
            DEFAULT_VERIFIED_MIN_TRANSACTIONS
        ),
    )

    x = build_critical_gates(
        x,
        critical_score_threshold=DEFAULT_CRITICAL_SCORE_THRESHOLD,
        critical_min_statistical_evidence=(
            DEFAULT_CRITICAL_MIN_STATISTICAL_EVIDENCE
        ),
        critical_min_temporal_evidence=(
            DEFAULT_CRITICAL_MIN_TEMPORAL_EVIDENCE
        ),
        critical_persistence_windows=(
            DEFAULT_CRITICAL_PERSISTENCE_WINDOWS
        ),
        critical_persistence_minutes=(
            DEFAULT_CRITICAL_PERSISTENCE_MINUTES
        ),
        critical_min_transactions=(
            DEFAULT_CRITICAL_MIN_TRANSACTIONS
        ),
        critical_min_unique_cards=(
            DEFAULT_CRITICAL_MIN_UNIQUE_CARDS
        ),
        critical_min_coordination_score=(
            DEFAULT_CRITICAL_MIN_COORDINATION_SCORE
        ),
    )

    rows = []

    for column in gate_columns:
        if column not in x.columns:
            continue

        mask = x[column].fillna(False).astype(bool)

        selected = x.loc[mask]

        rows.append(
            {
                "gate": column,
                "windows_selected": int(mask.sum()),
                "alert_rate": _safe_rate(
                    int(mask.sum()),
                    len(x),
                ),
                "fraud_positive_windows": int(
                    selected["fraud_positive_window"].sum()
                ),
                "fraud_window_recall": _safe_rate(
                    selected["fraud_positive_window"].sum(),
                    x["fraud_positive_window"].sum(),
                ),
                "actual_fraud_count": _safe_sum(
                    selected["actual_fraud_count"]
                ),
                "actual_fraud_amount": _safe_sum(
                    selected["actual_fraud_amount"]
                ),
                "fraud_amount_capture_rate": _safe_rate(
                    _safe_sum(
                        selected["actual_fraud_amount"]
                    ),
                    _safe_sum(
                        x["actual_fraud_amount"]
                    ),
                ),
                "precision": _safe_rate(
                    selected["fraud_positive_window"].sum(),
                    mask.sum(),
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Threshold-only analysis
# ---------------------------------------------------------------------

def run_score_threshold_analysis(
    windows: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Evaluate score-only thresholds.

    This helps separate the quality of the score from the quality
    of the state gates.
    """

    x = prepare_ground_truth(windows)

    if thresholds is None:
        thresholds = [
            50.0,
            55.0,
            57.150510779704376,
            60.0,
            65.0,
            70.0,
            75.15031235576491,
            80.0,
        ]

    score = _finite_or_zero(x["spike_score"])

    rows = []

    for threshold in thresholds:
        mask = score >= threshold

        selected = x.loc[mask]

        rows.append(
            {
                "threshold": float(threshold),
                "windows_selected": int(mask.sum()),
                "alert_rate": _safe_rate(
                    mask.sum(),
                    len(x),
                ),
                "fraud_positive_windows": int(
                    selected["fraud_positive_window"].sum()
                ),
                "fraud_window_recall": _safe_rate(
                    selected["fraud_positive_window"].sum(),
                    x["fraud_positive_window"].sum(),
                ),
                "actual_fraud_count": _safe_sum(
                    selected["actual_fraud_count"]
                ),
                "actual_fraud_amount": _safe_sum(
                    selected["actual_fraud_amount"]
                ),
                "fraud_amount_capture_rate": _safe_rate(
                    _safe_sum(
                        selected["actual_fraud_amount"]
                    ),
                    _safe_sum(
                        x["actual_fraud_amount"]
                    ),
                ),
                "precision": _safe_rate(
                    selected["fraud_positive_window"].sum(),
                    mask.sum(),
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Complete diagnostic
# ---------------------------------------------------------------------

def run_gate_ablation(
    windows: pd.DataFrame,
    verified_score_threshold: float = DEFAULT_VERIFIED_SCORE_THRESHOLD,
    critical_score_threshold: float = DEFAULT_CRITICAL_SCORE_THRESHOLD,
    verified_min_statistical_evidence: float = (
        DEFAULT_VERIFIED_MIN_STATISTICAL_EVIDENCE
    ),
    verified_min_temporal_evidence: float = (
        DEFAULT_VERIFIED_MIN_TEMPORAL_EVIDENCE
    ),
    critical_min_statistical_evidence: float = (
        DEFAULT_CRITICAL_MIN_STATISTICAL_EVIDENCE
    ),
    critical_min_temporal_evidence: float = (
        DEFAULT_CRITICAL_MIN_TEMPORAL_EVIDENCE
    ),
    verified_persistence_windows: int = (
        DEFAULT_VERIFIED_PERSISTENCE_WINDOWS
    ),
    critical_persistence_windows: int = (
        DEFAULT_CRITICAL_PERSISTENCE_WINDOWS
    ),
    verified_persistence_minutes: float = (
        DEFAULT_VERIFIED_PERSISTENCE_MINUTES
    ),
    critical_persistence_minutes: float = (
        DEFAULT_CRITICAL_PERSISTENCE_MINUTES
    ),
    verified_min_transactions: int = (
        DEFAULT_VERIFIED_MIN_TRANSACTIONS
    ),
    critical_min_transactions: int = (
        DEFAULT_CRITICAL_MIN_TRANSACTIONS
    ),
    critical_min_unique_cards: int = (
        DEFAULT_CRITICAL_MIN_UNIQUE_CARDS
    ),
    critical_min_coordination_score: float = (
        DEFAULT_CRITICAL_MIN_COORDINATION_SCORE
    ),
) -> Dict[str, pd.DataFrame]:
    """
    Run the complete Phase-2 gate-ablation diagnostic.

    Returns
    -------
    Dictionary containing:

        verified_cumulative
        critical_cumulative
        independent
        score_thresholds
        diagnostic_windows
    """

    verified_windows, verified_cumulative = (
        run_verified_ablation(
            windows,
            verified_score_threshold=verified_score_threshold,
            verified_min_statistical_evidence=(
                verified_min_statistical_evidence
            ),
            verified_min_temporal_evidence=(
                verified_min_temporal_evidence
            ),
            verified_persistence_windows=(
                verified_persistence_windows
            ),
            verified_persistence_minutes=(
                verified_persistence_minutes
            ),
            verified_min_transactions=(
                verified_min_transactions
            ),
        )
    )

    critical_windows, critical_cumulative = (
        run_critical_ablation(
            windows,
            critical_score_threshold=critical_score_threshold,
            critical_min_statistical_evidence=(
                critical_min_statistical_evidence
            ),
            critical_min_temporal_evidence=(
                critical_min_temporal_evidence
            ),
            critical_persistence_windows=(
                critical_persistence_windows
            ),
            critical_persistence_minutes=(
                critical_persistence_minutes
            ),
            critical_min_transactions=(
                critical_min_transactions
            ),
            critical_min_unique_cards=(
                critical_min_unique_cards
            ),
            critical_min_coordination_score=(
                critical_min_coordination_score
            ),
        )
    )

    independent = run_independent_gate_analysis(
        windows
    )

    score_thresholds = run_score_threshold_analysis(
        windows
    )

    # Use critical_windows because it contains both persistence
    # and all critical gate diagnostics.
    diagnostic_windows = critical_windows.copy()

    # Add verified gate columns if they are not already present.
    verified_gate_columns = [
        c
        for c in verified_windows.columns
        if c.startswith("verified_gate")
    ]

    for col in verified_gate_columns:
        if col not in diagnostic_windows.columns:
            diagnostic_windows[col] = (
                verified_windows[col].values
            )

    diagnostic_windows = diagnostic_windows.reset_index(
        drop=True
    )

    return {
        "verified_cumulative": verified_cumulative,
        "critical_cumulative": critical_cumulative,
        "independent": independent,
        "score_thresholds": score_thresholds,
        "diagnostic_windows": diagnostic_windows,
    }


# ---------------------------------------------------------------------
# Compact summary
# ---------------------------------------------------------------------

def build_gate_summary(
    verified_cumulative: pd.DataFrame,
    critical_cumulative: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact bottleneck summary.

    The largest reduction_rate identifies the strongest
    cumulative bottleneck.
    """

    frames = []

    for df in [
        verified_cumulative,
        critical_cumulative,
    ]:
        if df is None or df.empty:
            continue

        temp = df.copy()

        temp["bottleneck_score"] = (
            temp["reduction_rate"]
            .fillna(0.0)
        )

        frames.append(temp)

    if not frames:
        return pd.DataFrame()

    summary = pd.concat(
        frames,
        ignore_index=True,
    )

    return summary.sort_values(
        [
            "state",
            "bottleneck_score",
        ],
        ascending=[True, False],
    ).reset_index(drop=True)


__all__ = [
    "run_gate_ablation",
    "run_verified_ablation",
    "run_critical_ablation",
    "run_independent_gate_analysis",
    "run_score_threshold_analysis",
    "build_gate_summary",
    "calculate_persistence",
    "prepare_ground_truth",
]



