"""
FraudSentinel AI
Phase 2 — Temporal / Segment Spike Intelligence

Detector
--------
Converts continuous Phase 2 spike scores and supporting evidence into
operational states:

    NORMAL
        |
        +----> EARLY_WARNING
                  |
                  +----> VERIFIED_FRAUD_SPIKE
                              |
                              +----> CRITICAL_ACTIVE_SPIKE

Design principles
-----------------
1. Do not make EARLY_WARNING unnecessarily strict.
2. EARLY_WARNING is intended to catch sparse / fast fraud.
3. VERIFIED is based on sustained evidence across windows rather than
   requiring multiple transactions inside one 15-minute window.
4. Multiple cards are NOT mandatory for VERIFIED.
5. CRITICAL remains selective because it represents severe active risk.
6. Persistence is merchant-local and gap-aware.
7. No row-by-row iteration is used.
8. The function remains backward compatible with the previous detector API.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ============================================================================
# OPERATIONAL THRESHOLDS
# ============================================================================

# Calibrated Phase 2 score threshold.
#
# Approximately the 95th percentile of the calibration score distribution.
#
# This is intentionally used as the EARLY_WARNING threshold because the
# purpose of early warning is broad anomaly capture rather than confirmation.
PHASE2_EARLY_WARNING_SCORE_THRESHOLD = 57.150510779704376

# VERIFIED uses the same score threshold as EARLY_WARNING.
#
# The difference between the states comes from persistence/evidence rather
# than an unnecessarily higher score threshold.
PHASE2_VERIFIED_SCORE_THRESHOLD = 57.150510779704376

# CRITICAL remains a severe-risk state.
PHASE2_CRITICAL_SCORE_THRESHOLD = 75.15031235576491


# ============================================================================
# EVIDENCE THRESHOLDS
# ============================================================================

# EARLY_WARNING deliberately has NO statistical/temporal evidence gate.
#
# The spike score itself already contains information from the Phase 2
# statistical, temporal and breadth components.
PHASE2_EARLY_WARNING_MIN_STATISTICAL_EVIDENCE = 0.0
PHASE2_EARLY_WARNING_MIN_TEMPORAL_EVIDENCE = 0.0


# VERIFIED is intentionally moderate rather than strict.
#
# Either statistical OR temporal evidence can contribute to verification,
# while persistence provides the main distinction from EARLY_WARNING.
PHASE2_VERIFIED_MIN_STATISTICAL_EVIDENCE = 0.20
PHASE2_VERIFIED_MIN_TEMPORAL_EVIDENCE = 0.10


# CRITICAL requires stronger evidence.
PHASE2_CRITICAL_MIN_STATISTICAL_EVIDENCE = 0.40
PHASE2_CRITICAL_MIN_TEMPORAL_EVIDENCE = 0.20


# ============================================================================
# ACTIVITY / PERSISTENCE PARAMETERS
# ============================================================================

# IMPORTANT:
#
# VERIFIED no longer requires:
#
#     transaction_count >= 2
#
# in the SAME 15-minute window.
#
# A sparse fraud pattern can legitimately contain one transaction per
# window. Persistence across windows is therefore a better signal.
#
# These values remain defined for backward compatibility with older callers.
PHASE2_CANDIDATE_MIN_TRANSACTIONS = 1
PHASE2_VERIFIED_MIN_TRANSACTIONS = 1
PHASE2_CRITICAL_MIN_TRANSACTIONS = 1


# A verified spike should have evidence in at least two merchant windows.
PHASE2_VERIFIED_PERSISTENCE_WINDOWS = 2

# The windows do not have to be continuously dense. A modest temporal span
# is enough to distinguish persistent activity from a single isolated alert.
PHASE2_VERIFIED_PERSISTENCE_MINUTES = 15.0


# CRITICAL should be persistent, but not excessively strict.
PHASE2_CRITICAL_PERSISTENCE_WINDOWS = 2
PHASE2_CRITICAL_PERSISTENCE_MINUTES = 30.0


# Gap after which a merchant's sequence is considered broken.
#
# This is deliberately larger than the 15-minute window size so that
# adjacent activity is not incorrectly fragmented because of normal
# timestamp/window behavior.
PHASE2_MAX_GAP_MINUTES = 30.0


# ============================================================================
# CRITICAL COORDINATION PARAMETERS
# ============================================================================

# Multiple cards remain useful for CRITICAL because the state represents
# stronger evidence of coordinated activity.
PHASE2_CRITICAL_MIN_UNIQUE_CARDS = 2

# Coordination is retained, but the threshold is lower than the previous
# strict configuration.
PHASE2_CRITICAL_MIN_COORDINATION_SCORE = 0.30


# ============================================================================
# STATE NAMES
# ============================================================================

NORMAL = "NORMAL"
EARLY_WARNING = "EARLY_WARNING"
VERIFIED_FRAUD_SPIKE = "VERIFIED_FRAUD_SPIKE"
CRITICAL_ACTIVE_SPIKE = "CRITICAL_ACTIVE_SPIKE"

VALID_STATES = {
    NORMAL,
    EARLY_WARNING,
    VERIFIED_FRAUD_SPIKE,
    CRITICAL_ACTIVE_SPIKE,
}


# ============================================================================
# INTERNAL UTILITIES
# ============================================================================


def _numeric_series(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    """
    Return a finite numeric Series aligned to df.index.

    Missing/non-numeric/infinite values are safely replaced.
    """
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")

    values = pd.to_numeric(df[column], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.fillna(default)

    return values.astype("float64")


def _boolean_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Safely convert a column into boolean values.
    """
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)

    values = df[column]

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)

    numeric = pd.to_numeric(values, errors="coerce")

    if numeric.notna().any():
        return numeric.fillna(0).ne(0)

    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def _timestamp_series(
    df: pd.DataFrame,
    timestamp_column: str,
) -> pd.Series:
    """
    Convert timestamp column to UTC-aware timestamps.

    Invalid timestamps become NaT.
    """
    if timestamp_column not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    return pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
        utc=True,
    )


# ============================================================================
# PERSISTENCE
# ============================================================================


def _calculate_persistence(
    df: pd.DataFrame,
    *,
    merchant_column: str = "merchant_id",
    timestamp_column: str = "window_start",
    signal_column: str = "spike_score",
    signal_threshold: float = PHASE2_EARLY_WARNING_SCORE_THRESHOLD,
    max_gap_minutes: float = PHASE2_MAX_GAP_MINUTES,
) -> pd.DataFrame:
    """
    Calculate merchant-local persistence for score-qualified windows.

    Persistence is defined over windows where:

        spike_score >= signal_threshold

    A sequence continues while:

        same merchant
        AND
        timestamp gap <= max_gap_minutes

    For every row the function returns:

        persistence_windows
        persistence_minutes
        persistence_flag
        sequence_id

    The implementation is fully vectorized.

    No iterrows(), apply(axis=1), or repeated .loc assignment is used.
    """

    n = len(df)

    if n == 0:
        return pd.DataFrame(
            {
                "persistence_windows": pd.Series(
                    dtype="int64",
                    index=df.index,
                ),
                "persistence_minutes": pd.Series(
                    dtype="float64",
                    index=df.index,
                ),
                "persistence_flag": pd.Series(
                    dtype=bool,
                    index=df.index,
                ),
                "sequence_id": pd.Series(
                    dtype="int64",
                    index=df.index,
                ),
            }
        )

    merchant = (
        df[merchant_column]
        if merchant_column in df.columns
        else pd.Series("UNKNOWN", index=df.index)
    )

    timestamp = _timestamp_series(df, timestamp_column)

    score = _numeric_series(df, signal_column)

    # Preserve original positional order rather than relying on the index
    # being unique.
    original_position = np.arange(n, dtype=np.int64)

    work = pd.DataFrame(
        {
            "_merchant": merchant.to_numpy(),
            "_timestamp": timestamp.to_numpy(),
            "_score": score.to_numpy(),
            "_original_position": original_position,
        }
    )

    # Stable ordering is important for deterministic behavior.
    work = work.sort_values(
        ["_merchant", "_timestamp", "_original_position"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Score-qualified windows.
    work["_qualified"] = (
        work["_score"] >= float(signal_threshold)
    )

    # ----------------------------------------------------------------------
    # Merchant-local timestamp gap
    # ----------------------------------------------------------------------

    work["_previous_timestamp"] = work.groupby(
        "_merchant",
        sort=False,
        dropna=False,
    )["_timestamp"].shift(1)

    work["_gap_minutes"] = (
        work["_timestamp"] - work["_previous_timestamp"]
    ).dt.total_seconds() / 60.0

    same_merchant = work["_previous_timestamp"].notna()

    continuous_gap = (
        same_merchant
        & work["_gap_minutes"].notna()
        & (work["_gap_minutes"] >= 0.0)
        & (work["_gap_minutes"] <= float(max_gap_minutes))
    )

    # ----------------------------------------------------------------------
    # Build sequence boundaries
    # ----------------------------------------------------------------------

    previous_qualified = work.groupby(
        "_merchant",
        sort=False,
        dropna=False,
    )["_qualified"].shift(1)

    # A new sequence starts when:
    #
    # 1. current row is qualified AND
    # 2. previous row is not a compatible qualified row.
    #
    sequence_start = (
        work["_qualified"]
        & ~(
            work["_qualified"].shift(1, fill_value=False)
            & previous_qualified.fillna(False)
            & continuous_gap
        )
    )

    # The shift above can cross merchants, so explicitly force the first
    # row of every merchant to start a sequence when qualified.
    merchant_start = work["_merchant"].ne(
        work["_merchant"].shift(1)
    )

    sequence_start = (
        work["_qualified"]
        & (
            merchant_start
            | ~(
                work["_qualified"].shift(1, fill_value=False)
                & continuous_gap
            )
        )
    )

    # Give every row a sequence number.
    work["_sequence_number"] = sequence_start.astype(np.int64).cumsum()

    # Non-qualified rows do not belong to a persistence sequence.
    work["_sequence_key"] = np.where(
        work["_qualified"],
        work["_sequence_number"],
        -1,
    )

    qualified = work["_qualified"]

    # ----------------------------------------------------------------------
    # Sequence size
    # ----------------------------------------------------------------------

    sequence_sizes = (
        work.loc[qualified]
        .groupby("_sequence_key", sort=False)
        .size()
    )

    work["_persistence_windows"] = (
        work["_sequence_key"]
        .map(sequence_sizes)
        .fillna(0)
        .astype(np.int64)
    )

    # ----------------------------------------------------------------------
    # Sequence duration
    # ----------------------------------------------------------------------

    sequence_first_timestamp = (
        work.loc[qualified]
        .groupby("_sequence_key", sort=False)["_timestamp"]
        .transform("min")
    )

    work["_sequence_first_timestamp"] = sequence_first_timestamp

    work["_persistence_minutes"] = (
        work["_timestamp"] - work["_sequence_first_timestamp"]
    ).dt.total_seconds() / 60.0

    work["_persistence_minutes"] = (
        work["_persistence_minutes"]
        .fillna(0.0)
        .clip(lower=0.0)
    )

    # ----------------------------------------------------------------------
    # Persistence flag
    # ----------------------------------------------------------------------

    work["_persistence_flag"] = (
        qualified
        & (
            work["_persistence_windows"]
            >= 2
        )
    )

    # ----------------------------------------------------------------------
    # Restore original order
    # ----------------------------------------------------------------------

    work = work.sort_values(
        "_original_position",
        kind="mergesort",
    )

    result = pd.DataFrame(
        {
            "persistence_windows": work[
                "_persistence_windows"
            ].to_numpy(dtype=np.int64),

            "persistence_minutes": work[
                "_persistence_minutes"
            ].to_numpy(dtype=np.float64),

            "persistence_flag": work[
                "_persistence_flag"
            ].to_numpy(dtype=bool),

            "sequence_id": work[
                "_sequence_number"
            ].to_numpy(dtype=np.int64),
        },
        index=df.index,
    )

    return result


# ============================================================================
# STATE CLASSIFICATION
# ============================================================================


def classify_spike_state(
    df: pd.DataFrame,
    *,
    # ------------------------------------------------------------------
    # Backward-compatible thresholds
    # ------------------------------------------------------------------
    candidate_score_threshold: Optional[float] = None,
    verified_score_threshold: Optional[float] = None,
    critical_score_threshold: Optional[float] = None,

    early_warning_score_threshold: Optional[float] = None,

    # ------------------------------------------------------------------
    # Backward-compatible activity gates
    # ------------------------------------------------------------------
    candidate_min_transactions: int = PHASE2_CANDIDATE_MIN_TRANSACTIONS,
    verified_min_transactions: int = PHASE2_VERIFIED_MIN_TRANSACTIONS,
    critical_min_transactions: int = PHASE2_CRITICAL_MIN_TRANSACTIONS,

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    verified_persistence_windows: int = PHASE2_VERIFIED_PERSISTENCE_WINDOWS,
    critical_persistence_windows: int = PHASE2_CRITICAL_PERSISTENCE_WINDOWS,

    verified_persistence_minutes: float = PHASE2_VERIFIED_PERSISTENCE_MINUTES,
    critical_persistence_minutes: float = PHASE2_CRITICAL_PERSISTENCE_MINUTES,

    max_gap_minutes: float = PHASE2_MAX_GAP_MINUTES,

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------
    early_warning_min_statistical_evidence: float = (
        PHASE2_EARLY_WARNING_MIN_STATISTICAL_EVIDENCE
    ),

    early_warning_min_temporal_evidence: float = (
        PHASE2_EARLY_WARNING_MIN_TEMPORAL_EVIDENCE
    ),

    verified_min_statistical_evidence: float = (
        PHASE2_VERIFIED_MIN_STATISTICAL_EVIDENCE
    ),

    verified_min_temporal_evidence: float = (
        PHASE2_VERIFIED_MIN_TEMPORAL_EVIDENCE
    ),

    critical_min_statistical_evidence: float = (
        PHASE2_CRITICAL_MIN_STATISTICAL_EVIDENCE
    ),

    critical_min_temporal_evidence: float = (
        PHASE2_CRITICAL_MIN_TEMPORAL_EVIDENCE
    ),

    # ------------------------------------------------------------------
    # Critical coordination
    # ------------------------------------------------------------------
    critical_min_unique_cards: int = PHASE2_CRITICAL_MIN_UNIQUE_CARDS,

    critical_min_coordination_score: float = (
        PHASE2_CRITICAL_MIN_COORDINATION_SCORE
    ),

    # ------------------------------------------------------------------
    # Column names
    # ------------------------------------------------------------------
    score_column: str = "spike_score",
    statistical_column: str = "statistical_evidence",
    temporal_column: str = "temporal_evidence",
    transaction_count_column: str = "transaction_count",
    unique_cards_column: str = "unique_cards",
    coordination_column: str = "coordination_score",
    merchant_column: str = "merchant_id",
    timestamp_column: str = "window_start",
) -> pd.DataFrame:
    """
    Classify Phase 2 windows into operational spike states.

    State logic
    -----------

    NORMAL
        score below EARLY_WARNING threshold.

    EARLY_WARNING
        score >= EARLY_WARNING threshold.

        No additional statistical, temporal, activity, persistence,
        card-count or coordination gate is required.

        This intentionally captures sparse/fast anomalies.

    VERIFIED_FRAUD_SPIKE
        score >= verified threshold
        AND
        moderate statistical OR temporal evidence
        AND
        persistence across at least two merchant-local windows.

        No same-window transaction-count requirement is imposed.

    CRITICAL_ACTIVE_SPIKE
        score >= critical threshold
        AND
        stronger statistical evidence
        AND
        stronger temporal evidence
        AND
        persistence
        AND
        multiple cards
        AND
        moderate coordination.

    Priority
    --------
        CRITICAL > VERIFIED > EARLY_WARNING > NORMAL
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if len(df) == 0:
        result = df.copy()

        result["spike_state"] = pd.Series(
            dtype="object",
            index=result.index,
        )

        result["early_warning_flag"] = pd.Series(
            dtype=bool,
            index=result.index,
        )

        result["verified_spike_flag"] = pd.Series(
            dtype=bool,
            index=result.index,
        )

        result["critical_spike_flag"] = pd.Series(
            dtype=bool,
            index=result.index,
        )

        result["candidate_spike_flag"] = pd.Series(
            dtype=bool,
            index=result.index,
        )

        result["persistence_windows"] = pd.Series(
            dtype="int64",
            index=result.index,
        )

        result["persistence_minutes"] = pd.Series(
            dtype="float64",
            index=result.index,
        )

        result["persistence_flag"] = pd.Series(
            dtype=bool,
            index=result.index,
        )

        result["sequence_id"] = pd.Series(
            dtype="int64",
            index=result.index,
        )

        result["state_reason"] = pd.Series(
            dtype="object",
            index=result.index,
        )

        return result

    result = df.copy()

    # ======================================================================
    # Resolve thresholds
    # ======================================================================

    # Old validation code passes candidate_score_threshold.
    #
    # New architecture calls this EARLY_WARNING.
    if early_warning_score_threshold is None:
        if candidate_score_threshold is not None:
            early_warning_score_threshold = float(
                candidate_score_threshold
            )
        else:
            early_warning_score_threshold = (
                PHASE2_EARLY_WARNING_SCORE_THRESHOLD
            )

    if verified_score_threshold is None:
        verified_score_threshold = (
            PHASE2_VERIFIED_SCORE_THRESHOLD
        )

    if critical_score_threshold is None:
        critical_score_threshold = (
            PHASE2_CRITICAL_SCORE_THRESHOLD
        )

    # ======================================================================
    # Input signals
    # ======================================================================

    score = _numeric_series(
        result,
        score_column,
    )

    statistical = _numeric_series(
        result,
        statistical_column,
    ).clip(0.0, 1.0)

    temporal = _numeric_series(
        result,
        temporal_column,
    ).clip(0.0, 1.0)

    transaction_count = _numeric_series(
        result,
        transaction_count_column,
    ).clip(lower=0.0)

    unique_cards = _numeric_series(
        result,
        unique_cards_column,
    ).clip(lower=0.0)

    coordination = _numeric_series(
        result,
        coordination_column,
    ).clip(0.0, 1.0)

    # ======================================================================
    # Persistence
    # ======================================================================

    persistence = _calculate_persistence(
        result,
        merchant_column=merchant_column,
        timestamp_column=timestamp_column,
        signal_column=score_column,
        signal_threshold=float(early_warning_score_threshold),
        max_gap_minutes=float(max_gap_minutes),
    )

    # Align by position/index.
    result["persistence_windows"] = (
        persistence["persistence_windows"].to_numpy()
    )

    result["persistence_minutes"] = (
        persistence["persistence_minutes"].to_numpy()
    )

    result["persistence_flag"] = (
        persistence["persistence_flag"].to_numpy()
    )

    result["sequence_id"] = (
        persistence["sequence_id"].to_numpy()
    )

    persistence_windows = result["persistence_windows"]
    persistence_minutes = result["persistence_minutes"]

    # ======================================================================
    # EARLY WARNING
    # ======================================================================
    #
    # IMPORTANT:
    #
    # This is deliberately simple.
    #
    # The Phase 2 score already combines multiple dimensions of anomaly
    # intelligence. Requiring statistical evidence again here creates
    # unnecessary double-gating.
    #
    # Therefore:
    #
    #     EARLY_WARNING = score >= calibrated threshold
    #

    early_warning = (
        score >= float(early_warning_score_threshold)
    )

    # ======================================================================
    # VERIFIED
    # ======================================================================
    #
    # Verification requires:
    #
    # 1. score is elevated
    # 2. there is at least moderate statistical OR temporal evidence
    # 3. anomaly persists across multiple merchant-local windows
    #
    # No same-window transaction-count gate.
    #
    # No multi-card gate.
    #
    # This is intentionally tolerant of sparse/spread-out fraud.
    #

    moderate_evidence = (
        (statistical >= float(verified_min_statistical_evidence))
        |
        (temporal >= float(verified_min_temporal_evidence))
    )

    verified_persistence = (
        (persistence_windows >= int(verified_persistence_windows))
        |
        (
            (persistence_windows >= 2)
            & (
                persistence_minutes
                >= float(verified_persistence_minutes)
            )
        )
    )

    verified = (
        (score >= float(verified_score_threshold))
        & moderate_evidence
        & verified_persistence
    )

    # ======================================================================
    # CRITICAL
    # ======================================================================
    #
    # Critical deliberately remains more selective.
    #
    # We want strong evidence that this is not merely a single unusual
    # transaction or small anomaly.
    #

    strong_evidence = (
        (statistical >= float(critical_min_statistical_evidence))
        &
        (temporal >= float(critical_min_temporal_evidence))
    )

    critical_persistence = (
        (persistence_windows >= int(critical_persistence_windows))
        |
        (
            (persistence_windows >= 2)
            & (
                persistence_minutes
                >= float(critical_persistence_minutes)
            )
        )
    )

    critical_activity = (
        (unique_cards >= float(critical_min_unique_cards))
        |
        (
            transaction_count >= 2.0
        )
    )

    critical_coordination = (
        coordination
        >= float(critical_min_coordination_score)
    )

    critical = (
        (score >= float(critical_score_threshold))
        & strong_evidence
        & critical_persistence
        & critical_activity
        & critical_coordination
    )

    # ======================================================================
    # STATE PRIORITY
    # ======================================================================
    #
    # CRITICAL > VERIFIED > EARLY_WARNING > NORMAL
    #

    states = np.full(
        len(result),
        NORMAL,
        dtype=object,
    )

    states[early_warning.to_numpy()] = EARLY_WARNING
    states[verified.to_numpy()] = VERIFIED_FRAUD_SPIKE
    states[critical.to_numpy()] = CRITICAL_ACTIVE_SPIKE

    result["spike_state"] = states

    # ======================================================================
    # FLAGS
    # ======================================================================

    result["early_warning_flag"] = (
        result["spike_state"]
        .eq(EARLY_WARNING)
    )

    result["verified_spike_flag"] = (
        result["spike_state"]
        .eq(VERIFIED_FRAUD_SPIKE)
    )

    result["critical_spike_flag"] = (
        result["spike_state"]
        .eq(CRITICAL_ACTIVE_SPIKE)
    )

    # Backward-compatible candidate flag.
    #
    # Existing downstream validation may still expect this column.
    result["candidate_spike_flag"] = (
        result["early_warning_flag"]
    )

    # ======================================================================
    # STATE REASON
    # ======================================================================

    reason = np.full(
        len(result),
        "below_early_warning_threshold",
        dtype=object,
    )

    reason[
        early_warning.to_numpy()
    ] = "elevated_spike_score"

    reason[
        verified.to_numpy()
    ] = (
        "elevated_score_with_sustained_statistical_or_temporal_evidence"
    )

    reason[
        critical.to_numpy()
    ] = (
        "severe_persistent_multi_card_coordinated_spike"
    )

    result["state_reason"] = reason

    # ======================================================================
    # FINAL SANITY / SAFETY INVARIANTS
    # ======================================================================

    # 1. Row count must never change.
    if len(result) != len(df):
        raise RuntimeError(
            "Detector changed row count."
        )

    # 2. Index must remain unchanged.
    if not result.index.equals(df.index):
        raise RuntimeError(
            "Detector changed input index."
        )

    # 3. States must be valid.
    invalid_states = set(
        result["spike_state"].dropna().unique()
    ) - VALID_STATES

    if invalid_states:
        raise RuntimeError(
            f"Invalid spike states generated: {invalid_states}"
        )

    # 4. Numeric outputs must remain finite.
    numeric_columns = [
        "persistence_windows",
        "persistence_minutes",
    ]

    for column in numeric_columns:
        values = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        if not np.isfinite(values.to_numpy()).all():
            raise RuntimeError(
                f"Non-finite detector output in '{column}'."
            )

    # 5. Boolean flags must be internally consistent.
    if not (
        result["candidate_spike_flag"]
        .equals(result["early_warning_flag"])
    ):
        raise RuntimeError(
            "candidate_spike_flag is not aligned with early_warning_flag."
        )

    # 6. State hierarchy invariant.
    #
    # A CRITICAL row must not simultaneously be represented as a lower
    # operational state.
    #
    # The individual evidence flags are still available independently.
    if (
        result["critical_spike_flag"]
        & result["spike_state"].ne(CRITICAL_ACTIVE_SPIKE)
    ).any():
        raise RuntimeError(
            "Critical state hierarchy invariant failed."
        )

    return result


# ============================================================================
# BACKWARD-COMPATIBLE ALIAS
# ============================================================================


def detect_spike_state(
    df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    Backward-compatible alias for classify_spike_state().
    """
    return classify_spike_state(
        df,
        **kwargs,
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "NORMAL",
    "EARLY_WARNING",
    "VERIFIED_FRAUD_SPIKE",
    "CRITICAL_ACTIVE_SPIKE",
    "VALID_STATES",

    "PHASE2_EARLY_WARNING_SCORE_THRESHOLD",
    "PHASE2_VERIFIED_SCORE_THRESHOLD",
    "PHASE2_CRITICAL_SCORE_THRESHOLD",

    "classify_spike_state",
    "detect_spike_state",
]
