from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FusionConfig
from .decision import (
    assign_alert_flag,
    assign_priority,
    assign_response_action,
    assign_risk_band,
)
from .impact import (
    calculate_fused_expected_exposure,
    calculate_transaction_expected_loss,
)
from .normalization import (
    normalize_phase2_score,
    phase2_confidence,
    probability_to_score,
    state_multiplier,
)


# =====================================================================
# PHASE 2 PREPARATION
# =====================================================================

def _prepare_phase2_windows(
    phase2_windows: pd.DataFrame,
    config: FusionConfig,
) -> pd.DataFrame:
    """
    Prepare Phase 2 windows for point-in-time transaction alignment.

    Phase 2 is temporal intelligence. The transaction-level Phase 1
    risk remains the primary transaction signal, while Phase 2 provides
    merchant-level temporal context.
    """

    windows = phase2_windows.copy()

    required_columns = [
        "merchant_id",
        "window_start",
        "window_end",
        "spike_score",
        "spike_state",
    ]

    missing = [
        column
        for column in required_columns
        if column not in windows.columns
    ]

    if missing:
        raise ValueError(
            "Phase 2 windows are missing required columns: "
            + ", ".join(missing)
        )

    # ---------------------------------------------------------------
    # Numeric normalization
    # ---------------------------------------------------------------

    numeric_columns = [
        "spike_score",
        "expected_fraud_count",
        "expected_fraud_amount",
        "statistical_evidence",
        "financial_evidence",
        "temporal_evidence",
        "breadth_score",
        "coordination_score",
        "tas",
        "fas",
        "materiality",
        "history_confidence",
        "risk_rate",
        "transaction_count",
        "total_amount",
        "unique_cards",
        "high_risk_count",
        "new_cards",
        "new_card_rate",
        "history_count",
        "ewma_value",
        "ewma_residual",
        "cusum_value",
        "cusum_signal",
        "acceleration",
    ]

    for column in numeric_columns:
        if column in windows.columns:
            windows[column] = pd.to_numeric(
                windows[column],
                errors="coerce",
            ).fillna(0.0)

    # ---------------------------------------------------------------
    # Timestamp normalization
    # ---------------------------------------------------------------

    windows["window_start"] = pd.to_datetime(
        windows["window_start"],
        errors="coerce",
        utc=True,
    )

    windows["window_end"] = pd.to_datetime(
        windows["window_end"],
        errors="coerce",
        utc=True,
    )

    if windows["window_start"].isna().any():
        raise ValueError(
            "Phase 2 contains invalid window_start timestamps."
        )

    # ---------------------------------------------------------------
    # Phase 2 score
    # ---------------------------------------------------------------

    windows["phase2_score"] = normalize_phase2_score(
        windows["spike_score"]
    )

    # ---------------------------------------------------------------
    # Phase 2 confidence
    # ---------------------------------------------------------------

    windows["phase2_confidence"] = phase2_confidence(
        windows,
        config,
    )

    windows["phase2_confidence"] = (
        pd.to_numeric(
            windows["phase2_confidence"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    # ---------------------------------------------------------------
    # State multiplier
    # ---------------------------------------------------------------

    windows["phase2_state_multiplier"] = state_multiplier(
        windows["spike_state"],
        config,
    )

    windows["phase2_state_multiplier"] = (
        pd.to_numeric(
            windows["phase2_state_multiplier"],
            errors="coerce",
        )
        .fillna(config.state_multiplier_normal)
    )

    # ---------------------------------------------------------------
    # Keep required + contextual columns
    # ---------------------------------------------------------------

    keep_columns = [
        "merchant_id",
        "window_start",
        "window_end",
        "phase2_score",
        "spike_state",
        "phase2_confidence",
        "phase2_state_multiplier",

        # Phase 2 evidence
        "expected_fraud_count",
        "expected_fraud_amount",
        "statistical_evidence",
        "financial_evidence",
        "temporal_evidence",
        "breadth_score",
        "coordination_score",
        "tas",
        "fas",
        "materiality",

        # Window context
        "risk_rate",
        "transaction_count",
        "total_amount",
        "unique_cards",
        "high_risk_count",
        "new_cards",
        "new_card_rate",
        "history_count",
        "baseline_ready",

        # Temporal context
        "ewma_value",
        "ewma_residual",
        "cusum_value",
        "cusum_signal",
        "acceleration",
    ]

    keep_columns = [
        column
        for column in keep_columns
        if column in windows.columns
    ]

    windows = windows[keep_columns].copy()

    # ---------------------------------------------------------------
    # Remove duplicate merchant/window keys
    # ---------------------------------------------------------------

    windows = (
        windows
        .sort_values(
            ["merchant_id", "window_start"],
            kind="mergesort",
        )
        .drop_duplicates(
            subset=["merchant_id", "window_start"],
            keep="last",
        )
    )

    return windows


# =====================================================================
# POINT-IN-TIME TRANSACTION ALIGNMENT
# =====================================================================

def _align_transactions_to_windows(
    phase1: pd.DataFrame,
    phase2_windows: pd.DataFrame,
    config: FusionConfig,
) -> pd.DataFrame:
    """
    Point-in-time alignment between Phase 1 transactions and Phase 2
    merchant windows.

    A Phase 2 window may influence a transaction only when:

        phase2.window_start <= transaction.timestamp

    and the Phase 2 window is not older than the configured maximum
    window age.

    This prevents future temporal intelligence from leaking backward
    into transaction-level risk.
    """

    transactions = phase1.copy()

    required_columns = [
        "transaction_id",
        "timestamp",
        "merchant_id",
        "amount",
        "fraud_probability",
    ]

    missing = [
        column
        for column in required_columns
        if column not in transactions.columns
    ]

    if missing:
        raise ValueError(
            "Phase 1 data is missing required columns: "
            + ", ".join(missing)
        )

    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"],
        errors="coerce",
        utc=True,
    )

    if transactions["timestamp"].isna().any():
        raise ValueError(
            "Phase 1 contains invalid transaction timestamps."
        )

    transactions["_transaction_order"] = np.arange(
        len(transactions)
    )

    windows = phase2_windows.copy()

    windows["window_start"] = pd.to_datetime(
        windows["window_start"],
        errors="coerce",
        utc=True,
    )

    windows["window_end"] = pd.to_datetime(
        windows["window_end"],
        errors="coerce",
        utc=True,
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    # pandas merge_asof requires the temporal merge key to be
    # globally sorted.
    # ---------------------------------------------------------------

    left = transactions.sort_values(
        ["timestamp", "merchant_id"],
        kind="mergesort",
    )

    right = windows.sort_values(
        ["window_start", "merchant_id"],
        kind="mergesort",
    )

    aligned = pd.merge_asof(
        left,
        right,
        left_on="timestamp",
        right_on="window_start",
        by="merchant_id",
        direction="backward",
        tolerance=pd.Timedelta(
            minutes=config.max_window_age_minutes
        ),
    )

    aligned["_phase2_window_matched"] = (
        aligned["window_start"].notna()
    )

    aligned["_phase2_window_age_minutes"] = (
        (
            aligned["timestamp"]
            - aligned["window_start"]
        )
        .dt.total_seconds()
        .div(60.0)
    )

    # ---------------------------------------------------------------
    # Safety invariant
    # ---------------------------------------------------------------

    invalid_future_match = (
        aligned["_phase2_window_age_minutes"] < 0
    )

    if invalid_future_match.any():
        raise RuntimeError(
            "Fusion detected a future Phase 2 window match."
        )

    # ---------------------------------------------------------------
    # Restore original transaction ordering
    # ---------------------------------------------------------------

    aligned = aligned.sort_values(
        "_transaction_order",
        kind="mergesort",
    )

    return aligned


# =====================================================================
# EVENT ATTACHMENT
# =====================================================================

def _attach_events(
    unified: pd.DataFrame,
    phase2_events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach the Phase 2 event containing the transaction's matched
    Phase 2 window.

    Event membership requires:

        event.start_time <= phase2.window_start <= event.end_time

    and matching merchant_id.

    No future event is attached to a transaction.
    """

    result = unified.copy()

    events = phase2_events.copy()

    if events.empty:
        result["event_id"] = pd.NA
        result["event_active"] = False
        result["max_state"] = pd.NA
        result["peak_spike_score"] = np.nan
        result["duration_minutes"] = np.nan

        return result

    required_columns = [
        "event_id",
        "merchant_id",
        "start_time",
        "end_time",
    ]

    missing = [
        column
        for column in required_columns
        if column not in events.columns
    ]

    if missing:
        raise ValueError(
            "Phase 2 events are missing required columns: "
            + ", ".join(missing)
        )

    events["start_time"] = pd.to_datetime(
        events["start_time"],
        errors="coerce",
        utc=True,
    )

    events["end_time"] = pd.to_datetime(
        events["end_time"],
        errors="coerce",
        utc=True,
    )

    event_columns = [
        "event_id",
        "merchant_id",
        "start_time",
        "end_time",
        "duration_minutes",
        "peak_spike_score",
        "max_state",
        "windows",
        "transaction_count",
        "total_amount",
        "expected_fraud_count",
        "expected_fraud_amount",
        "unique_cards",
        "max_coordination_score",
        "max_tas",
        "max_fas",
    ]

    event_columns = [
        column
        for column in event_columns
        if column in events.columns
    ]

    events = events[event_columns].copy()

    events = events.rename(
        columns={
            "start_time": "_event_start",
            "end_time": "_event_end",
        }
    )

    # ---------------------------------------------------------------
    # Only transactions with a valid Phase 2 window can receive an
    # event attachment.
    # ---------------------------------------------------------------

    left = result[
        [
            "merchant_id",
            "window_start",
        ]
    ].copy()

    left["_row_id"] = np.arange(len(left))

    left = left.dropna(
        subset=["window_start"]
    )

    left = left.sort_values(
        ["window_start", "merchant_id"],
        kind="mergesort",
    )

    right = events.sort_values(
        ["_event_start", "merchant_id"],
        kind="mergesort",
    )

    if right.empty:
        result["event_id"] = pd.NA
        result["event_active"] = False
        result["max_state"] = pd.NA
        result["peak_spike_score"] = np.nan
        result["duration_minutes"] = np.nan

        return result

    merged = pd.merge_asof(
        left,
        right,
        left_on="window_start",
        right_on="_event_start",
        by="merchant_id",
        direction="backward",
    )

    # ---------------------------------------------------------------
    # Event must actually contain the matched Phase 2 window.
    # ---------------------------------------------------------------

    merged["_event_active"] = (
        merged["_event_start"].notna()
        & merged["window_start"].notna()
        & merged["window_start"].le(
            merged["_event_end"]
        )
    )

    merged = merged.sort_values(
        "_row_id",
        kind="mergesort",
    )

    event_columns_to_attach = [
        column
        for column in merged.columns
        if column not in {
            "merchant_id",
            "window_start",
            "_row_id",
            "_event_start",
            "_event_end",
        }
    ]

    event_data = merged[
        ["_row_id"] + event_columns_to_attach
    ].copy()

    result["_row_id"] = np.arange(len(result))

    result = result.merge(
        event_data,
        on="_row_id",
        how="left",
        sort=False,
        suffixes=("", "_event"),
    )

    # ---------------------------------------------------------------
    # Clear event data for inactive matches.
    # ---------------------------------------------------------------

    inactive = ~(
        result["_event_active"]
        .fillna(False)
        .astype(bool)
    )

    event_data_columns = [
        column
        for column in event_columns_to_attach
        if column != "_event_active"
    ]

    for column in event_data_columns:
        if column in result.columns:
            result.loc[inactive, column] = pd.NA

    result["event_active"] = (
        result["_event_active"]
        .fillna(False)
        .astype(bool)
    )

    result = result.drop(
        columns=[
            "_row_id",
            "_event_active",
        ],
        errors="ignore",
    )

    return result


# =====================================================================
# CORE UNIFIED RISK BUILD
# =====================================================================

def build_unified_risk(
    phase1: pd.DataFrame,
    phase2_windows: pd.DataFrame,
    phase2_events: pd.DataFrame,
    config: FusionConfig,
) -> pd.DataFrame:
    """
    Build the final transaction-level unified risk artifact.

    Pipeline:

        Phase 1 probability
            +
        Phase 2 temporal intelligence
            ->
        confidence-adjusted fusion
            ->
        state escalation
            ->
        expected loss
            ->
        risk decision
            ->
        unified transaction risk
    """

    # ---------------------------------------------------------------
    # Prepare Phase 2
    # ---------------------------------------------------------------

    windows = _prepare_phase2_windows(
        phase2_windows,
        config,
    )

    # ---------------------------------------------------------------
    # Point-in-time transaction alignment
    # ---------------------------------------------------------------

    aligned = _align_transactions_to_windows(
        phase1,
        windows,
        config,
    )

    # ---------------------------------------------------------------
    # Attach Phase 2 event context
    # ---------------------------------------------------------------

    aligned = _attach_events(
        aligned,
        phase2_events,
    )

    # ---------------------------------------------------------------
    # Phase 1 score
    # ---------------------------------------------------------------

    aligned["phase1_score"] = probability_to_score(
        aligned["fraud_probability"]
    )

    aligned["phase1_score"] = (
        pd.to_numeric(
            aligned["phase1_score"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(0.0, 100.0)
    )

    # ---------------------------------------------------------------
    # Phase 2 score
    # ---------------------------------------------------------------

    aligned["phase2_score"] = pd.to_numeric(
        aligned.get(
            "phase2_score",
            pd.Series(
                0.0,
                index=aligned.index,
            ),
        ),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 100.0)

    # ---------------------------------------------------------------
    # Phase 2 confidence
    # ---------------------------------------------------------------

    aligned["phase2_confidence"] = pd.to_numeric(
        aligned.get(
            "phase2_confidence",
            pd.Series(
                0.0,
                index=aligned.index,
            ),
        ),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)

    # ---------------------------------------------------------------
    # Base weighted fusion
    # ---------------------------------------------------------------

    aligned["base_fused_score"] = (
        config.phase1_weight
        * aligned["phase1_score"]
        +
        config.phase2_weight
        * aligned["phase2_score"]
    )

    # ---------------------------------------------------------------
    # Confidence-adjusted fusion
    # ---------------------------------------------------------------

    effective_phase2_weight = (
        config.phase2_weight
        * aligned["phase2_confidence"]
    )

    effective_phase1_weight = (
        config.phase1_weight
        +
        config.phase2_weight
        * (
            1.0
            - aligned["phase2_confidence"]
        )
    )

    denominator = (
        effective_phase1_weight
        + effective_phase2_weight
    ).replace(0.0, np.nan)

    aligned["confidence_adjusted_score"] = (
        (
            effective_phase1_weight
            * aligned["phase1_score"]
            +
            effective_phase2_weight
            * aligned["phase2_score"]
        )
        / denominator
    ).fillna(
        aligned["phase1_score"]
    )

    aligned["confidence_adjusted_score"] = (
        aligned["confidence_adjusted_score"]
        .clip(0.0, 100.0)
    )

    # ---------------------------------------------------------------
    # Phase 2 state escalation
    # ---------------------------------------------------------------

    aligned["phase2_state_multiplier"] = pd.to_numeric(
        aligned.get(
            "phase2_state_multiplier",
            pd.Series(
                config.state_multiplier_normal,
                index=aligned.index,
            ),
        ),
        errors="coerce",
    ).fillna(
        config.state_multiplier_normal
    )

    aligned["unified_risk_score"] = (
        aligned["confidence_adjusted_score"]
        * aligned["phase2_state_multiplier"]
    ).clip(
        lower=0.0,
        upper=100.0,
    )

    # ---------------------------------------------------------------
    # Expected financial impact
    # ---------------------------------------------------------------

    aligned["transaction_expected_loss"] = (
        calculate_transaction_expected_loss(
            aligned["amount"],
            aligned["fraud_probability"],
            config,
        )
    )

    aligned["phase2_expected_fraud_amount"] = pd.to_numeric(
        aligned.get(
            "expected_fraud_amount",
            pd.Series(
                0.0,
                index=aligned.index,
            ),
        ),
        errors="coerce",
    ).fillna(0.0)

    aligned["expected_fraud_exposure"] = (
        calculate_fused_expected_exposure(
            aligned["transaction_expected_loss"],
            aligned["phase2_expected_fraud_amount"],
        )
    )

    # ---------------------------------------------------------------
    # Decision layer
    # ---------------------------------------------------------------

    spike_state = (
        aligned.get(
            "spike_state",
            pd.Series(
                "NORMAL",
                index=aligned.index,
            ),
        )
        .fillna("NORMAL")
        .astype(str)
    )

    aligned["risk_band"] = assign_risk_band(
        aligned["unified_risk_score"],
        config,
    )

    aligned["response_action"] = assign_response_action(
        aligned["unified_risk_score"],
        spike_state,
        config,
    )

    aligned["response_priority"] = assign_priority(
        aligned["unified_risk_score"],
        spike_state,
        config,
    )

    aligned["alert_flag"] = assign_alert_flag(
        aligned["unified_risk_score"],
        spike_state,
        config,
    )

    # ---------------------------------------------------------------
    # Fusion confidence
    #
    # This is the overall confidence exposed to downstream systems.
    # It is based on Phase 2 evidence availability and confidence.
    # Phase 1 itself is assumed to be the calibrated transaction model.
    # ---------------------------------------------------------------

    aligned["fusion_confidence"] = (
        config.phase1_weight
        +
        config.phase2_weight
        * aligned["phase2_confidence"]
    ).clip(0.0, 1.0)

    # ---------------------------------------------------------------
    # Explainability-ready signals
    # ---------------------------------------------------------------

    aligned["phase1_primary_signal"] = (
        aligned["phase1_score"]
        >= aligned["phase2_score"]
    )

    aligned["phase2_primary_signal"] = (
        aligned["phase2_score"]
        > aligned["phase1_score"]
    )

    aligned["temporal_spike_signal"] = (
        pd.to_numeric(
            aligned.get(
                "spike_score",
                pd.Series(
                    0.0,
                    index=aligned.index,
                ),
            ),
            errors="coerce",
        )
        .fillna(0.0)
        >= 57.150510779704376
    )

    aligned["verified_spike_signal"] = (
        spike_state
        == "VERIFIED_FRAUD_SPIKE"
    )

    aligned["critical_spike_signal"] = (
        spike_state
        == "CRITICAL_ACTIVE_SPIKE"
    )

    aligned["early_warning_signal"] = (
        spike_state
        == "EARLY_WARNING"
    )

    aligned["active_event_signal"] = (
        aligned.get(
            "event_active",
            pd.Series(
                False,
                index=aligned.index,
            ),
        )
        .fillna(False)
        .astype(bool)
    )

    # ---------------------------------------------------------------
    # Model/version governance
    # ---------------------------------------------------------------

    aligned["phase1_model_version"] = (
        config.phase1_version
    )

    aligned["phase2_model_version"] = (
        config.phase2_version
    )

    aligned["fusion_model_version"] = (
        config.fusion_version
    )

    # ---------------------------------------------------------------
    # Numerical safety
    # ---------------------------------------------------------------

    numeric_columns = aligned.select_dtypes(
        include=[np.number]
    ).columns

    aligned[numeric_columns] = (
        aligned[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # ---------------------------------------------------------------
    # Final column ordering
    # ---------------------------------------------------------------

    preferred_columns = [
        # Transaction identity
        "transaction_id",
        "timestamp",
        "merchant_id",
        "card_id",
        "amount",

        # Phase 1
        "fraud_probability",
        "risk_score",
        "risk_band",
        "predicted_fraud",
        "phase1_score",

        # Phase 2 window
        "window_start",
        "window_end",
        "phase2_score",
        "spike_state",
        "phase2_confidence",
        "phase2_state_multiplier",

        # Phase 2 evidence
        "statistical_evidence",
        "financial_evidence",
        "temporal_evidence",
        "breadth_score",
        "coordination_score",
        "tas",
        "fas",
        "materiality",

        # Phase 2 context
        "expected_fraud_count",
        "expected_fraud_amount",
        "risk_rate",
        "transaction_count",
        "total_amount",
        "unique_cards",
        "high_risk_count",
        "new_cards",
        "new_card_rate",
        "history_count",
        "baseline_ready",

        # Temporal context
        "ewma_value",
        "ewma_residual",
        "cusum_value",
        "cusum_signal",
        "acceleration",

        # Event
        "event_id",
        "event_active",
        "max_state",
        "peak_spike_score",
        "duration_minutes",

        # Fusion
        "base_fused_score",
        "confidence_adjusted_score",
        "fusion_confidence",
        "unified_risk_score",

        # Impact
        "transaction_expected_loss",
        "phase2_expected_fraud_amount",
        "expected_fraud_exposure",

        # Decisions
        "response_action",
        "response_priority",
        "alert_flag",

        # XAI-ready flags
        "phase1_primary_signal",
        "phase2_primary_signal",
        "temporal_spike_signal",
        "early_warning_signal",
        "verified_spike_signal",
        "critical_spike_signal",
        "active_event_signal",

        # Governance
        "phase1_model_version",
        "phase2_model_version",
        "fusion_model_version",
    ]

    existing = [
        column
        for column in preferred_columns
        if column in aligned.columns
    ]

    remaining = [
        column
        for column in aligned.columns
        if column not in existing
    ]

    aligned = aligned[
        existing + remaining
    ]

    return aligned


# =====================================================================
# PUBLIC ENGINE API
# =====================================================================

class RiskFusionEngine:
    """
    Public Fusion Engine API.

    This class provides the interface expected by pipeline.py:

        engine = RiskFusionEngine(config=config)

        result = engine.transform(
            phase1,
            phase2_windows,
            phase2_events,
        )

    It delegates the actual fusion mathematics to build_unified_risk().
    """

    def __init__(
        self,
        config: FusionConfig | None = None,
    ) -> None:
        self.config = config or FusionConfig()
        

    def transform(
        self,
        phase1: pd.DataFrame,
        phase2_windows: pd.DataFrame,
        phase2_events: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Transform Phase 1 and Phase 2 artifacts into unified risk.

        Parameters
        ----------
        phase1:
            Transaction-level Phase 1 risk dataframe.

        phase2_windows:
            Merchant-level Phase 2 temporal windows.

        phase2_events:
            Phase 2 detected temporal events.

        Returns
        -------
        pd.DataFrame
            Transaction-level unified risk dataframe.
        """

        if phase2_events is None:
            phase2_events = pd.DataFrame(
                columns=[
                    "event_id",
                    "merchant_id",
                    "start_time",
                    "end_time",
                ]
            )

        return build_unified_risk(
            phase1=phase1,
            phase2_windows=phase2_windows,
            phase2_events=phase2_events,
            config=self.config,
        )


