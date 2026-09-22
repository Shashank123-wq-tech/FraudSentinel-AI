from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import FusionConfig
from .fusion import RiskFusionEngine


# =====================================================================
# INPUT LOADING
# =====================================================================

def _load_csv(
    path: str | Path,
    name: str,
) -> pd.DataFrame:
    """
    Load a CSV artifact with clear error reporting.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{name} artifact not found:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{name} path is not a file:\n{path}"
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read {name} artifact:\n{path}\n"
            f"Original error: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            f"{name} artifact is empty:\n{path}"
        )

    return df


# =====================================================================
# INPUT VALIDATION
# =====================================================================

def _validate_phase1(
    df: pd.DataFrame,
) -> None:
    """
    Validate the Phase 1 transaction-risk artifact.
    """

    required = [
        "transaction_id",
        "timestamp",
        "merchant_id",
        "amount",
        "fraud_probability",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Phase 1 transaction-risk artifact is missing "
            "required columns: "
            + ", ".join(missing)
        )


def _validate_phase2_windows(
    df: pd.DataFrame,
) -> None:
    """
    Validate the Phase 2 temporal-window artifact.
    """

    required = [
        "merchant_id",
        "window_start",
        "window_end",
        "spike_score",
        "spike_state",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Phase 2 windows artifact is missing required "
            "columns: "
            + ", ".join(missing)
        )


def _validate_phase2_events(
    df: pd.DataFrame,
) -> None:
    """
    Validate the Phase 2 event artifact.

    An empty event dataframe is valid in principle, although
    the production pipeline normally expects the artifact to
    contain event rows.
    """

    required = [
        "event_id",
        "merchant_id",
        "start_time",
        "end_time",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Phase 2 events artifact is missing required "
            "columns: "
            + ", ".join(missing)
        )


# =====================================================================
# FUSION EVENT BUILDER
# =====================================================================

def _build_fusion_events(
    fused_transactions: pd.DataFrame,
    phase2_events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build fusion-level event artifacts.

    Phase 2 is the source of truth for event identity.
    Fusion enriches those events with transaction-level unified-risk
    information when a compatible event identifier is available.

    The function intentionally does not assume that Phase 2 calls
    its identifier `event_id`.
    """

    if phase2_events is None or phase2_events.empty:
        return pd.DataFrame()

    events = phase2_events.copy()

    # ------------------------------------------------------------
    # 1. Detect Phase 2 event identifier
    # ------------------------------------------------------------

    possible_event_id_columns = [
        "event_id",
        "spike_event_id",
        "fraud_spike_event_id",
        "event",
        "episode_id",
        "spike_id",
    ]

    phase2_event_id = next(
        (
            col
            for col in possible_event_id_columns
            if col in events.columns
        ),
        None,
    )

    # ------------------------------------------------------------
    # 2. If Phase 2 has no event identifier, preserve its
    #    event artifact rather than crashing Fusion.
    # ------------------------------------------------------------

    if phase2_event_id is None:
        events = events.copy()

        events.insert(
            0,
            "fusion_event_id",
            [f"FUSION_EVT_{i:06d}" for i in range(len(events))],
        )

        events["fusion_event_id_source"] = "generated_from_phase2_row"

        return events

    # ------------------------------------------------------------
    # 3. Normalize event identifier
    # ------------------------------------------------------------

    events["fusion_event_id"] = events[phase2_event_id].astype(str)

    events["fusion_event_id_source"] = phase2_event_id

    # ------------------------------------------------------------
    # 4. Look for an event identifier in fused transactions
    # ------------------------------------------------------------

    transaction_event_id = next(
        (
            col
            for col in possible_event_id_columns
            if col in fused_transactions.columns
        ),
        None,
    )

    if transaction_event_id is None:
        return events

    tx = fused_transactions.copy()

    tx["fusion_event_id"] = tx[transaction_event_id].astype(str)

    # ------------------------------------------------------------
    # 5. Select only metrics that actually exist
    # ------------------------------------------------------------

    candidate_metrics = [
        "amount",
        "fraud_probability",
        "unified_risk",
        "risk_score",
        "phase1_score",
        "phase2_score",
        "spike_score",
        "tas",
        "fas",
        "coordination_score",
        "expected_fraud_amount",
        "expected_fraud_count",
        "predicted_fraud",
    ]

    available_metrics = [
        col
        for col in candidate_metrics
        if col in tx.columns
    ]

    # Nothing to aggregate
    if not available_metrics:
        return events

    # ------------------------------------------------------------
    # 6. Build transaction-level event aggregates
    # ------------------------------------------------------------

    aggregation = {}

    if "amount" in available_metrics:
        aggregation["amount"] = ["sum", "mean", "max"]

    if "fraud_probability" in available_metrics:
        aggregation["fraud_probability"] = ["mean", "max"]

    if "unified_risk" in available_metrics:
        aggregation["unified_risk"] = ["mean", "max"]

    if "risk_score" in available_metrics:
        aggregation["risk_score"] = ["mean", "max"]

    if "phase1_score" in available_metrics:
        aggregation["phase1_score"] = ["mean", "max"]

    if "phase2_score" in available_metrics:
        aggregation["phase2_score"] = ["mean", "max"]

    if "spike_score" in available_metrics:
        aggregation["spike_score"] = ["mean", "max"]

    if "tas" in available_metrics:
        aggregation["tas"] = ["mean", "max"]

    if "fas" in available_metrics:
        aggregation["fas"] = ["mean", "max"]

    if "coordination_score" in available_metrics:
        aggregation["coordination_score"] = ["mean", "max"]

    if "expected_fraud_amount" in available_metrics:
        aggregation["expected_fraud_amount"] = ["sum", "mean", "max"]

    if "expected_fraud_count" in available_metrics:
        aggregation["expected_fraud_count"] = ["sum", "mean", "max"]

    if not aggregation:
        return events

    summary = (
        tx.groupby("fusion_event_id")
        .agg(aggregation)
        .reset_index()
    )

    # ------------------------------------------------------------
    # 7. Flatten MultiIndex columns
    # ------------------------------------------------------------

    flattened_columns = []

    for column in summary.columns:
        if isinstance(column, tuple):
            base, statistic = column

            if statistic:
                flattened_columns.append(
                    f"{base}_{statistic}"
                )
            else:
                flattened_columns.append(base)
        else:
            flattened_columns.append(column)

    summary.columns = flattened_columns

    # ------------------------------------------------------------
    # 8. Add transaction count
    # ------------------------------------------------------------

    transaction_counts = (
        tx.groupby("fusion_event_id")
        .size()
        .reset_index(name="fusion_transaction_count")
    )

    summary = summary.merge(
        transaction_counts,
        on="fusion_event_id",
        how="left",
    )

    # ------------------------------------------------------------
    # 9. Merge Phase 2 events with Fusion evidence
    # ------------------------------------------------------------

    events = events.merge(
        summary,
        on="fusion_event_id",
        how="left",
    )

    return events


# =====================================================================
# PIPELINE
# =====================================================================

def run_fusion_pipeline(
    phase1_path: str | Path,
    phase2_path: str | Path,
    phase2_events_path: str | Path | None = None,
    output_path: str | Path | None = None,
    config: FusionConfig | None = None,
) -> pd.DataFrame:
    """
    Execute the complete FraudSentinel AI Risk Fusion pipeline.

    Outputs:

        fraudsentinel_unified_risk.csv
        fusion_events.csv
        fusion_summary.json
    """

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

    config = config or FusionConfig()

    phase1_path = Path(
        phase1_path
    )

    phase2_path = Path(
        phase2_path
    )

    # ---------------------------------------------------------------
    # Resolve Phase 2 events path
    # ---------------------------------------------------------------

    if phase2_events_path is None:

        phase2_events_path = (
            phase2_path.parent
            / "phase2_events.csv"
        )

    else:

        phase2_events_path = Path(
            phase2_events_path
        )

    # ---------------------------------------------------------------
    # Load artifacts
    # ---------------------------------------------------------------

    phase1 = _load_csv(
        phase1_path,
        "Phase 1 transaction-risk",
    )

    phase2_windows = _load_csv(
        phase2_path,
        "Phase 2 windows",
    )

    phase2_events = _load_csv(
        phase2_events_path,
        "Phase 2 events",
    )

    # ---------------------------------------------------------------
    # Validate artifacts
    # ---------------------------------------------------------------

    _validate_phase1(
        phase1
    )

    _validate_phase2_windows(
        phase2_windows
    )

    _validate_phase2_events(
        phase2_events
    )

    # ---------------------------------------------------------------
    # Build engine
    # ---------------------------------------------------------------

    engine = RiskFusionEngine(
        config=config,
    )

    # ---------------------------------------------------------------
    # Execute fusion
    # ---------------------------------------------------------------

    result = engine.transform(
        phase1=phase1,
        phase2_windows=phase2_windows,
        phase2_events=phase2_events,
    )

    # ---------------------------------------------------------------
    # Resolve output
    # ---------------------------------------------------------------

    if output_path is None:

        output_path = (
            config.output_dir
            / "fraudsentinel_unified_risk.csv"
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Save unified transaction artifact
    # ---------------------------------------------------------------

    result.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Build Fusion event artifact
    # ---------------------------------------------------------------

    fusion_events = _build_fusion_events(
        fused_transactions=result,
        phase2_events=phase2_events,
    )

    fusion_events_path = (
        output_path.parent
        / "fusion_events.csv"
    )

    fusion_events.to_csv(
        fusion_events_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------------

    _write_summary(
        result,
        output_path.parent,
        config,
    )

    # ---------------------------------------------------------------
    # Final artifact diagnostics
    # ---------------------------------------------------------------

    print(
        f"Fusion transaction artifact: "
        f"{output_path}"
    )

    print(
        f"Fusion event artifact       : "
        f"{fusion_events_path}"
    )

    print(
        f"Fusion events               : "
        f"{len(fusion_events):,}"
    )

    return result


# =====================================================================
# SUMMARY
# =====================================================================

def _write_summary(
    df: pd.DataFrame,
    output_dir: Path,
    config: FusionConfig,
) -> None:
    """
    Write a compact machine-readable Fusion summary.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # State distribution
    # ---------------------------------------------------------------

    if "spike_state" in df.columns:

        state_distribution = (
            df["spike_state"]
            .fillna("NORMAL")
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

    else:

        state_distribution = {}

    state_distribution = {
        str(key): int(value)
        for key, value
        in state_distribution.items()
    }

    # ---------------------------------------------------------------
    # Risk-band distribution
    # ---------------------------------------------------------------

    if "risk_band" in df.columns:

        risk_distribution = (
            df["risk_band"]
            .fillna("UNKNOWN")
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

    else:

        risk_distribution = {}

    risk_distribution = {
        str(key): int(value)
        for key, value
        in risk_distribution.items()
    }

    # ---------------------------------------------------------------
    # Response distribution
    # ---------------------------------------------------------------

    if "response_action" in df.columns:

        response_distribution = (
            df["response_action"]
            .fillna("UNKNOWN")
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

    else:

        response_distribution = {}

    response_distribution = {
        str(key): int(value)
        for key, value
        in response_distribution.items()
    }

    # ---------------------------------------------------------------
    # Alert distribution
    # ---------------------------------------------------------------

    if "alert_flag" in df.columns:

        alert_distribution = (
            df["alert_flag"]
            .fillna(False)
            .astype(bool)
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

    else:

        alert_distribution = {}

    alert_distribution = {
        str(key): int(value)
        for key, value
        in alert_distribution.items()
    }

    # ---------------------------------------------------------------
    # Safe numerical helper
    # ---------------------------------------------------------------

    def safe_float(
        series_name: str,
        default: float = 0.0,
    ) -> float:

        if series_name not in df.columns:
            return default

        value = pd.to_numeric(
            df[series_name],
            errors="coerce",
        ).mean()

        if pd.isna(value):
            return default

        return float(value)

    def safe_sum(
        series_name: str,
        default: float = 0.0,
    ) -> float:

        if series_name not in df.columns:
            return default

        value = pd.to_numeric(
            df[series_name],
            errors="coerce",
        ).sum()

        if pd.isna(value):
            return default

        return float(value)

    def safe_median(
        series_name: str,
        default: float = 0.0,
    ) -> float:

        if series_name not in df.columns:
            return default

        value = pd.to_numeric(
            df[series_name],
            errors="coerce",
        ).median()

        if pd.isna(value):
            return default

        return float(value)

    def safe_max(
        series_name: str,
        default: float = 0.0,
    ) -> float:

        if series_name not in df.columns:
            return default

        value = pd.to_numeric(
            df[series_name],
            errors="coerce",
        ).max()

        if pd.isna(value):
            return default

        return float(value)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary = {
        "project": "FraudSentinel AI",

        "component": "Risk Fusion Layer",

        "fusion_version":
            config.fusion_version,

        "phase1_version":
            config.phase1_version,

        "phase2_version":
            config.phase2_version,

        "rows":
            int(len(df)),

        "mean_unified_risk_score":
            safe_float(
                "unified_risk_score"
            ),

        "median_unified_risk_score":
            safe_median(
                "unified_risk_score"
            ),

        "max_unified_risk_score":
            safe_max(
                "unified_risk_score"
            ),

        "mean_fusion_confidence":
            safe_float(
                "fusion_confidence"
            ),

        "mean_phase2_confidence":
            safe_float(
                "phase2_confidence"
            ),

        "total_expected_fraud_exposure":
            safe_sum(
                "expected_fraud_exposure"
            ),

        "total_transaction_expected_loss":
            safe_sum(
                "transaction_expected_loss"
            ),

        "state_distribution":
            state_distribution,

        "risk_band_distribution":
            risk_distribution,

        "response_distribution":
            response_distribution,

        "alert_distribution":
            alert_distribution,

        "weights": {
            "phase1":
                float(
                    config.phase1_weight
                ),

            "phase2":
                float(
                    config.phase2_weight
                ),
        },
    }

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------

    summary_path = (
        output_dir
        / "fusion_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )