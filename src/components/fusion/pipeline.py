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

    An empty event dataframe is valid; however, when rows exist,
    the event identity and temporal boundaries must be present.
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

    Parameters
    ----------
    phase1_path:
        Phase 1 transaction-risk CSV.

    phase2_path:
        Phase 2 temporal-window CSV.

    phase2_events_path:
        Phase 2 event CSV.

        If omitted, the pipeline looks for:

            <phase2_path directory>/phase2_events.csv

    output_path:
        Optional destination for the unified risk CSV.

        If omitted:

            config.output_dir/
                fraudsentinel_unified_risk.csv

    config:
        Optional FusionConfig.

    Returns
    -------
    pd.DataFrame
        Transaction-level unified risk dataframe.
    """

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

    config = config or FusionConfig()
    

    phase1_path = Path(phase1_path)
    phase2_path = Path(phase2_path)

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

    # Phase 2 events are allowed to be empty in principle, but the
    # artifact should normally exist in the production pipeline.
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
    # Save unified artifact
    # ---------------------------------------------------------------

    result.to_csv(
        output_path,
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
        for key, value in state_distribution.items()
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
        for key, value in risk_distribution.items()
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
        for key, value in response_distribution.items()
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
        for key, value in alert_distribution.items()
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

        "fusion_version": config.fusion_version,

        "phase1_version": config.phase1_version,

        "phase2_version": config.phase2_version,

        "rows": int(len(df)),

        "mean_unified_risk_score": safe_float(
            "unified_risk_score"
        ),

        "median_unified_risk_score": safe_median(
            "unified_risk_score"
        ),

        "max_unified_risk_score": safe_max(
            "unified_risk_score"
        ),

        "mean_fusion_confidence": safe_float(
            "fusion_confidence"
        ),

        "mean_phase2_confidence": safe_float(
            "phase2_confidence"
        ),

        "total_expected_fraud_exposure": safe_sum(
            "expected_fraud_exposure"
        ),

        "total_transaction_expected_loss": safe_sum(
            "transaction_expected_loss"
        ),

        "state_distribution": state_distribution,

        "risk_band_distribution": risk_distribution,

        "response_distribution": response_distribution,

        "alert_distribution": alert_distribution,

        "weights": {
            "phase1": float(
                config.phase1_weight
            ),
            "phase2": float(
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
