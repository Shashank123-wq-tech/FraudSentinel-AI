"""
FraudSentinel AI
Phase 2 — Temporal Fraud-Spike Intelligence Pipeline

Pipeline
--------

Phase-1 Risk
    ↓
Temporal Context
    ↓
5m / 15m / 60m Merchant Windows
    ↓
Merchant Baselines
    ↓
Statistical Signals
    ↓
EWMA + CUSUM
    ↓
Breadth
    ↓
Persistence
    ↓
Materiality
    ↓
Financial Evidence
    ↓
BASE TAS
    ↓
Acceleration from BASE TAS
    ↓
FINAL TAS
    ↓
Multiscale Corroboration
    ↓
Robust Vectorized Verification
    ├── Normal Multi-Transaction Verification
    └── Exceptional Single-Transaction Verification
    ↓
Fraud Attack Score / Exceptional Risk Score
    ↓
Operational Hysteresis
    ↓
15m Event Construction
    ↓
Explanation
    ↓
Evaluation

Memory Strategy
---------------

Only one enriched resolution is processed at a time.

5m:
    rich frame -> persist -> retain merchant/window/TAS only

15m:
    rich frame -> retain as final decision frame

60m:
    rich frame -> persist -> retain merchant/window/TAS only

This prevents three large rich DataFrames from simultaneously occupying RAM.

Verification Strategy
---------------------

NORMAL VERIFIED FRAUD SPIKE
    requires:
        candidate
        + minimum transaction activity
        + verified TAS
        + materiality
        + history confidence
        + at least two independent temporal signals
        + coordination/breadth evidence
        + composite evidence score

EXCEPTIONAL SINGLE TRANSACTION
    requires:
        single transaction
        + exceptional expected fraud amount
        + exceptional TAS
        + supporting materiality/history evidence

CRITICAL FRAUD SPIKE
    requires:
        normal multi-transaction verification
        + critical TAS
        + critical materiality
        + all three temporal evidence dimensions
        + critical composite evidence score

Important
---------

This pipeline does not maintain a second verification implementation.

Verification mathematics live in:

    src/components/phase2/verification.py

This file is responsible for orchestration, memory management,
score integration, event construction, persistence, explanations,
and evaluation.
"""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.exception.exception import (
    ComputationError,
    PipelineError,
)

from src.logger.logger import get_logger

from src.components.phase2.risk_output_builder import (
    load_phase1_risk_output,
)

from src.components.phase2.temporal_features import (
    add_temporal_context,
)

from src.components.phase2.temporal_aggregation import (
    aggregate_merchant_windows,
)

from src.components.phase2.merchant_baseline import (
    build_merchant_baselines,
)

from src.components.phase2.statistical_signals import (
    add_statistical_signals,
)

from src.components.phase2.ewma_detector import (
    add_ewma,
)

from src.components.phase2.cusum_detector import (
    add_cusum,
)

from src.components.phase2.breadth_scoring import (
    add_breadth_score,
)

from src.components.phase2.persistence_detector import (
    add_persistence,
)

from src.components.phase2.acceleration_detector import (
    add_acceleration,
)

from src.components.phase2.scoring import (
    add_materiality,
)

from src.components.phase2.multiscale_verification import (
    add_multiscale_corroboration,
)

from src.components.phase2.verification import (
    add_verification_columns,
)

from src.components.phase2.explanation_builder import (
    build_explanations,
)

from src.components.phase2.evaluation import (
    evaluate_transaction_level,
    evaluate_events,
    write_evaluation,
)


# =============================================================================
# CONSTANTS
# =============================================================================

EPSILON = 1e-9

DEFAULT_EWMA_ALPHA = 0.25
DEFAULT_EWMA_TAU = 0.02
DEFAULT_CUSUM_TAU = 2.5

DEFAULT_ATTACK_GAMMA = 1.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _numeric(
    dataframe: pd.DataFrame,
    column_name: str,
    default: float = 0.0,
) -> pd.Series:
    """
    Safely convert a dataframe column to numeric values.

    Invalid values, NaN and infinities are replaced with `default`.
    """

    if column_name not in dataframe.columns:
        return pd.Series(
            default,
            index=dataframe.index,
            dtype="float64",
        )

    values = pd.to_numeric(
        dataframe[column_name],
        errors="coerce",
    )

    values = values.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return values.fillna(
        default
    )


def _clip01(
    values: pd.Series,
) -> pd.Series:
    """
    Convert values to finite [0, 1] range.
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    values = values.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return values.fillna(
        0.0
    ).clip(
        lower=0.0,
        upper=1.0,
    )


def _bounded_positive(
    values: pd.Series,
    tau: float,
) -> pd.Series:
    """
    Transform positive evidence into [0, 1]:

        score = 1 - exp(-x / tau)
    """

    safe_tau = max(
        float(tau),
        EPSILON,
    )

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    numeric_values = numeric_values.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).fillna(
        0.0
    )

    numeric_values = numeric_values.clip(
        lower=0.0,
    )

    return (
        1.0
        -
        np.exp(
            -numeric_values
            /
            safe_tau
        )
    ).clip(
        lower=0.0,
        upper=1.0,
    )


def _downcast_numeric(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reduce RAM usage of numeric columns.
    """

    for column_name in dataframe.columns:

        series = dataframe[column_name]

        if pd.api.types.is_float_dtype(
            series
        ):

            dataframe[
                column_name
            ] = pd.to_numeric(
                series,
                downcast="float",
            )

        elif pd.api.types.is_integer_dtype(
            series
        ):

            dataframe[
                column_name
            ] = pd.to_numeric(
                series,
                downcast="integer",
            )

    return dataframe


def _resolve_path(
    value: str,
) -> Path:
    """
    Resolve absolute or project-relative path.
    """

    path = Path(
        value
    )

    if path.is_absolute():
        return path.resolve()

    return (
        Path.cwd()
        /
        path
    ).resolve()


def _ensure_dir(
    path: Path,
) -> None:
    """
    Ensure directory exists.
    """

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def _finite_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert a value to a finite float.
    """

    try:
        numeric_value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return float(default)

    if not math.isfinite(
        numeric_value
    ):
        return float(default)

    return numeric_value


def _safe_sum(
    dataframe: pd.DataFrame,
    column_name: str,
) -> float:
    """
    Safely sum a dataframe column.
    """

    if column_name not in dataframe.columns:
        return 0.0

    values = _numeric(
        dataframe,
        column_name,
    )

    return _finite_float(
        values.sum(),
        default=0.0,
    )


def _safe_max(
    dataframe: pd.DataFrame,
    column_name: str,
) -> float:
    """
    Safely calculate maximum.
    """

    if column_name not in dataframe.columns:
        return 0.0

    values = _numeric(
        dataframe,
        column_name,
    )

    if values.empty:
        return 0.0

    return _finite_float(
        values.max(),
        default=0.0,
    )


def _safe_mean(
    dataframe: pd.DataFrame,
    column_name: str,
) -> float:
    """
    Safely calculate mean.
    """

    if column_name not in dataframe.columns:
        return 0.0

    values = _numeric(
        dataframe,
        column_name,
    )

    if values.empty:
        return 0.0

    return _finite_float(
        values.mean(),
        default=0.0,
    )


# =============================================================================
# PHASE 2 PIPELINE
# =============================================================================

class Phase2Pipeline:
    """
    Complete Phase-2 temporal fraud-spike intelligence pipeline.
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        cfg: dict,
    ) -> None:

        self.cfg = cfg

        logging_config = cfg.get(
            "logging",
            {},
        )

        self.logger = get_logger(
            "phase2",
            logging_config.get(
                "level",
                "INFO",
            ),
            logging_config.get(
                "file",
            ),
        )

        self._validate_config()

    # =========================================================================
    # CONFIGURATION VALIDATION
    # =========================================================================

    def _validate_config(
        self,
    ) -> None:

        required_sections = [
            "pipeline",
            "input",
            "windows",
            "baseline",
            "scoring",
            "breadth",
            "persistence",
            "acceleration",
            "materiality",
            "verification",
            "multiscale",
            "alerts",
            "output",
        ]

        missing_sections = [
            section_name
            for section_name in required_sections
            if section_name not in self.cfg
        ]

        if missing_sections:

            raise ValueError(
                "Phase-2 configuration is missing sections: "
                f"{missing_sections}"
            )

        # ---------------------------------------------------------------------
        # Windows
        # ---------------------------------------------------------------------

        configured_windows = {
            int(value)
            for value
            in self.cfg[
                "windows"
            ][
                "sizes_minutes"
            ]
        }

        if configured_windows != {
            5,
            15,
            60,
        }:

            raise ValueError(
                "Phase-2 requires exactly 5m, 15m and 60m windows."
            )

        event_window_minutes = int(
            self.cfg[
                "windows"
            ][
                "event_window_minutes"
            ]
        )

        if event_window_minutes not in {
            5,
            15,
            60,
        }:

            raise ValueError(
                "event_window_minutes must be one of 5, 15 or 60."
            )

        # ---------------------------------------------------------------------
        # TAS weights
        # ---------------------------------------------------------------------

        scoring_config = self.cfg[
            "scoring"
        ]

        scoring_weights = [
            float(
                scoring_config[
                    "intensity_weight"
                ]
            ),
            float(
                scoring_config[
                    "persistence_weight"
                ]
            ),
            float(
                scoring_config[
                    "acceleration_weight"
                ]
            ),
            float(
                scoring_config[
                    "breadth_weight"
                ]
            ),
            float(
                scoring_config[
                    "financial_weight"
                ]
            ),
        ]

        if any(
            weight < 0.0
            for weight in scoring_weights
        ):

            raise ValueError(
                "Phase-2 scoring weights cannot be negative."
            )

        if self.cfg.get(
            "runtime",
            {},
        ).get(
            "enforce_weight_sum",
            True,
        ):

            if not np.isclose(
                sum(scoring_weights),
                1.0,
                atol=1e-6,
            ):

                raise ValueError(
                    "Phase-2 scoring weights must sum to 1.0."
                )

        # ---------------------------------------------------------------------
        # Verification configuration
        # ---------------------------------------------------------------------

        verification_config = self.cfg[
            "verification"
        ]

        required_verification_keys = [
            "candidate_tas",
            "verified_tas",
            "critical_tas",
            "min_history_confidence",
            "persistence_min",
            "corroboration_min",
            "minimum_transactions",
            "allow_single_txn_exception",
            "single_txn_max_materiality_without_exception",
        ]

        missing_verification_keys = [
            key
            for key in required_verification_keys
            if key not in verification_config
        ]

        if missing_verification_keys:

            raise ValueError(
                "Phase-2 verification configuration is missing: "
                f"{missing_verification_keys}"
            )

        candidate_tas = float(
            verification_config[
                "candidate_tas"
            ]
        )

        verified_tas = float(
            verification_config[
                "verified_tas"
            ]
        )

        critical_tas = float(
            verification_config[
                "critical_tas"
            ]
        )

        if not (
            candidate_tas
            <= verified_tas
            <= critical_tas
        ):

            raise ValueError(
                "TAS thresholds must satisfy "
                "candidate_tas <= verified_tas <= critical_tas."
            )

        minimum_transactions = int(
            verification_config[
                "minimum_transactions"
            ]
        )

        if minimum_transactions < 2:

            raise ValueError(
                "minimum_transactions must be >= 2."
            )

        # ---------------------------------------------------------------------
        # Output configuration
        # ---------------------------------------------------------------------

        output_config = self.cfg[
            "output"
        ]

        required_output_keys = [
            "root",
            "windows_dir",
            "events_dir",
            "evaluation_dir",
            "metadata_file",
            "event_file",
            "window_file",
            "explanation_file",
        ]

        missing_output_keys = [
            key
            for key in required_output_keys
            if key not in output_config
        ]

        if missing_output_keys:

            raise ValueError(
                "Phase-2 output configuration is missing: "
                f"{missing_output_keys}"
            )

    # =========================================================================
    # PUBLIC RUN
    # =========================================================================

    def run(
        self,
    ) -> dict:

        started_at = time.time()

        try:

            return self._run_impl(
                started_at
            )

        except (
            ComputationError,
            PipelineError,
        ):

            raise

        except Exception as exc:

            raise PipelineError(
                "Phase-2 pipeline execution failed",
                cause=exc,
            ) from exc

    # =========================================================================
    # MAIN IMPLEMENTATION
    # =========================================================================

    def _run_impl(
        self,
        started_at: float,
    ) -> dict:

        # =====================================================================
        # PATHS
        # =====================================================================

        input_path = _resolve_path(
            self.cfg[
                "input"
            ][
                "phase1_risk_output"
            ]
        )

        if not input_path.exists():

            raise FileNotFoundError(
                "Phase-1 risk output not found:\n"
                f"{input_path}"
            )

        output_config = self.cfg[
            "output"
        ]

        root_directory = _resolve_path(
            output_config[
                "root"
            ]
        )

        windows_directory = _resolve_path(
            output_config[
                "windows_dir"
            ]
        )

        events_directory = _resolve_path(
            output_config[
                "events_dir"
            ]
        )

        evaluation_directory = _resolve_path(
            output_config[
                "evaluation_dir"
            ]
        )

        _ensure_dir(
            root_directory
        )

        _ensure_dir(
            windows_directory
        )

        _ensure_dir(
            events_directory
        )

        _ensure_dir(
            evaluation_directory
        )

        # =====================================================================
        # EFFECTIVE CONFIG
        # =====================================================================

        runtime_config = self.cfg.get(
            "runtime",
            {},
        )

        if runtime_config.get(
            "write_effective_config",
            True,
        ):

            effective_config_path = (
                root_directory
                /
                "effective_config.json"
            )

            effective_config_path.write_text(
                json.dumps(
                    self.cfg,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

        # =====================================================================
        # LOAD PHASE-1 RISK OUTPUT
        # =====================================================================

        self.logger.info(
            "Loading Phase-1 risk output: %s",
            input_path,
        )

        raw_transactions = load_phase1_risk_output(
            str(
                input_path
            )
        )

        if raw_transactions.empty:

            raise ValueError(
                "Phase-1 risk output contains zero rows."
            )

        # =====================================================================
        # TEMPORAL CONTEXT
        # =====================================================================

        raw_transactions = add_temporal_context(
            raw_transactions
        )

        self._validate_phase1_input(
            raw_transactions
        )

        input_row_count = int(
            len(raw_transactions)
        )

        merchant_count = int(
            raw_transactions[
                "merchant_id"
            ].nunique()
        )

        self.logger.info(
            "Phase-1 input validated | rows=%d merchants=%d",
            input_row_count,
            merchant_count,
        )

        # =====================================================================
        # WINDOW PROCESSING
        # =====================================================================

        multiscale_tas_frames: dict[
            int,
            pd.DataFrame,
        ] = {}

        decision_frame: pd.DataFrame | None = None

        event_window_minutes = int(
            self.cfg[
                "windows"
            ][
                "event_window_minutes"
            ]
        )

        for window_minutes in (
            5,
            15,
            60,
        ):

            self.logger.info(
                "Processing %sm merchant windows...",
                window_minutes,
            )

            window_frequency = (
                f"{window_minutes}min"
            )

            # -----------------------------------------------------------------
            # Aggregate merchant windows
            # -----------------------------------------------------------------

            window_frame = aggregate_merchant_windows(
                raw_transactions,
                window_frequency,
            )

            if window_frame.empty:

                raise ValueError(
                    f"{window_minutes}m aggregation produced zero rows."
                )

            # -----------------------------------------------------------------
            # Schema aliases
            # -----------------------------------------------------------------

            self._apply_window_schema_aliases(
                window_frame
            )

            # -----------------------------------------------------------------
            # Expected fraud amount
            # -----------------------------------------------------------------

            self._ensure_expected_fraud_amount(
                window_frame
            )

            # -----------------------------------------------------------------
            # Merchant baseline
            # -----------------------------------------------------------------

            baseline_config = self.cfg[
                "baseline"
            ]

            window_frame = build_merchant_baselines(
                window_frame,
                lookback_hours=int(
                    baseline_config[
                        "lookback_hours"
                    ]
                ),
                min_history_windows=int(
                    baseline_config[
                        "min_history_windows"
                    ]
                ),
                global_reference=window_frame,
                shrinkage_kappa=float(
                    baseline_config[
                        "shrinkage_kappa"
                    ]
                ),
                epsilon=float(
                    baseline_config[
                        "epsilon"
                    ]
                ),
                robust_scale_floor=float(
                    baseline_config[
                        "robust_scale_floor"
                    ]
                ),
            )

            # -----------------------------------------------------------------
            # Statistical signals
            # -----------------------------------------------------------------

            window_frame = add_statistical_signals(
                window_frame,
                epsilon=float(
                    baseline_config[
                        "epsilon"
                    ]
                ),
                transform_tau=float(
                    self.cfg[
                        "scoring"
                    ][
                        "transform_tau"
                    ]
                ),
            )

            # -----------------------------------------------------------------
            # EWMA
            # -----------------------------------------------------------------

            window_frame = add_ewma(
                window_frame,
                value_col="risk_rate",
                alpha=DEFAULT_EWMA_ALPHA,
            )

            if "ewma_residual" not in window_frame.columns:

                window_frame[
                    "ewma_residual"
                ] = 0.0

            window_frame[
                "ewma_evidence"
            ] = _bounded_positive(
                window_frame[
                    "ewma_residual"
                ],
                tau=DEFAULT_EWMA_TAU,
            )

            # -----------------------------------------------------------------
            # CUSUM
            # -----------------------------------------------------------------

            window_frame = add_cusum(
                window_frame,
                signal_col="risk_rate_log_ratio",
                drift=0.5,
            )

            if "cusum_positive" not in window_frame.columns:

                window_frame[
                    "cusum_positive"
                ] = 0.0

            window_frame[
                "cusum_evidence"
            ] = _bounded_positive(
                window_frame[
                    "cusum_positive"
                ],
                tau=DEFAULT_CUSUM_TAU,
            )

            # -----------------------------------------------------------------
            # Statistical intensity fusion
            # -----------------------------------------------------------------

            if "intensity_evidence_raw" not in window_frame.columns:

                window_frame[
                    "intensity_evidence_raw"
                ] = 0.0

            base_intensity_evidence = _clip01(
                window_frame[
                    "intensity_evidence_raw"
                ]
            )

            temporal_signal_confirmation = (
                0.60
                *
                window_frame[
                    "ewma_evidence"
                ]
                +
                0.40
                *
                window_frame[
                    "cusum_evidence"
                ]
            ).clip(
                0.0,
                1.0,
            )

            window_frame[
                "intensity_evidence_raw"
            ] = (
                0.80
                *
                base_intensity_evidence
                +
                0.20
                *
                temporal_signal_confirmation
            ).clip(
                0.0,
                1.0,
            )

            window_frame[
                "statistical_intensity"
            ] = window_frame[
                "intensity_evidence_raw"
            ]

            # Compatibility names.
            window_frame[
                "statistical_signal"
            ] = window_frame[
                "statistical_intensity"
            ]

            window_frame[
                "intensity_signal"
            ] = window_frame[
                "statistical_intensity"
            ]

            # -----------------------------------------------------------------
            # Breadth
            # -----------------------------------------------------------------

            breadth_config = self.cfg[
                "breadth"
            ]

            if breadth_config.get(
                "enabled",
                True,
            ):

                window_frame = add_breadth_score(
                    window_frame,
                    float(
                        breadth_config[
                            "card_saturation"
                        ]
                    ),
                    float(
                        breadth_config[
                            "tx_saturation"
                        ]
                    ),
                    float(
                        breadth_config[
                            "volume_activation"
                        ]
                    ),
                    float(
                        breadth_config[
                            "new_card_weight"
                        ]
                    ),
                    float(
                        breadth_config[
                            "unique_card_weight"
                        ]
                    ),
                )

            else:

                window_frame[
                    "breadth_score"
                ] = 0.0

            if "breadth_score" not in window_frame.columns:

                window_frame[
                    "breadth_score"
                ] = 0.0

            window_frame[
                "breadth_score"
            ] = _clip01(
                window_frame[
                    "breadth_score"
                ]
            )

            # -----------------------------------------------------------------
            # Persistence
            # -----------------------------------------------------------------

            persistence_config = self.cfg[
                "persistence"
            ]

            if persistence_config.get(
                "enabled",
                True,
            ):

                window_frame = add_persistence(
                    window_frame,
                    evidence_col="intensity_evidence_raw",
                    decay=float(
                        persistence_config[
                            "decay"
                        ]
                    ),
                    evidence_threshold=float(
                        persistence_config[
                            "evidence_threshold"
                        ]
                    ),
                    accumulation_step=float(
                        persistence_config[
                            "accumulation_step"
                        ]
                    ),
                    reset_penalty=float(
                        persistence_config[
                            "reset_penalty"
                        ]
                    ),
                )

            else:

                window_frame[
                    "persistence_score"
                ] = 0.0

                window_frame[
                    "persistence_streak"
                ] = 0

            if "persistence_score" not in window_frame.columns:

                window_frame[
                    "persistence_score"
                ] = 0.0

            window_frame[
                "persistence_score"
            ] = _clip01(
                window_frame[
                    "persistence_score"
                ]
            )

            # -----------------------------------------------------------------
            # Materiality
            # -----------------------------------------------------------------

            window_frame = add_materiality(
                window_frame,
                float(
                    self.cfg[
                        "materiality"
                    ][
                        "tau_expected_fraud_amount"
                    ]
                ),
            )

            if "materiality_score" not in window_frame.columns:

                window_frame[
                    "materiality_score"
                ] = 0.0

            window_frame[
                "materiality_score"
            ] = _clip01(
                window_frame[
                    "materiality_score"
                ]
            )

            # -----------------------------------------------------------------
            # Financial evidence
            # -----------------------------------------------------------------

            if "financial_stat_raw" in window_frame.columns:

                financial_statistical_evidence = _clip01(
                    window_frame[
                        "financial_stat_raw"
                    ]
                )

            elif "financial_anomaly" in window_frame.columns:

                financial_statistical_evidence = _clip01(
                    window_frame[
                        "financial_anomaly"
                    ]
                )

            else:

                financial_statistical_evidence = pd.Series(
                    0.0,
                    index=window_frame.index,
                )

            window_frame[
                "financial_stat_raw"
            ] = financial_statistical_evidence

            window_frame[
                "financial_score"
            ] = (
                0.60
                *
                window_frame[
                    "financial_stat_raw"
                ]
                +
                0.40
                *
                window_frame[
                    "materiality_score"
                ]
            ).clip(
                0.0,
                1.0,
            )

            # -----------------------------------------------------------------
            # BASE TAS
            #
            # Acceleration is deliberately excluded.
            # -----------------------------------------------------------------

            scoring_config = self.cfg[
                "scoring"
            ]

            intensity_weight = float(
                scoring_config[
                    "intensity_weight"
                ]
            )

            persistence_weight = float(
                scoring_config[
                    "persistence_weight"
                ]
            )

            breadth_weight = float(
                scoring_config[
                    "breadth_weight"
                ]
            )

            financial_weight = float(
                scoring_config[
                    "financial_weight"
                ]
            )

            acceleration_weight = float(
                scoring_config[
                    "acceleration_weight"
                ]
            )

            base_weight_sum = (
                intensity_weight
                +
                persistence_weight
                +
                breadth_weight
                +
                financial_weight
            )

            if base_weight_sum <= EPSILON:

                raise ValueError(
                    "BASE TAS component weight sum must be positive."
                )

            normalized_intensity_weight = (
                intensity_weight
                /
                base_weight_sum
            )

            normalized_persistence_weight = (
                persistence_weight
                /
                base_weight_sum
            )

            normalized_breadth_weight = (
                breadth_weight
                /
                base_weight_sum
            )

            normalized_financial_weight = (
                financial_weight
                /
                base_weight_sum
            )

            window_frame[
                "intensity_score"
            ] = _clip01(
                window_frame[
                    "intensity_evidence_raw"
                ]
            )

            window_frame[
                "persistence_component"
            ] = _clip01(
                window_frame[
                    "persistence_score"
                ]
            )

            window_frame[
                "breadth_component"
            ] = _clip01(
                window_frame[
                    "breadth_score"
                ]
            )

            window_frame[
                "tas_base"
            ] = (
                100.0
                *
                (
                    normalized_intensity_weight
                    *
                    window_frame[
                        "intensity_score"
                    ]
                    +
                    normalized_persistence_weight
                    *
                    window_frame[
                        "persistence_component"
                    ]
                    +
                    normalized_breadth_weight
                    *
                    window_frame[
                        "breadth_component"
                    ]
                    +
                    normalized_financial_weight
                    *
                    window_frame[
                        "financial_score"
                    ]
                )
            ).clip(
                0.0,
                100.0,
            )

            # -----------------------------------------------------------------
            # ACCELERATION FROM BASE TAS
            # -----------------------------------------------------------------

            acceleration_config = self.cfg[
                "acceleration"
            ]

            if acceleration_config.get(
                "enabled",
                True,
            ):

                window_frame = add_acceleration(
                    window_frame,
                    tas_base_col=str(
                        acceleration_config[
                            "tas_base_column"
                        ]
                    ),
                    max_delta_for_full_score=float(
                        acceleration_config[
                            "max_delta_for_full_score"
                        ]
                    ),
                    min_previous_tas=float(
                        acceleration_config[
                            "min_previous_tas"
                        ]
                    ),
                )

            else:

                window_frame[
                    "acceleration_score"
                ] = 0.0

            if "acceleration_score" not in window_frame.columns:

                window_frame[
                    "acceleration_score"
                ] = 0.0

            window_frame[
                "acceleration_score"
            ] = _clip01(
                window_frame[
                    "acceleration_score"
                ]
            )

            # -----------------------------------------------------------------
            # FINAL TAS
            # -----------------------------------------------------------------

            window_frame[
                "acceleration_component"
            ] = (
                window_frame[
                    "acceleration_score"
                ]
                *
                window_frame[
                    "persistence_component"
                ]
            ).clip(
                0.0,
                1.0,
            )

            window_frame[
                "tas"
            ] = (
                100.0
                *
                (
                    intensity_weight
                    *
                    window_frame[
                        "intensity_score"
                    ]
                    +
                    persistence_weight
                    *
                    window_frame[
                        "persistence_component"
                    ]
                    +
                    acceleration_weight
                    *
                    window_frame[
                        "acceleration_component"
                    ]
                    +
                    breadth_weight
                    *
                    window_frame[
                        "breadth_component"
                    ]
                    +
                    financial_weight
                    *
                    window_frame[
                        "financial_score"
                    ]
                )
            ).clip(
                0.0,
                100.0,
            )

            window_frame[
                "temporal_anomaly_score"
            ] = window_frame[
                "tas"
            ]

            # -----------------------------------------------------------------
            # Downstream schema
            # -----------------------------------------------------------------

            self._ensure_downstream_window_columns(
                window_frame
            )

            # -----------------------------------------------------------------
            # Downcast
            # -----------------------------------------------------------------

            window_frame = _downcast_numeric(
                window_frame
            )

            # -----------------------------------------------------------------
            # Persist resolution artifact
            # -----------------------------------------------------------------

            resolution_output_path = (
                windows_directory
                /
                f"merchant_spike_windows_{window_minutes}m.csv"
            )

            window_frame.to_csv(
                resolution_output_path,
                index=False,
            )

            self.logger.info(
                "%sm window artifact written | rows=%d",
                window_minutes,
                len(window_frame),
            )

            # -----------------------------------------------------------------
            # Retain minimum multiscale frame
            # -----------------------------------------------------------------

            multiscale_frame = window_frame[
                [
                    "merchant_id",
                    "window_start",
                    "tas",
                ]
            ].copy()

            multiscale_frame[
                "merchant_id"
            ] = multiscale_frame[
                "merchant_id"
            ].astype(
                str
            )

            multiscale_frame[
                "window_start"
            ] = pd.to_datetime(
                multiscale_frame[
                    "window_start"
                ],
                errors="coerce",
                utc=True,
            )

            multiscale_tas_frames[
                window_minutes
            ] = multiscale_frame

            # -----------------------------------------------------------------
            # Retain rich 15m decision frame only
            # -----------------------------------------------------------------

            if (
                window_minutes
                == event_window_minutes
            ):

                decision_frame = window_frame

            else:

                del window_frame

                gc.collect()

        # =====================================================================
        # 15M DECISION FRAME CHECK
        # =====================================================================

        if decision_frame is None:

            raise RuntimeError(
                "15m decision frame was not produced."
            )

        # =====================================================================
        # MULTISCALE CORROBORATION
        # =====================================================================

        self.logger.info(
            "Building multiscale corroboration..."
        )

        multiscale_config = self.cfg[
            "multiscale"
        ]

        corroboration_frame = (
            add_multiscale_corroboration(
                multiscale_tas_frames,
                float(
                    multiscale_config[
                        "fast_weight"
                    ]
                ),
                float(
                    multiscale_config[
                        "medium_weight"
                    ]
                ),
                float(
                    multiscale_config[
                        "slow_weight"
                    ]
                ),
                float(
                    multiscale_config[
                        "redundancy_penalty_fast_medium"
                    ]
                ),
                float(
                    multiscale_config[
                        "redundancy_penalty_medium_slow"
                    ]
                ),
                float(
                    multiscale_config[
                        "minimum_incremental_signal"
                    ]
                ),
            )
        )

        del multiscale_tas_frames

        gc.collect()

        # ---------------------------------------------------------------------
        # Select only required corroboration columns
        # ---------------------------------------------------------------------

        desired_corroboration_columns = [
            "merchant_id",
            "window_start",
            "tas_5m",
            "tas_15m",
            "tas_60m",
            "incremental_fast",
            "incremental_medium",
            "incremental_slow",
            "corroboration_score",
        ]

        available_corroboration_columns = [
            column_name
            for column_name
            in desired_corroboration_columns
            if column_name
            in corroboration_frame.columns
        ]

        if (
            "merchant_id"
            not in available_corroboration_columns
            or
            "window_start"
            not in available_corroboration_columns
        ):

            raise ValueError(
                "Multiscale corroboration output is missing "
                "merchant_id/window_start keys."
            )

        # ---------------------------------------------------------------------
        # Ensure one-to-one join
        # ---------------------------------------------------------------------

        duplicate_key_mask = (
            corroboration_frame
            .duplicated(
                subset=[
                    "merchant_id",
                    "window_start",
                ],
                keep=False,
            )
        )

        if duplicate_key_mask.any():

            duplicate_key_count = int(
                duplicate_key_mask.sum()
            )

            raise ValueError(
                "Multiscale corroboration contains duplicate "
                "merchant/window keys: "
                f"{duplicate_key_count}"
            )

        # ---------------------------------------------------------------------
        # Merge multiscale information
        # ---------------------------------------------------------------------

        decision_frame = decision_frame.merge(
            corroboration_frame[
                available_corroboration_columns
            ],
            on=[
                "merchant_id",
                "window_start",
            ],
            how="left",
            validate="one_to_one",
        )

        del corroboration_frame

        gc.collect()

        if (
            "corroboration_score"
            not in decision_frame.columns
        ):

            decision_frame[
                "corroboration_score"
            ] = 0.0

        decision_frame[
            "corroboration_score"
        ] = _clip01(
            decision_frame[
                "corroboration_score"
            ]
        )

        # =====================================================================
        # ROBUST VERIFICATION
        # =====================================================================

        self.logger.info(
            "Applying robust vectorized verification..."
        )

        decision_frame = self._apply_verification(
            decision_frame
        )

        self._log_verification_summary(
            decision_frame
        )

        # =====================================================================
        # FRAUD ATTACK SCORE
        # =====================================================================

        self.logger.info(
            "Calculating attack risk scores..."
        )

        decision_frame = self._apply_attack_scores(
            decision_frame
        )

        # =====================================================================
        # OPERATIONAL HYSTERESIS
        # =====================================================================

        decision_frame = self._apply_hysteresis(
            decision_frame
        )

        # =====================================================================
        # FINAL 15M WINDOW OUTPUT
        # =====================================================================

        final_window_output_path = _resolve_path(
            output_config[
                "window_file"
            ]
        )

        _ensure_dir(
            final_window_output_path.parent
        )

        decision_frame.to_csv(
            final_window_output_path,
            index=False,
        )

        self.logger.info(
            "Final 15m decision windows written: %s",
            final_window_output_path,
        )

        # =====================================================================
        # EVENT CONSTRUCTION
        # =====================================================================

        self.logger.info(
            "Building fraud-spike events..."
        )

        events = self._build_events(
            decision_frame
        )

        # =====================================================================
        # EXPLANATIONS
        # =====================================================================

        if self.cfg.get(
            "explanations",
            {},
        ).get(
            "enabled",
            True,
        ):

            try:

                events = build_explanations(
                    events
                )

            except Exception as exc:

                self.logger.warning(
                    "Explanation builder failed; "
                    "preserving pipeline-generated explanations: %s",
                    exc,
                )

        # =====================================================================
        # EVENT OUTPUTS
        # =====================================================================

        event_output_path = _resolve_path(
            output_config[
                "event_file"
            ]
        )

        explanation_output_path = _resolve_path(
            output_config[
                "explanation_file"
            ]
        )

        _ensure_dir(
            event_output_path.parent
        )

        _ensure_dir(
            explanation_output_path.parent
        )

        events.to_csv(
            event_output_path,
            index=False,
        )

        if events.empty:

            explanation_dataframe = pd.DataFrame(
                columns=[
                    "event_id",
                    "merchant_id",
                    "detection_time",
                    "peak_state",
                    "explanation",
                ]
            )

        else:

            explanation_columns = [
                column_name
                for column_name
                in [
                    "event_id",
                    "merchant_id",
                    "detection_time",
                    "peak_state",
                    "explanation",
                ]
                if column_name in events.columns
            ]

            explanation_dataframe = events[
                explanation_columns
            ].copy()

        explanation_dataframe.to_csv(
            explanation_output_path,
            index=False,
        )

        # =====================================================================
        # EVALUATION
        # =====================================================================

        transaction_metrics: dict = {}
        event_metrics: dict = {}

        evaluation_config = self.cfg.get(
            "evaluation",
            {},
        )

        if evaluation_config.get(
            "enabled",
            True,
        ):

            if evaluation_config.get(
                "transaction_level",
                {},
            ).get(
                "enabled",
                True,
            ):

                transaction_metrics = (
                    evaluate_transaction_level(
                        decision_frame
                    )
                )

            if evaluation_config.get(
                "event_level",
                {},
            ).get(
                "enabled",
                True,
            ):

                event_metrics = (
                    evaluate_events(
                        events
                    )
                )

        # =====================================================================
        # FINAL METRICS
        # =====================================================================

        runtime_seconds = (
            time.time()
            -
            started_at
        )

        metrics = {
            "pipeline_version":
                self.cfg[
                    "pipeline"
                ][
                    "version"
                ],

            "detector_version":
                self.cfg[
                    "pipeline"
                ][
                    "detector_version"
                ],

            "input_phase1_risk_output":
                str(
                    input_path
                ),

            "input_rows":
                input_row_count,

            "merchant_count":
                merchant_count,

            "window_counts":
                self._calculate_window_counts(
                    windows_directory,
                    decision_frame,
                ),

            "verification_summary":
                self._build_verification_summary(
                    decision_frame
                ),

            "score_summary":
                self._build_score_summary(
                    decision_frame
                ),

            "events":
                self._event_summary(
                    events
                ),

            "event_severity_distribution":
                self._event_severity_distribution(
                    events
                ),

            "transaction_level":
                transaction_metrics,

            "event_level":
                event_metrics,

            "runtime_seconds":
                runtime_seconds,
        }

        # =====================================================================
        # METRICS FILE
        # =====================================================================

        metrics_output_path = (
            evaluation_directory
            /
            "phase2_metrics.json"
        )

        write_evaluation(
            metrics,
            str(
                metrics_output_path
            ),
        )

        # =====================================================================
        # RUN METADATA
        # =====================================================================

        metadata_output_path = _resolve_path(
            output_config[
                "metadata_file"
            ]
        )

        _ensure_dir(
            metadata_output_path.parent
        )

        metadata = {
            **metrics,

            "effective_config":
                self.cfg,

            "verification_design":
                {
                    "normal_path":
                        "multi_transaction_evidence",

                    "exceptional_path":
                        "single_transaction_high_loss",

                    "temporal_evidence":
                        "at_least_two_of_persistence_corroboration_acceleration",

                    "coordination_evidence":
                        "unique_cards_or_new_cards_or_breadth_score",

                    "composite_evidence":
                        "weighted_geometric_mean",
                },
        }

        metadata_output_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        # =====================================================================
        # FINAL LOGGING
        # =====================================================================

        verified_window_count = int(
            decision_frame[
                "verified_spike"
            ].sum()
        )

        normal_verified_window_count = int(
            decision_frame[
                "normal_verification_gate"
            ].sum()
        )

        exceptional_window_count = int(
            decision_frame[
                "exceptional_verification_gate"
            ].sum()
        )

        critical_window_count = int(
            decision_frame[
                "critical_spike"
            ].sum()
        )

        self.logger.info(
            "Phase 2 complete | "
            "candidate=%d | "
            "verified=%d | "
            "normal_verified=%d | "
            "exceptional=%d | "
            "critical=%d | "
            "events=%d | "
            "runtime=%.2fs",
            int(
                decision_frame[
                    "candidate_spike"
                ].sum()
            ),
            verified_window_count,
            normal_verified_window_count,
            exceptional_window_count,
            critical_window_count,
            len(events),
            runtime_seconds,
        )

        # =====================================================================
        # CLEANUP
        # =====================================================================

        del raw_transactions

        gc.collect()

        return metrics

    # =========================================================================
    # WINDOW SCHEMA ALIASES
    # =========================================================================

    @staticmethod
    def _apply_window_schema_aliases(
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Normalize known aggregation aliases.
        """

        if (
            "risk_rate"
            not in dataframe.columns
            and
            "fraud_risk_rate"
            in dataframe.columns
        ):

            dataframe[
                "risk_rate"
            ] = dataframe[
                "fraud_risk_rate"
            ]

        if (
            "transaction_count"
            not in dataframe.columns
            and
            "txn_count"
            in dataframe.columns
        ):

            dataframe[
                "transaction_count"
            ] = dataframe[
                "txn_count"
            ]

        if (
            "total_amount"
            not in dataframe.columns
            and
            "amount"
            in dataframe.columns
        ):

            dataframe[
                "total_amount"
            ] = dataframe[
                "amount"
            ]

    # =========================================================================
    # EXPECTED FRAUD AMOUNT
    # =========================================================================

    @staticmethod
    def _ensure_expected_fraud_amount(
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Ensure expected_fraud_amount is available.

        Existing calculated values are preserved.
        """

        if (
            "expected_fraud_amount"
            in dataframe.columns
        ):

            dataframe[
                "expected_fraud_amount"
            ] = _numeric(
                dataframe,
                "expected_fraud_amount",
            ).clip(
                lower=0.0
            )

            return

        if (
            "expected_fraud_count"
            not in dataframe.columns
        ):

            dataframe[
                "expected_fraud_count"
            ] = 0.0

        if (
            "transaction_count"
            not in dataframe.columns
        ):

            dataframe[
                "transaction_count"
            ] = 0.0

        if (
            "total_amount"
            not in dataframe.columns
        ):

            dataframe[
                "total_amount"
            ] = 0.0

        expected_fraud_count = _numeric(
            dataframe,
            "expected_fraud_count",
        ).clip(
            lower=0.0
        )

        transaction_count = _numeric(
            dataframe,
            "transaction_count",
        ).clip(
            lower=1.0
        )

        total_amount = _numeric(
            dataframe,
            "total_amount",
        ).clip(
            lower=0.0
        )

        average_transaction_amount = (
            total_amount
            /
            transaction_count
        )

        dataframe[
            "expected_fraud_amount"
        ] = (
            expected_fraud_count
            *
            average_transaction_amount
        ).clip(
            lower=0.0
        )

    # =========================================================================
    # DOWNSTREAM SCHEMA
    # =========================================================================

    @staticmethod
    def _ensure_downstream_window_columns(
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Guarantee columns required by downstream components.
        """

        default_columns = {
            "transaction_count": 0,
            "total_amount": 0.0,
            "expected_fraud_count": 0.0,
            "expected_fraud_amount": 0.0,
            "baseline_expected_fraud_amount_exposure": 0.0,
            "unique_cards": 0,
            "new_cards": 0,
        }

        for column_name, default_value in default_columns.items():

            if column_name not in dataframe.columns:

                dataframe[
                    column_name
                ] = default_value

        # Compatibility with alternate baseline naming.
        if (
            "baseline_expected_fraud_amount_exposure"
            not in dataframe.columns
            and
            "baseline_expected_fraud_amount"
            in dataframe.columns
        ):

            dataframe[
                "baseline_expected_fraud_amount_exposure"
            ] = _numeric(
                dataframe,
                "baseline_expected_fraud_amount",
            )

    # =========================================================================
    # PHASE-1 INPUT VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_phase1_input(
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Validate Phase-1 output contract.
        """

        required_columns = [
            "transaction_id",
            "timestamp",
            "merchant_id",
            "card_id",
            "amount",
            "fraud_probability",
            "risk_score",
            "risk_band",
            "predicted_fraud",
            "model_version",
            "threshold",
        ]

        missing_columns = [
            column_name
            for column_name in required_columns
            if column_name not in dataframe.columns
        ]

        if missing_columns:

            raise ValueError(
                "Phase-1 risk output is missing required columns: "
                f"{missing_columns}"
            )

        if dataframe.empty:

            raise ValueError(
                "Phase-1 risk output is empty."
            )

        if dataframe[
            "transaction_id"
        ].isna().any():

            raise ValueError(
                "Phase-1 risk output contains null transaction IDs."
            )

        if dataframe[
            "transaction_id"
        ].duplicated().any():

            duplicate_count = int(
                dataframe[
                    "transaction_id"
                ].duplicated().sum()
            )

            raise ValueError(
                "Phase-1 risk output contains duplicate transaction IDs: "
                f"{duplicate_count}"
            )

        fraud_probability = _clip01(
            dataframe[
                "fraud_probability"
            ]
        )

        if not np.isfinite(
            fraud_probability.to_numpy(
                dtype=float
            )
        ).all():

            raise ValueError(
                "Phase-1 fraud_probability contains non-finite values."
            )

        risk_score = _numeric(
            dataframe,
            "risk_score",
        )

        if (
            (
                risk_score
                < 0.0
            )
            |
            (
                risk_score
                > 100.0
            )
        ).any():

            raise ValueError(
                "Phase-1 risk_score contains values outside [0, 100]."
            )

    # =========================================================================
    # VERIFICATION
    # =========================================================================

    def _apply_verification(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply the authoritative Phase-2 verification implementation.

        Verification mathematics live in:

            src.components.phase2.verification

        This function only performs integration and compatibility mapping.
        """

        output = add_verification_columns(
            dataframe,
            self.cfg,
        )

        # ---------------------------------------------------------------------
        # New canonical score
        # ---------------------------------------------------------------------

        if (
            "verification_evidence_score"
            in output.columns
        ):

            output[
                "verification_evidence_score"
            ] = _clip01(
                output[
                    "verification_evidence_score"
                ]
            )

            # Backward-compatible alias.
            output[
                "verification_score"
            ] = output[
                "verification_evidence_score"
            ]

        elif (
            "verification_score"
            in output.columns
        ):

            output[
                "verification_score"
            ] = _clip01(
                output[
                    "verification_score"
                ]
            )

            output[
                "verification_evidence_score"
            ] = output[
                "verification_score"
            ]

        else:

            output[
                "verification_evidence_score"
            ] = 0.0

            output[
                "verification_score"
            ] = 0.0

        # ---------------------------------------------------------------------
        # Ensure core verification fields exist
        # ---------------------------------------------------------------------

        required_verification_outputs = {
            "candidate_spike": False,
            "verified_spike": False,
            "critical_spike": False,
            "exceptional_evidence": 0.0,
            "spike_state": "ANOMALY_OR_NORMAL",
            "verification_reason":
                "Verification information unavailable",
        }

        for column_name, default_value in (
            required_verification_outputs.items()
        ):

            if column_name not in output.columns:

                output[
                    column_name
                ] = default_value

        # ---------------------------------------------------------------------
        # Normalize flags
        # ---------------------------------------------------------------------

        for column_name in [
            "candidate_spike",
            "verified_spike",
            "critical_spike",
        ]:

            output[
                column_name
            ] = output[
                column_name
            ].astype(bool)

        output[
            "exceptional_evidence"
        ] = _numeric(
            output,
            "exceptional_evidence",
        ).clip(
            0.0,
            1.0,
        )

        # ---------------------------------------------------------------------
        # Compatibility flags
        # ---------------------------------------------------------------------

        output[
            "exceptional_verification_gate"
        ] = (
            output[
                "exceptional_evidence"
            ]
            > 0.0
        )

        # Normal verified path explicitly excludes exceptional transactions.
        output[
            "normal_verification_gate"
        ] = (
            output[
                "verified_spike"
            ]
            &
            (
                ~output[
                    "exceptional_verification_gate"
                ]
            )
        )

        # ---------------------------------------------------------------------
        # Ensure expected diagnostic gates exist
        # ---------------------------------------------------------------------

        verification_config = self.cfg[
            "verification"
        ]

        materiality_config = self.cfg[
            "materiality"
        ]

        output[
            "verification_tas_gate"
        ] = (
            _numeric(
                output,
                "tas",
            )
            >= float(
                verification_config[
                    "verified_tas"
                ]
            )
        )

        output[
            "verification_materiality_gate"
        ] = (
            _clip01(
                output[
                    "materiality_score"
                ]
            )
            >= float(
                materiality_config[
                    "verified_min"
                ]
            )
        )

        output[
            "verification_history_gate"
        ] = (
            _clip01(
                output[
                    "history_confidence"
                ]
            )
            >= float(
                verification_config[
                    "min_history_confidence"
                ]
            )
        )

        output[
            "verification_persistence_gate"
        ] = (
            _clip01(
                output[
                    "persistence_score"
                ]
            )
            >= float(
                verification_config[
                    "persistence_min"
                ]
            )
        )

        output[
            "verification_corroboration_gate"
        ] = (
            _clip01(
                output[
                    "corroboration_score"
                ]
            )
            >= float(
                verification_config[
                    "corroboration_min"
                ]
            )
        )

        output[
            "multi_transaction_gate"
        ] = (
            _numeric(
                output,
                "transaction_count",
            )
            >= int(
                verification_config[
                    "minimum_transactions"
                ]
            )
        )

        return output

    # =========================================================================
    # VERIFICATION SUMMARY LOGGING
    # =========================================================================

    def _log_verification_summary(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Log verification and threshold diagnostics.
        """

        verification_config = self.cfg[
            "verification"
        ]

        candidate_count = int(
            dataframe[
                "candidate_spike"
            ].sum()
        )

        verified_count = int(
            dataframe[
                "verified_spike"
            ].sum()
        )

        normal_verified_count = int(
            dataframe[
                "normal_verification_gate"
            ].sum()
        )

        exceptional_count = int(
            dataframe[
                "exceptional_verification_gate"
            ].sum()
        )

        critical_count = int(
            dataframe[
                "critical_spike"
            ].sum()
        )

        max_tas = _safe_max(
            dataframe,
            "tas",
        )

        max_corroboration = _safe_max(
            dataframe,
            "corroboration_score",
        )

        max_persistence = _safe_max(
            dataframe,
            "persistence_score",
        )

        max_evidence_score = _safe_max(
            dataframe,
            "verification_evidence_score",
        )

        self.logger.info(
            "Verification summary | "
            "candidate=%d | "
            "verified=%d | "
            "normal_verified=%d | "
            "exceptional=%d | "
            "critical=%d | "
            "max_tas=%.4f | "
            "max_corroboration=%.4f | "
            "max_persistence=%.4f | "
            "max_evidence=%.4f",
            candidate_count,
            verified_count,
            normal_verified_count,
            exceptional_count,
            critical_count,
            max_tas,
            max_corroboration,
            max_persistence,
            max_evidence_score,
        )

        self.logger.info(
            "Verification thresholds | "
            "candidate_tas=%.3f | "
            "verified_tas=%.3f | "
            "critical_tas=%.3f | "
            "persistence_min=%.3f | "
            "corroboration_min=%.3f | "
            "minimum_transactions=%d",
            float(
                verification_config[
                    "candidate_tas"
                ]
            ),
            float(
                verification_config[
                    "verified_tas"
                ]
            ),
            float(
                verification_config[
                    "critical_tas"
                ]
            ),
            float(
                verification_config[
                    "persistence_min"
                ]
            ),
            float(
                verification_config[
                    "corroboration_min"
                ]
            ),
            int(
                verification_config[
                    "minimum_transactions"
                ]
            ),
        )

    # =========================================================================
    # ATTACK / EXCEPTIONAL RISK SCORES
    # =========================================================================

    def _apply_attack_scores(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate:

            fraud_attack_score
            exceptional_risk_score
            event_peak_risk_score

        The two paths are deliberately separated.

        fraud_attack_score:
            coordinated verified fraud attack

        exceptional_risk_score:
            exceptional single transaction

        event_peak_risk_score:
            unified operational peak score
        """

        attack_config = self.cfg.get(
            "fraud_attack_score",
            {},
        )

        if not attack_config.get(
            "enabled",
            True,
        ):

            dataframe[
                "fraud_attack_score"
            ] = 0.0

            dataframe[
                "exceptional_risk_score"
            ] = 0.0

            dataframe[
                "fraud_attack_score_normalized"
            ] = 0.0

            dataframe[
                "exceptional_risk_score_normalized"
            ] = 0.0

            dataframe[
                "event_peak_risk_score"
            ] = 0.0

            return dataframe

        attack_gamma = float(
            attack_config.get(
                "attack_gamma",
                DEFAULT_ATTACK_GAMMA,
            )
        )

        # ---------------------------------------------------------------------
        # TAS strength
        # ---------------------------------------------------------------------

        tas_normalized = (
            _numeric(
                dataframe,
                "tas",
            )
            /
            100.0
        ).clip(
            0.0,
            1.0,
        )

        tas_strength = (
            tas_normalized
            **
            attack_gamma
        ).clip(
            0.0,
            1.0,
        )

        # ---------------------------------------------------------------------
        # Verification evidence
        # ---------------------------------------------------------------------

        verification_evidence = _clip01(
            dataframe[
                "verification_evidence_score"
            ]
        )

        materiality_score = _clip01(
            dataframe[
                "materiality_score"
            ]
        )

        # ---------------------------------------------------------------------
        # Joint coordinated attack evidence
        # ---------------------------------------------------------------------

        dataframe[
            "joint_attack_evidence"
        ] = (
            verification_evidence
            *
            materiality_score
        ).clip(
            0.0,
            1.0,
        )

        raw_attack_score = (
            100.0
            *
            tas_strength
            *
            dataframe[
                "joint_attack_evidence"
            ]
        ).clip(
            0.0,
            100.0,
        )

        coordinated_verified_path = (
            dataframe[
                "normal_verification_gate"
            ]
            |
            dataframe[
                "critical_spike"
            ]
        )

        dataframe[
            "fraud_attack_score"
        ] = raw_attack_score.where(
            coordinated_verified_path,
            0.0,
        )

        # ---------------------------------------------------------------------
        # Exceptional single-transaction risk
        # ---------------------------------------------------------------------

        materiality_config = self.cfg[
            "materiality"
        ]

        exceptional_tas_threshold = float(
            materiality_config.get(
                "exceptional_min_tas",
                45.0,
            )
        )

        amount_scale = max(
            float(
                materiality_config.get(
                    "tau_expected_fraud_amount",
                    25.0,
                )
            ),
            EPSILON,
        )

        expected_fraud_amount = _numeric(
            dataframe,
            "expected_fraud_amount",
        ).clip(
            lower=0.0
        )

        exceptional_amount_evidence = (
            1.0
            -
            np.exp(
                -expected_fraud_amount
                /
                amount_scale
            )
        ).clip(
            0.0,
            1.0,
        )

        exceptional_tas_evidence = (
            _numeric(
                dataframe,
                "tas",
            )
            /
            max(
                exceptional_tas_threshold,
                EPSILON,
            )
        ).clip(
            0.0,
            1.0,
        )

        exceptional_materiality_evidence = (
            materiality_score
        )

        exceptional_risk_score = (
            100.0
            *
            (
                exceptional_tas_evidence
                *
                exceptional_amount_evidence
                *
                exceptional_materiality_evidence
            )
            **
            (1.0 / 3.0)
        ).clip(
            0.0,
            100.0,
        )

        dataframe[
            "exceptional_risk_score"
        ] = exceptional_risk_score.where(
            dataframe[
                "exceptional_verification_gate"
            ],
            0.0,
        )

        # ---------------------------------------------------------------------
        # Normalized score outputs
        # ---------------------------------------------------------------------

        dataframe[
            "fraud_attack_score_normalized"
        ] = (
            dataframe[
                "fraud_attack_score"
            ]
            /
            100.0
        ).clip(
            0.0,
            1.0,
        )

        dataframe[
            "exceptional_risk_score_normalized"
        ] = (
            dataframe[
                "exceptional_risk_score"
            ]
            /
            100.0
        ).clip(
            0.0,
            1.0,
        )

        # ---------------------------------------------------------------------
        # Unified event risk
        # ---------------------------------------------------------------------

        dataframe[
            "event_peak_risk_score"
        ] = np.maximum(
            dataframe[
                "fraud_attack_score"
            ].to_numpy(
                dtype=float
            ),
            dataframe[
                "exceptional_risk_score"
            ].to_numpy(
                dtype=float
            ),
        )

        dataframe[
            "event_peak_risk_score"
        ] = pd.Series(
            dataframe[
                "event_peak_risk_score"
            ],
            index=dataframe.index,
        ).clip(
            0.0,
            100.0,
        )

        return dataframe

    # =========================================================================
    # OPERATIONAL HYSTERESIS
    # =========================================================================

    def _apply_hysteresis(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply operational hysteresis.

        Analytical verification fields are not overwritten.
        """

        sorted_dataframe = dataframe.sort_values(
            [
                "merchant_id",
                "window_start",
            ]
        ).copy()

        hysteresis_config = self.cfg.get(
            "hysteresis",
            {},
        )

        high_enter_threshold = float(
            hysteresis_config.get(
                "high_enter",
                40.0,
            )
        )

        high_exit_threshold = float(
            hysteresis_config.get(
                "high_exit",
                32.0,
            )
        )

        critical_enter_threshold = float(
            hysteresis_config.get(
                "critical_enter",
                70.0,
            )
        )

        critical_exit_threshold = float(
            hysteresis_config.get(
                "critical_exit",
                58.0,
            )
        )

        operational_states: dict[Any, str] = {}

        for merchant_id, merchant_windows in sorted_dataframe.groupby(
            "merchant_id",
            sort=False,
        ):

            high_state_active = False
            critical_state_active = False

            for index, row in merchant_windows.iterrows():

                tas_value = _finite_float(
                    row.get(
                        "tas",
                        0.0,
                    )
                )

                # -------------------------------------------------------------
                # Critical hysteresis
                # -------------------------------------------------------------

                if critical_state_active:

                    if (
                        tas_value
                        <
                        critical_exit_threshold
                    ):

                        critical_state_active = False

                elif (
                    tas_value
                    >= critical_enter_threshold
                ):

                    critical_state_active = True

                # -------------------------------------------------------------
                # High hysteresis
                # -------------------------------------------------------------

                if high_state_active:

                    if (
                        tas_value
                        <
                        high_exit_threshold
                    ):

                        high_state_active = False

                elif (
                    tas_value
                    >= high_enter_threshold
                ):

                    high_state_active = True

                # -------------------------------------------------------------
                # Analytical classification takes precedence.
                # -------------------------------------------------------------

                if bool(
                    row.get(
                        "critical_spike",
                        False,
                    )
                ):

                    state = (
                        "CRITICAL_ACTIVE_SPIKE"
                    )

                elif bool(
                    row.get(
                        "normal_verification_gate",
                        False,
                    )
                ):

                    state = (
                        "VERIFIED_FRAUD_SPIKE"
                    )

                elif bool(
                    row.get(
                        "exceptional_verification_gate",
                        False,
                    )
                ):

                    state = (
                        "EXCEPTIONAL_SINGLE_TRANSACTION"
                    )

                elif bool(
                    row.get(
                        "candidate_spike",
                        False,
                    )
                ):

                    state = (
                        "CANDIDATE_SPIKE"
                    )

                elif critical_state_active:

                    state = (
                        "CRITICAL_ACTIVE_STATE"
                    )

                elif high_state_active:

                    state = (
                        "HIGH_RISK_STATE"
                    )

                else:

                    state = "NORMAL"

                operational_states[
                    index
                ] = state

        sorted_dataframe[
            "operational_state"
        ] = [
            operational_states.get(
                index,
                "NORMAL",
            )
            for index
            in sorted_dataframe.index
        ]

        return sorted_dataframe

    # =========================================================================
    # EVENT CONSTRUCTION
    # =========================================================================

    def _build_events(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construct merchant-level fraud-spike events.

        Normal event:
            requires a verified multi-transaction anchor.

        Exceptional event:
            may be a single-window exceptional transaction.

        Candidate-only windows do not become final events without an anchor.
        """

        alert_config = self.cfg[
            "alerts"
        ]

        # ---------------------------------------------------------------------
        # IMPORTANT:
        # Read event_window_minutes HERE.
        #
        # It is local to this method and therefore cannot rely on the
        # variable created inside _run_impl().
        # ---------------------------------------------------------------------

        event_window_minutes = int(
            self.cfg[
                "windows"
            ][
                "event_window_minutes"
            ]
        )

        candidate_event_mask = (
            dataframe[
                "candidate_spike"
            ].astype(bool)
            |
            dataframe[
                "verified_spike"
            ].astype(bool)
            |
            dataframe[
                "exceptional_verification_gate"
            ].astype(bool)
        )

        active_windows = dataframe.loc[
            candidate_event_mask
        ].copy()

        if active_windows.empty:

            return self._empty_event_dataframe()

        active_windows = active_windows.sort_values(
            [
                "merchant_id",
                "window_start",
            ]
        )

        maximum_event_gap = pd.Timedelta(
            minutes=float(
                alert_config.get(
                    "max_event_gap_minutes",
                    event_window_minutes,
                )
            )
        )

        minimum_event_windows = int(
            alert_config.get(
                "minimum_event_windows",
                2,
            )
        )

        minimum_verified_windows = int(
            alert_config.get(
                "minimum_verified_windows",
                1,
            )
        )

        allow_exceptional_single_window = bool(
            alert_config.get(
                "allow_exceptional_single_window",
                True,
            )
        )

        require_verified_anchor = bool(
            alert_config.get(
                "require_verified_anchor",
                True,
            )
        )

        events: list[dict[str, Any]] = []

        event_number = 0

        # =====================================================================
        # GROUP BY MERCHANT
        # =====================================================================

        for merchant_id, merchant_windows in active_windows.groupby(
            "merchant_id",
            sort=False,
        ):

            merchant_windows = merchant_windows.sort_values(
                "window_start"
            ).copy()

            episode_group_ids = (
                merchant_windows[
                    "window_start"
                ]
                .diff()
                .gt(
                    maximum_event_gap
                )
                .cumsum()
            )

            # =================================================================
            # PROCESS EACH EPISODE
            # =================================================================

            for _, episode in merchant_windows.groupby(
                episode_group_ids,
                sort=False,
            ):

                episode = episode.sort_values(
                    "window_start"
                ).copy()

                exceptional_episode = bool(
                    episode[
                        "exceptional_verification_gate"
                    ]
                    .astype(bool)
                    .any()
                )

                normal_verified_episode = episode.loc[
                    episode[
                        "normal_verification_gate"
                    ].astype(bool)
                ]

                normal_verified_window_count = int(
                    len(
                        normal_verified_episode
                    )
                )

                # -------------------------------------------------------------
                # Verified anchor requirement
                # -------------------------------------------------------------

                if (
                    require_verified_anchor
                    and
                    normal_verified_window_count
                    <
                    minimum_verified_windows
                    and
                    not (
                        allow_exceptional_single_window
                        and
                        exceptional_episode
                    )
                ):

                    continue

                # -------------------------------------------------------------
                # Minimum episode length
                # -------------------------------------------------------------

                if (
                    len(episode)
                    <
                    minimum_event_windows
                    and
                    not (
                        allow_exceptional_single_window
                        and
                        exceptional_episode
                    )
                ):

                    continue

                event_number += 1

                # -------------------------------------------------------------
                # Event timing
                # -------------------------------------------------------------

                event_start_time = episode[
                    "window_start"
                ].min()

                last_window_start_time = episode[
                    "window_start"
                ].max()

                event_end_time = (
                    last_window_start_time
                    +
                    pd.Timedelta(
                        minutes=event_window_minutes
                    )
                )

                event_duration_minutes = float(
                    (
                        event_end_time
                        -
                        event_start_time
                    ).total_seconds()
                    /
                    60.0
                )

                # -------------------------------------------------------------
                # Detection time
                #
                # Normal:
                #     first verified multi-transaction window.
                #
                # Exceptional-only:
                #     first exceptional window.
                # -------------------------------------------------------------

                if not normal_verified_episode.empty:

                    detection_row = (
                        normal_verified_episode
                        .sort_values(
                            "window_start"
                        )
                        .iloc[0]
                    )

                else:

                    exceptional_windows = episode.loc[
                        episode[
                            "exceptional_verification_gate"
                        ].astype(bool)
                    ]

                    if not exceptional_windows.empty:

                        detection_row = (
                            exceptional_windows
                            .sort_values(
                                "window_start"
                            )
                            .iloc[0]
                        )

                    else:

                        detection_row = (
                            episode
                            .sort_values(
                                "window_start"
                            )
                            .iloc[0]
                        )

                # -------------------------------------------------------------
                # Peak risk
                # -------------------------------------------------------------

                peak_risk_scores = _numeric(
                    episode,
                    "event_peak_risk_score",
                )

                if peak_risk_scores.empty:

                    peak_index = (
                        _numeric(
                            episode,
                            "tas",
                        )
                        .idxmax()
                    )

                else:

                    peak_index = (
                        peak_risk_scores
                        .idxmax()
                    )

                peak_row = episode.loc[
                    peak_index
                ]

                peak_tas = _finite_float(
                    peak_row.get(
                        "tas",
                        0.0,
                    )
                )

                peak_fraud_attack_score = _finite_float(
                    peak_row.get(
                        "fraud_attack_score",
                        0.0,
                    )
                )

                peak_exceptional_risk_score = _finite_float(
                    peak_row.get(
                        "exceptional_risk_score",
                        0.0,
                    )
                )

                peak_event_risk_score = _finite_float(
                    peak_row.get(
                        "event_peak_risk_score",
                        max(
                            peak_fraud_attack_score,
                            peak_exceptional_risk_score,
                        ),
                    )
                )

                # -------------------------------------------------------------
                # Event-wide unique cards
                # -------------------------------------------------------------

                (
                    unique_card_count,
                    card_union_available,
                ) = self._calculate_event_unique_cards(
                    episode
                )

                # -------------------------------------------------------------
                # Event-wide new cards
                # -------------------------------------------------------------

                new_card_count = (
                    self._calculate_event_new_cards(
                        episode
                    )
                )

                # -------------------------------------------------------------
                # Financial aggregates
                # -------------------------------------------------------------

                event_transaction_count = int(
                    _safe_sum(
                        episode,
                        "transaction_count",
                    )
                )

                event_total_amount = _safe_sum(
                    episode,
                    "total_amount",
                )

                event_expected_fraud_count = _safe_sum(
                    episode,
                    "expected_fraud_count",
                )

                event_expected_fraud_amount = _safe_sum(
                    episode,
                    "expected_fraud_amount",
                )

                baseline_expected_fraud_amount = _safe_sum(
                    episode,
                    "baseline_expected_fraud_amount_exposure",
                )

                # -------------------------------------------------------------
                # Spike ratio
                # -------------------------------------------------------------

                if (
                    baseline_expected_fraud_amount
                    <= EPSILON
                ):

                    if (
                        event_expected_fraud_amount
                        >
                        EPSILON
                    ):

                        spike_ratio = float(
                            "inf"
                        )

                    else:

                        spike_ratio = 0.0

                else:

                    spike_ratio = (
                        event_expected_fraud_amount
                        /
                        baseline_expected_fraud_amount
                    )

                # -------------------------------------------------------------
                # Verification reasons
                # -------------------------------------------------------------

                if (
                    "verification_reason"
                    in episode.columns
                ):

                    verification_reasons = (
                        episode[
                            "verification_reason"
                        ]
                        .astype(str)
                        .drop_duplicates()
                        .tolist()
                    )

                else:

                    verification_reasons = []

                verification_path = (
                    " | ".join(
                        verification_reasons
                    )
                )

                # -------------------------------------------------------------
                # Verification class
                # -------------------------------------------------------------

                if (
                    exceptional_episode
                    and
                    normal_verified_window_count == 0
                ):

                    verification_class = (
                        "EXCEPTIONAL_SINGLE_TRANSACTION"
                    )

                elif normal_verified_window_count > 0:

                    verification_class = (
                        "VERIFIED_FRAUD_SPIKE"
                    )

                else:

                    verification_class = (
                        "CANDIDATE_SPIKE"
                    )

                # -------------------------------------------------------------
                # Event severity
                # -------------------------------------------------------------

                if bool(
                    peak_row.get(
                        "critical_spike",
                        False,
                    )
                ):

                    event_severity = (
                        "CRITICAL_ACTIVE_SPIKE"
                    )

                elif normal_verified_window_count > 0:

                    event_severity = (
                        "VERIFIED_FRAUD_SPIKE"
                    )

                elif exceptional_episode:

                    event_severity = (
                        "EXCEPTIONAL_SINGLE_TRANSACTION"
                    )

                else:

                    event_severity = (
                        "CANDIDATE_SPIKE"
                    )

                # -------------------------------------------------------------
                # Event ID
                # -------------------------------------------------------------

                event_id = (
                    f"FS2-{event_number:06d}"
                )

                # -------------------------------------------------------------
                # Event-level transaction interpretation
                # -------------------------------------------------------------

                single_transaction_event = (
                    event_transaction_count
                    ==
                    1
                )

                # -------------------------------------------------------------
                # Explanation
                # -------------------------------------------------------------

                explanation = (
                    f"Merchant {merchant_id} "
                    f"entered {event_severity}. "
                    f"Peak TAS={peak_tas:.2f}, "
                    f"event risk score={peak_event_risk_score:.2f}, "
                    f"expected fraud amount="
                    f"{event_expected_fraud_amount:.2f}, "
                    f"transactions={event_transaction_count}, "
                    f"unique cards={unique_card_count}, "
                    f"new cards={new_card_count}, "
                    f"windows={len(episode)}, "
                    f"duration={event_duration_minutes:.1f} minutes."
                )

                if exceptional_episode:

                    explanation += (
                        " Exceptional single-transaction evidence "
                        "was present."
                    )

                if normal_verified_window_count > 0:

                    explanation += (
                        f" Verified multi-transaction anchor count="
                        f"{normal_verified_window_count}."
                    )

                if card_union_available:

                    explanation += (
                        " Unique cards were calculated using "
                        "event-wide union."
                    )

                # -------------------------------------------------------------
                # Append event record
                # -------------------------------------------------------------

                events.append(
                    {
                        "event_id":
                            event_id,

                        "merchant_id":
                            merchant_id,

                        "attack_start_time":
                            event_start_time,

                        "attack_end_time":
                            event_end_time,

                        "detection_time":
                            detection_row[
                                "window_start"
                            ],

                        "detection_state":
                            detection_row.get(
                                "spike_state",
                                "UNKNOWN",
                            ),

                        "verification_class":
                            verification_class,

                        "peak_state":
                            event_severity,

                        "peak_tas":
                            peak_tas,

                        "peak_fraud_attack_score":
                            peak_fraud_attack_score,

                        "peak_exceptional_risk_score":
                            peak_exceptional_risk_score,

                        "peak_event_risk_score":
                            peak_event_risk_score,

                        "transaction_count":
                            event_transaction_count,

                        "total_amount":
                            event_total_amount,

                        "expected_fraud_count":
                            event_expected_fraud_count,

                        "expected_fraud_amount":
                            event_expected_fraud_amount,

                        "baseline_expected_fraud_amount":
                            baseline_expected_fraud_amount,

                        "unique_cards":
                            unique_card_count,

                        "new_cards":
                            new_card_count,

                        "window_count":
                            int(
                                len(
                                    episode
                                )
                            ),

                        "event_duration_minutes":
                            event_duration_minutes,

                        "single_transaction_event":
                            single_transaction_event,

                        "exceptional_event":
                            exceptional_episode,

                        "spike_ratio":
                            spike_ratio,

                        "normal_verified_window_count":
                            normal_verified_window_count,

                        "verification_path":
                            verification_path,

                        "card_union_available":
                            card_union_available,

                        "explanation":
                            explanation,
                    }
                )

        return pd.DataFrame(
            events
        )

    # =========================================================================
    # EVENT CARD UNION
    # =========================================================================

    @staticmethod
    def _calculate_event_unique_cards(
        episode: pd.DataFrame,
    ) -> tuple[int, bool]:
        """
        Calculate event-wide unique cards.

        Preferred:
            union of card sets.

        Fallback:
            maximum window-level unique-card count.

        We intentionally do not sum unique-card counts because overlapping
        windows can count the same card repeatedly.
        """

        card_union: set[str] = set()

        card_set_available = False

        if "card_set" in episode.columns:

            for card_set in episode[
                "card_set"
            ]:

                if isinstance(
                    card_set,
                    (
                        set,
                        frozenset,
                        list,
                        tuple,
                    ),
                ):

                    card_set_available = True

                    card_union.update(
                        str(card_id)
                        for card_id
                        in card_set
                    )

        if card_union:

            return (
                len(
                    card_union
                ),
                True,
            )

        if "unique_cards" in episode.columns:

            maximum_unique_card_count = int(
                _safe_max(
                    episode,
                    "unique_cards",
                )
            )

            return (
                maximum_unique_card_count,
                False,
            )

        return (
            0,
            False,
        )

    # =========================================================================
    # EVENT NEW CARDS
    # =========================================================================

    @staticmethod
    def _calculate_event_new_cards(
        episode: pd.DataFrame,
    ) -> int:
        """
        Calculate event-wide new-card count.

        Prefer an actual new_card_set union where available.
        Otherwise sum the window-level new-card counts.
        """

        if "new_card_set" in episode.columns:

            new_card_union: set[str] = set()

            usable_set_found = False

            for new_card_set in episode[
                "new_card_set"
            ]:

                if isinstance(
                    new_card_set,
                    (
                        set,
                        frozenset,
                        list,
                        tuple,
                    ),
                ):

                    usable_set_found = True

                    new_card_union.update(
                        str(card_id)
                        for card_id
                        in new_card_set
                    )

            if usable_set_found:

                return int(
                    len(
                        new_card_union
                    )
                )

        return int(
            _safe_sum(
                episode,
                "new_cards",
            )
        )

    # =========================================================================
    # EMPTY EVENT SCHEMA
    # =========================================================================

    @staticmethod
    def _empty_event_dataframe() -> pd.DataFrame:
        """
        Stable empty event dataframe schema.
        """

        return pd.DataFrame(
            columns=[
                "event_id",
                "merchant_id",
                "attack_start_time",
                "attack_end_time",
                "detection_time",
                "detection_state",
                "verification_class",
                "peak_state",
                "peak_tas",
                "peak_fraud_attack_score",
                "peak_exceptional_risk_score",
                "peak_event_risk_score",
                "transaction_count",
                "total_amount",
                "expected_fraud_count",
                "expected_fraud_amount",
                "baseline_expected_fraud_amount",
                "unique_cards",
                "new_cards",
                "window_count",
                "event_duration_minutes",
                "single_transaction_event",
                "exceptional_event",
                "spike_ratio",
                "normal_verified_window_count",
                "verification_path",
                "card_union_available",
                "explanation",
            ]
        )

    # =========================================================================
    # WINDOW COUNTS
    # =========================================================================

    @staticmethod
    def _calculate_window_counts(
        windows_directory: Path,
        decision_frame: pd.DataFrame,
    ) -> dict:
        """
        Calculate 5m/15m/60m window counts.

        Only the CSV line count is used for 5m and 60m to avoid loading the
        complete artifacts into memory.
        """

        window_counts: dict[str, int] = {}

        for window_minutes in (
            5,
            15,
            60,
        ):

            if window_minutes == 15:

                window_counts[
                    "15m"
                ] = int(
                    len(
                        decision_frame
                    )
                )

                continue

            artifact_path = (
                windows_directory
                /
                f"merchant_spike_windows_{window_minutes}m.csv"
            )

            if not artifact_path.exists():

                window_counts[
                    f"{window_minutes}m"
                ] = 0

                continue

            row_count = 0

            try:

                with artifact_path.open(
                    "r",
                    encoding="utf-8",
                ) as artifact_file:

                    for _ in artifact_file:

                        row_count += 1

                row_count = max(
                    row_count - 1,
                    0,
                )

            except OSError:

                row_count = 0

            window_counts[
                f"{window_minutes}m"
            ] = int(
                row_count
            )

        return window_counts

    # =========================================================================
    # VERIFICATION SUMMARY
    # =========================================================================

    @staticmethod
    def _build_verification_summary(
        dataframe: pd.DataFrame,
    ) -> dict:
        """
        Build verification summary.
        """

        summary = {
            "candidate_windows":
                int(
                    dataframe[
                        "candidate_spike"
                    ].sum()
                ),

            "verified_windows":
                int(
                    dataframe[
                        "verified_spike"
                    ].sum()
                ),

            "normal_verified_windows":
                int(
                    dataframe[
                        "normal_verification_gate"
                    ].sum()
                ),

            "critical_windows":
                int(
                    dataframe[
                        "critical_spike"
                    ].sum()
                ),

            "exceptional_windows":
                int(
                    dataframe[
                        "exceptional_verification_gate"
                    ].sum()
                ),

            "multi_transaction_windows":
                int(
                    dataframe[
                        "multi_transaction_gate"
                    ].sum()
                ),
        }

        if (
            "strong_temporal_evidence"
            in dataframe.columns
        ):

            summary[
                "strong_temporal_windows"
            ] = int(
                dataframe[
                    "strong_temporal_evidence"
                ].sum()
            )

        if (
            "coordination_evidence_gate"
            in dataframe.columns
        ):

            summary[
                "coordination_evidence_windows"
            ] = int(
                dataframe[
                    "coordination_evidence_gate"
                ].sum()
            )

        return summary

    # =========================================================================
    # SCORE SUMMARY
    # =========================================================================

    @staticmethod
    def _build_score_summary(
        dataframe: pd.DataFrame,
    ) -> dict:
        """
        Build score distribution summary.
        """

        return {
            "mean_tas":
                _safe_mean(
                    dataframe,
                    "tas",
                ),

            "max_tas":
                _safe_max(
                    dataframe,
                    "tas",
                ),

            "mean_verification_evidence":
                _safe_mean(
                    dataframe,
                    "verification_evidence_score",
                ),

            "max_verification_evidence":
                _safe_max(
                    dataframe,
                    "verification_evidence_score",
                ),

            "mean_fraud_attack_score":
                _safe_mean(
                    dataframe,
                    "fraud_attack_score",
                ),

            "max_fraud_attack_score":
                _safe_max(
                    dataframe,
                    "fraud_attack_score",
                ),

            "mean_exceptional_risk_score":
                _safe_mean(
                    dataframe,
                    "exceptional_risk_score",
                ),

            "max_exceptional_risk_score":
                _safe_max(
                    dataframe,
                    "exceptional_risk_score",
                ),

            "max_event_peak_risk_score":
                _safe_max(
                    dataframe,
                    "event_peak_risk_score",
                ),

            "mean_corroboration":
                _safe_mean(
                    dataframe,
                    "corroboration_score",
                ),

            "max_corroboration":
                _safe_max(
                    dataframe,
                    "corroboration_score",
                ),

            "mean_persistence":
                _safe_mean(
                    dataframe,
                    "persistence_score",
                ),

            "max_persistence":
                _safe_max(
                    dataframe,
                    "persistence_score",
                ),

            "mean_breadth":
                _safe_mean(
                    dataframe,
                    "breadth_score",
                ),

            "max_breadth":
                _safe_max(
                    dataframe,
                    "breadth_score",
                ),
        }

    # =========================================================================
    # EVENT SUMMARY
    # =========================================================================

    @staticmethod
    def _event_summary(
        events: pd.DataFrame,
    ) -> dict:
        """
        Build event-level summary.
        """

        if events.empty:

            return {
                "count": 0,
                "merchants": 0,
                "single_transaction_event_rate": 0.0,
                "exceptional_event_rate": 0.0,
                "mean_duration_minutes": 0.0,
                "total_expected_fraud_amount": 0.0,
                "total_event_risk_score": 0.0,
            }

        return {
            "count":
                int(
                    len(events)
                ),

            "merchants":
                int(
                    events[
                        "merchant_id"
                    ].nunique()
                ),

            "single_transaction_event_rate":
                float(
                    events[
                        "single_transaction_event"
                    ].mean()
                ),

            "exceptional_event_rate":
                float(
                    events[
                        "exceptional_event"
                    ].mean()
                ),

            "mean_duration_minutes":
                float(
                    events[
                        "event_duration_minutes"
                    ].mean()
                ),

            "total_expected_fraud_amount":
                float(
                    events[
                        "expected_fraud_amount"
                    ].sum()
                ),

            "total_event_risk_score":
                float(
                    events[
                        "peak_event_risk_score"
                    ].sum()
                ),
        }

    # =========================================================================
    # EVENT SEVERITY DISTRIBUTION
    # =========================================================================

    @staticmethod
    def _event_severity_distribution(
        events: pd.DataFrame,
    ) -> dict:
        """
        Return event severity counts.
        """

        if events.empty:
            return {}

        if "peak_state" not in events.columns:
            return {}

        return {
            str(state):
                int(count)
            for state, count
            in events[
                "peak_state"
            ]
            .value_counts()
            .to_dict()
            .items()
        }


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "Phase2Pipeline",
]