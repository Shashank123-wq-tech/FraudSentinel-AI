from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pandas as pd

from src.components.phase2.acceleration_detector import (
    add_acceleration,
)
from src.components.phase2.cusum_detector import (
    compute_cusum,
)
from src.components.phase2.ewma_detector import (
    compute_ewma,
)
from src.components.phase2.event_builder import (
    add_explanations,
    build_merchant_spike_events,
)
from src.components.phase2.merchant_baseline import (
    build_merchant_baseline,
)
from src.components.phase2.multi_window_detector import (
    fuse_multi_window,
)
from src.components.phase2.persistence_detector import (
    add_persistence,
)
from src.components.phase2.risk_output_builder import (
    load_phase1_risk_output,
)
from src.components.phase2.rolling_anomaly import (
    compute_zscores,
)
from src.components.phase2.spike_evaluation import (
    summarize_events,
)
from src.components.phase2.spike_scoring import (
    recompute_spike_score_with_persistence,
    score_windows,
)
from src.components.phase2.spike_verification import (
    add_severity,
)
from src.components.phase2.temporal_aggregation import (
    add_expected_fraud_exposure,
    attach_exact_expected_amount,
    build_all_merchant_windows,
)
from src.components.phase2.temporal_features import (
    add_temporal_context,
)
from src.entity.spike_event import (
    SeverityPolicy,
    SpikeScoreWeights,
)
from src.exception import (
    ArtifactError,
    PipelineError,
)
from src.logger.logger import get_logger
from src.utils.config_utils import (
    load_phase2_config,
)


class Phase2Pipeline:
    """
    FraudSentinel AI Phase-2 pipeline.

    Input:
        Phase-1 transaction risk artifact.

    Output:
        Merchant temporal fraud-spike events.
    """

    def __init__(
        self,
        config_path: str = "config/phase2.yaml",
        output_dir: str = "artifacts/phase2",
    ) -> None:

        self.config_path = Path(
            config_path
        )

        self.output_dir = Path(
            output_dir
        )

        self.config = load_phase2_config(
            self.config_path
        )

        logging_config = self.config.get(
            "logging",
            {},
        )

        self.logger = get_logger(
            name="fraudsentinel.phase2",
            log_dir=logging_config.get(
                "log_directory",
                "artifacts/logs",
            ),
            log_filename=logging_config.get(
                "log_filename",
                "phase2.log",
            ),
            level=logging_config.get(
                "level",
                "INFO",
            ),
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.pipeline_version = self.config.get(
            "pipeline_version",
            "phase2_pipeline_v2",
        )

        self.detector_version = self.config[
            "detector_version"
        ]

        self.weights = SpikeScoreWeights(
            **{
                key:
                self.config[
                    "spike_score_weights"
                ][key]
                for key
                in SpikeScoreWeights.__dataclass_fields__
            }
        )

        self.policy = SeverityPolicy(
            **{
                key:
                self.config[
                    "severity_policy"
                ][key]
                for key
                in SeverityPolicy.__dataclass_fields__
            }
        )

    def _write_parquet(
        self,
        df: pd.DataFrame,
        path: Path,
    ) -> None:

        try:
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            df.to_parquet(
                path,
                index=False,
            )

        except Exception as exc:
            raise ArtifactError(
                f"Failed to write artifact: {path}",
                cause=exc,
            ) from exc

    def _write_json(
        self,
        payload: dict,
        path: Path,
    ) -> None:

        try:
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            path.write_text(
                json.dumps(
                    payload,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

        except Exception as exc:
            raise ArtifactError(
                f"Failed to write JSON artifact: {path}",
                cause=exc,
            ) from exc

    def run(
        self,
        phase1_risk_output: str,
    ) -> dict:

        run_id = uuid4().hex[:12]

        started = perf_counter()

        self.logger.info(
            "[run=%s] Phase 2 STARTED",
            run_id,
        )

        self.logger.info(
            "[run=%s] input=%s",
            run_id,
            phase1_risk_output,
        )

        try:

            # =================================================
            # STEP 1
            # Load Phase-1 risk
            # =================================================

            transactions = (
                load_phase1_risk_output(
                    phase1_risk_output
                )
            )

            self.logger.info(
                "[run=%s] Phase-1 input validated | "
                "rows=%d | merchants=%d",
                run_id,
                len(transactions),
                transactions["merchant_id"].nunique(),
            )

            # =================================================
            # STEP 2
            # Temporal context
            # =================================================

            transactions = (
                add_temporal_context(
                    transactions
                )
            )

            self._write_parquet(
                transactions,
                self.output_dir
                / "01_phase2_transaction_risk.parquet",
            )

            # =================================================
            # STEP 3
            # Merchant windows
            # =================================================

            windows = (
                build_all_merchant_windows(
                    transactions,
                    self.config[
                        "window_sizes"
                    ],
                )
            )

            final_windows = {}

            # =================================================
            # STEP 4
            # Run every time resolution
            # =================================================

            for resolution, frame in windows.items():

                self.logger.info(
                    "[run=%s] Processing resolution=%s",
                    run_id,
                    resolution,
                )

                frame = (
                    add_expected_fraud_exposure(
                        frame
                    )
                )

                frame = (
                    attach_exact_expected_amount(
                        frame,
                        transactions,
                        self.config[
                            "window_sizes"
                        ][resolution],
                    )
                )

                baseline_config = (
                    self.config["baseline"]
                )

                frame = build_merchant_baseline(
                    frame,
                    resolution=resolution,
                    history_hours=int(
                        baseline_config[
                            "history_hours"
                        ][resolution]
                    ),
                    min_history_windows=int(
                        baseline_config[
                            "min_history_windows"
                        ][resolution]
                    ),
                    minimum_std=float(
                        baseline_config.get(
                            "minimum_baseline_std",
                            1e-12,
                        )
                    ),
                )

                frame = compute_zscores(
                    frame,
                    minimum_std=float(
                        baseline_config.get(
                            "minimum_baseline_std",
                            1e-12,
                        )
                    ),
                )

                frame = compute_ewma(
                    frame,
                    span=int(
                        self.config[
                            "ewma"
                        ]["span"]
                    ),
                )

                frame = compute_cusum(
                    frame,
                    k=float(
                        self.config[
                            "cusum"
                        ]["k"]
                    ),
                )

                frame = add_persistence(
                    frame,
                    z_threshold=float(
                        self.config[
                            "persistence"
                        ][
                            "anomaly_zscore_threshold"
                        ]
                    ),
                    max_streak=int(
                        self.config[
                            "persistence"
                        ].get(
                            "max_streak",
                            5,
                        )
                    ),
                )

                # Base score.
                frame = score_windows(
                    frame,
                    self.weights,
                )

                frame = add_acceleration(
                    frame,
                    max_positive_change=float(
                        self.config[
                            "acceleration"
                        ].get(
                            "max_positive_change",
                            25.0,
                        )
                    ),
                )

                # Final score.
                frame = (
                    recompute_spike_score_with_persistence(
                        frame,
                        self.weights,
                    )
                )

                frame = add_severity(
                    frame,
                    self.policy,
                )

                final_windows[
                    resolution
                ] = frame

                self._write_parquet(
                    frame,
                    self.output_dir
                    / f"{resolution}_merchant_time_features.parquet",
                )

                self.logger.info(
                    "[run=%s] resolution=%s completed | "
                    "windows=%d | baseline_ready=%d | "
                    "verified=%d",
                    run_id,
                    resolution,
                    len(frame),
                    int(
                        frame[
                            "baseline_ready"
                        ].sum()
                    ),
                    int(
                        frame[
                            "verified_spike"
                        ].sum()
                    ),
                )

            # =================================================
            # STEP 5
            # Multi-window fusion
            # =================================================

            multi_config = (
                self.config[
                    "multi_window"
                ]
            )

            if multi_config.get(
                "enabled",
                True,
            ):

                fused = fuse_multi_window(
                    final_windows,
                    short_threshold=float(
                        multi_config[
                            "short_window_threshold"
                        ]
                    ),
                    medium_threshold=float(
                        multi_config[
                            "medium_window_threshold"
                        ]
                    ),
                    medium_confirming_short=float(
                        multi_config[
                            "medium_confirming_short_threshold"
                        ]
                    ),
                    long_threshold=float(
                        multi_config[
                            "long_window_threshold"
                        ]
                    ),
                )

            else:

                fused = pd.DataFrame()

            self._write_parquet(
                fused,
                self.output_dir
                / "06_multi_window_fusion.parquet",
            )

            # =================================================
            # STEP 6
            # Event stream = 15-minute resolution
            # Enriched with multi-window evidence.
            # =================================================

            event_windows = (
                final_windows["15m"]
                .merge(
                    fused[
                        [
                            "merchant_id",
                            "window_start",
                            "multi_window_score",
                            "multi_window_verified",
                            "multi_window_mode",
                        ]
                    ],
                    on=[
                        "merchant_id",
                        "window_start",
                    ],
                    how="left",
                    validate="one_to_one",
                )
            )

            event_windows[
                "multi_window_verified"
            ] = (
                event_windows[
                    "multi_window_verified"
                ]
                .fillna(False)
            )

            event_windows[
                "verified_spike"
            ] = (
                event_windows[
                    "verified_spike"
                ]
                |
                event_windows[
                    "multi_window_verified"
                ]
            )

            # =================================================
            # STEP 7
            # Event construction
            # =================================================

            event_config = (
                self.config[
                    "event_builder"
                ]
            )

            events = (
                build_merchant_spike_events(
                    event_windows,
                    min_severity=event_config[
                        "min_severity_for_event"
                    ],
                    gap_tolerance_windows=int(
                        event_config[
                            "gap_tolerance_windows"
                        ]
                    ),
                    resolution=event_config[
                        "resolution"
                    ],
                )
            )

            events = add_explanations(
                events,
                event_windows,
            )

            self._write_parquet(
                events,
                self.output_dir
                / "merchant_spike_events.parquet",
            )

            # =================================================
            # STEP 8
            # Summary
            # =================================================

            metrics = summarize_events(
                events
            )

            runtime = (
                perf_counter()
                - started
            )

            metadata = {
                "run_id": run_id,

                "pipeline_version":
                    self.pipeline_version,

                "detector_version":
                    self.detector_version,

                "input_phase1_risk_output":
                    str(
                        Path(
                            phase1_risk_output
                        )
                    ),

                "input_rows":
                    int(
                        len(transactions)
                    ),

                "merchant_count":
                    int(
                        transactions[
                            "merchant_id"
                        ].nunique()
                    ),

                "window_counts":
                    {
                        resolution:
                        int(len(frame))
                        for resolution, frame
                        in final_windows.items()
                    },

                **metrics,

                "runtime_seconds":
                    round(
                        runtime,
                        3,
                    ),
            }

            self._write_json(
                metadata,
                self.output_dir
                / "run_metadata.json",
            )

            self.logger.info(
                "[run=%s] Phase 2 COMPLETED | "
                "events=%d | runtime=%.3fs",
                run_id,
                len(events),
                runtime,
            )

            return {
                "transactions":
                    transactions,

                "windows":
                    final_windows,

                "fused":
                    fused,

                "events":
                    events,

                "metadata":
                    metadata,
            }

        except Exception as exc:

            self.logger.exception(
                "[run=%s] Phase 2 FAILED",
                run_id,
            )

            if isinstance(
                exc,
                PipelineError,
            ):
                raise

            raise PipelineError(
                "Phase 2 execution failed.",
                cause=exc,
            ) from exc


def run_phase2_pipeline(
    phase1_risk_output: str,
    output_dir: str = "artifacts/phase2",
    config_path: str = "config/phase2.yaml",
) -> dict:

    pipeline = Phase2Pipeline(
        config_path=config_path,
        output_dir=output_dir,
    )

    return pipeline.run(
        phase1_risk_output
    )