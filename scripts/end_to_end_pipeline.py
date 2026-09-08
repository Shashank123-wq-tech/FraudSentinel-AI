"""
FraudSentinel AI
End-to-End Phase 1 -> Phase 2 Pipeline

Execution
---------
1. Validate repository.
2. Remove stale Phase-1 and Phase-2 artifacts.
3. Validate raw dataset.
4. Record the 80/20 temporal split metadata.
5. Run Phase-1 training.
6. Run Phase-1 prediction.
7. Verify canonical transaction_risk.csv.
8. Pass ONLY the fresh transaction_risk.csv to Phase 2.
9. Write an end-to-end lineage manifest.

No --phase1-train-command,
no --phase1-predict-command,
and no --phase2-command are required.

Windows-safe paths are used throughout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)


# ============================================================================
# DEFAULT PATHS
# ============================================================================

DEFAULT_RAW_DATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fraud_transactions.csv"
)

DEFAULT_PHASE2_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "phase2.yaml"
)


# ============================================================================
# ARTIFACT PATHS
# ============================================================================

PHASE1_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "phase1"
)

PHASE2_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "phase2"
)

PHASE1_MODEL = (
    PHASE1_ROOT
    / "models"
    / "fraudsentinel_phase1_xgboost.json"
)

PHASE1_PREDICTION = (
    PHASE1_ROOT
    / "predictions"
    / "phase1_predictions_legacy.csv"
)

PHASE1_RISK_OUTPUT = (
    PHASE1_ROOT
    / "risk_output"
    / "transaction_risk.csv"
)

PHASE1_SPLIT_METADATA = (
    PHASE1_ROOT
    / "data_split"
    / "split_metadata.json"
)

PHASE2_METRICS = (
    PHASE2_ROOT
    / "evaluation"
    / "phase2_metrics.json"
)

E2E_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "end_to_end_manifest.json"
)


# ============================================================================
# EXACT PHASE-1 -> PHASE-2 CONTRACT
# ============================================================================

REQUIRED_PHASE1_COLUMNS = [
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


# ============================================================================
# LOGGING
# ============================================================================

def log(message: str) -> None:

    print(
        f"[E2E] {message}",
        flush=True,
    )


def section(title: str) -> None:

    print()

    log(
        "=" * 80
    )

    log(
        title
    )

    log(
        "=" * 80
    )


# ============================================================================
# HASH
# ============================================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):

            digest.update(
                block
            )

    return digest.hexdigest()


# ============================================================================
# COMMAND EXECUTION
# ============================================================================

def run_command(
    command: list[str],
) -> None:

    command = [
        str(item)
        for item in command
    ]

    command_text = (
        subprocess.list2cmdline(
            command
        )
    )

    log(
        f"Running: {command_text}"
    )

    # ------------------------------------------------------------------------
    # Make repository root importable.
    # ------------------------------------------------------------------------

    env = os.environ.copy()

    existing_pythonpath = env.get(
        "PYTHONPATH",
        "",
    )

    if existing_pythonpath:

        env["PYTHONPATH"] = (
            str(PROJECT_ROOT)
            + os.pathsep
            + existing_pythonpath
        )

    else:

        env["PYTHONPATH"] = str(
            PROJECT_ROOT
        )

    # ------------------------------------------------------------------------
    # Execute from repository root.
    # ------------------------------------------------------------------------

    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"Command failed with exit code "
            f"{result.returncode}: "
            f"{command_text}"
        )


# ============================================================================
# REPOSITORY VALIDATION
# ============================================================================

def validate_repository(
    raw_data: Path,
    phase2_config: Path,
) -> None:

    section(
        "REPOSITORY VALIDATION"
    )

    required_paths = {
        "raw dataset": raw_data,

        "phase2 config": phase2_config,

        "train_phase1.py": (
            PROJECT_ROOT
            / "scripts"
            / "train_phase1.py"
        ),

        "predict_phase1.py": (
            PROJECT_ROOT
            / "scripts"
            / "predict_phase1.py"
        ),

        "run_phase2.py": (
            PROJECT_ROOT
            / "scripts"
            / "run_phase2.py"
        ),

        "src directory": (
            PROJECT_ROOT
            / "src"
        ),

        "phase1 training pipeline": (
            PROJECT_ROOT
            / "src"
            / "pipeline"
            / "phase1_training_pipeline.py"
        ),

        "phase1 prediction pipeline": (
            PROJECT_ROOT
            / "src"
            / "pipeline"
            / "phase1_prediction_pipeline.py"
        ),

        "phase2 pipeline": (
            PROJECT_ROOT
            / "src"
            / "pipeline"
            / "phase2_pipeline.py"
        ),
    }

    for name, path in required_paths.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Required {name} not found:\n"
                f"{path}"
            )

        log(
            f"{name}: OK"
        )


# ============================================================================
# CLEAN
# ============================================================================

def clean_artifacts() -> None:

    section(
        "REMOVING STALE ARTIFACTS"
    )

    if PHASE1_ROOT.exists():

        log(
            f"Removing stale artifacts: "
            f"{PHASE1_ROOT}"
        )

        shutil.rmtree(
            PHASE1_ROOT
        )

    if PHASE2_ROOT.exists():

        log(
            f"Removing stale artifacts: "
            f"{PHASE2_ROOT}"
        )

        shutil.rmtree(
            PHASE2_ROOT
        )

    if E2E_MANIFEST.exists():

        log(
            f"Removing stale manifest: "
            f"{E2E_MANIFEST}"
        )

        E2E_MANIFEST.unlink()

    PHASE1_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    PHASE2_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# RAW DATA VALIDATION
# ============================================================================

def validate_raw_dataset(
    raw_data: Path,
) -> dict:

    section(
        "RAW DATA VALIDATION"
    )

    if not raw_data.exists():

        raise FileNotFoundError(
            f"Raw dataset not found:\n"
            f"{raw_data}"
        )

    df = pd.read_csv(
        raw_data
    )

    if df.empty:

        raise ValueError(
            "Raw dataset is empty."
        )

    required_columns = [
        "trans_date_trans_time",
        "is_fraud",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Raw dataset is missing required columns: "
            f"{missing}"
        )

    timestamp = pd.to_datetime(
        df[
            "trans_date_trans_time"
        ],
        errors="coerce",
        utc=True,
    )

    if timestamp.isna().any():

        raise ValueError(
            "Raw dataset contains invalid timestamps."
        )

    target = pd.to_numeric(
        df[
            "is_fraud"
        ],
        errors="coerce",
    )

    if target.isna().any():

        raise ValueError(
            "Raw dataset contains invalid is_fraud values."
        )

    info = {
        "path": str(
            raw_data
        ),
        "rows": int(
            len(df)
        ),
        "columns": int(
            len(df.columns)
        ),
        "sha256": sha256_file(
            raw_data
        ),
        "timestamp_column":
            "trans_date_trans_time",
        "target_column":
            "is_fraud",
        "timestamp_min": str(
            timestamp.min()
        ),
        "timestamp_max": str(
            timestamp.max()
        ),
        "fraud_rows": int(
            target.sum()
        ),
        "fraud_rate": float(
            target.mean()
        ),
    }

    log(
        f"Rows: {info['rows']:,}"
    )

    log(
        f"Columns: {info['columns']:,}"
    )

    log(
        f"Fraud rows: {info['fraud_rows']:,}"
    )

    log(
        f"Fraud rate: {info['fraud_rate']:.6%}"
    )

    log(
        f"Time range: "
        f"{info['timestamp_min']} -> "
        f"{info['timestamp_max']}"
    )

    return info


# ============================================================================
# TEMPORAL SPLIT METADATA
# ============================================================================

def create_split_metadata(
    raw_data: Path,
    split_quantile: float,
) -> dict:

    section(
        "PHASE 1 — TEMPORAL DATA SPLIT"
    )

    df = pd.read_csv(
        raw_data,
        usecols=[
            "trans_date_trans_time",
            "is_fraud",
        ],
    )

    timestamps = pd.to_datetime(
        df[
            "trans_date_trans_time"
        ],
        errors="coerce",
        utc=True,
    )

    target = pd.to_numeric(
        df[
            "is_fraud"
        ],
        errors="coerce",
    )

    split_timestamp = timestamps.quantile(
        split_quantile
    )

    train_mask = (
        timestamps < split_timestamp
    )

    test_mask = (
        timestamps >= split_timestamp
    )

    train_rows = int(
        train_mask.sum()
    )

    test_rows = int(
        test_mask.sum()
    )

    if train_rows == 0:
        raise ValueError(
            "Temporal split produced zero training rows."
        )

    if test_rows == 0:
        raise ValueError(
            "Temporal split produced zero test rows."
        )

    metadata = {
        "split_type":
            "time_based",

        "split_method":
            "timestamp_quantile",

        "split_quantile":
            split_quantile,

        "timestamp_column":
            "trans_date_trans_time",

        "target_column":
            "is_fraud",

        "split_timestamp":
            str(split_timestamp),

        "total_rows":
            int(len(df)),

        "train_rows":
            train_rows,

        "test_rows":
            test_rows,

        "train_fraction":
            float(
                train_rows / len(df)
            ),

        "test_fraction":
            float(
                test_rows / len(df)
            ),

        "train_start":
            str(
                timestamps.loc[
                    train_mask
                ].min()
            ),

        "train_end":
            str(
                timestamps.loc[
                    train_mask
                ].max()
            ),

        "test_start":
            str(
                timestamps.loc[
                    test_mask
                ].min()
            ),

        "test_end":
            str(
                timestamps.loc[
                    test_mask
                ].max()
            ),

        "train_fraud_rows":
            int(
                target.loc[
                    train_mask
                ].sum()
            ),

        "test_fraud_rows":
            int(
                target.loc[
                    test_mask
                ].sum()
            ),

        "train_fraud_rate":
            float(
                target.loc[
                    train_mask
                ].mean()
            ),

        "test_fraud_rate":
            float(
                target.loc[
                    test_mask
                ].mean()
            ),

        "random_seed":
            42,

        "raw_dataset":
            str(raw_data),

        "raw_dataset_sha256":
            sha256_file(raw_data),
    }

    output_dir = (
        PHASE1_ROOT
        / "data_split"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        output_dir
        / "split_metadata.json"
    )

    summary_path = (
        output_dir
        / "split_summary.csv"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "split": "train",
                "rows": train_rows,
                "start":
                    metadata[
                        "train_start"
                    ],
                "end":
                    metadata[
                        "train_end"
                    ],
                "fraud_rows":
                    metadata[
                        "train_fraud_rows"
                    ],
                "fraud_rate":
                    metadata[
                        "train_fraud_rate"
                    ],
            },
            {
                "split": "test",
                "rows": test_rows,
                "start":
                    metadata[
                        "test_start"
                    ],
                "end":
                    metadata[
                        "test_end"
                    ],
                "fraud_rows":
                    metadata[
                        "test_fraud_rows"
                    ],
                "fraud_rate":
                    metadata[
                        "test_fraud_rate"
                    ],
            },
        ]
    ).to_csv(
        summary_path,
        index=False,
    )

    log(
        f"Split timestamp: "
        f"{metadata['split_timestamp']}"
    )

    log(
        f"Train rows: "
        f"{train_rows:,}"
    )

    log(
        f"Test rows: "
        f"{test_rows:,}"
    )

    log(
        f"Train fraud rate: "
        f"{metadata['train_fraud_rate']:.6%}"
    )

    log(
        f"Test fraud rate: "
        f"{metadata['test_fraud_rate']:.6%}"
    )

    log(
        f"Split metadata: "
        f"{metadata_path}"
    )

    return metadata


# ============================================================================
# PHASE 1 TRAINING
# ============================================================================

def run_phase1_training() -> None:

    section(
        "PHASE 1 — TRAINING / CALIBRATION / EVALUATION"
    )

    script = (
        PROJECT_ROOT
        / "scripts"
        / "train_phase1.py"
    )

    if not script.exists():

        raise FileNotFoundError(
            f"Training script not found:\n"
            f"{script}"
        )

    # IMPORTANT:
    # Pass the full script path.
    #
    # Correct:
    #     python C:\...\scripts\train_phase1.py
    #
    # Incorrect:
    #     python scripts train_phase1.py

    run_command(
        [
            sys.executable,
            str(script),
        ]
    )

    if not PHASE1_MODEL.exists():

        raise FileNotFoundError(
            "Phase-1 training finished but "
            "the expected model was not created:\n"
            f"{PHASE1_MODEL}"
        )

    log(
        f"Fresh Phase-1 model created:\n"
        f"{PHASE1_MODEL}"
    )


# ============================================================================
# PHASE 1 PREDICTION
# ============================================================================

def run_phase1_prediction() -> None:

    section(
        "PHASE 1 — TEST / PREDICTION"
    )

    script = (
        PROJECT_ROOT
        / "scripts"
        / "predict_phase1.py"
    )

    if not script.exists():

        raise FileNotFoundError(
            f"Prediction script not found:\n"
            f"{script}"
        )

    run_command(
        [
            sys.executable,
            str(script),
        ]
    )

    if not PHASE1_RISK_OUTPUT.exists():

        raise FileNotFoundError(
            "Phase-1 prediction finished but "
            "canonical transaction_risk.csv "
            "was not created:\n"
            f"{PHASE1_RISK_OUTPUT}"
        )

    log(
        f"Fresh canonical Phase-1 risk output:\n"
        f"{PHASE1_RISK_OUTPUT}"
    )


# ============================================================================
# VALIDATE PHASE-1 -> PHASE-2 CONTRACT
# ============================================================================

def validate_phase1_contract(
    path: Path,
) -> dict:

    section(
        "PHASE 1 -> PHASE 2 CONTRACT VALIDATION"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Phase-1 risk output not found:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    # ------------------------------------------------------------------------
    # Exact schema
    # ------------------------------------------------------------------------

    actual_columns = list(
        df.columns
    )

    if actual_columns != REQUIRED_PHASE1_COLUMNS:

        raise ValueError(
            "Phase-1 -> Phase-2 contract failed.\n\n"
            f"Expected:\n{REQUIRED_PHASE1_COLUMNS}\n\n"
            f"Actual:\n{actual_columns}"
        )

    # ------------------------------------------------------------------------
    # Empty
    # ------------------------------------------------------------------------

    if df.empty:

        raise ValueError(
            "transaction_risk.csv is empty."
        )

    # ------------------------------------------------------------------------
    # Transaction IDs
    # ------------------------------------------------------------------------

    if df[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction_id values found."
        )

    # ------------------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------------------

    timestamp = pd.to_datetime(
        df[
            "timestamp"
        ],
        errors="coerce",
        utc=True,
    )

    if timestamp.isna().any():

        raise ValueError(
            "Invalid timestamp values found."
        )

    # ------------------------------------------------------------------------
    # Numeric
    # ------------------------------------------------------------------------

    for column in [
        "amount",
        "fraud_probability",
        "risk_score",
        "predicted_fraud",
        "threshold",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if df[column].isna().any():

            raise ValueError(
                f"{column} contains invalid values."
            )

    # ------------------------------------------------------------------------
    # Probability
    # ------------------------------------------------------------------------

    if not df[
        "fraud_probability"
    ].between(
        0.0,
        1.0,
    ).all():

        raise ValueError(
            "fraud_probability must be within [0,1]."
        )

    # ------------------------------------------------------------------------
    # Risk score
    # ------------------------------------------------------------------------

    if not df[
        "risk_score"
    ].between(
        0.0,
        100.0,
    ).all():

        raise ValueError(
            "risk_score must be within [0,100]."
        )

    # ------------------------------------------------------------------------
    # Risk-score relationship
    # ------------------------------------------------------------------------

    expected_score = (
        df[
            "fraud_probability"
        ]
        * 100.0
    )

    if not (
        (
            df[
                "risk_score"
            ]
            - expected_score
        ).abs()
        <= 1e-8
    ).all():

        raise ValueError(
            "risk_score is inconsistent with "
            "100 * fraud_probability."
        )

    # ------------------------------------------------------------------------
    # Amount
    # ------------------------------------------------------------------------

    if (
        df["amount"] < 0
    ).any():

        raise ValueError(
            "Negative transaction amounts found."
        )

    # ------------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------------

    if not df[
        "predicted_fraud"
    ].isin(
        [0, 1]
    ).all():

        raise ValueError(
            "predicted_fraud must contain only 0/1."
        )

    # ------------------------------------------------------------------------
    # Required identifiers
    # ------------------------------------------------------------------------

    for column in [
        "transaction_id",
        "merchant_id",
        "card_id",
        "risk_band",
        "model_version",
    ]:

        if (
            df[column]
            .astype(str)
            .str.strip()
            .eq("")
            .any()
        ):

            raise ValueError(
                f"{column} contains blank values."
            )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    info = {
        "path": str(path),

        "rows": int(
            len(df)
        ),

        "columns": int(
            len(df.columns)
        ),

        "exact_11_column_contract":
            True,

        "transaction_count": int(
            df[
                "transaction_id"
            ].nunique()
        ),

        "merchant_count": int(
            df[
                "merchant_id"
            ].nunique()
        ),

        "card_count": int(
            df[
                "card_id"
            ].nunique()
        ),

        "predicted_fraud_count": int(
            df[
                "predicted_fraud"
            ].sum()
        ),

        "predicted_fraud_rate": float(
            df[
                "predicted_fraud"
            ].mean()
        ),

        "fraud_probability_mean": float(
            df[
                "fraud_probability"
            ].mean()
        ),

        "fraud_probability_max": float(
            df[
                "fraud_probability"
            ].max()
        ),

        "risk_score_mean": float(
            df[
                "risk_score"
            ].mean()
        ),

        "risk_score_max": float(
            df[
                "risk_score"
            ].max()
        ),

        "timestamp_min": str(
            timestamp.min()
        ),

        "timestamp_max": str(
            timestamp.max()
        ),

        "sha256": sha256_file(
            path
        ),

        "columns":
            REQUIRED_PHASE1_COLUMNS,
    }

    log(
        "PHASE-1 -> PHASE-2 CONTRACT: PASSED"
    )

    log(
        f"Rows: {info['rows']:,}"
    )

    log(
        f"Transactions: "
        f"{info['transaction_count']:,}"
    )

    log(
        f"Merchants: "
        f"{info['merchant_count']:,}"
    )

    log(
        f"Cards: "
        f"{info['card_count']:,}"
    )

    log(
        f"Predicted fraud: "
        f"{info['predicted_fraud_count']:,}"
    )

    return info


# ============================================================================
# PHASE 2
# ============================================================================

def run_phase2(
    phase2_config: Path,
) -> dict | None:

    section(
        "PHASE 2 — TEMPORAL FRAUD-SPIKE INTELLIGENCE"
    )

    # ------------------------------------------------------------------------
    # Verify fresh input BEFORE launching Phase 2.
    # ------------------------------------------------------------------------

    contract_info = validate_phase1_contract(
        PHASE1_RISK_OUTPUT
    )

    log(
        "Phase 2 will consume:"
    )

    log(
        str(
            PHASE1_RISK_OUTPUT
        )
    )

    log(
        f"Input rows: "
        f"{contract_info['rows']:,}"
    )

    # ------------------------------------------------------------------------
    # Validate config.
    # ------------------------------------------------------------------------

    if not phase2_config.exists():

        raise FileNotFoundError(
            f"Phase-2 configuration not found:\n"
            f"{phase2_config}"
        )

    # ------------------------------------------------------------------------
    # Phase-2 runner.
    # ------------------------------------------------------------------------

    script = (
        PROJECT_ROOT
        / "scripts"
        / "run_phase2.py"
    )

    if not script.exists():

        raise FileNotFoundError(
            f"Phase-2 runner not found:\n"
            f"{script}"
        )

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Phase 2 gets the exact fresh canonical file explicitly.
    # ------------------------------------------------------------------------

    run_command(
        [
            sys.executable,
            str(script),
            "--config",
            str(phase2_config),
            "--phase1-input",
            str(
                PHASE1_RISK_OUTPUT
            ),
        ]
    )

    # ------------------------------------------------------------------------
    # Metrics.
    # ------------------------------------------------------------------------

    if not PHASE2_METRICS.exists():

        log(
            "Phase 2 completed, but "
            "phase2_metrics.json was not found."
        )

        return None

    try:

        metrics = json.loads(
            PHASE2_METRICS.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "Phase-2 metrics file is invalid JSON:\n"
            f"{PHASE2_METRICS}"
        ) from exc

    return metrics


# ============================================================================
# MANIFEST
# ============================================================================

def write_manifest(
    *,
    started_at: float,
    raw_info: dict,
    split_info: dict,
    phase1_info: dict,
    phase2_config: Path,
    phase2_result: dict | None,
) -> None:

    runtime_seconds = (
        time.time()
        - started_at
    )

    manifest = {
        "project":
            "FraudSentinel AI",

        "pipeline":
            "phase1_to_phase2",

        "status":
            "SUCCESS",

        "python_executable":
            sys.executable,

        "python_version":
            sys.version,

        "project_root":
            str(PROJECT_ROOT),

        "runtime_seconds":
            runtime_seconds,

        "raw_data":
            raw_info,

        "phase1":
            {
                "training_script":
                    "scripts/train_phase1.py",

                "prediction_script":
                    "scripts/predict_phase1.py",

                "model":
                    str(
                        PHASE1_MODEL.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "prediction":
                    str(
                        PHASE1_PREDICTION.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "risk_output":
                    str(
                        PHASE1_RISK_OUTPUT.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "split_metadata":
                    split_info,

                "contract_validation":
                    phase1_info,
            },

        "phase2":
            {
                "runner":
                    "scripts/run_phase2.py",

                "config":
                    str(
                        phase2_config.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "input":
                    str(
                        PHASE1_RISK_OUTPUT.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "metrics_file":
                    (
                        str(
                            PHASE2_METRICS.relative_to(
                                PROJECT_ROOT
                            )
                        )
                        if PHASE2_METRICS.exists()
                        else None
                    ),

                "metrics":
                    phase2_result,
            },
    }

    E2E_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    E2E_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    log(
        f"E2E manifest written:\n"
        f"{E2E_MANIFEST}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "FraudSentinel AI — "
            "Phase 1 -> Phase 2 End-to-End Pipeline"
        )
    )

    parser.add_argument(
        "--raw-data",
        default=str(
            DEFAULT_RAW_DATA
        ),
        help=(
            "Raw dataset path. "
            "Default: data/raw/fraud_transactions.csv"
        ),
    )

    parser.add_argument(
        "--split-quantile",
        type=float,
        default=0.80,
        help=(
            "Temporal train/test split quantile. "
            "Default: 0.80"
        ),
    )

    parser.add_argument(
        "--phase2-config",
        default=str(
            DEFAULT_PHASE2_CONFIG
        ),
        help=(
            "Phase-2 configuration. "
            "Default: config/phase2.yaml"
        ),
    )

    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help=(
            "Do not delete existing Phase-1 and Phase-2 artifacts."
        ),
    )

    args = parser.parse_args()

    started_at = time.time()

    section(
        "FRAUDSENTINEL AI — END-TO-END PIPELINE"
    )

    log(
        f"Project root: "
        f"{PROJECT_ROOT}"
    )

    log(
        f"Python executable: "
        f"{sys.executable}"
    )

    # ------------------------------------------------------------------------
    # Resolve raw path.
    # ------------------------------------------------------------------------

    raw_data = Path(
        args.raw_data
    )

    if not raw_data.is_absolute():

        raw_data = (
            PROJECT_ROOT
            / raw_data
        )

    raw_data = raw_data.resolve()

    # ------------------------------------------------------------------------
    # Resolve Phase-2 config.
    # ------------------------------------------------------------------------

    phase2_config = Path(
        args.phase2_config
    )

    if not phase2_config.is_absolute():

        phase2_config = (
            PROJECT_ROOT
            / phase2_config
        )

    phase2_config = (
        phase2_config
        .resolve()
    )

    # ------------------------------------------------------------------------
    # Validate split quantile.
    # ------------------------------------------------------------------------

    if not (
        0.0
        < args.split_quantile
        < 1.0
    ):

        raise ValueError(
            "--split-quantile must be "
            "between 0 and 1."
        )

    # ------------------------------------------------------------------------
    # Repository.
    # ------------------------------------------------------------------------

    validate_repository(
        raw_data,
        phase2_config,
    )

    # ------------------------------------------------------------------------
    # Clean.
    # ------------------------------------------------------------------------

    if args.skip_clean:

        log(
            "WARNING: stale-artifact cleanup disabled."
        )

    else:

        clean_artifacts()

    # ------------------------------------------------------------------------
    # Raw data.
    # ------------------------------------------------------------------------

    raw_info = validate_raw_dataset(
        raw_data
    )

    # ------------------------------------------------------------------------
    # Split metadata.
    # ------------------------------------------------------------------------

    split_info = create_split_metadata(
        raw_data,
        args.split_quantile,
    )

    # ------------------------------------------------------------------------
    # Phase 1 training.
    # ------------------------------------------------------------------------

    run_phase1_training()

    # ------------------------------------------------------------------------
    # Phase 1 prediction + normalization.
    # ------------------------------------------------------------------------

    run_phase1_prediction()

    # ------------------------------------------------------------------------
    # Phase 1 contract.
    # ------------------------------------------------------------------------

    phase1_info = validate_phase1_contract(
        PHASE1_RISK_OUTPUT
    )

    # ------------------------------------------------------------------------
    # Phase 2.
    # ------------------------------------------------------------------------

    phase2_result = run_phase2(
        phase2_config
    )

    # ------------------------------------------------------------------------
    # Manifest.
    # ------------------------------------------------------------------------

    write_manifest(
        started_at=started_at,
        raw_info=raw_info,
        split_info=split_info,
        phase1_info=phase1_info,
        phase2_config=phase2_config,
        phase2_result=phase2_result,
    )

    # ------------------------------------------------------------------------
    # Final status.
    # ------------------------------------------------------------------------

    runtime = (
        time.time()
        - started_at
    )

    section(
        "END-TO-END PIPELINE COMPLETED SUCCESSFULLY"
    )

    log(
        f"Runtime: {runtime:.2f} seconds"
    )

    log(
        "Fresh Phase-1 model:"
    )

    log(
        str(PHASE1_MODEL)
    )

    log(
        "Fresh Phase-1 prediction:"
    )

    log(
        str(PHASE1_PREDICTION)
    )

    log(
        "Fresh Phase-1 -> Phase-2 input:"
    )

    log(
        str(PHASE1_RISK_OUTPUT)
    )

    log(
        "Temporal split metadata:"
    )

    log(
        str(PHASE1_SPLIT_METADATA)
    )

    log(
        "Phase-2 configuration:"
    )

    log(
        str(phase2_config)
    )

    log(
        "E2E manifest:"
    )

    log(
        str(E2E_MANIFEST)
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            f"\n[E2E][FATAL] "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        raise