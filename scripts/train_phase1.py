"""
FraudSentinel AI
Phase 1 Training Runner

This script:
- resolves the repository root reliably on Windows,
- makes src importable,
- validates the raw dataset,
- records the actual 80/20 temporal split metadata,
- runs the existing Phase1TrainingPipeline,
- verifies that the fresh XGBoost model was created.

The existing Phase1TrainingPipeline remains responsible for:
- feature engineering
- training
- calibration
- evaluation
- model persistence
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


# ============================================================================
# REPOSITORY ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.pipeline.phase1_training_pipeline import (  # noqa: E402
    Phase1TrainingPipeline,
)


# ============================================================================
# PATHS
# ============================================================================

RAW_DATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fraud_transactions.csv"
)

PHASE1_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "phase1"
)

MODEL_PATH = (
    PHASE1_ROOT
    / "models"
    / "fraudsentinel_phase1_xgboost.json"
)

EVALUATION_PATH = (
    PHASE1_ROOT
    / "evaluations"
    / "phase1_validation_test_metrics.csv"
)

SPLIT_DIR = (
    PHASE1_ROOT
    / "data_split"
)

SPLIT_METADATA_PATH = (
    SPLIT_DIR
    / "split_metadata.json"
)

SPLIT_SUMMARY_PATH = (
    SPLIT_DIR
    / "split_summary.csv"
)


# ============================================================================
# PHASE 1 POLICY
# ============================================================================

THRESHOLD = 0.84
SPLIT_QUANTILE = 0.80
RANDOM_SEED = 42


# ============================================================================
# LOGGING
# ============================================================================

def log(message: str) -> None:
    print(
        f"[PHASE1-TRAIN] {message}",
        flush=True,
    )


# ============================================================================
# SHA256
# ============================================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


# ============================================================================
# RAW DATA VALIDATION
# ============================================================================

def load_and_validate_raw_data() -> pd.DataFrame:

    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_DATA}"
        )

    if not RAW_DATA.is_file():
        raise FileNotFoundError(
            f"Raw dataset is not a file:\n{RAW_DATA}"
        )

    log(f"Raw dataset: {RAW_DATA}")

    df = pd.read_csv(RAW_DATA)

    if df.empty:
        raise ValueError(
            "The raw transaction dataset is empty."
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

    timestamps = pd.to_datetime(
        df["trans_date_trans_time"],
        errors="coerce",
        utc=True,
    )

    if timestamps.isna().any():
        invalid_count = int(
            timestamps.isna().sum()
        )

        raise ValueError(
            f"Raw dataset contains "
            f"{invalid_count:,} invalid timestamps."
        )

    target = pd.to_numeric(
        df["is_fraud"],
        errors="coerce",
    )

    if target.isna().any():
        invalid_count = int(
            target.isna().sum()
        )

        raise ValueError(
            f"Raw dataset contains "
            f"{invalid_count:,} invalid is_fraud values."
        )

    return df


# ============================================================================
# TEMPORAL SPLIT METADATA
# ============================================================================

def persist_temporal_split_metadata(
    df: pd.DataFrame,
) -> dict:

    timestamps = pd.to_datetime(
        df["trans_date_trans_time"],
        errors="coerce",
        utc=True,
    )

    target = pd.to_numeric(
        df["is_fraud"],
        errors="coerce",
    )

    split_timestamp = timestamps.quantile(
        SPLIT_QUANTILE
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
        "split_type": "time_based",
        "split_method": "timestamp_quantile",
        "split_quantile": SPLIT_QUANTILE,
        "timestamp_column": "trans_date_trans_time",
        "target_column": "is_fraud",
        "split_timestamp": str(
            split_timestamp
        ),
        "total_rows": int(
            len(df)
        ),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_fraction": float(
            train_rows / len(df)
        ),
        "test_fraction": float(
            test_rows / len(df)
        ),
        "train_start": str(
            timestamps.loc[train_mask].min()
        ),
        "train_end": str(
            timestamps.loc[train_mask].max()
        ),
        "test_start": str(
            timestamps.loc[test_mask].min()
        ),
        "test_end": str(
            timestamps.loc[test_mask].max()
        ),
        "train_fraud_rows": int(
            target.loc[train_mask].sum()
        ),
        "test_fraud_rows": int(
            target.loc[test_mask].sum()
        ),
        "train_fraud_rate": float(
            target.loc[train_mask].mean()
        ),
        "test_fraud_rate": float(
            target.loc[test_mask].mean()
        ),
        "random_seed": RANDOM_SEED,
        "threshold": THRESHOLD,
        "raw_dataset": str(
            RAW_DATA
        ),
        "raw_dataset_sha256": sha256_file(
            RAW_DATA
        ),
    }

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SPLIT_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    split_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": train_rows,
                "start": metadata["train_start"],
                "end": metadata["train_end"],
                "fraud_rows": metadata["train_fraud_rows"],
                "fraud_rate": metadata["train_fraud_rate"],
            },
            {
                "split": "test",
                "rows": test_rows,
                "start": metadata["test_start"],
                "end": metadata["test_end"],
                "fraud_rows": metadata["test_fraud_rows"],
                "fraud_rate": metadata["test_fraud_rate"],
            },
        ]
    )

    split_summary.to_csv(
        SPLIT_SUMMARY_PATH,
        index=False,
    )

    log(
        f"Temporal split timestamp: "
        f"{metadata['split_timestamp']}"
    )

    log(
        f"Train rows: "
        f"{train_rows:,} "
        f"({metadata['train_fraction']:.2%})"
    )

    log(
        f"Test rows: "
        f"{test_rows:,} "
        f"({metadata['test_fraction']:.2%})"
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
        f"Split metadata written: "
        f"{SPLIT_METADATA_PATH}"
    )

    return metadata


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "FRAUDSENTINEL AI — PHASE 1 TRAINING"
    )
    print("=" * 80)

    log(
        f"Project root: {PROJECT_ROOT}"
    )

    log(
        f"Python executable: {sys.executable}"
    )

    # ------------------------------------------------------------------------
    # Validate raw data
    # ------------------------------------------------------------------------

    df = load_and_validate_raw_data()

    log(
        f"Raw rows: {len(df):,}"
    )

    # ------------------------------------------------------------------------
    # Persist temporal split
    # ------------------------------------------------------------------------

    split_metadata = (
        persist_temporal_split_metadata(df)
    )

    # ------------------------------------------------------------------------
    # Run actual training pipeline
    # ------------------------------------------------------------------------

    log(
        "Starting Phase1TrainingPipeline..."
    )

    pipeline = Phase1TrainingPipeline(
        data_path=str(RAW_DATA),
        threshold=THRESHOLD,
    )

    results = pipeline.run()

    # ------------------------------------------------------------------------
    # Verify model artifact
    # ------------------------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Phase-1 pipeline completed but the expected "
            f"model was not created:\n{MODEL_PATH}"
        )

    log(
        f"Fresh model created: {MODEL_PATH}"
    )

    # ------------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)

    print(
        f"Operating threshold: "
        f"{results.get('threshold', THRESHOLD)}"
    )

    print(
        f"Best iteration: "
        f"{results.get('best_iteration', 'N/A')}"
    )

    print()
    print("Model:")
    print(MODEL_PATH)

    print()
    print("Evaluation:")
    print(EVALUATION_PATH)

    print()
    print("Temporal split:")
    print(SPLIT_METADATA_PATH)

    print()
    print(
        f"Train rows: "
        f"{split_metadata['train_rows']:,}"
    )

    print(
        f"Test rows: "
        f"{split_metadata['test_rows']:,}"
    )

    print(
        f"Split timestamp: "
        f"{split_metadata['split_timestamp']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\n[PHASE1-TRAIN][FATAL] "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
