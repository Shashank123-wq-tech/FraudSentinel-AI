"""
FraudSentinel AI
Phase 1 Prediction Runner

The underlying Phase1PredictionPipeline currently produces:

    trans_date_trans_time
    cc_num
    merchant
    amt
    fraud_probability
    fraud_prediction
    risk_level

This runner normalizes that output into the canonical Phase-1 -> Phase-2
contract:

    transaction_id
    timestamp
    merchant_id
    card_id
    amount
    fraud_probability
    risk_score
    risk_band
    predicted_fraud
    model_version
    threshold
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.pipeline.phase1_prediction_pipeline import (  # noqa: E402
    Phase1PredictionPipeline,
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

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase1"
    / "models"
    / "fraudsentinel_phase1_xgboost.json"
)

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "phase1"
    / "predictions"
)

LEGACY_PREDICTION_OUTPUT = (
    PREDICTIONS_DIR
    / "phase1_predictions_legacy.csv"
)

CANONICAL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "phase1"
    / "risk_output"
)

CANONICAL_OUTPUT = (
    CANONICAL_OUTPUT_DIR
    / "transaction_risk.csv"
)


# ============================================================================
# PHASE-1 POLICY
# ============================================================================

THRESHOLD = 0.84

MODEL_VERSION = "phase1_xgboost_v1"


# ============================================================================
# FINAL CONTRACT
# ============================================================================

REQUIRED_COLUMNS = [
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
# UTILITIES
# ============================================================================

def make_transaction_id(
    index: int,
) -> str:
    """
    Generate a deterministic transaction identifier.

    This is used only because the current raw dataset / prediction
    output does not expose a transaction_id field.

    The identifier is stable for a fixed input row ordering.
    """
    return f"TXN_{index:09d}"


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_prediction_output(
    legacy_path: Path,
) -> pd.DataFrame:

    if not legacy_path.exists():
        raise FileNotFoundError(
            f"Prediction file was not generated:\n"
            f"{legacy_path}"
        )

    df = pd.read_csv(
        legacy_path
    )

    print(
        f"[PHASE1-PREDICT] Legacy prediction rows: "
        f"{len(df):,}"
    )

    expected_legacy_columns = [
        "trans_date_trans_time",
        "cc_num",
        "merchant",
        "amt",
        "fraud_probability",
        "fraud_prediction",
        "risk_level",
    ]

    missing = [
        column
        for column in expected_legacy_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "The underlying Phase-1 prediction pipeline "
            "returned an unexpected schema.\n\n"
            f"Missing legacy columns: {missing}\n"
            f"Actual columns: {list(df.columns)}"
        )

    # ------------------------------------------------------------------------
    # Build canonical fields
    # ------------------------------------------------------------------------

    canonical = pd.DataFrame()

    # Deterministic transaction ID.
    canonical[
        "transaction_id"
    ] = [
        make_transaction_id(i)
        for i in range(len(df))
    ]

    # Timestamp.
    canonical[
        "timestamp"
    ] = pd.to_datetime(
        df["trans_date_trans_time"],
        errors="coerce",
        utc=True,
    )

    # Merchant.
    canonical[
        "merchant_id"
    ] = df[
        "merchant"
    ].astype(str)

    # Card.
    canonical[
        "card_id"
    ] = df[
        "cc_num"
    ].astype(str)

    # Amount.
    canonical[
        "amount"
    ] = pd.to_numeric(
        df["amt"],
        errors="coerce",
    )

    # Probability.
    canonical[
        "fraud_probability"
    ] = pd.to_numeric(
        df["fraud_probability"],
        errors="coerce",
    )

    # Risk score = probability × 100.
    canonical[
        "risk_score"
    ] = (
        canonical[
            "fraud_probability"
        ]
        * 100.0
    )

    # Risk band from the existing Phase-1 risk level.
    canonical[
        "risk_band"
    ] = df[
        "risk_level"
    ].astype(str)

    # Prediction.
    canonical[
        "predicted_fraud"
    ] = pd.to_numeric(
        df["fraud_prediction"],
        errors="coerce",
    )

    # Model version.
    canonical[
        "model_version"
    ] = MODEL_VERSION

    # Operating threshold.
    canonical[
        "threshold"
    ] = THRESHOLD

    # Ensure exact column order.
    canonical = canonical[
        REQUIRED_COLUMNS
    ]

    return canonical


# ============================================================================
# VALIDATION
# ============================================================================

def validate_canonical_output(
    df: pd.DataFrame,
) -> dict:

    # ------------------------------------------------------------------------
    # Exact columns
    # ------------------------------------------------------------------------

    if list(df.columns) != REQUIRED_COLUMNS:

        raise ValueError(
            "Canonical Phase-1 output schema is incorrect.\n\n"
            f"Expected:\n{REQUIRED_COLUMNS}\n\n"
            f"Actual:\n{list(df.columns)}"
        )

    if df.empty:
        raise ValueError(
            "Canonical Phase-1 output is empty."
        )

    # ------------------------------------------------------------------------
    # IDs
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

    if df[
        "timestamp"
    ].isna().any():

        raise ValueError(
            "Invalid timestamp values found."
        )

    # ------------------------------------------------------------------------
    # Numeric validation
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
                f"{column} contains "
                "missing/non-numeric values."
            )

    # ------------------------------------------------------------------------
    # Probability bounds
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
    # Risk score bounds
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
    # Verify score relationship
    # ------------------------------------------------------------------------

    expected_score = (
        df["fraud_probability"]
        * 100.0
    )

    score_difference = (
        df["risk_score"]
        - expected_score
    ).abs()

    if not (
        score_difference <= 1e-8
    ).all():

        raise ValueError(
            "risk_score is not consistent with "
            "100 × fraud_probability."
        )

    # ------------------------------------------------------------------------
    # Prediction bounds
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
    # Amount
    # ------------------------------------------------------------------------

    if (
        df["amount"] < 0
    ).any():

        raise ValueError(
            "Negative transaction amounts found."
        )

    # ------------------------------------------------------------------------
    # String fields
    # ------------------------------------------------------------------------

    for column in [
        "transaction_id",
        "merchant_id",
        "card_id",
        "risk_band",
        "model_version",
    ]:

        blank = (
            df[column]
            .astype(str)
            .str.strip()
            .eq("")
        )

        if blank.any():

            raise ValueError(
                f"{column} contains blank values."
            )

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "transactions": int(
            df["transaction_id"].nunique()
        ),
        "merchants": int(
            df["merchant_id"].nunique()
        ),
        "cards": int(
            df["card_id"].nunique()
        ),
        "predicted_fraud_count": int(
            df["predicted_fraud"].sum()
        ),
        "predicted_fraud_rate": float(
            df["predicted_fraud"].mean()
        ),
        "fraud_probability_mean": float(
            df["fraud_probability"].mean()
        ),
        "risk_score_mean": float(
            df["risk_score"].mean()
        ),
        "risk_score_max": float(
            df["risk_score"].max()
        ),
        "columns": REQUIRED_COLUMNS,
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "FRAUDSENTINEL AI — PHASE 1 PREDICTION"
    )
    print("=" * 80)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Python executable: {sys.executable}"
    )

    # ------------------------------------------------------------------------
    # Validate inputs.
    # ------------------------------------------------------------------------

    if not RAW_DATA.exists():

        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_DATA}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Phase-1 model not found:\n"
            f"{MODEL_PATH}"
        )

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CANONICAL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Run underlying Phase-1 prediction pipeline.
    # ------------------------------------------------------------------------

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Input: {RAW_DATA}"
    )

    print(
        f"Legacy output: {LEGACY_PREDICTION_OUTPUT}"
    )

    pipeline = Phase1PredictionPipeline(
        model_path=str(
            MODEL_PATH
        ),
        threshold=THRESHOLD,
    )

    pipeline.run(
        input_path=str(
            RAW_DATA
        ),
        output_path=str(
            LEGACY_PREDICTION_OUTPUT
        ),
    )

    # ------------------------------------------------------------------------
    # Normalize to canonical contract.
    # ------------------------------------------------------------------------

    print()
    print(
        "Normalizing Phase-1 prediction schema..."
    )

    canonical_df = normalize_prediction_output(
        LEGACY_PREDICTION_OUTPUT
    )

    # ------------------------------------------------------------------------
    # Validate canonical dataframe.
    # ------------------------------------------------------------------------

    info = validate_canonical_output(
        canonical_df
    )

    # ------------------------------------------------------------------------
    # Persist canonical transaction_risk.csv.
    # ------------------------------------------------------------------------

    canonical_df.to_csv(
        CANONICAL_OUTPUT,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Validate file after writing.
    # ------------------------------------------------------------------------

    written_df = pd.read_csv(
        CANONICAL_OUTPUT
    )

    written_info = validate_canonical_output(
        written_df
    )

    # ------------------------------------------------------------------------
    # Final summary.
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "PHASE 1 PREDICTION COMPLETE"
    )
    print("=" * 80)

    print(
        f"Legacy prediction: "
        f"{LEGACY_PREDICTION_OUTPUT}"
    )

    print(
        f"Canonical risk output: "
        f"{CANONICAL_OUTPUT}"
    )

    print(
        f"Rows: "
        f"{written_info['rows']:,}"
    )

    print(
        f"Transactions: "
        f"{written_info['transactions']:,}"
    )

    print(
        f"Merchants: "
        f"{written_info['merchants']:,}"
    )

    print(
        f"Cards: "
        f"{written_info['cards']:,}"
    )

    print(
        f"Predicted fraud: "
        f"{written_info['predicted_fraud_count']:,}"
    )

    print(
        f"Predicted fraud rate: "
        f"{written_info['predicted_fraud_rate']:.6%}"
    )

    print()
    print(
        "Canonical 11-column contract: PASSED"
    )

    print()
    print(
        "Columns:"
    )

    for column in REQUIRED_COLUMNS:
        print(
            f"  ✓ {column}"
        )


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print(
            f"\n[PHASE1-PREDICT][FATAL] "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        raise