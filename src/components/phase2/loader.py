# src/components/phase2/loader.py

from pathlib import Path

import pandas as pd

from .schema import REQUIRED_PHASE1_COLUMNS


def load_phase1_risk(
    input_path: str,
) -> pd.DataFrame:
    """
    Load and validate Phase-1 transaction risk output.

    Expected columns:

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

    Phase 2 uses fraud_probability as the primary
    transaction-level fraud-risk signal.
    """

    # ============================================================
    # 1. CHECK FILE
    # ============================================================

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Phase-1 risk file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Input path is not a file: {path}"
        )

    # ============================================================
    # 2. LOAD DATA
    # ============================================================

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "Phase-1 risk file is empty."
        )

    # ============================================================
    # 3. CHECK REQUIRED COLUMNS
    # ============================================================

    missing_columns = (
        set(REQUIRED_PHASE1_COLUMNS)
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Phase-1 risk file is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    # ============================================================
    # 4. TIMESTAMP
    # ============================================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    bad_timestamp = df["timestamp"].isna().sum()

    if bad_timestamp > 0:
        raise ValueError(
            f"Invalid timestamps found: {bad_timestamp}"
        )

    # ============================================================
    # 5. NUMERIC COLUMNS
    # ============================================================

    numeric_columns = [
        "amount",
        "fraud_probability",
        "risk_score",
        "predicted_fraud",
        "threshold",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        bad_values = df[column].isna().sum()

        if bad_values > 0:
            raise ValueError(
                f"Invalid numeric values in "
                f"{column}: {bad_values}"
            )

    # ============================================================
    # 6. AMOUNT VALIDATION
    # ============================================================

    if (df["amount"] < 0).any():
        raise ValueError(
            "Negative transaction amounts detected."
        )

    # ============================================================
    # 7. FRAUD PROBABILITY VALIDATION
    # ============================================================

    if (
        (df["fraud_probability"] < 0)
        | (df["fraud_probability"] > 1)
    ).any():

        raise ValueError(
            "fraud_probability must be between 0 and 1."
        )

    # ============================================================
    # 8. PREDICTED FRAUD VALIDATION
    # ============================================================

    if not df["predicted_fraud"].isin([0, 1]).all():

        raise ValueError(
            "predicted_fraud must contain only 0 or 1."
        )

    # ============================================================
    # 9. TRANSACTION ID VALIDATION
    # ============================================================

    if df["transaction_id"].isna().any():

        raise ValueError(
            "transaction_id contains missing values."
        )

    duplicate_ids = df["transaction_id"].duplicated().sum()

    if duplicate_ids > 0:

        raise ValueError(
            f"Duplicate transaction IDs detected: "
            f"{duplicate_ids}"
        )

    # ============================================================
    # 10. MERCHANT / CARD VALIDATION
    # ============================================================

    for column in [
        "merchant_id",
        "card_id",
    ]:

        if df[column].isna().any():

            raise ValueError(
                f"{column} contains missing values."
            )

    # ============================================================
    # 11. SORT CHRONOLOGICALLY
    # ============================================================

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # ============================================================
    # 12. USE FLOAT32 WHERE APPROPRIATE
    # ============================================================

    df["amount"] = df["amount"].astype("float64")

    df["fraud_probability"] = (
        df["fraud_probability"]
        .astype("float64")
    )

    df["risk_score"] = (
        df["risk_score"]
        .astype("float64")
    )

    # ============================================================
    # 13. DIAGNOSTIC INFORMATION
    # ============================================================

    print("\n[PHASE 2 LOADER]")

    print(
        f"Rows                 : {len(df):,}"
    )

    print(
        f"Transactions         : "
        f"{df['transaction_id'].nunique():,}"
    )

    print(
        f"Merchants            : "
        f"{df['merchant_id'].nunique():,}"
    )

    print(
        f"Cards                : "
        f"{df['card_id'].nunique():,}"
    )

    print(
    f"Time range           : "
    f"{df['timestamp'].min()} -> "
    f"{df['timestamp'].max()}"
    )

    print(
        f"Mean fraud probability: "
        f"{df['fraud_probability'].mean():.6f}"
    )

    print(
        f"Max fraud probability : "
        f"{df['fraud_probability'].max():.6f}"
    )

    print(
        f"Predicted fraud rows  : "
        f"{int(df['predicted_fraud'].sum()):,}"
    )

    print(
        f"Total transaction amount: "
        f"{df['amount'].sum():,.2f}"
    )

    print(
        "[PHASE 2 LOADER] Validation PASSED"
    )

    return df