from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    """
    Find the first matching column from a list of candidates.
    Matching is case-insensitive.
    """

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def load_ground_truth_transactions(
    input_path: str | Path,
) -> pd.DataFrame:
    """
    Load the original labeled transaction dataset.

    Expected logical fields:

        timestamp
        merchant_id / merchant
        amount
        is_fraud

    The function supports common alternative names.
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Ground-truth dataset is empty.")

    timestamp_col = _find_column(
        df,
        [
            "timestamp",
            "trans_date_trans_time",
            "transaction_timestamp",
            "datetime",
            "date",
        ],
    )

    merchant_col = _find_column(
        df,
        [
            "merchant_id",
            "merchant",
            "merchant_name",
        ],
    )

    amount_col = _find_column(
        df,
        [
            "amount",
            "amt",
            "transaction_amount",
        ],
    )

    fraud_col = _find_column(
        df,
        [
            "is_fraud",
            "fraud",
            "label",
            "fraud_label",
        ],
    )

    out = pd.DataFrame()

    out["timestamp"] = pd.to_datetime(
        df[timestamp_col],
        errors="coerce",
        utc=True,
    )

    out["merchant_id"] = (
        df[merchant_col]
        .astype(str)
        .str.strip()
    )

    out["amount"] = pd.to_numeric(
        df[amount_col],
        errors="coerce",
    )

    out["is_fraud"] = pd.to_numeric(
        df[fraud_col],
        errors="coerce",
    )

    out = out.dropna(
        subset=[
            "timestamp",
            "merchant_id",
            "amount",
            "is_fraud",
        ]
    )

    out["is_fraud"] = (
        out["is_fraud"]
        .astype(int)
        .clip(0, 1)
    )

    out["amount"] = out["amount"].clip(lower=0)

    out = out.sort_values(
        ["merchant_id", "timestamp"]
    ).reset_index(drop=True)

    print("\nGround-truth dataset")
    print("-" * 60)
    print(f"Rows:       {len(out):,}")
    print(f"Merchants:  {out['merchant_id'].nunique():,}")
    print(f"Fraud rows: {out['is_fraud'].sum():,}")
    print(
        f"Fraud rate: {out['is_fraud'].mean():.6%}"
    )
    print(
        f"Fraud amount: "
        f"{out.loc[out['is_fraud'] == 1, 'amount'].sum():,.2f}"
    )

    return out


def build_phase2_ground_truth(
    input_path: str | Path,
    window_minutes: int = 15,
    min_fraud_transactions: int = 1,
    coordinated_fraud_transactions: int = 2,
    coordinated_unique_cards: int = 2,
) -> pd.DataFrame:
    """
    Construct actual fraud ground truth at the same merchant/time-window
    resolution used by Phase 2.

    Important:
        Ground truth is based ONLY on actual is_fraud labels.

    A window is:

        fraud_positive = actual fraud count >= min_fraud_transactions

    A window is:

        coordinated_fraud_positive =
            actual fraud count >= coordinated_fraud_transactions
            OR
            unique fraudulent cards >= coordinated_unique_cards

    These definitions are configurable and should be treated as
    validation policy, not as mathematical truth.
    """

    df = load_ground_truth_transactions(input_path)

    freq = f"{window_minutes}min"

    df["window_start"] = (
        df["timestamp"]
        .dt.floor(freq)
    )

    df["window_end"] = (
        df["window_start"]
        + pd.Timedelta(minutes=window_minutes)
    )

    df["fraud_amount"] = (
        df["amount"] * df["is_fraud"]
    )

    grouped = (
        df.groupby(
            [
                "merchant_id",
                "window_start",
                "window_end",
            ],
            observed=True,
        )
        .agg(
            actual_transaction_count=(
                "amount",
                "size",
            ),
            actual_total_amount=(
                "amount",
                "sum",
            ),
            actual_fraud_count=(
                "is_fraud",
                "sum",
            ),
            actual_fraud_amount=(
                "fraud_amount",
                "sum",
            ),
        )
        .reset_index()
    )

    fraud_cards = (
        df.loc[df["is_fraud"] == 1]
        .groupby(
            [
                "merchant_id",
                "window_start",
            ],
            observed=True,
        )["is_fraud"]
        .size()
        .rename("fraud_transaction_count_check")
        .reset_index()
    )

    # Calculate unique fraudulent cards only when card_id exists.
    raw = pd.read_csv(input_path, nrows=5)

    card_col = _find_column(
        raw,
        [
            "card_id",
            "card",
            "account",
            "customer_id",
        ],
        required=False,
    )

    if card_col is not None:
        full = pd.read_csv(input_path)

        full_timestamp_col = _find_column(
            full,
            [
                "timestamp",
                "trans_date_trans_time",
                "transaction_timestamp",
                "datetime",
                "date",
            ],
        )

        full_merchant_col = _find_column(
            full,
            [
                "merchant_id",
                "merchant",
                "merchant_name",
            ],
        )

        full_fraud_col = _find_column(
            full,
            [
                "is_fraud",
                "fraud",
                "label",
                "fraud_label",
            ],
        )

        card_df = pd.DataFrame()

        card_df["timestamp"] = pd.to_datetime(
            full[full_timestamp_col],
            errors="coerce",
            utc=True,
        )

        card_df["merchant_id"] = (
            full[full_merchant_col]
            .astype(str)
            .str.strip()
        )

        card_df["card_id"] = (
            full[card_col]
            .astype(str)
            .str.strip()
        )

        card_df["is_fraud"] = pd.to_numeric(
            full[full_fraud_col],
            errors="coerce",
        )

        card_df = card_df.dropna(
            subset=[
                "timestamp",
                "merchant_id",
                "card_id",
                "is_fraud",
            ]
        )

        card_df["is_fraud"] = (
            card_df["is_fraud"]
            .astype(int)
            .clip(0, 1)
        )

        card_df["window_start"] = (
            card_df["timestamp"]
            .dt.floor(freq)
        )

        fraud_card_counts = (
            card_df.loc[
                card_df["is_fraud"] == 1
            ]
            .groupby(
                [
                    "merchant_id",
                    "window_start",
                ],
                observed=True,
            )["card_id"]
            .nunique()
            .rename("actual_unique_fraud_cards")
            .reset_index()
        )

        grouped = grouped.merge(
            fraud_card_counts,
            on=[
                "merchant_id",
                "window_start",
            ],
            how="left",
        )

    else:
        grouped["actual_unique_fraud_cards"] = 0

    grouped["actual_unique_fraud_cards"] = (
        grouped["actual_unique_fraud_cards"]
        .fillna(0)
        .astype(int)
    )

    grouped["actual_fraud_rate"] = np.where(
        grouped["actual_transaction_count"] > 0,
        grouped["actual_fraud_count"]
        / grouped["actual_transaction_count"],
        0.0,
    )

    grouped["fraud_positive"] = (
        grouped["actual_fraud_count"]
        >= min_fraud_transactions
    )

    grouped["coordinated_fraud_positive"] = (
        (
            grouped["actual_fraud_count"]
            >= coordinated_fraud_transactions
        )
        |
        (
            grouped["actual_unique_fraud_cards"]
            >= coordinated_unique_cards
        )
    )

    grouped["fraud_amount_positive"] = (
        grouped["actual_fraud_amount"] > 0
    )

    grouped["fraud_amount_share"] = np.where(
        grouped["actual_total_amount"] > 0,
        grouped["actual_fraud_amount"]
        / grouped["actual_total_amount"],
        0.0,
    )

    grouped = grouped.sort_values(
        [
            "merchant_id",
            "window_start",
        ]
    ).reset_index(drop=True)

    print("\nPhase 2 ground truth")
    print("-" * 60)

    print(
        f"Windows: "
        f"{len(grouped):,}"
    )

    print(
        f"Fraud-positive windows: "
        f"{grouped['fraud_positive'].sum():,}"
    )

    print(
        f"Coordinated-fraud windows: "
        f"{grouped['coordinated_fraud_positive'].sum():,}"
    )

    print(
        f"Fraud amount: "
        f"{grouped['actual_fraud_amount'].sum():,.2f}"
    )

    return grouped