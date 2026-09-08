# src/components/phase2/windows.py

import numpy as np
import pandas as pd


def add_new_card_features(
    transactions: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate merchant-level new-card activity.

    A card is considered new for a merchant if its first
    transaction with that merchant occurs inside the current window.
    """

    x = transactions.copy()

    window_minutes = int(
        windows["window_minutes"].iloc[0]
    )

    x["window_start"] = (
        x["timestamp"]
        .dt.floor(f"{window_minutes}min")
    )

    # ------------------------------------------------------------
    # First transaction of card with merchant
    # ------------------------------------------------------------

    first_seen = (
        x
        .groupby(
            ["merchant_id", "card_id"],
            observed=True,
        )["timestamp"]
        .min()
        .rename("card_first_seen")
        .reset_index()
    )

    x = x.merge(
        first_seen,
        on=[
            "merchant_id",
            "card_id",
        ],
        how="left",
    )

    x["is_new_card"] = (
        x["timestamp"]
        == x["card_first_seen"]
    ).astype(int)

    # ------------------------------------------------------------
    # Aggregate new cards
    # ------------------------------------------------------------

    new_cards = (
        x
        .groupby(
            [
                "merchant_id",
                "window_start",
            ],
            observed=True,
        )
        .agg(
            new_cards=(
                "is_new_card",
                "sum",
            )
        )
        .reset_index()
    )

    # ------------------------------------------------------------
    # Merge with windows
    # ------------------------------------------------------------

    result = windows.merge(
        new_cards,
        on=[
            "merchant_id",
            "window_start",
        ],
        how="left",
    )

    result["new_cards"] = (
        result["new_cards"]
        .fillna(0)
        .astype(int)
    )

    result["new_card_rate"] = (
        result["new_cards"]
        / result["unique_cards"].clip(lower=1)
    ).clip(0, 1)

    return result


def build_transaction_window_features(
    df: pd.DataFrame,
    window_minutes: int,
) -> pd.DataFrame:
    """
    Convert transaction-level Phase-1 risk output into
    merchant-level fixed time windows.

    Core quantities:

        E[F] = Σ p_i

        E[$F] = Σ p_i * amount_i

        risk_rate = Σ p_i / N

    where p_i is the Phase-1 fraud probability.
    """

    if window_minutes <= 0:
        raise ValueError(
            "window_minutes must be positive."
        )

    x = df.copy()

    # ============================================================
    # VALIDATION
    # ============================================================

    required_columns = [
        "transaction_id",
        "timestamp",
        "merchant_id",
        "card_id",
        "amount",
        "fraud_probability",
        "risk_score",
        "predicted_fraud",
    ]

    missing = [
        column
        for column in required_columns
        if column not in x.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns for window construction: "
            f"{missing}"
        )

    # ============================================================
    # FIXED UTC WINDOW
    # ============================================================

    x["window_start"] = (
        x["timestamp"]
        .dt.floor(f"{window_minutes}min")
    )

    # ============================================================
    # WINDOW END
    #
    # This is required by the event builder.
    # ============================================================

    x["window_end"] = (
        x["window_start"]
        + pd.Timedelta(
            minutes=window_minutes
        )
    )

    # ============================================================
    # EXPECTED FRAUD AMOUNT
    # ============================================================

    x["expected_fraud_amount_tx"] = (
        x["fraud_probability"]
        * x["amount"]
    )

    # ============================================================
    # MERCHANT-WINDOW GROUPING
    # ============================================================

    grouped = (
        x
        .groupby(
            [
                "merchant_id",
                "window_start",
            ],
            sort=True,
            observed=True,
        )
    )

    windows = (
        grouped
        .agg(
            transaction_count=(
                "transaction_id",
                "size",
            ),

            total_amount=(
                "amount",
                "sum",
            ),

            expected_fraud_count=(
                "fraud_probability",
                "sum",
            ),

            expected_fraud_amount=(
                "expected_fraud_amount_tx",
                "sum",
            ),

            avg_fraud_probability=(
                "fraud_probability",
                "mean",
            ),

            avg_risk_score=(
                "risk_score",
                "mean",
            ),

            max_risk_score=(
                "risk_score",
                "max",
            ),

            unique_cards=(
                "card_id",
                "nunique",
            ),

            high_risk_count=(
                "predicted_fraud",
                "sum",
            ),
        )
        .reset_index()
    )

    # ============================================================
    # WINDOW END
    # ============================================================

    windows["window_end"] = (
        windows["window_start"]
        + pd.Timedelta(
            minutes=window_minutes
        )
    )

    # ============================================================
    # EXPECTED FRAUD RATE
    # ============================================================

    windows["risk_rate"] = (
        windows["expected_fraud_count"]
        / windows["transaction_count"].clip(lower=1)
    )

    # ============================================================
    # EXPECTED FRAUD LOSS PER TRANSACTION
    # ============================================================

    windows["expected_loss_per_txn"] = (
        windows["expected_fraud_amount"]
        / windows["transaction_count"].clip(lower=1)
    )

    # ============================================================
    # METADATA
    # ============================================================

    windows["window_minutes"] = window_minutes

    # ============================================================
    # NUMERICAL SAFETY
    # ============================================================

    numeric_columns = [
        "transaction_count",
        "total_amount",
        "expected_fraud_count",
        "expected_fraud_amount",
        "avg_fraud_probability",
        "avg_risk_score",
        "max_risk_score",
        "unique_cards",
        "high_risk_count",
        "risk_rate",
        "expected_loss_per_txn",
    ]

    for column in numeric_columns:

        windows[column] = (
            pd.to_numeric(
                windows[column],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

    return windows


# ================================================================
# PIPELINE INTERFACE
# ================================================================

def build_merchant_windows(
    df: pd.DataFrame,
    window_minutes: int = 15,
) -> pd.DataFrame:
    """
    Standard Phase-2 pipeline interface.

    Converts transaction-level Phase-1 risk output into
    merchant-level fixed 15-minute windows.

    This wrapper exists so pipeline.py has a stable function
    name while the underlying implementation remains
    build_transaction_window_features().
    """

    windows = build_transaction_window_features(
        df=df,
        window_minutes=window_minutes,
    )

    # ------------------------------------------------------------
    # Add new-card activity
    # ------------------------------------------------------------

    windows = add_new_card_features(
        transactions=df,
        windows=windows,
    )

    # ------------------------------------------------------------
    # Final required columns
    # ------------------------------------------------------------

    required_columns = [
        "merchant_id",
        "window_start",
        "window_end",
        "transaction_count",
        "total_amount",
        "expected_fraud_count",
        "expected_fraud_amount",
        "risk_rate",
        "unique_cards",
        "new_cards",
        "new_card_rate",
        "window_minutes",
    ]

    missing = [
        column
        for column in required_columns
        if column not in windows.columns
    ]

    if missing:
        raise RuntimeError(
            "Window builder failed to produce required "
            f"columns: {missing}"
        )

    return windows