from __future__ import annotations

import gc
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

TIMESTAMP_COL = "trans_date_trans_time"
TARGET_COL = "is_fraud"
CARD_COL = "cc_num"

RANDOM_STATE = 42

NIGHT_HOURS = [22, 23, 0, 1, 2, 3]

EPS = 1e-8


# ============================================================
# FEATURE GROUPS
# ============================================================

FEATURE_GROUPS: Dict[str, List[str]] = {

    "amount": [
        "amt_log",
        "is_high_value",
        "amount_vs_card_avg",
    ],

    "temporal": [
        "hour_sin",
        "hour_cos",
        "is_night",
        "is_peak_night",
        "is_weekend",
    ],

    "card_history": [
        "time_since_last_txn",
        "card_txn_count_so_far",
        "card_avg_amt_so_far",
        "card_std_amt_so_far",
        "amt_zscore_vs_card",
    ],

    "velocity": [
        "card_txn_count_1h",
        "card_txn_count_24h",
        "card_spend_1h",
        "card_spend_24h",
        "card_avg_amount_1h",
        "card_avg_amount_24h",
        "card_velocity_ratio",
        "card_spend_velocity_ratio",
    ],

    "novelty": [
        "is_first_card_transaction",
        "is_first_card_merchant",
        "is_first_card_category",
    ],

    "geography": [
        "distance_km",
        "distance_log",
        "distance_zscore_vs_card",
    ],

    "context": [
        "city_pop_log",
        "age",
    ],

    "interactions": [
        "peak_night_high_value",
        "novel_high_value",
    ],
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _safe_numeric(
    series: pd.Series,
    default: float = 0.0,
) -> pd.Series:

    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(default)
    )


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:

    numerator = _safe_numeric(numerator)
    denominator = _safe_numeric(denominator)

    return (
        numerator
        /
        denominator.replace(
            0,
            np.nan,
        )
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)


def _downcast_numeric(
    df: pd.DataFrame,
) -> pd.DataFrame:

    for column in df.columns:

        if pd.api.types.is_integer_dtype(
            df[column]
        ):

            df[column] = pd.to_numeric(
                df[column],
                downcast="integer",
            )

        elif pd.api.types.is_float_dtype(
            df[column]
        ):

            df[column] = pd.to_numeric(
                df[column],
                downcast="float",
            )

    return df


def _haversine_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:

    lat1 = np.radians(
        _safe_numeric(lat1).to_numpy(
            dtype=np.float64
        )
    )

    lon1 = np.radians(
        _safe_numeric(lon1).to_numpy(
            dtype=np.float64
        )
    )

    lat2 = np.radians(
        _safe_numeric(lat2).to_numpy(
            dtype=np.float64
        )
    )

    lon2 = np.radians(
        _safe_numeric(lon2).to_numpy(
            dtype=np.float64
        )
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    a = np.clip(
        a,
        0.0,
        1.0,
    )

    c = 2.0 * np.arcsin(
        np.sqrt(a)
    )

    return pd.Series(
        6371.0088 * c,
        index=lat2.shape and None,
    )


# ============================================================
# TEMPORAL FEATURES
# ============================================================

def add_temporal_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    hour = df[TIMESTAMP_COL].dt.hour

    weekday = df[TIMESTAMP_COL].dt.dayofweek

    df["hour_sin"] = (
        np.sin(
            2.0
            * np.pi
            * hour
            / 24.0
        )
        .astype("float32")
    )

    df["hour_cos"] = (
        np.cos(
            2.0
            * np.pi
            * hour
            / 24.0
        )
        .astype("float32")
    )

    night_mask = hour.isin(
        NIGHT_HOURS
    )

    df["is_night"] = (
        night_mask
        .astype("int8")
    )

    df["is_peak_night"] = (
        night_mask
        .astype("int8")
    )

    df["is_weekend"] = (
        (weekday >= 5)
        .astype("int8")
    )

    return df


# ============================================================
# AMOUNT FEATURES
# ============================================================

def add_amount_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    amount = _safe_numeric(
        df["amt"]
    )

    df["amt_log"] = (
        np.log1p(
            amount.clip(
                lower=0.0
            )
        )
        .astype("float32")
    )

    # --------------------------------------------------------
    # High-value transaction
    #
    # Uses a global transaction-level percentile calculated
    # without target information.
    # --------------------------------------------------------

    high_value_threshold = float(
        amount.quantile(
            0.95
        )
    )

    df["is_high_value"] = (
        amount >= high_value_threshold
    ).astype("int8")

    return df


# ============================================================
# CARD HISTORICAL FEATURES
# ============================================================

def add_card_historical_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    grouped = df.groupby(
        CARD_COL,
        sort=False,
    )

    # --------------------------------------------------------
    # Previous timestamp
    # --------------------------------------------------------

    previous_timestamp = (
        grouped[TIMESTAMP_COL]
        .shift(1)
    )

    time_delta = (
        df[TIMESTAMP_COL]
        -
        previous_timestamp
    )

    df["time_since_last_txn"] = (
        time_delta
        .dt.total_seconds()
        .fillna(0.0)
        .clip(
            lower=0.0
        )
        .astype("float32")
    )

    # --------------------------------------------------------
    # Historical count
    # --------------------------------------------------------

    df["card_txn_count_so_far"] = (
        grouped.cumcount()
        .astype("float32")
    )

    # --------------------------------------------------------
    # Historical amount statistics
    #
    # Shift first so the current transaction is never
    # included in its own history.
    # --------------------------------------------------------

    amount = _safe_numeric(
        df["amt"]
    )

    previous_amount = (
        grouped["amt"]
        .shift(1)
    )

    historical_count = (
        grouped["amt"]
        .cumcount()
        .astype("float64")
    )

    historical_sum = (
        previous_amount
        .fillna(0.0)
        .groupby(
            df[CARD_COL],
            sort=False,
        )
        .cumsum()
    )

    historical_sum_sq = (
        previous_amount
        .fillna(0.0)
        .pow(2)
        .groupby(
            df[CARD_COL],
            sort=False,
        )
        .cumsum()
    )

    historical_mean = _safe_divide(
        historical_sum,
        historical_count,
    )

    variance_numerator = (
        historical_sum_sq
        -
        historical_count
        *
        historical_mean.pow(2)
    )

    historical_variance = _safe_divide(
        variance_numerator,
        (
            historical_count - 1
        ).clip(
            lower=1
        ),
    )

    historical_std = (
        np.sqrt(
            historical_variance.clip(
                lower=0.0
            )
        )
        .fillna(0.0)
    )

    df["card_avg_amt_so_far"] = (
        historical_mean
        .fillna(amount)
        .astype("float32")
    )

    df["card_std_amt_so_far"] = (
        historical_std
        .astype("float32")
    )

    # --------------------------------------------------------
    # Current amount relative to card history
    # --------------------------------------------------------

    df["amt_zscore_vs_card"] = (
        (
            amount
            -
            df["card_avg_amt_so_far"]
        )
        /
        (
            df["card_std_amt_so_far"]
            +
            EPS
        )
    ).clip(
        -20.0,
        20.0,
    ).astype("float32")

    df["amount_vs_card_avg"] = (
        _safe_divide(
            amount,
            df["card_avg_amt_so_far"]
            + EPS,
        )
        .clip(
            0.0,
            100.0,
        )
        .astype("float32")
    )

    return df


# ============================================================
# VELOCITY FEATURES
# ============================================================

def add_rolling_velocity_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    work = df[
        [
            CARD_COL,
            TIMESTAMP_COL,
            "amt",
        ]
    ].copy()

    work["__row_id"] = np.arange(
        len(work),
        dtype=np.int64,
    )

    work = work.sort_values(
        [
            CARD_COL,
            TIMESTAMP_COL,
            "__row_id",
        ],
        kind="mergesort",
    )

    # --------------------------------------------------------
    # Historical rolling calculations.
    #
    # closed="left" excludes current transaction.
    # --------------------------------------------------------

    grouped = (
        work
        .set_index(
            TIMESTAMP_COL
        )
        .groupby(
            CARD_COL,
            sort=False,
        )["amt"]
    )

    count_1h = (
        grouped
        .rolling(
            "1h",
            closed="left",
        )
        .count()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    count_24h = (
        grouped
        .rolling(
            "24h",
            closed="left",
        )
        .count()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    spend_1h = (
        grouped
        .rolling(
            "1h",
            closed="left",
        )
        .sum()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    spend_24h = (
        grouped
        .rolling(
            "24h",
            closed="left",
        )
        .sum()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    # --------------------------------------------------------
    # The rolling results are indexed by timestamp.
    # Because duplicate timestamps can exist, use a stable
    # row-id based merge rather than relying on positional
    # assignment.
    # --------------------------------------------------------

    rolling = work[
        [
            CARD_COL,
            TIMESTAMP_COL,
            "__row_id",
        ]
    ].copy()

    rolling["card_txn_count_1h"] = (
        count_1h.to_numpy()
    )

    rolling["card_txn_count_24h"] = (
        count_24h.to_numpy()
    )

    rolling["card_spend_1h"] = (
        spend_1h.to_numpy()
    )

    rolling["card_spend_24h"] = (
        spend_24h.to_numpy()
    )

    rolling = rolling[
        [
            "__row_id",
            "card_txn_count_1h",
            "card_txn_count_24h",
            "card_spend_1h",
            "card_spend_24h",
        ]
    ]

    df["__row_id"] = np.arange(
        len(df),
        dtype=np.int64,
    )

    df = df.merge(
        rolling,
        on="__row_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )

    df.drop(
        columns="__row_id",
        inplace=True,
    )

    for column in [
        "card_txn_count_1h",
        "card_txn_count_24h",
        "card_spend_1h",
        "card_spend_24h",
    ]:

        df[column] = (
            _safe_numeric(
                df[column]
            )
            .clip(
                lower=0.0
            )
            .astype("float32")
        )

    df["card_avg_amount_1h"] = (
        _safe_divide(
            df["card_spend_1h"],
            df["card_txn_count_1h"],
        )
        .astype("float32")
    )

    df["card_avg_amount_24h"] = (
        _safe_divide(
            df["card_spend_24h"],
            df["card_txn_count_24h"],
        )
        .astype("float32")
    )

    df["card_velocity_ratio"] = (
        _safe_divide(
            df["card_txn_count_1h"],
            df["card_txn_count_24h"],
        )
        .clip(
            0.0,
            1.0,
        )
        .astype("float32")
    )

    df["card_spend_velocity_ratio"] = (
        _safe_divide(
            df["card_spend_1h"],
            df["card_spend_24h"],
        )
        .clip(
            0.0,
            1.0,
        )
        .astype("float32")
    )

    del work
    del rolling

    gc.collect()

    return df


# ============================================================
# NOVELTY FEATURES
# ============================================================

def add_novelty_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    # --------------------------------------------------------
    # First-ever transaction for card
    # --------------------------------------------------------

    card_count_before = (
        df.groupby(
            CARD_COL,
            sort=False,
        )
        .cumcount()
    )

    df["is_first_card_transaction"] = (
        card_count_before == 0
    ).astype("int8")

    # --------------------------------------------------------
    # First card-merchant interaction
    # --------------------------------------------------------

    card_merchant_seen = (
        df.groupby(
            [
                CARD_COL,
                "merchant",
            ],
            sort=False,
        )
        .cumcount()
    )

    df["is_first_card_merchant"] = (
        card_merchant_seen == 0
    ).astype("int8")

    # --------------------------------------------------------
    # First card-category interaction
    # --------------------------------------------------------

    card_category_seen = (
        df.groupby(
            [
                CARD_COL,
                "category",
            ],
            sort=False,
        )
        .cumcount()
    )

    df["is_first_card_category"] = (
        card_category_seen == 0
    ).astype("int8")

    return df


# ============================================================
# GEOGRAPHIC FEATURES
# ============================================================

def add_geographic_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    distance = _haversine_km(
        df["lat"],
        df["long"],
        df["merch_lat"],
        df["merch_long"],
    )

    # Restore original dataframe index
    distance.index = df.index

    distance = (
        distance
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .clip(
            lower=0.0
        )
    )

    df["distance_km"] = (
        distance
        .astype("float32")
    )

    df["distance_log"] = (
        np.log1p(
            distance
        )
        .astype("float32")
    )

    # --------------------------------------------------------
    # Historical distance deviation
    # --------------------------------------------------------

    previous_distance = (
        df.groupby(
            CARD_COL,
            sort=False,
        )["distance_km"]
        .shift(1)
    )

    cumulative_sum = (
        previous_distance
        .fillna(0.0)
        .groupby(
            df[CARD_COL],
            sort=False,
        )
        .cumsum()
    )

    count = (
        df.groupby(
            CARD_COL,
            sort=False,
        )
        .cumcount()
        .astype("float64")
    )

    historical_mean = _safe_divide(
        cumulative_sum,
        count,
    )

    previous_squared = (
        previous_distance
        .fillna(0.0)
        .pow(2)
    )

    cumulative_squared = (
        previous_squared
        .groupby(
            df[CARD_COL],
            sort=False,
        )
        .cumsum()
    )

    variance = _safe_divide(
        (
            cumulative_squared
            -
            count
            *
            historical_mean.pow(2)
        ),
        (
            count - 1
        ).clip(
            lower=1
        ),
    )

    std = (
        np.sqrt(
            variance.clip(
                lower=0.0
            )
        )
        .fillna(0.0)
    )

    distance_zscore = (
        (
            df["distance_km"]
            -
            historical_mean
        )
        /
        (
            std + EPS
        )
    )

    distance_zscore = (
        distance_zscore
        .where(
            count > 0,
            0.0,
        )
        .clip(
            -20.0,
            20.0,
        )
    )

    df["distance_zscore_vs_card"] = (
        distance_zscore
        .astype("float32")
    )

    return df


# ============================================================
# CONTEXT FEATURES
# ============================================================

def add_context_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df["city_pop_log"] = (
        np.log1p(
            _safe_numeric(
                df["city_pop"]
            )
            .clip(
                lower=0.0
            )
        )
        .astype("float32")
    )

    birth_date = pd.to_datetime(
        df["dob"],
        errors="coerce",
    )

    age = (
        (
            df[TIMESTAMP_COL]
            -
            birth_date
        )
        .dt.days
        /
        365.25
    )

    df["age"] = (
        age
        .fillna(
            age.median()
            if age.notna().any()
            else 0.0
        )
        .clip(
            0.0,
            120.0,
        )
        .astype("float32")
    )

    return df


# ============================================================
# INTERACTION FEATURES
# ============================================================

def add_interaction_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df["peak_night_high_value"] = (
        (
            df["is_peak_night"]
            == 1
        )
        &
        (
            df["is_high_value"]
            == 1
        )
    ).astype("int8")

    novelty = (
        (
            df["is_first_card_transaction"]
            == 1
        )
        |
        (
            df["is_first_card_merchant"]
            == 1
        )
        |
        (
            df["is_first_card_category"]
            == 1
        )
    )

    df["novel_high_value"] = (
        novelty
        &
        (
            df["is_high_value"]
            == 1
        )
    ).astype("int8")

    return df


# ============================================================
# FEATURE MATRIX
# ============================================================

def _build_feature_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:

    feature_names = [
        feature
        for group in FEATURE_GROUPS.values()
        for feature in group
    ]

    missing = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing engineered Phase 1 features: "
            f"{missing}"
        )

    X = df[
        feature_names
    ].copy()

    # --------------------------------------------------------
    # Numeric safety
    # --------------------------------------------------------

    for column in X.columns:

        X[column] = (
            pd.to_numeric(
                X[column],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

    X = _downcast_numeric(
        X
    )

    return X


# ============================================================
# FEATURE ENGINEERING ENTRY POINT
# ============================================================

def engineer_phase1_features(
    df: pd.DataFrame,
    restore_time_order: bool = True,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:

    logger.info(
        "Starting Phase 1 feature engineering"
    )

    if df is None:
        raise ValueError(
            "Input dataframe is None."
        )

    if df.empty:
        raise ValueError(
            "Input dataframe is empty."
        )

    required_columns = [
        TIMESTAMP_COL,
        TARGET_COL,
        CARD_COL,
        "amt",
        "merchant",
        "category",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "city_pop",
        "dob",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required raw columns for "
            f"Phase 1 feature engineering: {missing}"
        )

    # --------------------------------------------------------
    # Copy input
    # --------------------------------------------------------

    df = df.copy()

    # --------------------------------------------------------
    # Datetime conversion
    # --------------------------------------------------------

    df[TIMESTAMP_COL] = pd.to_datetime(
        df[TIMESTAMP_COL],
        errors="coerce",
    )

    df["dob"] = pd.to_datetime(
        df["dob"],
        errors="coerce",
    )

    if df[TIMESTAMP_COL].isna().any():
        raise ValueError(
            "Invalid timestamps found in "
            f"{TIMESTAMP_COL}."
        )

    # --------------------------------------------------------
    # Stable chronological order
    # --------------------------------------------------------

    if restore_time_order:

        df.sort_values(
            TIMESTAMP_COL,
            kind="mergesort",
            inplace=True,
        )

        df.reset_index(
            drop=True,
            inplace=True,
        )

    # --------------------------------------------------------
    # Downcast raw numeric columns
    # --------------------------------------------------------

    df = _downcast_numeric(
        df
    )

    logger.info(
        "Input after preparation: shape=%s",
        df.shape,
    )

    # ========================================================
    # 1. TEMPORAL
    # ========================================================

    df = add_temporal_features(
        df
    )

    # ========================================================
    # 2. AMOUNT
    # ========================================================

    df = add_amount_features(
        df
    )

    # ========================================================
    # 3. CARD HISTORY
    # ========================================================

    df = add_card_historical_features(
        df
    )

    gc.collect()

    # ========================================================
    # 4. VELOCITY
    # ========================================================

    df = add_rolling_velocity_features(
        df
    )

    gc.collect()

    # ========================================================
    # 5. NOVELTY
    # ========================================================

    df = add_novelty_features(
        df
    )

    # ========================================================
    # 6. GEOGRAPHY
    # ========================================================

    df = add_geographic_features(
        df
    )

    # ========================================================
    # 7. CONTEXT
    # ========================================================

    df = add_context_features(
        df
    )

    # ========================================================
    # 8. INTERACTIONS
    # ========================================================

    df = add_interaction_features(
        df
    )

    gc.collect()

    # ========================================================
    # BUILD X / y
    # ========================================================

    X = _build_feature_matrix(
        df
    )

    y = (
        pd.to_numeric(
            df[TARGET_COL],
            errors="raise",
        )
        .astype("int8")
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if len(df) != len(X):
        raise ValueError(
            "Dataframe and feature matrix "
            "row counts do not match."
        )

    if len(X) != len(y):
        raise ValueError(
            "Feature matrix and target "
            "row counts do not match."
        )

    if not set(
        y.unique()
    ).issubset({0, 1}):

        raise ValueError(
            "Target column must contain only 0/1."
        )

    if X.isna().any().any():
        raise ValueError(
            "NaN values remain in Phase 1 "
            "feature matrix."
        )

    if np.isinf(
        X.to_numpy(
            dtype=np.float64
        )
    ).any():

        raise ValueError(
            "Infinite values remain in Phase 1 "
            "feature matrix."
        )

    logger.info(
        "Phase 1 feature engineering completed | "
        "rows=%d | features=%d",
        len(X),
        X.shape[1],
    )

    logger.info(
        "Feature names=%s",
        list(X.columns),
    )

    return (
        df,
        X,
        y,
    )


# ============================================================
# BACKWARD-COMPATIBLE ALIAS
# ============================================================

def build_phase1_features(
    df: pd.DataFrame,
    restore_time_order: bool = True,
):
    """
    Backward-compatible alias.

    The existing Phase 1 runner uses
    engineer_phase1_features().
    """

    return engineer_phase1_features(
        df,
        restore_time_order=restore_time_order,
    )


# ============================================================
# FEATURE MATRIX VALIDATION
# ============================================================

def validate_feature_matrix(
    X: pd.DataFrame,
) -> Dict[str, object]:

    if X is None:
        raise ValueError(
            "Feature matrix is None."
        )

    if X.empty:
        raise ValueError(
            "Feature matrix is empty."
        )

    numeric_columns = list(
        X.select_dtypes(
            include=[np.number]
        ).columns
    )

    non_numeric = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    missing_count = int(
        X.isna().sum().sum()
    )

    infinite_count = int(
        np.isinf(
            X.to_numpy(
                dtype=np.float64
            )
        ).sum()
    )

    return {
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "non_numeric_columns": non_numeric,
        "missing_values": missing_count,
        "infinite_values": infinite_count,
        "feature_names": list(X.columns),
    }