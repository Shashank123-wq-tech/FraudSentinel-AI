import gc
import numpy as np
import pandas as pd

from src.logger.logger import get_logger
from src.exception.exception import FraudSentinelException


logger = get_logger(__name__)


FEATURE_COLS = [

    "amt_log",
    "amt_percentile",

    "hour_sin",
    "hour_cos",
    "is_night",
    "is_peak_night",
    "is_weekend",

    "distance_km",

    "time_since_last_txn",

    "card_txn_count_so_far",
    "card_avg_amt_so_far",
    "card_std_amt_so_far",
    "amt_zscore_vs_card",

    "card_txn_count_1h",
    "card_txn_count_24h",

    "card_spend_1h",
    "card_spend_24h",

    "category_risk_encoded",
    "merchant_risk_encoded",

    "city_pop_log",
    "age",
]


def report_memory(df, label):

    mb = df.memory_usage(deep=True).sum() / 1e6

    logger.info(
        "[%s] shape=%s memory=%.1f MB",
        label,
        df.shape,
        mb
    )


def downcast_df(df):

    for col in df.select_dtypes(
        include=["float64"]
    ).columns:

        df[col] = pd.to_numeric(
            df[col],
            downcast="float"
        )

    for col in df.select_dtypes(
        include=["int64"]
    ).columns:

        df[col] = pd.to_numeric(
            df[col],
            downcast="integer"
        )

    return df


# ============================================================
# 1. TEMPORAL
# ============================================================

def add_temporal_features(df):

    df["hour"] = (
        df["trans_date_trans_time"]
        .dt.hour
        .astype("int8")
    )

    df["dayofweek"] = (
        df["trans_date_trans_time"]
        .dt.dayofweek
        .astype("int8")
    )

    df["is_weekend"] = (
        df["dayofweek"]
        .isin([5, 6])
        .astype("int8")
    )

    df["is_night"] = (
        df["hour"]
        .isin([22, 23, 0, 1, 2, 3])
        .astype("int8")
    )

    df["is_peak_night"] = (
        df["hour"]
        .isin([22, 23])
        .astype("int8")
    )

    df["hour_sin"] = (
        np.sin(
            2 * np.pi * df["hour"] / 24
        )
        .astype("float32")
    )

    df["hour_cos"] = (
        np.cos(
            2 * np.pi * df["hour"] / 24
        )
        .astype("float32")
    )

    return df


# ============================================================
# 2. AMOUNT
# ============================================================

def add_amount_log(df):

    df["amt_log"] = (
        np.log1p(df["amt"])
        .astype("float32")
    )

    return df


def add_expanding_amt_percentile(
    df,
    freq="D"
):

    date_bucket = (
        df["trans_date_trans_time"]
        .dt.floor(freq)
    )

    amt = df["amt"].to_numpy()

    percentiles = np.full(
        len(df),
        0.5,
        dtype="float32"
    )

    seen = np.empty(
        0,
        dtype="float32"
    )

    for _, idx in df.groupby(
        date_bucket
    ).indices.items():

        idx = np.asarray(idx)

        if seen.size > 0:

            percentiles[idx] = (
                np.searchsorted(
                    seen,
                    amt[idx]
                )
                / seen.size
            )

        seen = np.sort(
            np.concatenate(
                [
                    seen,
                    amt[idx].astype(
                        "float32"
                    )
                ]
            )
        )

    df["amt_percentile"] = percentiles

    return df


# ============================================================
# 3. GEO
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [
            lat1,
            lon1,
            lat2,
            lon2
        ]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return (
        2
        * 6371
        * np.arcsin(
            np.sqrt(a)
        )
    ).astype("float32")


def add_geo_features(df):

    df["distance_km"] = haversine_km(
        df["lat"],
        df["long"],
        df["merch_lat"],
        df["merch_long"]
    )

    df["distance_log"] = (
        np.log1p(
            df["distance_km"]
        )
        .astype("float32")
    )

    return df


# ============================================================
# 4. CARD EXPANDING FEATURES
# ============================================================

def add_card_expanding_stats(df):

    df.sort_values(
        [
            "cc_num",
            "trans_date_trans_time"
        ],
        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    grp = df.groupby(
        "cc_num"
    )

    df["time_since_last_txn"] = (
        grp[
            "trans_date_trans_time"
        ]
        .diff()
        .dt.total_seconds()
        / 60
    ).astype("float32")

    df["time_since_last_txn"] = (
        df["time_since_last_txn"]
        .fillna(
            df["time_since_last_txn"]
            .median()
        )
    )

    df["card_txn_count_so_far"] = (
        grp.cumcount()
        .astype("int32")
    )

    df["card_avg_amt_so_far"] = (
        grp["amt"]
        .transform(
            lambda s:
            s.shift(1)
            .expanding()
            .mean()
        )
        .astype("float32")
    )

    df["card_std_amt_so_far"] = (
        grp["amt"]
        .transform(
            lambda s:
            s.shift(1)
            .expanding()
            .std()
        )
        .astype("float32")
    )

    df["amt_zscore_vs_card"] = (
        (
            df["amt"]
            -
            df["card_avg_amt_so_far"]
        )
        /
        df["card_std_amt_so_far"]
        .replace(0, np.nan)
    ).fillna(0).astype("float32")

    df["card_avg_amt_so_far"] = (
        df["card_avg_amt_so_far"]
        .fillna(df["amt"])
    )

    df["card_std_amt_so_far"] = (
        df["card_std_amt_so_far"]
        .fillna(0)
    )

    return df


# ============================================================
# 5. ROLLING
# ============================================================

def add_rolling_velocity(
    df,
    window,
    col_name,
    agg="count"
):

    s = (
        df
        .set_index(
            "trans_date_trans_time"
        )
        .groupby("cc_num")["amt"]
    )

    roller = s.rolling(
        window,
        closed="left"
    )

    if agg == "count":

        result = roller.count()

    else:

        result = roller.sum()

    df[col_name] = (
        result
        .to_numpy(dtype="float32")
    )

    del result
    del roller
    del s

    return df


# ============================================================
# 6. CATEGORY / MERCHANT RISK
# ============================================================

def add_expanding_risk_encoding(
    df,
    group_col,
    target_col="is_fraud",
    smoothing_k=20,
    out_col=None
):

    out_col = (
        out_col
        or f"{group_col}_risk_encoded"
    )

    df.sort_values(
        "trans_date_trans_time",
        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    grp = df.groupby(
        group_col
    )[target_col]

    cum_sum = (
        grp.cumsum()
        .shift(1)
    )

    cum_count = (
        grp.cumcount()
    )

    running_global_mean = (
        df[target_col]
        .expanding()
        .mean()
        .shift(1)
        .fillna(
            df[target_col].mean()
        )
    )

    df[out_col] = (
        (
            cum_sum.fillna(0)
            +
            running_global_mean
            * smoothing_k
        )
        /
        (
            cum_count
            +
            smoothing_k
        )
    ).astype("float32")

    del cum_sum
    del cum_count
    del running_global_mean

    return df


# ============================================================
# 7. DEMOGRAPHICS
# ============================================================

def add_limited_demographic_features(df):

    df["age"] = (
        (
            df["trans_date_trans_time"]
            - df["dob"]
        )
        .dt.days
        // 365
    ).astype("int16")

    df["city_pop_log"] = (
        np.log1p(
            df["city_pop"]
        )
        .astype("float32")
    )

    return df


# ============================================================
# COMPLETE FEATURE ENGINEERING
# ============================================================

def engineer_phase1_features(
    df,
    restore_time_order=True
):

    try:

        logger.info(
            "Starting Phase 1 feature engineering"
        )

        df["trans_date_trans_time"] = (
            pd.to_datetime(
                df["trans_date_trans_time"]
            )
        )

        df["dob"] = (
            pd.to_datetime(
                df["dob"]
            )
        )

        df = downcast_df(df)

        df.sort_values(
            "trans_date_trans_time",
            inplace=True
        )

        df.reset_index(
            drop=True,
            inplace=True
        )

        report_memory(
            df,
            "after load + downcast"
        )

        # ----------------------------------------------------
        # Temporal
        # ----------------------------------------------------

        df = add_temporal_features(df)

        gc.collect()

        report_memory(
            df,
            "after temporal features"
        )

        # ----------------------------------------------------
        # Amount
        # ----------------------------------------------------

        df = add_amount_log(df)

        df = add_expanding_amt_percentile(
            df,
            freq="D"
        )

        gc.collect()

        report_memory(
            df,
            "after amount features"
        )

        # ----------------------------------------------------
        # Geo
        # ----------------------------------------------------

        df = add_geo_features(df)

        gc.collect()

        report_memory(
            df,
            "after geo features"
        )

        # ----------------------------------------------------
        # Card
        # ----------------------------------------------------

        df = add_card_expanding_stats(df)

        gc.collect()

        report_memory(
            df,
            "after card expanding stats"
        )

        # ----------------------------------------------------
        # Rolling
        # ----------------------------------------------------

        df = add_rolling_velocity(
            df,
            "1h",
            "card_txn_count_1h",
            agg="count"
        )

        gc.collect()

        df = add_rolling_velocity(
            df,
            "24h",
            "card_txn_count_24h",
            agg="count"
        )

        gc.collect()

        df = add_rolling_velocity(
            df,
            "1h",
            "card_spend_1h",
            agg="sum"
        )

        gc.collect()

        df = add_rolling_velocity(
            df,
            "24h",
            "card_spend_24h",
            agg="sum"
        )

        gc.collect()

        report_memory(
            df,
            "after rolling velocity/spend"
        )

        # ----------------------------------------------------
        # Risk encoding
        # ----------------------------------------------------

        df = add_expanding_risk_encoding(
            df,
            "category",
            out_col="category_risk_encoded"
        )

        gc.collect()

        df = add_expanding_risk_encoding(
            df,
            "merchant",
            out_col="merchant_risk_encoded"
        )

        gc.collect()

        report_memory(
            df,
            "after category/merchant risk encoding"
        )

        # ----------------------------------------------------
        # Demographics
        # ----------------------------------------------------

        df = add_limited_demographic_features(
            df
        )

        gc.collect()

        report_memory(
            df,
            "after demographic features"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Restore chronological ordering.
        #
        # The notebook's feature calculations use temporary
        # card/time sorting. We return the final dataframe
        # chronologically so the 80/20 future split remains
        # correctly aligned.
        # ----------------------------------------------------

        if restore_time_order:

            df.sort_values(
                "trans_date_trans_time",
                inplace=True
            )

            df.reset_index(
                drop=True,
                inplace=True
            )

        X = (
            df[FEATURE_COLS]
            .fillna(0)
            .astype("float32")
        )

        y = (
            df["is_fraud"]
            .astype("int8")
        )

        logger.info(
            "Phase 1 feature engineering completed"
        )

        logger.info(
            "X shape=%s",
            X.shape
        )

        logger.info(
            "Feature count=%d",
            len(FEATURE_COLS)
        )

        logger.info(
            "NaN count=%d",
            X.isna().sum().sum()
        )

        return df, X, y

    except Exception as e:

        logger.exception(
            "Phase 1 feature engineering failed"
        )

        raise FraudSentinelException(
            "Phase 1 feature engineering failed",
            e
        )