import pandas as pd

from src.logger.logger import get_logger
from src.exception.exception import FraudSentinelException


logger = get_logger(__name__)


REQUIRED_COLUMNS = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "city",
    "lat",
    "long",
    "city_pop",
    "dob",
    "merch_lat",
    "merch_long",
    "is_fraud",
]


class Phase1DataValidator:

    def validate(self, df):

        try:

            logger.info("Starting Phase 1 data validation")

            missing_columns = [
                col
                for col in REQUIRED_COLUMNS
                if col not in df.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing required columns: {missing_columns}"
                )

            if df.empty:

                raise ValueError(
                    "Input dataframe is empty"
                )

            if df["is_fraud"].isna().any():

                raise ValueError(
                    "Target column contains NaN values"
                )

            if not set(df["is_fraud"].unique()).issubset({0, 1}):

                raise ValueError(
                    "is_fraud must contain only 0 and 1"
                )

            if (df["amt"] < 0).any():

                raise ValueError(
                    "Negative transaction amounts detected"
                )

            logger.info(
                "Validation successful. Rows=%s, Columns=%s",
                len(df),
                len(df.columns)
            )

            return True

        except Exception as e:

            logger.exception(
                "Phase 1 data validation failed"
            )

            raise FraudSentinelException(
                "Phase 1 data validation failed",
                e
            )