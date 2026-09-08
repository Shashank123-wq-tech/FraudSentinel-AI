import pandas as pd
import numpy as np

from src.logger.logger import get_logger
from src.exception.exception import FraudSentinelException


logger = get_logger(__name__)


class Phase1Predictor:

    def __init__(
        self,
        model,
        threshold=0.84
    ):

        self.model = model
        self.threshold = threshold

    def predict(
        self,
        X
    ):

        try:

            probabilities = (
                self.model
                .predict_proba(X)[:, 1]
            )

            predictions = (
                probabilities
                >= self.threshold
            ).astype("int8")

            logger.info(
                "Generated %d predictions",
                len(predictions)
            )

            logger.info(
                "Alerts=%d | Alert rate=%.4f%%",
                predictions.sum(),
                predictions.mean() * 100
            )

            return probabilities, predictions

        except Exception as e:

            logger.exception(
                "Prediction failed"
            )

            raise FraudSentinelException(
                "Phase 1 prediction failed",
                e
            )

    def build_output(
        self,
        df,
        probabilities,
        predictions
    ):

        output = pd.DataFrame({

            "trans_date_trans_time":
                df[
                    "trans_date_trans_time"
                ].values,

            "cc_num":
                df["cc_num"].values,

            "merchant":
                df["merchant"].values,

            "amt":
                df["amt"].values,

            "fraud_probability":
                probabilities,

            "fraud_prediction":
                predictions
        })

        output["risk_level"] = np.select(

            [
                output[
                    "fraud_probability"
                ] >= 0.84,

                output[
                    "fraud_probability"
                ] >= 0.50
            ],

            [
                "HIGH",
                "MEDIUM"
            ],

            default="LOW"
        )

        return output