import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.logger.logger import get_logger


logger = get_logger(__name__)


class Phase1ThresholdOptimizer:

    def evaluate_threshold(
        self,
        y_true,
        probabilities,
        threshold=0.84
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions
        ).ravel()

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        )

        fnr = (
            fn / (fn + tp)
            if (fn + tp) > 0
            else 0
        )

        alert_rate = (
            (predictions == 1).mean()
        )

        return {

            "threshold": threshold,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "fpr": fpr,

            "fnr": fnr,

            "tn": tn,

            "fp": fp,

            "fn": fn,

            "tp": tp,

            "alert_volume":
                int(predictions.sum()),

            "alert_rate":
                alert_rate
        }