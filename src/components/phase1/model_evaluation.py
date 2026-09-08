import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)

from src.logger.logger import get_logger
from src.common.serialization import save_json


logger = get_logger(__name__)


class Phase1Evaluator:

    def evaluate(
        self,
        y_true,
        probabilities,
        threshold=0.84,
        amounts=None
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = (
            confusion_matrix(
                y_true,
                predictions
            ).ravel()
        )

        roc_auc = roc_auc_score(
            y_true,
            probabilities
        )

        pr_auc = average_precision_score(
            y_true,
            probabilities
        )

        brier = brier_score_loss(
            y_true,
            probabilities
        )

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

        accuracy = accuracy_score(
            y_true,
            predictions
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_true,
                predictions
            )
        )

        mcc = matthews_corrcoef(
            y_true,
            predictions
        )

        fpr = (
            fp / (fp + tn)
            if fp + tn > 0
            else 0
        )

        fnr = (
            fn / (fn + tp)
            if fn + tp > 0
            else 0
        )

        alert_volume = int(
            predictions.sum()
        )

        alert_rate = (
            alert_volume
            /
            len(predictions)
        )

        result = {

            "ROC-AUC":
                roc_auc,

            "PR-AUC":
                pr_auc,

            "Brier Score":
                brier,

            "Accuracy":
                accuracy,

            "Balanced Accuracy":
                balanced_accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1-score":
                f1,

            "MCC":
                mcc,

            "FPR":
                fpr,

            "FNR":
                fnr,

            "Operating Threshold":
                threshold,

            "Alert Volume":
                alert_volume,

            "Alert Rate":
                alert_rate,

            "TN": int(tn),

            "FP": int(fp),

            "FN": int(fn),

            "TP": int(tp),
        }

        # ----------------------------------------------------
        # MONETARY METRICS
        # ----------------------------------------------------

        if amounts is not None:

            amounts = np.asarray(
                amounts
            )

            fraud_mask = (
                y_true == 1
            )

            legit_mask = (
                y_true == 0
            )

            tp_mask = (
                (y_true == 1)
                &
                (predictions == 1)
            )

            fn_mask = (
                (y_true == 1)
                &
                (predictions == 0)
            )

            fp_mask = (
                (y_true == 0)
                &
                (predictions == 1)
            )

            tn_mask = (
                (y_true == 0)
                &
                (predictions == 0)
            )

            total_fraud_amount = (
                amounts[fraud_mask].sum()
            )

            detected_fraud_amount = (
                amounts[tp_mask].sum()
            )

            missed_fraud_amount = (
                amounts[fn_mask].sum()
            )

            fp_amount = (
                amounts[fp_mask].sum()
            )

            tn_amount = (
                amounts[tn_mask].sum()
            )

            fraud_amount_capture_rate = (
                detected_fraud_amount
                /
                total_fraud_amount
                if total_fraud_amount > 0
                else 0
            )

            result.update({

                "Total Actual Fraud Amount":
                    total_fraud_amount,

                "Detected Fraud Amount":
                    detected_fraud_amount,

                "Fraud Amount Capture Rate":
                    fraud_amount_capture_rate,

                "Missed Fraud Amount":
                    missed_fraud_amount,

                "False Positive Amount":
                    fp_amount,

                "Legitimate Amount Correctly Allowed":
                    tn_amount,

                "Average Captured Fraud Value":
                    (
                        detected_fraud_amount / tp
                        if tp > 0
                        else 0
                    ),

                "Average Missed Fraud Value":
                    (
                        missed_fraud_amount / fn
                        if fn > 0
                        else 0
                    ),

                "Fraud Value per Alert":
                    (
                        detected_fraud_amount
                        /
                        alert_volume
                        if alert_volume > 0
                        else 0
                    )
            })

        return result

    def compare(
        self,
        validation_metrics,
        test_metrics
    ):

        rows = []

        all_metrics = list(
            dict.fromkeys(
                list(validation_metrics.keys())
                +
                list(test_metrics.keys())
            )
        )

        for metric in all_metrics:

            rows.append({

                "Metric":
                    metric,

                "Validation":
                    validation_metrics.get(
                        metric,
                        np.nan
                    ),

                "Future Test":
                    test_metrics.get(
                        metric,
                        np.nan
                    )
            })

        return pd.DataFrame(rows)

    def save_metrics(
        self,
        metrics,
        path
    ):

        save_json(
            metrics,
            path
        )

        logger.info(
            "Evaluation metrics saved: %s",
            path
        )