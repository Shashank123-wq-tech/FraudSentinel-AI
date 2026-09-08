import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split

from src.logger.logger import get_logger
from src.exception.exception import FraudSentinelException


logger = get_logger(__name__)


BEST_PARAMS = {

    "max_depth": 6,

    "min_child_weight": 1,

    "learning_rate": 0.07091340754075995,

    "subsample": 0.8881945843174721,

    "colsample_bytree": 0.8445573399695375,

    "gamma": 0.25073469812212335,

    "reg_alpha": 1.0560705268797807e-06,

    "reg_lambda": 0.19158576751393638,
}


class Phase1ModelTrainer:

    def __init__(
        self,
        random_state=42,
        n_estimators=2000,
        early_stopping_rounds=30
    ):

        self.random_state = random_state

        self.n_estimators = n_estimators

        self.early_stopping_rounds = (
            early_stopping_rounds
        )

    def create_fit_validation_split(
        self,
        X_train,
        y_train
    ):

        logger.info(
            "Creating model fit/validation split"
        )

        X_fit, X_val, y_fit, y_val = (
            train_test_split(
                X_train,
                y_train,
                test_size=0.20,
                stratify=y_train,
                random_state=self.random_state
            )
        )

        logger.info(
            "X_fit=%s | X_val=%s",
            X_fit.shape,
            X_val.shape
        )

        logger.info(
            "Fit fraud rate=%.6f | "
            "Validation fraud rate=%.6f",
            y_fit.mean(),
            y_val.mean()
        )

        return (
            X_fit,
            X_val,
            y_fit,
            y_val
        )

    def train(
        self,
        X_fit,
        y_fit,
        X_val,
        y_val
    ):

        try:

            scale_pos_weight = (
                (y_fit == 0).sum()
                /
                (y_fit == 1).sum()
            )

            logger.info(
                "scale_pos_weight=%.4f",
                scale_pos_weight
            )

            params = BEST_PARAMS.copy()

            params.update({

                "objective":
                    "binary:logistic",

                "eval_metric":
                    "aucpr",

                "n_estimators":
                    self.n_estimators,

                "scale_pos_weight":
                    scale_pos_weight,

                "tree_method":
                    "hist",

                "device":
                    "cpu",

                "random_state":
                    self.random_state,

                "n_jobs":
                    -1
            })

            logger.info(
                "Training optimized XGBoost model"
            )

            logger.info(
                "XGBoost parameters: %s",
                params
            )

            model = xgb.XGBClassifier(
                **params,
                early_stopping_rounds=(
                    self.early_stopping_rounds
                )
            )

            model.fit(
                X_fit,
                y_fit,
                eval_set=[
                    (X_val, y_val)
                ],
                verbose=50
            )

            logger.info(
                "Training completed"
            )

            logger.info(
                "Best iteration=%s",
                model.best_iteration
            )

            logger.info(
                "Best validation score=%s",
                model.best_score
            )

            return model, scale_pos_weight

        except Exception as e:

            logger.exception(
                "XGBoost training failed"
            )

            raise FraudSentinelException(
                "Phase 1 model training failed",
                e
            )