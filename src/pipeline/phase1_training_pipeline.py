from pathlib import Path

import pandas as pd

from src.logger.logger import get_logger
from src.exception.exception import FraudSentinelException

from src.components.phase1.data_ingestion import (
    Phase1DataIngestion
)

from src.components.phase1.data_validation import (
    Phase1DataValidator
)

from src.components.phase1.feature_engineering import (
    engineer_phase1_features,
    FEATURE_COLS
)

from src.components.phase1.model_trainer import (
    Phase1ModelTrainer
)

from src.components.phase1.model_registry import (
    Phase1ModelRegistry
)

from src.components.phase1.model_evaluation import (
    Phase1Evaluator
)

from src.components.phase1.prediction import (
    Phase1Predictor
)


logger = get_logger(__name__)


class Phase1TrainingPipeline:

    def __init__(
        self,
        data_path,
        threshold=0.84
    ):

        self.data_path = data_path

        self.threshold = threshold

    def run(self):

        try:

            logger.info("=" * 70)
            logger.info(
                "FRAUDSENTINEL AI — PHASE 1 TRAINING"
            )
            logger.info("=" * 70)

            # ------------------------------------------------
            # 1. INGESTION
            # ------------------------------------------------

            ingestion = (
                Phase1DataIngestion(
                    self.data_path
                )
            )

            df = ingestion.load_data()

            # ------------------------------------------------
            # 2. VALIDATION
            # ------------------------------------------------

            validator = (
                Phase1DataValidator()
            )

            validator.validate(df)

            # ------------------------------------------------
            # 3. FEATURE ENGINEERING
            # ------------------------------------------------

            df, X, y = (
                engineer_phase1_features(
                    df
                )
            )

            # ------------------------------------------------
            # 4. CHRONOLOGICAL 80/20 SPLIT
            # ------------------------------------------------

            split_date = (
                df[
                    "trans_date_trans_time"
                ]
                .quantile(0.80)
            )

            train_mask = (
                df[
                    "trans_date_trans_time"
                ]
                <
                split_date
            )

            X_train = X.loc[
                train_mask
            ]

            X_test = X.loc[
                ~train_mask
            ]

            y_train = y.loc[
                train_mask
            ]

            y_test = y.loc[
                ~train_mask
            ]

            logger.info(
                "Chronological split=%s",
                split_date
            )

            logger.info(
                "X_train=%s | X_test=%s",
                X_train.shape,
                X_test.shape
            )

            logger.info(
                "Train fraud rate=%.4f%%",
                y_train.mean() * 100
            )

            logger.info(
                "Test fraud rate=%.4f%%",
                y_test.mean() * 100
            )

            # ------------------------------------------------
            # 5. FIT / VALIDATION SPLIT
            # ------------------------------------------------

            trainer = (
                Phase1ModelTrainer()
            )

            (
                X_fit,
                X_val,
                y_fit,
                y_val
            ) = trainer.create_fit_validation_split(
                X_train,
                y_train
            )

            # ------------------------------------------------
            # 6. MODEL TRAINING
            # ------------------------------------------------

            model, scale_pos_weight = (
                trainer.train(
                    X_fit,
                    y_fit,
                    X_val,
                    y_val
                )
            )

            # ------------------------------------------------
            # 7. SAVE MODEL
            # ------------------------------------------------

            registry = (
                Phase1ModelRegistry(
                    "artifacts/phase1/models/"
                    "fraudsentinel_phase1_xgboost.json"
                )
            )

            registry.save(model)

            # ------------------------------------------------
            # 8. SAVE FEATURES
            # ------------------------------------------------

            feature_path = Path(
                "artifacts/phase1/models/"
                "phase1_features.json"
            )

            feature_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            feature_path.write_text(
                pd.Series(
                    FEATURE_COLS
                ).to_json(),
                encoding="utf-8"
            )

            # ------------------------------------------------
            # 9. VALIDATION EVALUATION
            # ------------------------------------------------

            predictor = Phase1Predictor(
                model,
                self.threshold
            )

            val_probability, val_prediction = (
                predictor.predict(X_val)
            )

            evaluator = (
                Phase1Evaluator()
            )

            validation_metrics = (
                evaluator.evaluate(
                    y_val,
                    val_probability,
                    self.threshold,
                    amounts=df.loc[
                        X_val.index,
                        "amt"
                    ].values
                )
            )

            # ------------------------------------------------
            # 10. FUTURE TEST EVALUATION
            # ------------------------------------------------

            test_probability, test_prediction = (
                predictor.predict(X_test)
            )

            test_metrics = (
                evaluator.evaluate(
                    y_test,
                    test_probability,
                    self.threshold,
                    amounts=df.loc[
                        X_test.index,
                        "amt"
                    ].values
                )
            )

            # ------------------------------------------------
            # 11. COMPARISON TABLE
            # ------------------------------------------------

            comparison = (
                evaluator.compare(
                    validation_metrics,
                    test_metrics
                )
            )

            print("\n")
            print("=" * 90)
            print(
                "FRAUDSENTINEL AI — PHASE 1 "
                "VALIDATION / FUTURE TEST"
            )
            print("=" * 90)

            print(
                comparison.to_string(
                    index=False
                )
            )

            # ------------------------------------------------
            # 12. SAVE EVALUATIONS
            # ------------------------------------------------

            evaluation_dir = Path(
                "artifacts/phase1/evaluations"
            )

            evaluation_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            comparison.to_csv(
                evaluation_dir
                /
                "phase1_validation_test_metrics.csv",
                index=False
            )

            evaluator.save_metrics(
                validation_metrics,
                evaluation_dir
                /
                "validation_metrics.json"
            )

            evaluator.save_metrics(
                test_metrics,
                evaluation_dir
                /
                "future_test_metrics.json"
            )

            # ------------------------------------------------
            # 13. SAVE PREDICTIONS
            # ------------------------------------------------

            val_output = predictor.build_output(
                df.loc[
                    X_val.index
                ],
                val_probability,
                val_prediction
            )

            test_output = predictor.build_output(
                df.loc[
                    X_test.index
                ],
                test_probability,
                test_prediction
            )

            prediction_dir = Path(
                "artifacts/phase1/predictions"
            )

            prediction_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            val_output.to_csv(
                prediction_dir
                /
                "validation_predictions.csv",
                index=False
            )

            test_output.to_csv(
                prediction_dir
                /
                "future_test_predictions.csv",
                index=False
            )

            logger.info(
                "Phase 1 training pipeline completed"
            )

            return {

                "model": model,

                "validation_metrics":
                    validation_metrics,

                "test_metrics":
                    test_metrics,

                "comparison":
                    comparison,

                "threshold":
                    self.threshold,

                "best_iteration":
                    model.best_iteration
            }

        except Exception as e:

            logger.exception(
                "Phase 1 training pipeline failed"
            )

            raise FraudSentinelException(
                "Phase 1 training pipeline failed",
                e
            )