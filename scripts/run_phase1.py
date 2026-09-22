from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from src.logger.logger import get_logger
from src.exception.exception import FraudSentinelException

from src.components.phase1.data_ingestion import (
    Phase1DataIngestion,
)

from src.components.phase1.data_validation import (
    Phase1DataValidator,
)

from src.components.phase1.feature_engineering import (
    engineer_phase1_features,
)

from src.components.phase1.feature_selection import (
    select_features,
)

from src.components.phase1.model_trainer import (
    Phase1ModelTrainer,
)

from src.components.phase1.model_registry import (
    Phase1ModelRegistry,
)

from src.components.phase1.prediction import (
    Phase1Predictor,
)

from src.components.phase1.threshold_optimizer import (
    Phase1ThresholdOptimizer,
)

from src.components.phase1.model_evaluation import (
    Phase1Evaluator,
)


logger = get_logger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fraud_transactions.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "phase1"
    / "phase1_xgboost.json"
)

RISK_OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase1"
    / "risk_output"
    / "transaction_risk.csv"
)

METRICS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase1"
    / "metrics"
    / "phase1_metrics.json"
)

FEATURE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase1"
    / "metrics"
    / "selected_features.json"
)


# ============================================================
# PROJECT CONSTANTS
# ============================================================

RANDOM_STATE = 42

TRAIN_RATIO = 0.80

OPERATING_THRESHOLD = 0.84

MODEL_VERSION = "phase1_xgboost_v1"


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RISK_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# TIME-ORDERED SPLIT
# ============================================================

def time_ordered_split(
    df,
    X,
    y,
    train_ratio=TRAIN_RATIO,
):
    """
    Creates a chronological train/test split.

    The feature engineering step already restores
    chronological order.
    """

    if len(df) != len(X):
        raise ValueError(
            "Dataframe and feature matrix row counts do not match."
        )

    split_index = int(
        len(df) * train_ratio
    )

    if split_index <= 0 or split_index >= len(df):
        raise ValueError(
            "Invalid train/test split index."
        )

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    df_train = df.iloc[:split_index].copy()
    df_test = df.iloc[split_index:].copy()

    logger.info(
        "Time ordered split completed | "
        "train=%s | test=%s",
        len(X_train),
        len(X_test),
    )

    logger.info(
        "Train fraud rate=%.6f%% | "
        "Test fraud rate=%.6f%%",
        y_train.mean() * 100,
        y_test.mean() * 100,
    )

    return (
        df_train,
        df_test,
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# STANDARDIZE PHASE 1 OUTPUT
# ============================================================

def build_standardized_risk_output(
    df,
    probabilities,
    predictions,
):
    """
    Converts the native Phase-1 prediction output
    into the schema required by downstream Phase 2.
    """

    if len(df) != len(probabilities):
        raise ValueError(
            "Prediction length does not match dataframe."
        )

    output = pd.DataFrame(
        {
            "transaction_id": np.arange(
                len(df),
                dtype=np.int64,
            ),

            "timestamp": pd.to_datetime(
                df["trans_date_trans_time"]
            ).values,

            "merchant_id": (
                df["merchant"]
                .astype(str)
                .values
            ),

            "card_id": (
                df["cc_num"]
                .astype(str)
                .values
            ),

            "amount": pd.to_numeric(
                df["amt"],
                errors="coerce",
            ).fillna(0).values,

            "fraud_probability": (
                probabilities
            ),

            "predicted_fraud": (
                predictions.astype("int8")
            ),
        }
    )

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    output["risk_score"] = (
        output["fraud_probability"] * 100.0
    )

    # --------------------------------------------------------
    # Risk band
    # --------------------------------------------------------

    output["risk_band"] = np.select(
        [
            output["fraud_probability"] >= 0.84,
            output["fraud_probability"] >= 0.50,
        ],
        [
            "HIGH",
            "MEDIUM",
        ],
        default="LOW",
    )

    # --------------------------------------------------------
    # Governance
    # --------------------------------------------------------

    output["model_version"] = MODEL_VERSION

    output["threshold"] = OPERATING_THRESHOLD

    # --------------------------------------------------------
    # Column ordering
    # --------------------------------------------------------

    output = output[
        [
            "transaction_id",
            "timestamp",
            "merchant_id",
            "card_id",
            "amount",
            "fraud_probability",
            "risk_score",
            "risk_band",
            "predicted_fraud",
            "model_version",
            "threshold",
        ]
    ]

    return output


# ============================================================
# SAVE JSON
# ============================================================

def save_json(data, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=4,
            default=str,
        )


# ============================================================
# MAIN PHASE 1 PIPELINE
# ============================================================

def run_phase1():

    start_time = time.time()

    logger.info("=" * 70)
    logger.info("FraudSentinel AI - Phase 1 Pipeline")
    logger.info("=" * 70)

    try:

        create_directories()

        # ====================================================
        # 1. DATA INGESTION
        # ====================================================

        logger.info("[1/9] Data ingestion")

        ingestion = Phase1DataIngestion(
            data_path=str(DATA_PATH)
        )

        df = ingestion.load_data()

        logger.info(
            "Input dataset shape=%s",
            df.shape,
        )

        # ====================================================
        # 2. DATA VALIDATION
        # ====================================================

        logger.info("[2/9] Data validation")

        validator = Phase1DataValidator()

        validator.validate(df)

        # ====================================================
        # 3. FEATURE ENGINEERING
        # ====================================================

        logger.info("[3/9] Feature engineering")

        df, X, y = engineer_phase1_features(
            df,
            restore_time_order=True,
        )

        logger.info(
            "Feature matrix shape=%s",
            X.shape,
        )

        # ====================================================
        # 4. TIME ORDERED SPLIT
        # ====================================================

        logger.info("[4/9] Time ordered train/test split")

        (
            df_train,
            df_test,
            X_train,
            X_test,
            y_train,
            y_test,
        ) = time_ordered_split(
            df,
            X,
            y,
        )

        # ====================================================
        # 5. FEATURE SELECTION
        # ====================================================

        logger.info("[5/9] Feature selection")

        selected_features, removed_features = (
            select_features(
                X_train,
                y_train,
                correlation_threshold=0.85,
            )
        )

        

        logger.info(
            "Selected features=%d",
            len(selected_features),
        )

        logger.info(
            "Removed features=%s",
            removed_features,
        )

        # IMPORTANT:
        # Apply exactly the same selected feature
        # list to test and full dataset.
        X_train_selected = X_train[
            selected_features
        ].copy()
        
        X_test_selected = X_test[
            selected_features
        ].copy()

        X_full_selected = X[
            selected_features
        ].copy()

        save_json(
            {
                "model_version": MODEL_VERSION,
                "selected_features": selected_features,
                "removed_features": removed_features,
                "correlation_threshold": 0.85,
            },
            FEATURE_OUTPUT_PATH,
        )

        # ====================================================
        # 6. MODEL TRAINING
        # ====================================================

        logger.info("[6/9] XGBoost model training")

        trainer = Phase1ModelTrainer(
            random_state=RANDOM_STATE,
            n_estimators=2000,
            early_stopping_rounds=30,
        )

        (
            X_fit,
            X_val,
            y_fit,
            y_val,
        ) = trainer.create_fit_validation_split(
            X_train_selected,
            y_train,
        )

        model, scale_pos_weight = trainer.train(
            X_fit,
            y_fit,
            X_val,
            y_val,
        )

        logger.info(
            "Model trained successfully."
        )

        logger.info(
            "scale_pos_weight=%.6f",
            scale_pos_weight,
        )

        # ====================================================
        # 7. MODEL REGISTRY
        # ====================================================

        logger.info("[7/9] Model registry")

        registry = Phase1ModelRegistry(
            model_path=MODEL_PATH
        )

        registry.save(model)

        # ====================================================
        # 8. EVALUATION + THRESHOLD
        # ====================================================

        logger.info(
            "[8/9] Model evaluation and threshold optimization"
        )

        predictor = Phase1Predictor(
            model=model,
            threshold=OPERATING_THRESHOLD,
        )

        test_probabilities = (
            model.predict_proba(
                X_test_selected
            )[:, 1]
        )

        threshold_optimizer = (
            Phase1ThresholdOptimizer()
        )

        threshold_metrics = (
            threshold_optimizer.evaluate_threshold(
                y_true=y_test.values,
                probabilities=test_probabilities,
                threshold=OPERATING_THRESHOLD,
            )
        )

        evaluator = Phase1Evaluator()

        evaluation_metrics = evaluator.evaluate(
            y_true=y_test.values,
            probabilities=test_probabilities,
            threshold=OPERATING_THRESHOLD,
            amounts=df_test["amt"].values,
        )

        # ----------------------------------------------------
        # Combine evaluation information
        # ----------------------------------------------------

        metrics = {
            "project": "FraudSentinel AI",
            "component": "Phase 1 - Transaction Intelligence",
            "model_version": MODEL_VERSION,
            "threshold": OPERATING_THRESHOLD,

            "dataset": {
                "total_rows": int(len(df)),
                "train_rows": int(len(df_train)),
                "test_rows": int(len(df_test)),
                "n_features_before_selection": int(
                    X.shape[1]
                ),
                "n_features_after_selection": int(
                    len(selected_features)
                ),
            },

            "class_balance": {
                "train_fraud_rate": float(
                    y_train.mean()
                ),
                "test_fraud_rate": float(
                    y_test.mean()
                ),
            },

            "threshold_metrics": {
                key: (
                    float(value)
                    if isinstance(
                        value,
                        (np.floating, float)
                    )
                    else int(value)
                    if isinstance(
                        value,
                        (np.integer, int)
                    )
                    else value
                )
                for key, value
                in threshold_metrics.items()
            },

            "evaluation_metrics": {
                key: (
                    float(value)
                    if isinstance(
                        value,
                        (np.floating, float)
                    )
                    else int(value)
                    if isinstance(
                        value,
                        (np.integer, int)
                    )
                    else value
                )
                for key, value
                in evaluation_metrics.items()
            },

            "selected_features": selected_features,

            "removed_features": removed_features,

            "training": {
                "scale_pos_weight": float(
                    scale_pos_weight
                ),
                "random_state": RANDOM_STATE,
                "n_estimators": 2000,
                "early_stopping_rounds": 30,
            },
        }

        save_json(
            metrics,
            METRICS_OUTPUT_PATH,
        )

        # ====================================================
        # 9. FULL DATASET PREDICTION
        # ====================================================

        logger.info(
            "[9/9] Full dataset risk prediction"
        )

        full_probabilities, full_predictions = (
            predictor.predict(
                X_full_selected
            )
        )

        risk_output = (
            build_standardized_risk_output(
                df=df,
                probabilities=full_probabilities,
                predictions=full_predictions,
            )
        )

        risk_output.to_csv(
            RISK_OUTPUT_PATH,
            index=False,
        )

        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        elapsed = time.time() - start_time

        logger.info("=" * 70)
        logger.info(
            "PHASE 1 PIPELINE COMPLETED SUCCESSFULLY"
        )
        logger.info("=" * 70)

        logger.info(
            "Rows processed      : %d",
            len(df),
        )

        logger.info(
            "Features selected   : %d",
            len(selected_features),
        )
        
        logger.info(
           "Test ROC-AUC        : %s",
            evaluation_metrics.get(
             "roc_auc",
            evaluation_metrics.get("ROC-AUC", "N/A"),
            ),
        )

        logger.info(
        "Test PR-AUC         : %s",
        evaluation_metrics.get(
        "pr_auc",
        evaluation_metrics.get("PR-AUC", "N/A"),
    ),
)

        logger.info(
    "Test precision      : %s",
    evaluation_metrics.get(
        "precision",
        "N/A",
    ),
)

        logger.info(
          "Test recall         : %s",
          evaluation_metrics.get(
              "recall",
              "N/A",
          ),
        )

        logger.info(
         "Fraud amount capture: %s",
         evaluation_metrics.get(
        "fraud_amount_capture_rate",
        evaluation_metrics.get(
            "fraud_amount_capture",
            "N/A",
           ),
           ),
        )
        

        logger.info(
            "Risk output         : %s",
            RISK_OUTPUT_PATH,
        )

        logger.info(
            "Model               : %s",
            MODEL_PATH,
        )

        logger.info(
            "Metrics             : %s",
            METRICS_OUTPUT_PATH,
        )

        logger.info(
            "Elapsed time        : %.2f seconds",
            elapsed,
        )

        return {
            "status": "PASS",
            "model_version": MODEL_VERSION,
            "rows_processed": int(len(df)),
            "train_rows": int(len(df_train)),
            "test_rows": int(len(df_test)),
            "selected_features": selected_features,
            "risk_output": str(RISK_OUTPUT_PATH),
            "model_path": str(MODEL_PATH),
            "metrics_path": str(METRICS_OUTPUT_PATH),
            "elapsed_seconds": elapsed,
        }

    except Exception as e:

        logger.exception(
            "Phase 1 pipeline failed"
        )

        raise FraudSentinelException(
            "Phase 1 pipeline failed",
            {e},
        )


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    result = run_phase1()

    print("\n" + "=" * 70)
    print("FraudSentinel AI - Phase 1")
    print("=" * 70)
    print(
        f"Status              : {result['status']}"
    )
    print(
        f"Rows processed      : {result['rows_processed']}"
    )
    print(
        f"Train rows          : {result['train_rows']}"
    )
    print(
        f"Test rows           : {result['test_rows']}"
    )
    print(
        f"Selected features   : "
        f"{len(result['selected_features'])}"
    )
    print(
        f"Risk output         : "
        f"{result['risk_output']}"
    )
    print(
        f"Model               : "
        f"{result['model_path']}"
    )
    print(
        f"Metrics             : "
        f"{result['metrics_path']}"
    )
    print(
        f"Runtime             : "
        f"{result['elapsed_seconds']:.2f} sec"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()