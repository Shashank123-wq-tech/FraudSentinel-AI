"""
FraudSentinel AI
Phase 1 Gate Validation

Purpose
-------
Validate the EXISTING Phase 1 artifacts without changing
the existing training/evaluation/prediction pipeline.

This validator is read-only with respect to existing artifacts.

It checks:

1. Model / feature contract
2. Prediction artifact integrity
3. Performance
4. Calibration proxy (Brier score)
5. Operating threshold
6. Monetary performance
7. Risk-band consistency
8. Explainability availability
9. Validation -> Future-test robustness

IMPORTANT
---------
This file intentionally uses the existing FraudSentinel artifact
paths rather than assuming new artifact filenames.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# EXISTING ARTIFACT PATHS
# ============================================================

PHASE1_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "phase1"

MODEL_DIR = PHASE1_ARTIFACT_ROOT / "models"
PREDICTION_DIR = PHASE1_ARTIFACT_ROOT / "predictions"
EVALUATION_DIR = PHASE1_ARTIFACT_ROOT / "evaluations"


# ------------------------------------------------------------
# CURRENT AUTHORITATIVE MODEL
# ------------------------------------------------------------

MODEL_PATH = (
    MODEL_DIR / "fraudsentinel_phase1_xgboost.json"
)

FEATURE_SCHEMA_PATH = (
    MODEL_DIR / "phase1_features.json"
)


# ------------------------------------------------------------
# EXISTING PREDICTIONS
# ------------------------------------------------------------

VALIDATION_PREDICTIONS_PATH = (
    PREDICTION_DIR / "validation_predictions.csv"
)

FUTURE_TEST_PREDICTIONS_PATH = (
    PREDICTION_DIR / "future_test_predictions.csv"
)

RISK_OUTPUT_PATH = (
    PREDICTION_DIR / "phase1_risk_output_table.csv"
)

FUTURE_RISK_OUTPUT_PATH = (
    PREDICTION_DIR / "phase1_test_risk_output_table.csv"
)

TRANSACTION_RISK_PATH = (
    PREDICTION_DIR / "transaction_risk.csv"
)


# ------------------------------------------------------------
# EXISTING EVALUATION ARTIFACTS
# ------------------------------------------------------------

VALIDATION_METRICS_PATH = (
    EVALUATION_DIR / "validation_metrics.json"
)

FUTURE_TEST_METRICS_PATH = (
    EVALUATION_DIR / "future_test_metrics.json"
)

COMBINED_METRICS_PATH = (
    EVALUATION_DIR / "phase1_validation_test_metrics.csv"
)


# ============================================================
# PHASE 1 POLICY
# ============================================================

OPERATING_THRESHOLD = 0.84

RISK_BANDS = {
    "LOW": (0.00, 0.20),
    "MEDIUM": (0.20, 0.50),
    "HIGH": (0.50, 0.84),
    "CRITICAL": (0.84, 1.01),
}


# ============================================================
# GATE LIMITS
# ============================================================

GATE_LIMITS = {

    # Performance
    "min_roc_auc": 0.99,
    "min_pr_auc": 0.90,
    "max_brier": 0.01,

    # Classification
    "min_precision": 0.90,
    "min_recall": 0.70,
    "min_f1": 0.80,
    "min_mcc": 0.80,

    # Operational
    "max_fpr": 0.001,
    "max_fnr": 0.30,
    "max_alert_rate": 0.02,

    # Monetary
    "min_fraud_amount_capture": 0.80,

    # Robustness
    "max_roc_auc_drop": 0.02,
    "max_pr_auc_drop": 0.05,
}


# ============================================================
# EXPECTED CURRENT FEATURE CONTRACT
# ============================================================

EXPECTED_FEATURES = [
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


# ============================================================
# RESULT OBJECT
# ============================================================

class GateResult:

    def __init__(
        self,
        name: str,
        status: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:

        self.name = name
        self.status = status
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:

        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


# ============================================================
# UTILITY
# ============================================================

def load_json(path: Path) -> dict[str, Any]:

    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_file(path: Path) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact does not exist:\n{path}"
        )


def safe_float(value: Any) -> float:

    try:
        return float(value)
    except Exception:
        return float("nan")


def pass_fail(condition: bool) -> str:

    return "PASS" if condition else "FAIL"


# ============================================================
# 1. MODEL / FEATURE CONTRACT
# ============================================================

def validate_model_contract() -> GateResult:

    try:

        require_file(MODEL_PATH)
        require_file(FEATURE_SCHEMA_PATH)

        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))

        model_features = list(model.get_booster().feature_names)

        feature_json = load_json(FEATURE_SCHEMA_PATH)

        if isinstance(feature_json, dict):

            if "features" in feature_json:
                stored_features = feature_json["features"]

            elif "selected_features" in feature_json:
                stored_features = feature_json["selected_features"]

            else:
                stored_features = list(feature_json.values())

        else:
            stored_features = feature_json

        stored_features = list(stored_features)

        model_ok = model_features == EXPECTED_FEATURES
        schema_ok = stored_features == EXPECTED_FEATURES

        status = "PASS" if model_ok and schema_ok else "FAIL"

        return GateResult(
            name="MODEL_FEATURE_CONTRACT",
            status=status,
            message=(
                "Model and feature schema match the current "
                "21-feature Phase 1 contract."
                if status == "PASS"
                else
                "Model or stored feature schema does not match "
                "the current 21-feature Phase 1 contract."
            ),
            details={
                "model_feature_count": len(model_features),
                "stored_feature_count": len(stored_features),
                "expected_feature_count": len(EXPECTED_FEATURES),
                "model_features": model_features,
                "stored_features": stored_features,
                "expected_features": EXPECTED_FEATURES,
            },
        )

    except Exception as exc:

        return GateResult(
            "MODEL_FEATURE_CONTRACT",
            "FAIL",
            f"Contract validation failed: {exc}",
        )


# ============================================================
# 2. PREDICTION ARTIFACT INTEGRITY
# ============================================================

def validate_prediction_artifacts() -> GateResult:

    paths = [
        VALIDATION_PREDICTIONS_PATH,
        FUTURE_TEST_PREDICTIONS_PATH,
        RISK_OUTPUT_PATH,
        FUTURE_RISK_OUTPUT_PATH,
        TRANSACTION_RISK_PATH,
    ]

    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    if missing:

        return GateResult(
            "PREDICTION_ARTIFACTS",
            "FAIL",
            "One or more existing prediction artifacts are missing.",
            {"missing": missing},
        )

    try:

        validation = pd.read_csv(
            VALIDATION_PREDICTIONS_PATH
        )

        future = pd.read_csv(
            FUTURE_TEST_PREDICTIONS_PATH
        )

        required_probability_candidates = [
            "fraud_probability",
            "probability",
            "predicted_probability",
            "y_probability",
            "score",
        ]

        def find_probability_column(df):

            for col in required_probability_candidates:
                if col in df.columns:
                    return col

            return None

        val_prob_col = find_probability_column(validation)
        future_prob_col = find_probability_column(future)

        if val_prob_col is None or future_prob_col is None:

            return GateResult(
                "PREDICTION_ARTIFACTS",
                "FAIL",
                "Probability column could not be identified.",
                {
                    "validation_columns": list(validation.columns),
                    "future_columns": list(future.columns),
                },
            )

        val_probs = pd.to_numeric(
            validation[val_prob_col],
            errors="coerce",
        )

        future_probs = pd.to_numeric(
            future[future_prob_col],
            errors="coerce",
        )

        finite_ok = (
            np.isfinite(val_probs).all()
            and np.isfinite(future_probs).all()
        )

        range_ok = (
            ((val_probs >= 0) & (val_probs <= 1)).all()
            and
            ((future_probs >= 0) & (future_probs <= 1)).all()
        )

        status = "PASS" if finite_ok and range_ok else "FAIL"

        return GateResult(
            "PREDICTION_ARTIFACTS",
            status,
            (
                "Prediction artifacts contain valid probability values."
                if status == "PASS"
                else
                "Prediction artifacts contain invalid probabilities."
            ),
            {
                "validation_rows": len(validation),
                "future_test_rows": len(future),
                "validation_probability_column": val_prob_col,
                "future_probability_column": future_prob_col,
                "finite": bool(finite_ok),
                "range_0_1": bool(range_ok),
            },
        )

    except Exception as exc:

        return GateResult(
            "PREDICTION_ARTIFACTS",
            "FAIL",
            f"Prediction validation failed: {exc}",
        )


# ============================================================
# 3. PERFORMANCE
# ============================================================

def validate_performance() -> GateResult:

    try:

        metrics = load_json(FUTURE_TEST_METRICS_PATH)

        roc_auc = safe_float(
            metrics.get("roc_auc")
            or metrics.get("ROC-AUC")
            or metrics.get("roc_auc_score")
        )

        pr_auc = safe_float(
            metrics.get("pr_auc")
            or metrics.get("PR-AUC")
            or metrics.get("average_precision")
        )

        precision = safe_float(
            metrics.get("precision")
        )

        recall = safe_float(
            metrics.get("recall")
        )

        f1 = safe_float(
            metrics.get("f1")
            or metrics.get("f1_score")
            or metrics.get("F1-score")
        )

        mcc = safe_float(
            metrics.get("mcc")
            or metrics.get("MCC")
        )

        fpr = safe_float(
            metrics.get("fpr")
            or metrics.get("FPR")
        )

        fnr = safe_float(
            metrics.get("fnr")
            or metrics.get("FNR")
        )

        checks = {

            "roc_auc": (
                roc_auc >= GATE_LIMITS["min_roc_auc"]
            ),

            "pr_auc": (
                pr_auc >= GATE_LIMITS["min_pr_auc"]
            ),

            "precision": (
                precision >= GATE_LIMITS["min_precision"]
            ),

            "recall": (
                recall >= GATE_LIMITS["min_recall"]
            ),

            "f1": (
                f1 >= GATE_LIMITS["min_f1"]
            ),

            "mcc": (
                mcc >= GATE_LIMITS["min_mcc"]
            ),

            "fpr": (
                fpr <= GATE_LIMITS["max_fpr"]
            ),

            "fnr": (
                fnr <= GATE_LIMITS["max_fnr"]
            ),
        }

        status = (
            "PASS"
            if all(checks.values())
            else "FAIL"
        )

        return GateResult(
            "PERFORMANCE",
            status,
            (
                "Future-test performance satisfies Phase 1 policy."
                if status == "PASS"
                else
                "One or more future-test performance limits failed."
            ),
            {
                "metrics": {
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mcc": mcc,
                    "fpr": fpr,
                    "fnr": fnr,
                },
                "checks": checks,
            },
        )

    except Exception as exc:

        return GateResult(
            "PERFORMANCE",
            "FAIL",
            f"Performance validation failed: {exc}",
        )


# ============================================================
# 4. CALIBRATION
# ============================================================

def validate_calibration() -> GateResult:

    try:

        metrics = load_json(FUTURE_TEST_METRICS_PATH)

        brier = safe_float(
            metrics.get("brier_score")
            or metrics.get("brier")
            or metrics.get("Brier Score")
        )

        if math.isnan(brier):

            return GateResult(
                "CALIBRATION",
                "FAIL",
                "Brier score was not found in future-test metrics.",
            )

        passed = (
            brier <= GATE_LIMITS["max_brier"]
        )

        return GateResult(
            "CALIBRATION",
            pass_fail(passed),
            (
                "Brier score passes the calibration proxy threshold."
                if passed
                else
                "Brier score exceeds the allowed calibration threshold."
            ),
            {
                "brier_score": brier,
                "max_allowed": GATE_LIMITS["max_brier"],
                "note": (
                    "Brier score is a calibration/performance proxy. "
                    "A full reliability-curve audit is recommended."
                ),
            },
        )

    except Exception as exc:

        return GateResult(
            "CALIBRATION",
            "FAIL",
            f"Calibration validation failed: {exc}",
        )


# ============================================================
# 5. THRESHOLD
# ============================================================

def validate_threshold() -> GateResult:

    try:

        metrics = load_json(FUTURE_TEST_METRICS_PATH)

        threshold = safe_float(
            metrics.get("operating_threshold")
            or metrics.get("threshold")
            or metrics.get("Operating Threshold")
        )

        alert_rate = safe_float(
            metrics.get("alert_rate")
            or metrics.get("Alert Rate")
        )

        fpr = safe_float(
            metrics.get("fpr")
            or metrics.get("FPR")
        )

        threshold_ok = (
            not math.isnan(threshold)
            and abs(threshold - OPERATING_THRESHOLD) < 1e-9
        )

        alert_ok = (
            not math.isnan(alert_rate)
            and alert_rate <= GATE_LIMITS["max_alert_rate"]
        )

        fpr_ok = (
            not math.isnan(fpr)
            and fpr <= GATE_LIMITS["max_fpr"]
        )

        passed = (
            threshold_ok
            and alert_ok
            and fpr_ok
        )

        return GateResult(
            "THRESHOLD",
            pass_fail(passed),
            (
                "Operating threshold and alert constraints pass."
                if passed
                else
                "Threshold policy validation failed."
            ),
            {
                "configured_threshold": OPERATING_THRESHOLD,
                "evaluated_threshold": threshold,
                "alert_rate": alert_rate,
                "fpr": fpr,
                "threshold_ok": threshold_ok,
                "alert_rate_ok": alert_ok,
                "fpr_ok": fpr_ok,
            },
        )

    except Exception as exc:

        return GateResult(
            "THRESHOLD",
            "FAIL",
            f"Threshold validation failed: {exc}",
        )


# ============================================================
# 6. MONETARY
# ============================================================

def validate_monetary() -> GateResult:

    try:

        metrics = load_json(FUTURE_TEST_METRICS_PATH)

        capture = safe_float(
            metrics.get("fraud_amount_capture_rate")
            or metrics.get("Fraud Amount Capture Rate")
            or metrics.get("fraud_capture_rate")
        )

        actual_fraud = safe_float(
            metrics.get("total_actual_fraud_amount")
            or metrics.get("Total Actual Fraud Amount")
        )

        detected_fraud = safe_float(
            metrics.get("detected_fraud_amount")
            or metrics.get("Detected Fraud Amount")
        )

        missed_fraud = safe_float(
            metrics.get("missed_fraud_amount")
            or metrics.get("Missed Fraud Amount")
        )

        fp_amount = safe_float(
            metrics.get("false_positive_amount")
            or metrics.get("False Positive Amount")
        )

        passed = (
            not math.isnan(capture)
            and capture >= GATE_LIMITS["min_fraud_amount_capture"]
        )

        return GateResult(
            "MONETARY",
            pass_fail(passed),
            (
                "Fraud amount capture satisfies the monetary gate."
                if passed
                else
                "Fraud amount capture is below the monetary gate."
            ),
            {
                "fraud_amount_capture_rate": capture,
                "minimum_required": GATE_LIMITS[
                    "min_fraud_amount_capture"
                ],
                "total_actual_fraud_amount": actual_fraud,
                "detected_fraud_amount": detected_fraud,
                "missed_fraud_amount": missed_fraud,
                "false_positive_amount": fp_amount,
            },
        )

    except Exception as exc:

        return GateResult(
            "MONETARY",
            "FAIL",
            f"Monetary validation failed: {exc}",
        )


# ============================================================
# 7. RISK BANDS
# ============================================================

def expected_risk_band(probability: float) -> str:

    if probability < 0.20:
        return "LOW"

    if probability < 0.50:
        return "MEDIUM"

    if probability < 0.84:
        return "HIGH"

    return "CRITICAL"


def validate_risk_bands() -> GateResult:

    try:

        if not RISK_OUTPUT_PATH.exists():

            return GateResult(
                "RISK_BANDS",
                "FAIL",
                f"Risk output table missing: {RISK_OUTPUT_PATH}",
            )

        df = pd.read_csv(RISK_OUTPUT_PATH)

        probability_candidates = [
            "fraud_probability",
            "probability",
            "predicted_probability",
        ]

        probability_col = next(
            (
                c
                for c in probability_candidates
                if c in df.columns
            ),
            None,
        )

        if probability_col is None:

            return GateResult(
                "RISK_BANDS",
                "FAIL",
                "No fraud probability column found.",
                {"columns": list(df.columns)},
            )

        if "risk_band" not in df.columns:

            return GateResult(
                "RISK_BANDS",
                "FAIL",
                "risk_band column is missing.",
            )

        probabilities = pd.to_numeric(
            df[probability_col],
            errors="coerce",
        )

        expected = probabilities.apply(
            expected_risk_band
        )

        actual = (
            df["risk_band"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        matches = expected == actual

        accuracy = float(matches.mean())

        passed = (
            accuracy == 1.0
            and probabilities.notna().all()
        )

        return GateResult(
            "RISK_BANDS",
            pass_fail(passed),
            (
                "All risk bands match the canonical probability policy."
                if passed
                else
                "Risk-band assignments do not match the policy."
            ),
            {
                "rows": len(df),
                "probability_column": probability_col,
                "band_consistency": accuracy,
                "distribution": (
                    actual.value_counts()
                    .to_dict()
                ),
                "policy": {
                    "LOW": "< 0.20",
                    "MEDIUM": "0.20 <= p < 0.50",
                    "HIGH": "0.50 <= p < 0.84",
                    "CRITICAL": "p >= 0.84",
                },
            },
        )

    except Exception as exc:

        return GateResult(
            "RISK_BANDS",
            "FAIL",
            f"Risk-band validation failed: {exc}",
        )


# ============================================================
# 8. EXPLAINABILITY
# ============================================================

def validate_explainability() -> GateResult:

    try:

        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))

        booster = model.get_booster()

        gain = booster.get_score(
            importance_type="gain"
        )

        gain = {
            str(k): float(v)
            for k, v in gain.items()
        }

        if not gain:

            return GateResult(
                "EXPLAINABILITY",
                "FAIL",
                "XGBoost did not expose feature importance.",
            )

        top_features = sorted(
            gain.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return GateResult(
            "EXPLAINABILITY",
            "PASS",
            "Global XGBoost feature importance is available.",
            {
                "top_features_by_gain": top_features,
                "feature_count_with_importance": len(gain),
                "note": (
                    "This validates global model explainability. "
                    "Transaction-level SHAP explanations can be "
                    "added later."
                ),
            },
        )

    except Exception as exc:

        return GateResult(
            "EXPLAINABILITY",
            "FAIL",
            f"Explainability validation failed: {exc}",
        )


# ============================================================
# 9. ROBUSTNESS
# ============================================================

def validate_robustness() -> GateResult:

    try:

        validation = load_json(
            VALIDATION_METRICS_PATH
        )

        future = load_json(
            FUTURE_TEST_METRICS_PATH
        )

        def metric(d, *keys):

            for key in keys:
                if key in d:
                    return safe_float(d[key])

            return float("nan")

        val_roc = metric(
            validation,
            "roc_auc",
            "ROC-AUC",
        )

        future_roc = metric(
            future,
            "roc_auc",
            "ROC-AUC",
        )

        val_pr = metric(
            validation,
            "pr_auc",
            "PR-AUC",
            "average_precision",
        )

        future_pr = metric(
            future,
            "pr_auc",
            "PR-AUC",
            "average_precision",
        )

        roc_drop = val_roc - future_roc
        pr_drop = val_pr - future_pr

        roc_ok = (
            roc_drop <= GATE_LIMITS["max_roc_auc_drop"]
        )

        pr_ok = (
            pr_drop <= GATE_LIMITS["max_pr_auc_drop"]
        )

        passed = roc_ok and pr_ok

        return GateResult(
            "ROBUSTNESS",
            pass_fail(passed),
            (
                "Future-test degradation is within policy."
                if passed
                else
                "Future-test degradation exceeds policy."
            ),
            {
                "validation_roc_auc": val_roc,
                "future_test_roc_auc": future_roc,
                "roc_auc_drop": roc_drop,
                "max_roc_auc_drop": GATE_LIMITS[
                    "max_roc_auc_drop"
                ],
                "validation_pr_auc": val_pr,
                "future_test_pr_auc": future_pr,
                "pr_auc_drop": pr_drop,
                "max_pr_auc_drop": GATE_LIMITS[
                    "max_pr_auc_drop"
                ],
                "roc_ok": roc_ok,
                "pr_ok": pr_ok,
            },
        )

    except Exception as exc:

        return GateResult(
            "ROBUSTNESS",
            "FAIL",
            f"Robustness validation failed: {exc}",
        )


# ============================================================
# RUN ALL GATES
# ============================================================

def run_phase1_gate() -> dict[str, Any]:

    results = [

        validate_model_contract(),

        validate_prediction_artifacts(),

        validate_performance(),

        validate_calibration(),

        validate_threshold(),

        validate_monetary(),

        validate_risk_bands(),

        validate_explainability(),

        validate_robustness(),
    ]

    failed = [
        result
        for result in results
        if result.status == "FAIL"
    ]

    overall_status = (
        "FAIL"
        if failed
        else
        "PASS"
    )

    report = {

        "phase": "phase1",

        "gate": "FraudSentinel Phase 1 Gate",

        "overall_status": overall_status,

        "model": str(
            MODEL_PATH.relative_to(PROJECT_ROOT)
        ),

        "threshold": OPERATING_THRESHOLD,

        "feature_count": len(EXPECTED_FEATURES),

        "results": [
            result.to_dict()
            for result in results
        ],
    }

    output_path = (
        EVALUATION_DIR
        / "phase1_gate_report.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            default=str,
        )

    return report


# ============================================================
# CLI
# ============================================================

def main() -> None:

    print()
    print("=" * 72)
    print("FraudSentinel AI — PHASE 1 GATE")
    print("=" * 72)
    print()

    print(
        f"Model      : {MODEL_PATH}"
    )

    print(
        f"Predictions: {PREDICTION_DIR}"
    )

    print(
        f"Evaluations: {EVALUATION_DIR}"
    )

    print(
        f"Threshold  : {OPERATING_THRESHOLD}"
    )

    print()

    report = run_phase1_gate()

    print("-" * 72)

    for result in report["results"]:

        print(
            f"[{result['status']:4}] "
            f"{result['name']}"
        )

        print(
            f"       {result['message']}"
        )

    print("-" * 72)

    print()

    print(
        f"PHASE 1 GATE STATUS: "
        f"{report['overall_status']}"
    )

    print()

    print(
        "Report:"
    )

    print(
        EVALUATION_DIR
        / "phase1_gate_report.json"
    )

    print()


if __name__ == "__main__":
    main()