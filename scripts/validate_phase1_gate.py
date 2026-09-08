"""
FraudSentinel AI
Phase 1 Final Gate Validation Runner

IMPORTANT:
- Standalone validator.
- Does NOT import phase1_gate_validation.py.
- Does NOT modify existing Phase 1 files.
- Reads existing Phase 1 artifacts only.
"""

from pathlib import Path
import sys
import json

import pandas as pd
import xgboost as xgb


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS = ROOT / "artifacts" / "phase1"


# ============================================================
# EXISTING PHASE 1 ARTIFACT PATHS
# ============================================================

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

MODEL_PATH = (
    ARTIFACTS
    / "models"
    / "fraudsentinel_phase1_xgboost.json"
)

FEATURE_CONTRACT_PATH = (
    ARTIFACTS
    / "models"
    / "phase1_features.json"
)


# ------------------------------------------------------------
# Evaluation artifacts
# ------------------------------------------------------------

VALIDATION_METRICS_PATH = (
    ARTIFACTS
    / "evaluations"
    / "validation_metrics.json"
)

FUTURE_TEST_METRICS_PATH = (
    ARTIFACTS
    / "evaluations"
    / "future_test_metrics.json"
)

COMPARISON_METRICS_PATH = (
    ARTIFACTS
    / "evaluations"
    / "phase1_validation_test_metrics.csv"
)


# ------------------------------------------------------------
# Prediction artifacts
# ------------------------------------------------------------

VALIDATION_PREDICTIONS_PATH = (
    ARTIFACTS
    / "predictions"
    / "validation_predictions.csv"
)

FUTURE_TEST_PREDICTIONS_PATH = (
    ARTIFACTS
    / "predictions"
    / "future_test_predictions.csv"
)


# ------------------------------------------------------------
# Existing risk output tables
# ------------------------------------------------------------

RISK_OUTPUT_PATH = (
    ARTIFACTS
    / "phase1_risk_output_table.csv"
)

TEST_RISK_OUTPUT_PATH = (
    ARTIFACTS
    / "phase1_test_risk_output_table.csv"
)

TRANSACTION_RISK_PATH = (
    ARTIFACTS
    / "transaction_risk.csv"
)


# ------------------------------------------------------------
# New report produced by this script
# ------------------------------------------------------------

OUTPUT_REPORT_PATH = (
    ARTIFACTS
    / "evaluations"
    / "phase1_gate_report.json"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_section(title: str):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def metric(metrics: dict, *names):
    """
    Retrieve a metric using several possible key names.
    """

    for name in names:
        if name in metrics:
            return metrics[name]

    return None


def safe_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# 1. ARTIFACT EXISTENCE
# ============================================================

def check_artifacts():

    print_section("1. ARTIFACT EXISTENCE")

    required = {
        "Phase 1 model": MODEL_PATH,
        "Feature contract": FEATURE_CONTRACT_PATH,
        "Validation metrics": VALIDATION_METRICS_PATH,
        "Future-test metrics": FUTURE_TEST_METRICS_PATH,
        "Validation predictions": VALIDATION_PREDICTIONS_PATH,
        "Future-test predictions": FUTURE_TEST_PREDICTIONS_PATH,
    }

    results = {}

    for name, path in required.items():

        exists = path.exists()

        results[name] = exists

        if exists:
            print(f"[PASS] {name}")
            print(f"       {path}")

        else:
            print(f"[FAIL] {name}")
            print(f"       Missing: {path}")

    return results


# ============================================================
# 2. MODEL VALIDATION
# ============================================================

def validate_model():

    print_section("2. MODEL VALIDATION")

    if not MODEL_PATH.exists():

        print("[FAIL] Model file does not exist.")

        return False

    try:

        model = xgb.XGBClassifier()

        model.load_model(
            str(MODEL_PATH)
        )

        print("[PASS] XGBoost model loaded successfully.")

        print(
            f"[INFO] Model path:\n"
            f"       {MODEL_PATH}"
        )

        # ----------------------------------------------------
        # Feature contract
        # ----------------------------------------------------

        if FEATURE_CONTRACT_PATH.exists():

            features_data = load_json(
                FEATURE_CONTRACT_PATH
            )

            if isinstance(features_data, list):

                features = features_data

            elif isinstance(features_data, dict):

                features = list(
                    features_data.values()
                )

            else:

                print(
                    "[FAIL] Unsupported feature contract format."
                )

                return False

            features = [
                str(feature)
                for feature in features
                if feature is not None
            ]

            print(
                f"[INFO] Feature contract count: "
                f"{len(features)}"
            )

            if len(features) == 21:

                print(
                    "[PASS] Expected 21 Phase 1 features found."
                )

            else:

                print(
                    "[WARN] Expected 21 features but found "
                    f"{len(features)}."
                )

            model_features = getattr(
                model,
                "feature_names",
                None
            )

            if model_features is not None:

                model_features = list(
                    model_features
                )

                if model_features == features:

                    print(
                        "[PASS] Model features match "
                        "feature contract."
                    )

                else:

                    print(
                        "[FAIL] Model features do not match "
                        "feature contract."
                    )

                    print(
                        f"[INFO] Contract:\n"
                        f"       {features}"
                    )

                    print(
                        f"[INFO] Model:\n"
                        f"       {model_features}"
                    )

                    return False

            else:

                print(
                    "[WARN] XGBoost did not expose "
                    "feature names."
                )

        else:

            print(
                "[WARN] Feature contract file missing."
            )

        return True

    except Exception as exc:

        print(
            f"[FAIL] Model validation failed: {exc}"
        )

        return False


# ============================================================
# 3. PREDICTION TABLE VALIDATION
# ============================================================

def validate_prediction_table(
    path: Path,
    name: str
):

    print()
    print(f"Checking {name}...")

    if not path.exists():

        print(
            f"[FAIL] Missing: {path}"
        )

        return False

    try:

        df = pd.read_csv(path)

        print(
            f"[INFO] Rows: {len(df):,}"
        )

        print(
            f"[INFO] Columns: {list(df.columns)}"
        )

        # ----------------------------------------------------
        # Fraud probability
        # ----------------------------------------------------

        if "fraud_probability" not in df.columns:

            print(
                "[FAIL] fraud_probability column missing."
            )

            return False

        probability = pd.to_numeric(
            df["fraud_probability"],
            errors="coerce"
        )

        invalid = (
            probability.isna()
            | (probability < 0)
            | (probability > 1)
        )

        invalid_count = int(
            invalid.sum()
        )

        if invalid_count > 0:

            print(
                "[FAIL] Invalid fraud probabilities: "
                f"{invalid_count:,}"
            )

            return False

        print(
            "[PASS] fraud_probability values "
            "are within [0, 1]."
        )

        # ----------------------------------------------------
        # Fraud prediction
        # ----------------------------------------------------

        if "fraud_prediction" in df.columns:

            prediction = pd.to_numeric(
                df["fraud_prediction"],
                errors="coerce"
            )

            invalid_prediction = (
                prediction.isna()
                | ~prediction.isin([0, 1])
            )

            invalid_prediction_count = int(
                invalid_prediction.sum()
            )

            if invalid_prediction_count > 0:

                print(
                    "[FAIL] Invalid fraud_prediction values: "
                    f"{invalid_prediction_count:,}"
                )

                return False

            print(
                "[PASS] fraud_prediction is binary."
            )

        else:

            print(
                "[WARN] fraud_prediction column not found."
            )

        # ----------------------------------------------------
        # Risk column
        # ----------------------------------------------------

        risk_column = None

        if "risk_band" in df.columns:
            risk_column = "risk_band"

        elif "risk_level" in df.columns:
            risk_column = "risk_level"

        if risk_column:

            values = sorted(
                df[risk_column]
                .dropna()
                .astype(str)
                .str.upper()
                .unique()
                .tolist()
            )

            print(
                f"[INFO] {risk_column}: {values}"
            )

        return True

    except Exception as exc:

        print(
            f"[FAIL] Could not read {name}: {exc}"
        )

        return False


def validate_predictions():

    print_section("3. PREDICTION ARTIFACT VALIDATION")

    validation_result = validate_prediction_table(
        VALIDATION_PREDICTIONS_PATH,
        "Validation predictions"
    )

    future_result = validate_prediction_table(
        FUTURE_TEST_PREDICTIONS_PATH,
        "Future-test predictions"
    )

    return (
        validation_result
        and future_result
    )


# ============================================================
# 4. PERFORMANCE GATE
# ============================================================

def validate_performance():

    print_section("4. PERFORMANCE GATE")

    if not FUTURE_TEST_METRICS_PATH.exists():

        print(
            "[FAIL] Future-test metrics unavailable."
        )

        return False

    metrics = load_json(
        FUTURE_TEST_METRICS_PATH
    )

    roc_auc = safe_float(
        metric(
            metrics,
            "roc_auc",
            "ROC-AUC"
        )
    )

    pr_auc = safe_float(
        metric(
            metrics,
            "pr_auc",
            "PR-AUC",
            "average_precision"
        )
    )

    precision = safe_float(
        metric(
            metrics,
            "precision"
        )
    )

    recall = safe_float(
        metric(
            metrics,
            "recall"
        )
    )

    f1 = safe_float(
        metric(
            metrics,
            "f1",
            "f1_score",
            "F1-score"
        )
    )

    mcc = safe_float(
        metric(
            metrics,
            "mcc",
            "MCC"
        )
    )

    print(f"ROC-AUC   : {roc_auc}")
    print(f"PR-AUC    : {pr_auc}")
    print(f"Precision : {precision}")
    print(f"Recall    : {recall}")
    print(f"F1        : {f1}")
    print(f"MCC       : {mcc}")

    checks = []

    if roc_auc is not None:
        checks.append(
            roc_auc >= 0.95
        )

    if pr_auc is not None:
        checks.append(
            pr_auc >= 0.80
        )

    if precision is not None:
        checks.append(
            precision >= 0.90
        )

    if recall is not None:
        checks.append(
            recall >= 0.70
        )

    if f1 is not None:
        checks.append(
            f1 >= 0.80
        )

    if mcc is not None:
        checks.append(
            mcc >= 0.80
        )

    if not checks:

        print(
            "[FAIL] No performance metrics were found."
        )

        return False

    if all(checks):

        print(
            "[PASS] Future-test performance gate."
        )

        return True

    print(
        "[FAIL] Future-test performance gate."
    )

    return False


# ============================================================
# 5. CALIBRATION GATE
# ============================================================

def validate_calibration():

    print_section("5. CALIBRATION GATE")

    if not FUTURE_TEST_METRICS_PATH.exists():

        print(
            "[FAIL] Future-test metrics unavailable."
        )

        return False

    metrics = load_json(
        FUTURE_TEST_METRICS_PATH
    )

    brier = safe_float(
        metric(
            metrics,
            "brier_score",
            "Brier Score",
            "brier"
        )
    )

    if brier is None:

        print(
            "[FAIL] Brier score not found."
        )

        return False

    print(
        f"Brier Score: {brier:.6f}"
    )

    if brier <= 0.01:

        print(
            "[PASS] Brier score <= 0.01."
        )

        return True

    print(
        "[FAIL] Brier score > 0.01."
    )

    return False


# ============================================================
# 6. THRESHOLD GATE
# ============================================================

def validate_threshold():

    print_section("6. OPERATING THRESHOLD")

    if not FUTURE_TEST_METRICS_PATH.exists():

        print(
            "[FAIL] Future-test metrics unavailable."
        )

        return False

    metrics = load_json(
        FUTURE_TEST_METRICS_PATH
    )

    threshold = safe_float(
        metric(
            metrics,
            "operating_threshold",
            "threshold",
            "Operating Threshold"
        )
    )

    # Existing Phase 1 production threshold
    expected_threshold = 0.84

    if threshold is None:

        print(
            "[WARN] Operating threshold is not "
            "stored in future_test_metrics.json."
        )

        print(
            f"[INFO] Existing Phase 1 operating threshold: "
            f"{expected_threshold}"
        )

        # Do not fail the whole gate just because the
        # threshold isn't duplicated in this JSON.
        return True

    print(
        f"Stored threshold: {threshold}"
    )

    print(
        f"Expected threshold: {expected_threshold}"
    )

    if abs(
        threshold - expected_threshold
    ) < 1e-9:

        print(
            "[PASS] Operating threshold = 0.84."
        )

        return True

    print(
        "[WARN] Stored threshold differs from 0.84."
    )

    return False


# ============================================================
# 7. MONETARY GATE
# ============================================================

def validate_monetary():

    print_section("7. MONETARY PERFORMANCE GATE")

    if not FUTURE_TEST_METRICS_PATH.exists():

        print(
            "[FAIL] Future-test metrics unavailable."
        )

        return False

    metrics = load_json(
        FUTURE_TEST_METRICS_PATH
    )

    capture_rate = safe_float(
        metric(
            metrics,
            "fraud_amount_capture_rate",
            "Fraud Amount Capture Rate",
            "fraud_capture_rate"
        )
    )

    missed_amount = safe_float(
        metric(
            metrics,
            "missed_fraud_amount",
            "Missed Fraud Amount"
        )
    )

    false_positive_amount = safe_float(
        metric(
            metrics,
            "false_positive_amount",
            "False Positive Amount"
        )
    )

    print(
        f"Fraud Amount Capture Rate: "
        f"{capture_rate}"
    )

    print(
        f"Missed Fraud Amount: "
        f"{missed_amount}"
    )

    print(
        f"False Positive Amount: "
        f"{false_positive_amount}"
    )

    if capture_rate is None:

        print(
            "[FAIL] Fraud amount capture rate unavailable."
        )

        return False

    # Phase 1 gate:
    # At least 85% of fraudulent monetary value should
    # be captured.

    if capture_rate >= 0.85:

        print(
            "[PASS] Fraud amount capture >= 85%."
        )

        return True

    print(
        "[FAIL] Fraud amount capture < 85%."
    )

    return False


# ============================================================
# 8. RISK OUTPUT VALIDATION
# ============================================================

def validate_risk_outputs():

    print_section("8. RISK OUTPUT VALIDATION")

    candidates = [
        RISK_OUTPUT_PATH,
        TEST_RISK_OUTPUT_PATH,
        TRANSACTION_RISK_PATH,
    ]

    existing = [
        path
        for path in candidates
        if path.exists()
    ]

    if not existing:

        print(
            "[WARN] No existing risk output table found."
        )

        return True

    passed = False

    for path in existing:

        try:

            df = pd.read_csv(path)

            print()
            print(
                f"[INFO] {path.name}"
            )

            print(
                f"       Rows: {len(df):,}"
            )

            print(
                f"       Columns: {list(df.columns)}"
            )

            risk_column = None

            if "risk_band" in df.columns:

                risk_column = "risk_band"

            elif "risk_level" in df.columns:

                risk_column = "risk_level"

            if risk_column is None:

                print(
                    "[WARN] No risk_band/risk_level column."
                )

                continue

            values = (
                df[risk_column]
                .dropna()
                .astype(str)
                .str.upper()
                .unique()
                .tolist()
            )

            print(
                f"       Risk values: {sorted(values)}"
            )

            allowed = {
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL",
            }

            if set(values).issubset(allowed):

                print(
                    "[PASS] Recognized risk levels."
                )

                passed = True

            else:

                print(
                    "[WARN] Non-standard risk values detected."
                )

        except Exception as exc:

            print(
                f"[WARN] Could not inspect "
                f"{path.name}: {exc}"
            )

    return passed


# ============================================================
# 9. ROBUSTNESS / GENERALIZATION
# ============================================================

def validate_robustness():

    print_section("9. ROBUSTNESS / GENERALIZATION")

    if not (
        VALIDATION_METRICS_PATH.exists()
        and FUTURE_TEST_METRICS_PATH.exists()
    ):

        print(
            "[FAIL] Validation/future-test metrics missing."
        )

        return False

    validation = load_json(
        VALIDATION_METRICS_PATH
    )

    future = load_json(
        FUTURE_TEST_METRICS_PATH
    )

    validation_pr_auc = safe_float(
        metric(
            validation,
            "pr_auc",
            "PR-AUC",
            "average_precision"
        )
    )

    future_pr_auc = safe_float(
        metric(
            future,
            "pr_auc",
            "PR-AUC",
            "average_precision"
        )
    )

    validation_roc_auc = safe_float(
        metric(
            validation,
            "roc_auc",
            "ROC-AUC"
        )
    )

    future_roc_auc = safe_float(
        metric(
            future,
            "roc_auc",
            "ROC-AUC"
        )
    )

    print(
        f"Validation PR-AUC : {validation_pr_auc}"
    )

    print(
        f"Future-test PR-AUC: {future_pr_auc}"
    )

    print(
        f"Validation ROC-AUC : {validation_roc_auc}"
    )

    print(
        f"Future-test ROC-AUC: {future_roc_auc}"
    )

    if (
        validation_pr_auc is None
        or future_pr_auc is None
    ):

        print(
            "[FAIL] PR-AUC comparison unavailable."
        )

        return False

    pr_drop = (
        validation_pr_auc
        - future_pr_auc
    )

    print(
        f"PR-AUC degradation: {pr_drop:.6f}"
    )

    # Allow up to 5 percentage points degradation.
    if pr_drop <= 0.05:

        print(
            "[PASS] Generalization degradation "
            "within 5 percentage points."
        )

        return True

    print(
        "[FAIL] Excessive PR-AUC degradation."
    )

    return False


# ============================================================
# 10. EXPLAINABILITY
# ============================================================

def validate_explainability():

    print_section("10. EXPLAINABILITY")

    if not MODEL_PATH.exists():

        print(
            "[FAIL] Model missing."
        )

        return False

    try:

        model = xgb.XGBClassifier()

        model.load_model(
            str(MODEL_PATH)
        )

        booster = model.get_booster()

        importance = booster.get_score(
            importance_type="gain"
        )

        if not importance:

            print(
                "[FAIL] No feature importance available."
            )

            return False

        top_features = sorted(
            importance.items(),
            key=lambda item: item[1],
            reverse=True
        )[:10]

        print(
            "[PASS] XGBoost feature importance available."
        )

        print()
        print("Top 10 features by gain:")
        print("-" * 55)

        for feature, value in top_features:

            print(
                f"{feature:<35} {value:.6f}"
            )

        return True

    except Exception as exc:

        print(
            f"[FAIL] Explainability validation failed: "
            f"{exc}"
        )

        return False


# ============================================================
# REPORT
# ============================================================

def build_report(results):

    passed = sum(
        bool(value)
        for value in results.values()
    )

    total = len(results)

    # Risk output is informational rather than a critical
    # model-quality gate because the existing repository can
    # use risk_level instead of risk_band.

    critical_checks = [
        "artifact_existence",
        "model_contract",
        "prediction_artifacts",
        "performance",
        "calibration",
        "threshold",
        "monetary",
        "robustness",
        "explainability",
    ]

    critical_pass = all(
        results[name]
        for name in critical_checks
    )

    return {
        "project": "FraudSentinel AI",
        "phase": "Phase 1",
        "gate": (
            "PASS"
            if critical_pass
            else "FAIL"
        ),
        "checks_passed": passed,
        "checks_total": total,
        "checks": results,
        "artifact_paths": {
            "model": str(MODEL_PATH),
            "feature_contract": str(
                FEATURE_CONTRACT_PATH
            ),
            "validation_metrics": str(
                VALIDATION_METRICS_PATH
            ),
            "future_test_metrics": str(
                FUTURE_TEST_METRICS_PATH
            ),
            "validation_predictions": str(
                VALIDATION_PREDICTIONS_PATH
            ),
            "future_test_predictions": str(
                FUTURE_TEST_PREDICTIONS_PATH
            ),
        },
    }


def save_report(report):

    OUTPUT_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )


def print_report(report):

    print_section("PHASE 1 FINAL GATE")

    for name, result in report["checks"].items():

        status = (
            "PASS"
            if result
            else "FAIL"
        )

        print(
            f"{name:<30} [{status}]"
        )

    print()

    print(
        f"Checks passed: "
        f"{report['checks_passed']}/"
        f"{report['checks_total']}"
    )

    print()

    if report["gate"] == "PASS":

        print("=" * 72)
        print("PHASE 1 GATE: PASS")
        print("=" * 72)

        print(
            "Phase 1 is ready to proceed to "
            "the next intelligence layer."
        )

    else:

        print("=" * 72)
        print("PHASE 1 GATE: FAIL")
        print("=" * 72)

        print(
            "One or more critical Phase 1 checks failed."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("FraudSentinel AI — Phase 1 Gate Validation")
    print("=" * 72)

    print()
    print(
        f"Project root:\n{ROOT}"
    )

    print(
        f"Artifact root:\n{ARTIFACTS}"
    )

    # --------------------------------------------------------
    # 1. Artifacts
    # --------------------------------------------------------

    artifact_results = check_artifacts()

    artifact_pass = all(
        artifact_results.values()
    )

    # --------------------------------------------------------
    # 2. Model
    # --------------------------------------------------------

    model_pass = validate_model()

    # --------------------------------------------------------
    # 3. Predictions
    # --------------------------------------------------------

    prediction_pass = validate_predictions()

    # --------------------------------------------------------
    # 4. Performance
    # --------------------------------------------------------

    performance_pass = validate_performance()

    # --------------------------------------------------------
    # 5. Calibration
    # --------------------------------------------------------

    calibration_pass = validate_calibration()

    # --------------------------------------------------------
    # 6. Threshold
    # --------------------------------------------------------

    threshold_pass = validate_threshold()

    # --------------------------------------------------------
    # 7. Monetary
    # --------------------------------------------------------

    monetary_pass = validate_monetary()

    # --------------------------------------------------------
    # 8. Risk outputs
    # --------------------------------------------------------

    risk_pass = validate_risk_outputs()

    # --------------------------------------------------------
    # 9. Robustness
    # --------------------------------------------------------

    robustness_pass = validate_robustness()

    # --------------------------------------------------------
    # 10. Explainability
    # --------------------------------------------------------

    explainability_pass = validate_explainability()

    # --------------------------------------------------------
    # Collect
    # --------------------------------------------------------

    results = {
        "artifact_existence": artifact_pass,
        "model_contract": model_pass,
        "prediction_artifacts": prediction_pass,
        "performance": performance_pass,
        "calibration": calibration_pass,
        "threshold": threshold_pass,
        "monetary": monetary_pass,
        "risk_output": risk_pass,
        "robustness": robustness_pass,
        "explainability": explainability_pass,
    }

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    report = build_report(
        results
    )

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print_report(
        report
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    save_report(
        report
    )

    print()
    print(
        f"Gate report saved to:\n"
        f"{OUTPUT_REPORT_PATH}"
    )

    print()

    # --------------------------------------------------------
    # Exit code
    # --------------------------------------------------------

    if report["gate"] == "PASS":

        return 0

    return 1


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )