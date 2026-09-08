import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
)

# ============================================================
# 1. LOAD UNSEEN DATA
# ============================================================

unseen_df = pd.read_csv(
    "data/external/fraud_test.csv"
)

# ============================================================
# 2. LOAD PHASE-1 PREDICTIONS
# ============================================================

prediction_df = pd.read_csv(
    "artifacts/phase1/predictions/unseen_risk_output.csv"
)

# ============================================================
# 3. MERGE ACTUAL LABEL WITH PREDICTION
# ============================================================

eval_df = prediction_df.merge(
    unseen_df[["trans_num", "is_fraud"]],
    left_on="transaction_id",
    right_on="trans_num",
    how="inner",
)

print("Unseen rows:", len(unseen_df))
print("Prediction rows:", len(prediction_df))
print("Evaluation rows:", len(eval_df))

# Safety check
if len(eval_df) != len(prediction_df):
    raise ValueError(
        "Some prediction rows could not be matched "
        "with actual is_fraud labels."
    )

# ============================================================
# 4. ACTUAL / PREDICTED VALUES
# ============================================================

y_true = eval_df["is_fraud"].astype(int).to_numpy()
y_proba = eval_df["fraud_probability"].to_numpy()
y_pred = eval_df["predicted_fraud"].astype(int).to_numpy()
amount = eval_df["amount"].to_numpy()

threshold = float(
    eval_df["threshold"].iloc[0]
)

# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred
).ravel()

print("\n" + "=" * 60)
print("UNSEEN DATA — CONFUSION MATRIX")
print("=" * 60)

print(
    f"TP: {tp:,}\n"
    f"FN: {fn:,}\n"
    f"FP: {fp:,}\n"
    f"TN: {tn:,}"
)

# ============================================================
# 6. CORE METRICS
# ============================================================

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0,
)

accuracy = accuracy_score(
    y_true,
    y_pred,
)

balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred,
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else np.nan
)

fpr = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else np.nan
)

fnr = (
    fn / (fn + tp)
    if (fn + tp) > 0
    else np.nan
)

mcc = matthews_corrcoef(
    y_true,
    y_pred,
)

# ============================================================
# 7. PROBABILITY / RANKING METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_true,
    y_proba,
)

pr_auc = average_precision_score(
    y_true,
    y_proba,
)

brier = brier_score_loss(
    y_true,
    y_proba,
)

# ============================================================
# 8. MONETARY METRICS
# ============================================================

actual_fraud_mask = y_true == 1
tp_mask = (y_true == 1) & (y_pred == 1)
fn_mask = (y_true == 1) & (y_pred == 0)
fp_mask = (y_true == 0) & (y_pred == 1)
tn_mask = (y_true == 0) & (y_pred == 0)

total_actual_fraud_amount = (
    amount[actual_fraud_mask].sum()
)

detected_fraud_amount = (
    amount[tp_mask].sum()
)

missed_fraud_amount = (
    amount[fn_mask].sum()
)

false_positive_amount = (
    amount[fp_mask].sum()
)

legitimate_allowed_amount = (
    amount[tn_mask].sum()
)

fraud_amount_capture_rate = (
    detected_fraud_amount
    / total_actual_fraud_amount
    if total_actual_fraud_amount > 0
    else np.nan
)

# ============================================================
# 9. HIGH-VALUE FRAUD
# ============================================================

high_value_threshold = amount[
    amount >= np.quantile(amount, 0.95)
].min()

high_value_mask = (
    amount >= high_value_threshold
)

hv_true = y_true[high_value_mask]
hv_pred = y_pred[high_value_mask]
hv_amount = amount[high_value_mask]

hv_fraud_mask = hv_true == 1
hv_tp_mask = (
    (hv_true == 1) &
    (hv_pred == 1)
)

hv_fn_mask = (
    (hv_true == 1) &
    (hv_pred == 0)
)

high_value_transaction_capture = (
    hv_tp_mask.sum()
    /
    (hv_tp_mask.sum() + hv_fn_mask.sum())
    if (hv_tp_mask.sum() + hv_fn_mask.sum()) > 0
    else np.nan
)

high_value_actual_fraud_amount = (
    hv_amount[hv_fraud_mask].sum()
)

high_value_detected_fraud_amount = (
    hv_amount[hv_tp_mask].sum()
)

high_value_amount_capture = (
    high_value_detected_fraud_amount
    / high_value_actual_fraud_amount
    if high_value_actual_fraud_amount > 0
    else np.nan
)

# ============================================================
# 10. PRINT UNSEEN SCORECARD
# ============================================================

print("\n" + "=" * 60)
print("PHASE-1 — UNSEEN DATA TRANSACTION INTELLIGENCE SCORECARD")
print("=" * 60)

print(f"Operating threshold       : {threshold:.2f}")

print("\n--- Core Transaction Metrics ---")

print(f"TP                        : {tp:,}")
print(f"FN                        : {fn:,}")
print(f"FP                        : {fp:,}")
print(f"TN                        : {tn:,}")

print(f"Precision                 : {precision:.4f}")
print(f"Recall                    : {recall:.4f}")
print(f"F1 Score                  : {f1:.4f}")
print(f"Accuracy                  : {accuracy:.4f}")
print(f"Balanced Accuracy         : {balanced_accuracy:.4f}")
print(f"Specificity               : {specificity:.4f}")
print(f"False Positive Rate       : {fpr:.4%}")
print(f"False Negative Rate       : {fnr:.4%}")
print(f"MCC                       : {mcc:.4f}")

print("\n--- Probability / Ranking Metrics ---")

print(f"ROC-AUC                   : {roc_auc:.4f}")
print(f"PR-AUC                    : {pr_auc:.4f}")
print(f"Brier Score               : {brier:.6f}")

print("\n--- Monetary / Business Metrics ---")

print(
    f"Total Actual Fraud Amount: "
    f"{total_actual_fraud_amount:,.2f}"
)

print(
    f"Detected Fraud Amount    : "
    f"{detected_fraud_amount:,.2f}"
)

print(
    f"Fraud Amount Capture Rate: "
    f"{fraud_amount_capture_rate:.4%}"
)

print(
    f"Missed Fraud Amount      : "
    f"{missed_fraud_amount:,.2f}"
)

print(
    f"False-Positive Amount    : "
    f"{false_positive_amount:,.2f}"
)

print(
    f"Legitimate Amount Allowed: "
    f"{legitimate_allowed_amount:,.2f}"
)

print("\n--- High-Value Fraud Metrics ---")

print(
    f"High-Value Threshold      : "
    f"{high_value_threshold:,.2f}"
)

print(
    f"High-Value Transaction Capture: "
    f"{high_value_transaction_capture:.4%}"
)

print(
    f"High-Value Amount Capture : "
    f"{high_value_amount_capture:.4%}"
)

print("\n--- Classification Report ---")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Legit", "Fraud"],
        digits=4,
        zero_division=0,
    )
)