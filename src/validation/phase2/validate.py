from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ground_truth import (
    build_phase2_ground_truth,
)
from .metrics import (
    build_score_deciles,
    evaluate_phase2,
)


def load_phase2_windows(
    path: str | Path,
) -> pd.DataFrame:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Phase 2 windows file not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "Phase 2 windows file is empty."
        )

    required = [
        "merchant_id",
        "window_start",
        "window_end",
        "spike_score",
        "tas",
        "fas",
        "spike_state",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Phase 2 output is missing: {missing}"
        )

    df["window_start"] = pd.to_datetime(
        df["window_start"],
        errors="coerce",
        utc=True,
    )

    df["window_end"] = pd.to_datetime(
        df["window_end"],
        errors="coerce",
        utc=True,
    )

    numeric_columns = [
        "spike_score",
        "tas",
        "fas",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["merchant_id"] = (
        df["merchant_id"]
        .astype(str)
        .str.strip()
    )

    return df


def merge_predictions_with_ground_truth(
    phase2_windows: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:

    pred = phase2_windows.copy()

    truth = ground_truth.copy()

    pred["window_start"] = pd.to_datetime(
        pred["window_start"],
        utc=True,
    )

    truth["window_start"] = pd.to_datetime(
        truth["window_start"],
        utc=True,
    )

    merged = pred.merge(
        truth,
        on=[
            "merchant_id",
            "window_start",
        ],
        how="left",
        suffixes=(
            "",
            "_truth",
        ),
    )

    # Windows that contain no fraud are valid
    # ground-truth windows, not missing labels.
    fill_zero_columns = [
        "actual_transaction_count",
        "actual_total_amount",
        "actual_fraud_count",
        "actual_fraud_amount",
        "actual_unique_fraud_cards",
        "actual_fraud_rate",
        "fraud_amount_share",
    ]

    for col in fill_zero_columns:

        if col in merged.columns:
            merged[col] = (
                merged[col]
                .fillna(0)
            )

    boolean_columns = [
        "fraud_positive",
        "coordinated_fraud_positive",
        "fraud_amount_positive",
    ]

    for col in boolean_columns:

        if col in merged.columns:
            merged[col] = (
                merged[col]
                .fillna(False)
                .astype(bool)
            )

    return merged


def print_validation_report(
    report: dict,
) -> None:

    print("\n")
    print("=" * 80)
    print("PHASE 2 — OFFLINE VALIDATION")
    print("=" * 80)

    print(
        f"\nValidation windows: "
        f"{report['windows']:,}"
    )

    print(
        f"Fraud-positive windows: "
        f"{report['fraud_positive_windows']:,}"
    )

    print(
        f"Coordinated-fraud windows: "
        f"{report['coordinated_fraud_windows']:,}"
    )

    print(
        f"Actual fraud transactions: "
        f"{report['actual_fraud_transactions']:,}"
    )

    print(
        f"Actual fraud amount: "
        f"{report['actual_fraud_amount']:,.2f}"
    )

    print("\n")
    print("-" * 80)
    print("SCORE QUALITY")
    print("-" * 80)

    print(
        f"Spike Score ROC-AUC: "
        f"{report['spike_score_roc_auc']:.6f}"
    )

    print(
        f"Spike Score PR-AUC:  "
        f"{report['spike_score_pr_auc']:.6f}"
    )

    print(
        "Coordinated ROC-AUC:  "
        f"{report['spike_score_coordinated_roc_auc']:.6f}"
    )

    print(
        "Coordinated PR-AUC:   "
        f"{report['spike_score_coordinated_pr_auc']:.6f}"
    )

    print(
        "Fraud amount Spearman: "
        f"{report['spike_score_fraud_amount_spearman']:.6f}"
    )

    print("\n")
    print("-" * 80)
    print("SCORE DISTRIBUTION")
    print("-" * 80)

    print(
        f"Mean:   {report['score_mean']:.4f}"
    )

    print(
        f"Median: {report['score_median']:.4f}"
    )

    print(
        f"P90:    {report['score_p90']:.4f}"
    )

    print(
        f"P95:    {report['score_p95']:.4f}"
    )

    print(
        f"P99:    {report['score_p99']:.4f}"
    )

    print("\n")
    print("-" * 80)
    print("TOP-K FRAUD CAPTURE")
    print("-" * 80)

    for row in report["top_k"]:

        print(
            f"Top {row['top_fraction']:.0%}: "
            f""
            f"window recall="
            f"{row['fraud_window_recall']:.4f}, "
            f""
            f"amount capture="
            f"{row['fraud_amount_capture_rate']:.4f}"
        )

    print("\n")
    print("-" * 80)
    print("SPIKE STATE QUALITY")
    print("-" * 80)

    for row in report["state_metrics"]:

        print(
            f"{row['spike_state']:<25} "
            f"windows={row['windows']:,} "
            f""
            f"fraud_rate="
            f"{row['fraud_window_rate']:.4f} "
            f""
            f"coord_rate="
            f"{row['coordinated_window_rate']:.4f} "
            f""
            f"fraud_amount="
            f"{row['actual_fraud_amount']:,.2f}"
        )

    print("\n")
    print("=" * 80)


def run_validation(
    phase2_windows_path: str,
    ground_truth_path: str,
    output_dir: str,
    window_minutes: int = 15,
) -> dict:

    output = Path(output_dir)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading Phase 2 predictions...")

    phase2_windows = load_phase2_windows(
        phase2_windows_path
    )

    print(
        f"Phase 2 windows: "
        f"{len(phase2_windows):,}"
    )

    print("\nBuilding ground truth...")

    ground_truth = (
        build_phase2_ground_truth(
            ground_truth_path,
            window_minutes=window_minutes,
        )
    )

    # ----------------------------------------------------------
    # MERGE
    # ----------------------------------------------------------

    print("\nAligning predictions with ground truth...")

    merged = merge_predictions_with_ground_truth(
        phase2_windows,
        ground_truth,
    )

    print(
        f"Aligned windows: "
        f"{len(merged):,}"
    )

    # ----------------------------------------------------------
    # SAVE ALIGNED DATA
    # ----------------------------------------------------------

    merged_path = (
        output
        / "phase2_validation_windows.csv"
    )

    merged.to_csv(
        merged_path,
        index=False,
    )

    ground_truth_path_out = (
        output
        / "phase2_ground_truth_windows.csv"
    )

    ground_truth.to_csv(
        ground_truth_path_out,
        index=False,
    )

    # ----------------------------------------------------------
    # SCORE DECILES
    # ----------------------------------------------------------

    deciles = build_score_deciles(
        merged,
        score_col="spike_score",
        label_col="fraud_positive",
        fraud_amount_col="actual_fraud_amount",
    )

    deciles_path = (
        output
        / "phase2_score_deciles.csv"
    )

    deciles.to_csv(
        deciles_path,
        index=False,
    )

    # ----------------------------------------------------------
    # STATE TABLE
    # ----------------------------------------------------------

    state_metrics = (
        merged.groupby(
            "spike_state",
            observed=True,
        )
        .agg(
            windows=(
                "spike_state",
                "size",
            ),
            fraud_positive_windows=(
                "fraud_positive",
                "sum",
            ),
            coordinated_fraud_windows=(
                "coordinated_fraud_positive",
                "sum",
            ),
            actual_fraud_count=(
                "actual_fraud_count",
                "sum",
            ),
            actual_fraud_amount=(
                "actual_fraud_amount",
                "sum",
            ),
        )
        .reset_index()
    )

    state_metrics["fraud_window_rate"] = (
        state_metrics[
            "fraud_positive_windows"
        ]
        / state_metrics["windows"]
    )

    state_metrics[
        "coordinated_window_rate"
    ] = (
        state_metrics[
            "coordinated_fraud_windows"
        ]
        / state_metrics["windows"]
    )

    state_path = (
        output
        / "phase2_state_metrics.csv"
    )

    state_metrics.to_csv(
        state_path,
        index=False,
    )

    # ----------------------------------------------------------
    # COMPLETE METRICS
    # ----------------------------------------------------------

    report = evaluate_phase2(
        merged
    )

    report["configuration"] = {
        "window_minutes": window_minutes,
        "ground_truth_source": str(
            ground_truth_path
        ),
        "phase2_source": str(
            phase2_windows_path
        ),
    }

    # ----------------------------------------------------------
    # SAVE JSON
    # ----------------------------------------------------------

    report_path = (
        output
        / "phase2_validation_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            default=lambda x:
                float(x)
                if isinstance(
                    x,
                    np.floating,
                )
                else x,
        )

    print_validation_report(
        report
    )

    print("\nArtifacts written:")
    print(
        f"  {merged_path}"
    )
    print(
        f"  {ground_truth_path_out}"
    )
    print(
        f"  {deciles_path}"
    )
    print(
        f"  {state_path}"
    )
    print(
        f"  {report_path}"
    )

    return report


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Offline validation of "
            "FraudSentinel AI Phase 2"
        )
    )

    parser.add_argument(
        "--phase2-windows",
        required=True,
        help=(
            "Path to "
            "phase2_windows.csv"
        ),
    )

    parser.add_argument(
        "--ground-truth",
        required=True,
        help=(
            "Path to original labeled "
            "transaction dataset"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/phase2/validation"
        ),
        help=(
            "Validation output directory"
        ),
    )

    parser.add_argument(
        "--window-minutes",
        type=int,
        default=15,
        help=(
            "Phase 2 merchant window size"
        ),
    )

    args = parser.parse_args()

    run_validation(
        phase2_windows_path=(
            args.phase2_windows
        ),
        ground_truth_path=(
            args.ground_truth
        ),
        output_dir=(
            args.output_dir
        ),
        window_minutes=(
            args.window_minutes
        ),
    )


if __name__ == "__main__":
    main()