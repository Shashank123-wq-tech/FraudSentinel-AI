
# ============================================================
# FraudSentinel AI
# Phase 2 — Temporal Holdout Validation Runner
#
# File:
#     src/validation/phase2/run_temporal_holdout.py
#
# Run from PROJECT ROOT:
#
#     python -m src.validation.phase2.run_temporal_holdout
#
# ============================================================

from __future__ import annotations

from src.validation.phase2.temporal_holdout import (
    run_temporal_holdout_validation,
)


def main() -> None:

    # ========================================================
    # Configuration
    # ========================================================

    phase2_windows_path = (
        "artifacts/phase2/phase2_windows.csv"
    )

    ground_truth_path = (
        "data/raw/fraud_transactions.csv"
    )

    output_dir = (
        "artifacts/validation/phase2"
    )

    calibration_fraction = 0.70

    window_minutes = 15

    max_event_gap_minutes = 15.0

    # ========================================================
    # Run validation
    # ========================================================

    result = run_temporal_holdout_validation(
        phase2_windows_path=(
            phase2_windows_path
        ),
        ground_truth_path=(
            ground_truth_path
        ),
        output_dir=(
            output_dir
        ),
        calibration_fraction=(
            calibration_fraction
        ),
        window_minutes=(
            window_minutes
        ),
        max_event_gap_minutes=(
            max_event_gap_minutes
        ),
    )

    # ========================================================
    # Compact final summary
    # ========================================================

    print()
    print("=" * 72)
    print("FINAL PHASE-2 VALIDATION SUMMARY")
    print("=" * 72)

    temporal = result.get(
        "temporal_split",
        {},
    )

    performance = result.get(
        "holdout_score_performance",
        {},
    )

    events = result.get(
        "events",
        {},
    )

    calibration = result.get(
        "calibration",
        {},
    )

    print()

    print(
        "Temporal Split:"
    )

    print(
        f"  Calibration windows : "
        f"{temporal.get('calibration_windows', 0):,}"
    )

    print(
        f"  Holdout windows     : "
        f"{temporal.get('holdout_windows', 0):,}"
    )

    print(
        f"  Split time          : "
        f"{temporal.get('split_time', 'N/A')}"
    )

    print()

    print(
        "Frozen Thresholds:"
    )

    print(
        f"  Candidate           : "
        f"{calibration.get('candidate_score_threshold', 'N/A')}"
    )

    print(
        f"  Verified            : "
        f"{calibration.get('verified_score_threshold', 'N/A')}"
    )

    print(
        f"  Critical            : "
        f"{calibration.get('critical_score_threshold', 'N/A')}"
    )

    print()

    print(
        "Holdout Score:"
    )

    roc_auc = performance.get(
        "roc_auc"
    )

    pr_auc = performance.get(
        "pr_auc"
    )

    amount_spearman = performance.get(
        "fraud_amount_spearman"
    )

    if roc_auc is None:
        print(
            "  ROC-AUC             : N/A"
        )
    else:
        print(
            f"  ROC-AUC             : "
            f"{roc_auc:.6f}"
        )

    if pr_auc is None:
        print(
            "  PR-AUC              : N/A"
        )
    else:
        print(
            f"  PR-AUC              : "
            f"{pr_auc:.6f}"
        )

    if amount_spearman is None:
        print(
            "  Amount Spearman     : N/A"
        )
    else:
        print(
            f"  Amount Spearman     : "
            f"{amount_spearman:.6f}"
        )

    print()

    print(
        "Holdout Events:"
    )

    print(
        f"  Events detected     : "
        f"{events.get('event_count', 0):,}"
    )

    print()

    print("=" * 72)
    print(
        "TEMPORAL HOLDOUT VALIDATION COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
