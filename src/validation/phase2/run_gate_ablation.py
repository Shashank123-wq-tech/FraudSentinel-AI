"""
Run Phase 2 Gate Ablation Diagnostic
====================================

Usage
-----

From the FraudSentinel-AI project root:

    python -m src.validation.phase2.run_gate_ablation

Input
-----

The diagnostic uses the temporal holdout artifact:

    artifacts/validation/phase2/phase2_temporal_holdout_windows.csv

This is intentional.

We want to diagnose the gates on the SAME unseen holdout
that produced:

    ROC-AUC = 0.954735
    PR-AUC  = 0.079721
    VERIFIED = 24 windows
    CRITICAL = 0 windows

Outputs
-------

    artifacts/validation/phase2/
        phase2_gate_ablation_verified.csv
        phase2_gate_ablation_critical.csv
        phase2_gate_ablation_independent.csv
        phase2_gate_ablation_thresholds.csv
        phase2_gate_ablation_summary.csv
        phase2_gate_ablation_windows.csv
        phase2_gate_ablation.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from .gate_ablation import (
    build_gate_summary,
    run_gate_ablation,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "validation"
    / "phase2"
    / "phase2_temporal_holdout_windows.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "validation"
    / "phase2"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

VERIFIED_SCORE_THRESHOLD = 57.150510779704376
CRITICAL_SCORE_THRESHOLD = 75.15031235576491

VERIFIED_MIN_STATISTICAL_EVIDENCE = 0.35
VERIFIED_MIN_TEMPORAL_EVIDENCE = 0.20

CRITICAL_MIN_STATISTICAL_EVIDENCE = 0.55
CRITICAL_MIN_TEMPORAL_EVIDENCE = 0.30

VERIFIED_PERSISTENCE_WINDOWS = 2
CRITICAL_PERSISTENCE_WINDOWS = 2

VERIFIED_PERSISTENCE_MINUTES = 30.0
CRITICAL_PERSISTENCE_MINUTES = 45.0

VERIFIED_MIN_TRANSACTIONS = 2
CRITICAL_MIN_TRANSACTIONS = 2

CRITICAL_MIN_UNIQUE_CARDS = 2
CRITICAL_MIN_COORDINATION_SCORE = 0.45


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_holdout_windows(
    path: Path,
) -> pd.DataFrame:
    """Load and validate the temporal holdout artifact."""

    if not path.exists():
        raise FileNotFoundError(
            f"Holdout windows artifact not found:\n{path}\n\n"
            "Run temporal holdout validation first:\n"
            "python -m src.validation.phase2.run_temporal_holdout"
        )

    df = pd.read_csv(path)

    required = [
        "merchant_id",
        "window_start",
        "spike_score",
        "statistical_evidence",
        "temporal_evidence",
        "transaction_count",
        "unique_cards",
        "coordination_score",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Holdout artifact is missing required "
            f"Phase-2 columns: {missing}"
        )

    # The temporal holdout validator normally provides these.
    # We allow equivalent names so the diagnostic remains robust.
    if "actual_fraud_count" not in df.columns:
        if "fraud_count" in df.columns:
            df["actual_fraud_count"] = df["fraud_count"]
        else:
            raise ValueError(
                "Holdout artifact does not contain "
                "'actual_fraud_count' or 'fraud_count'."
            )

    if "actual_fraud_amount" not in df.columns:
        if "fraud_amount" in df.columns:
            df["actual_fraud_amount"] = df["fraud_amount"]
        else:
            raise ValueError(
                "Holdout artifact does not contain "
                "'actual_fraud_amount' or 'fraud_amount'."
            )

    if "fraud_positive_window" not in df.columns:
        df["fraud_positive_window"] = (
            pd.to_numeric(
                df["actual_fraud_count"],
                errors="coerce",
            )
            .fillna(0)
            > 0
        ).astype(int)

    return df


def _clean_for_json(
    value,
):
    """Convert NumPy/Pandas objects into JSON-safe values."""

    if isinstance(value, dict):
        return {
            str(k): _clean_for_json(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            _clean_for_json(v)
            for v in value
        ]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if pd.isna(value):
        return None

    return value


def _print_table(
    title: str,
    df: pd.DataFrame,
) -> None:
    """Print a readable diagnostic table."""

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    if df.empty:
        print("No rows.")
        return

    display = df.copy()

    numeric_columns = display.select_dtypes(
        include=["float", "float64", "float32"]
    ).columns

    for col in numeric_columns:
        display[col] = display[col].map(
            lambda x: (
                f"{x:.6f}"
                if pd.notna(x)
                else "NaN"
            )
        )

    print(
        display.to_string(
            index=False
        )
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_gate_ablation_diagnostic(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict:
    """
    Run the complete gate-ablation diagnostic.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 100)
    print("FraudSentinel AI — Phase 2 Gate Ablation Diagnostic")
    print("=" * 100)

    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    df = _load_holdout_windows(
        input_path
    )

    print()
    print(f"Holdout windows loaded: {len(df):,}")

    fraud_windows = int(
        pd.to_numeric(
            df["fraud_positive_window"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    fraud_amount = float(
        pd.to_numeric(
            df["actual_fraud_amount"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    print(
        f"Fraud-positive windows: {fraud_windows:,}"
    )

    print(
        f"Actual fraud amount: {fraud_amount:,.2f}"
    )

    # -------------------------------------------------------------
    # Run diagnostic
    # -------------------------------------------------------------

    results = run_gate_ablation(
        df,

        verified_score_threshold=(
            VERIFIED_SCORE_THRESHOLD
        ),

        critical_score_threshold=(
            CRITICAL_SCORE_THRESHOLD
        ),

        verified_min_statistical_evidence=(
            VERIFIED_MIN_STATISTICAL_EVIDENCE
        ),

        verified_min_temporal_evidence=(
            VERIFIED_MIN_TEMPORAL_EVIDENCE
        ),

        critical_min_statistical_evidence=(
            CRITICAL_MIN_STATISTICAL_EVIDENCE
        ),

        critical_min_temporal_evidence=(
            CRITICAL_MIN_TEMPORAL_EVIDENCE
        ),

        verified_persistence_windows=(
            VERIFIED_PERSISTENCE_WINDOWS
        ),

        critical_persistence_windows=(
            CRITICAL_PERSISTENCE_WINDOWS
        ),

        verified_persistence_minutes=(
            VERIFIED_PERSISTENCE_MINUTES
        ),

        critical_persistence_minutes=(
            CRITICAL_PERSISTENCE_MINUTES
        ),

        verified_min_transactions=(
            VERIFIED_MIN_TRANSACTIONS
        ),

        critical_min_transactions=(
            CRITICAL_MIN_TRANSACTIONS
        ),

        critical_min_unique_cards=(
            CRITICAL_MIN_UNIQUE_CARDS
        ),

        critical_min_coordination_score=(
            CRITICAL_MIN_COORDINATION_SCORE
        ),
    )

    verified = results[
        "verified_cumulative"
    ]

    critical = results[
        "critical_cumulative"
    ]

    independent = results[
        "independent"
    ]

    thresholds = results[
        "score_thresholds"
    ]

    diagnostic_windows = results[
        "diagnostic_windows"
    ]

    summary = build_gate_summary(
        verified,
        critical,
    )

    # -------------------------------------------------------------
    # Save CSV artifacts
    # -------------------------------------------------------------

    verified_path = (
        output_dir
        / "phase2_gate_ablation_verified.csv"
    )

    critical_path = (
        output_dir
        / "phase2_gate_ablation_critical.csv"
    )

    independent_path = (
        output_dir
        / "phase2_gate_ablation_independent.csv"
    )

    thresholds_path = (
        output_dir
        / "phase2_gate_ablation_thresholds.csv"
    )

    summary_path = (
        output_dir
        / "phase2_gate_ablation_summary.csv"
    )

    windows_path = (
        output_dir
        / "phase2_gate_ablation_windows.csv"
    )

    verified.to_csv(
        verified_path,
        index=False,
    )

    critical.to_csv(
        critical_path,
        index=False,
    )

    independent.to_csv(
        independent_path,
        index=False,
    )

    thresholds.to_csv(
        thresholds_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # Don't save every original column if the file becomes enormous.
    # Keep the diagnostic columns needed for investigation.
    diagnostic_columns = [
        "merchant_id",
        "window_start",
        "spike_score",
        "statistical_evidence",
        "temporal_evidence",
        "transaction_count",
        "unique_cards",
        "coordination_score",
        "actual_fraud_count",
        "actual_fraud_amount",
        "fraud_positive_window",
        "previous_window_start",
        "gap_minutes",
        "persistence_windows",
        "persistence_minutes",
    ]

    diagnostic_columns += [
        c
        for c in diagnostic_windows.columns
        if c.startswith("verified_gate_")
        or c.startswith("critical_gate_")
    ]

    diagnostic_columns = list(
        dict.fromkeys(
            c
            for c in diagnostic_columns
            if c in diagnostic_windows.columns
        )
    )

    diagnostic_windows[
        diagnostic_columns
    ].to_csv(
        windows_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Find likely bottlenecks
    # -------------------------------------------------------------

    verified_bottleneck = None
    critical_bottleneck = None

    if not verified.empty:
        temp = verified.dropna(
            subset=["reduction_rate"]
        )

        if not temp.empty:
            row = temp.loc[
                temp["reduction_rate"].idxmax()
            ]

            verified_bottleneck = {
                "gate": row["gate"],
                "reduction_rate": float(
                    row["reduction_rate"]
                ),
                "windows_removed": int(
                    row["windows_removed"]
                ),
            }

    if not critical.empty:
        temp = critical.dropna(
            subset=["reduction_rate"]
        )

        if not temp.empty:
            row = temp.loc[
                temp["reduction_rate"].idxmax()
            ]

            critical_bottleneck = {
                "gate": row["gate"],
                "reduction_rate": float(
                    row["reduction_rate"]
                ),
                "windows_removed": int(
                    row["windows_removed"]
                ),
            }

    # -------------------------------------------------------------
    # Build JSON report
    # -------------------------------------------------------------

    report = {
        "validation": "phase2_gate_ablation",

        "input": str(
            input_path
        ),

        "dataset": {
            "holdout_windows": int(
                len(df)
            ),
            "fraud_positive_windows": int(
                fraud_windows
            ),
            "actual_fraud_amount": float(
                fraud_amount
            ),
        },

        "configuration": {
            "verified_score_threshold": (
                VERIFIED_SCORE_THRESHOLD
            ),
            "critical_score_threshold": (
                CRITICAL_SCORE_THRESHOLD
            ),

            "verified_min_statistical_evidence": (
                VERIFIED_MIN_STATISTICAL_EVIDENCE
            ),

            "verified_min_temporal_evidence": (
                VERIFIED_MIN_TEMPORAL_EVIDENCE
            ),

            "critical_min_statistical_evidence": (
                CRITICAL_MIN_STATISTICAL_EVIDENCE
            ),

            "critical_min_temporal_evidence": (
                CRITICAL_MIN_TEMPORAL_EVIDENCE
            ),

            "verified_persistence_windows": (
                VERIFIED_PERSISTENCE_WINDOWS
            ),

            "critical_persistence_windows": (
                CRITICAL_PERSISTENCE_WINDOWS
            ),

            "verified_persistence_minutes": (
                VERIFIED_PERSISTENCE_MINUTES
            ),

            "critical_persistence_minutes": (
                CRITICAL_PERSISTENCE_MINUTES
            ),

            "verified_min_transactions": (
                VERIFIED_MIN_TRANSACTIONS
            ),

            "critical_min_transactions": (
                CRITICAL_MIN_TRANSACTIONS
            ),

            "critical_min_unique_cards": (
                CRITICAL_MIN_UNIQUE_CARDS
            ),

            "critical_min_coordination_score": (
                CRITICAL_MIN_COORDINATION_SCORE
            ),
        },

        "verified_cumulative": verified.to_dict(
            orient="records"
        ),

        "critical_cumulative": critical.to_dict(
            orient="records"
        ),

        "independent": independent.to_dict(
            orient="records"
        ),

        "score_thresholds": thresholds.to_dict(
            orient="records"
        ),

        "bottlenecks": {
            "verified": verified_bottleneck,
            "critical": critical_bottleneck,
        },

        "artifacts": {
            "verified": str(
                verified_path
            ),
            "critical": str(
                critical_path
            ),
            "independent": str(
                independent_path
            ),
            "thresholds": str(
                thresholds_path
            ),
            "summary": str(
                summary_path
            ),
            "windows": str(
                windows_path
            ),
        },
    }

    json_path = (
        output_dir
        / "phase2_gate_ablation.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            _clean_for_json(report),
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------

    _print_table(
        "VERIFIED — CUMULATIVE GATE ABLATION",
        verified,
    )

    _print_table(
        "CRITICAL — CUMULATIVE GATE ABLATION",
        critical,
    )

    _print_table(
        "INDEPENDENT GATE ANALYSIS",
        independent,
    )

    _print_table(
        "SCORE-ONLY THRESHOLD ANALYSIS",
        thresholds,
    )

    print()
    print("=" * 100)
    print("BOTTLENECK ANALYSIS")
    print("=" * 100)

    if verified_bottleneck:
        print(
            "Verified bottleneck:"
            f" {verified_bottleneck['gate']} "
            f"— removed "
            f"{verified_bottleneck['windows_removed']:,} "
            f"windows "
            f"({verified_bottleneck['reduction_rate']:.2%})"
        )
    else:
        print(
            "Verified bottleneck: unavailable"
        )

    if critical_bottleneck:
        print(
            "Critical bottleneck:"
            f" {critical_bottleneck['gate']} "
            f"— removed "
            f"{critical_bottleneck['windows_removed']:,} "
            f"windows "
            f"({critical_bottleneck['reduction_rate']:.2%})"
        )
    else:
        print(
            "Critical bottleneck: unavailable"
        )

    print()
    print("=" * 100)
    print("ARTIFACTS")
    print("=" * 100)

    print(
        f"Verified : {verified_path}"
    )

    print(
        f"Critical : {critical_path}"
    )

    print(
        f"Independent: {independent_path}"
    )

    print(
        f"Thresholds : {thresholds_path}"
    )

    print(
        f"Summary    : {summary_path}"
    )

    print(
        f"Windows    : {windows_path}"
    )

    print(
        f"JSON       : {json_path}"
    )

    print()
    print(
        "Gate-ablation diagnostic completed."
    )

    return report


def main() -> None:
    """CLI entry point."""

    run_gate_ablation_diagnostic(
        input_path=DEFAULT_INPUT,
        output_dir=DEFAULT_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
