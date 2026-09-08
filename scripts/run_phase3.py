# scripts/run_phase3.py

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.pipeline.phase3_network_pipeline import (
    Phase3NetworkIntelligence,
)


def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FRAUDSENTINEL AI"
    )

    print(
        "PHASE 3 NETWORK INTELLIGENCE"
    )

    print(
        "=" * 70
    )

    # =====================================================
    # INPUTS
    # =====================================================

    phase1_risk_path = (
        PROJECT_ROOT
        / "artifacts"
        / "phase1"
        / "risk_output"
        / "transaction_risk.csv"
    )

    phase2_events_path = (
        PROJECT_ROOT
        / "artifacts"
        / "phase2"
        / "spike_events"
        / "merchant_spike_events.parquet"
    )

    output_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "phase3"
    )

    # =====================================================
    # CHECK FILES
    # =====================================================

    print(
        "\nChecking input files..."
    )

    print(
        f"\nPhase 1:"
        f"\n  {phase1_risk_path}"
    )

    if not phase1_risk_path.exists():
        raise FileNotFoundError(
            "\nPhase 1 transaction risk file "
            "was not found:\n"
            f"{phase1_risk_path}"
        )

    print(
        "  ✓ Found"
    )

    print(
        f"\nPhase 2:"
        f"\n  {phase2_events_path}"
    )

    if not phase2_events_path.exists():
        raise FileNotFoundError(
            "\nPhase 2 spike event file "
            "was not found:\n"
            f"{phase2_events_path}"
        )

    print(
        "  ✓ Found"
    )

    # =====================================================
    # LOAD INPUTS
    # =====================================================

    print(
        "\nLoading Phase 1..."
    )

    phase1_df = pd.read_csv(
        phase1_risk_path
    )

    print(
        f"  Rows: "
        f"{len(phase1_df):,}"
    )

    print(
        f"  Columns: "
        f"{len(phase1_df.columns)}"
    )

    print(
        "\nLoading Phase 2..."
    )

    phase2_df = pd.read_parquet(
        phase2_events_path
    )

    print(
        f"  Rows: "
        f"{len(phase2_df):,}"
    )

    print(
        f"  Columns: "
        f"{len(phase2_df.columns)}"
    )

    # =====================================================
    # PRINT SCHEMA
    # =====================================================

    print(
        "\nPhase 1 schema:"
    )

    print(
        list(phase1_df.columns)
    )

    print(
        "\nPhase 2 schema:"
    )

    print(
        list(phase2_df.columns)
    )

    # =====================================================
    # INITIALIZE PIPELINE
    # =====================================================

    pipeline = (
        Phase3NetworkIntelligence(
            output_dir=output_dir,
            random_seed=42,
            model_version=(
                "network_intelligence_v1"
            ),
            pre_window_minutes=0,
            post_window_minutes=0,
        )
    )

    # =====================================================
    # RUN
    # =====================================================

    result = pipeline.run(
        phase1_df=phase1_df,
        phase2_df=phase2_df,
    )

    # =====================================================
    # FINAL INFORMATION
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PHASE 3 FINISHED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"  {output_dir / 'network_intelligence.parquet'}"
    )

    print(
        f"  {output_dir / 'network_nodes.parquet'}"
    )

    print(
        f"  {output_dir / 'network_edges.parquet'}"
    )

    print(
        f"  {output_dir / 'communities.parquet'}"
    )

    print(
        f"  {output_dir / 'phase3_summary.json'}"
    )


if __name__ == "__main__":
    main()