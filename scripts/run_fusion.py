from pathlib import Path

from src.components.fusion import (
    FusionConfig,
    run_fusion_pipeline,
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# INPUT ARTIFACTS
# ============================================================

PHASE1_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase1"
    / "risk_output"
    / "transaction_risk.csv"
)

PHASE2_WINDOWS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase2"
    / "phase2_windows.csv"
)

PHASE2_EVENTS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase2"
    / "phase2_events.csv"
)


# ============================================================
# OUTPUT ARTIFACTS
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "fusion"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "fraudsentinel_unified_risk.csv"
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("FraudSentinel AI — Risk Fusion Layer")
    print("=" * 70)

    print("\n[INPUT] Phase 1:")
    print(PHASE1_PATH)

    print("\n[INPUT] Phase 2 Windows:")
    print(PHASE2_WINDOWS_PATH)

    print("\n[INPUT] Phase 2 Events:")
    print(PHASE2_EVENTS_PATH)

    print("\n[OUTPUT]:")
    print(OUTPUT_PATH)

    # --------------------------------------------------------
    # Validate input artifacts before execution
    # --------------------------------------------------------

    for path, name in [
        (PHASE1_PATH, "Phase 1 transaction risk"),
        (PHASE2_WINDOWS_PATH, "Phase 2 windows"),
        (PHASE2_EVENTS_PATH, "Phase 2 events"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"\n{name} artifact not found:\n{path}"
            )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = FusionConfig()

    

    # --------------------------------------------------------
    # Run Fusion
    # --------------------------------------------------------

    result = run_fusion_pipeline(
        phase1_path=PHASE1_PATH,
        phase2_path=PHASE2_WINDOWS_PATH,
        phase2_events_path=PHASE2_EVENTS_PATH,
        output_path=OUTPUT_PATH,
        config=config,
    )

    # --------------------------------------------------------
    # Execution summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FUSION PIPELINE COMPLETED")
    print("=" * 70)

    print(f"\nRows processed       : {len(result):,}")

    if "unified_risk_score" in result.columns:
        print(
            "Mean unified risk   : "
            f"{result['unified_risk_score'].mean():.4f}"
        )

        print(
            "Max unified risk    : "
            f"{result['unified_risk_score'].max():.4f}"
        )

    if "alert_flag" in result.columns:
        alert_count = (
            result["alert_flag"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        print(
            "Alerts generated    : "
            f"{alert_count:,}"
        )

    print("\nOutput:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()


