from pathlib import Path

from src.components.xai import (
    XAIConfig,
    run_xai_pipeline,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


INPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "fusion"
    / "fraudsentinel_unified_risk.csv"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "xai"
)


def main() -> None:

    print("\n" + "=" * 72)
    print("FRAUDSENTINEL AI — EXPLAINABLE AI")
    print("=" * 72)

    print(
        f"[INPUT]  {INPUT_PATH}"
    )

    print(
        f"[OUTPUT] {OUTPUT_DIR}"
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "\nFusion output does not exist:\n"
            f"{INPUT_PATH}\n\n"
            "Run the Fusion pipeline first."
        )

    config = XAIConfig(
        input_path=INPUT_PATH,
        output_dir=OUTPUT_DIR,
    )

    result = run_xai_pipeline(
        input_path=INPUT_PATH,
        output_dir=OUTPUT_DIR,
        config=config,
    )

    print("\n" + "=" * 72)
    print("XAI RESULT")
    print("=" * 72)

    if not result.empty:

        rows_processed = int(
            result.loc[
                0,
                "rows_processed",
            ]
        )

        mean_unified_risk = float(
            result.loc[
                0,
                "unified_risk_score",
            ]
        )

        max_unified_risk = float(
            result.loc[
                0,
                "max_unified_risk",
            ]
        )

        alert_count = int(
            result.loc[
                0,
                "alert_count",
            ]
        )

        active_event_rows = int(
            result.loc[
                0,
                "active_event_rows",
            ]
        )

        print(
            f"Rows processed       : "
            f"{rows_processed:,}"
        )

        print(
            f"Mean unified risk    : "
            f"{mean_unified_risk:.4f}"
        )

        print(
            f"Max unified risk     : "
            f"{max_unified_risk:.4f}"
        )

        print(
            f"Alert rows           : "
            f"{alert_count:,}"
        )

        print(
            f"Active event rows    : "
            f"{active_event_rows:,}"
        )

    print("\nArtifacts:")

    print(
        f"  {OUTPUT_DIR / config.transaction_output}"
    )

    print(
        f"  {OUTPUT_DIR / config.spike_output}"
    )

    print(
        f"  {OUTPUT_DIR / config.unified_output}"
    )

    print(
        f"  {OUTPUT_DIR / config.summary_output}"
    )

    print("\nXAI execution finished successfully.")


if __name__ == "__main__":
    main()


