import json

from src.components.impact.config import (
    ImpactConfig,
)

from src.components.impact.pipeline import (
    ImpactPipeline,
)


def main():

    # =========================================================
    # CONFIG
    # =========================================================

    config = ImpactConfig()

    # =========================================================
    # PIPELINE
    # =========================================================

    pipeline = ImpactPipeline(
        config=config
    )

    result = pipeline.run()

    summary = result[
        "summary"
    ]

    # =========================================================
    # HEADER
    # =========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "FRAUDSENTINEL AI — "
        "LOSS / IMPACT FORECASTING"
    )

    print(
        "=" * 70
    )

    # =========================================================
    # GENERAL
    # =========================================================

    print(
        f"Events processed       : "
        f"{summary['events_processed']:,}"
    )

    print(
        f"Forecast rows          : "
        f"{summary['forecast_rows']:,}"
    )

    print(
        f"Scenario rows          : "
        f"{summary['scenario_rows']:,}"
    )

    print(
        f"Current expected loss  : "
        f"{summary['current_expected_loss']:,.2f}"
    )

    print(
        f"Expected fraud count   : "
        f"{summary['current_expected_fraud_count']:,.2f}"
    )

    print(
        f"Forecast confidence    : "
        f"{summary['mean_forecast_confidence']:.4f}"
    )

    print(
        f"Maximum spike score    : "
        f"{summary['max_spike_score']:.4f}"
    )

    # =========================================================
    # HORIZONS
    # =========================================================

    print()

    print(
        "-" * 70
    )

    print(
        "FORECAST HORIZONS"
    )

    print(
        "-" * 70
    )

    for horizon, values in (
        summary[
            "horizons"
        ].items()
    ):

        print(
            f"{horizon:>4} min | "
            f"Loss: "
            f"{values['total_projected_loss']:,.2f} | "
            f"Lower: "
            f"{values['lower_bound']:,.2f} | "
            f"Upper: "
            f"{values['upper_bound']:,.2f}"
        )

    # =========================================================
    # SCENARIOS
    # =========================================================

    print()

    print(
        "-" * 70
    )

    print(
        "SCENARIOS"
    )

    print(
        "-" * 70
    )

    for scenario, values in (
        summary[
            "scenarios"
        ].items()
    ):

        print(
            f"{scenario:<24} "
            f"Loss: "
            f"{values['projected_loss']:,.2f} | "
            f"Prevented: "
            f"{values['loss_prevented']:,.2f}"
        )

    # =========================================================
    # ARTIFACTS
    # =========================================================

    print()

    print(
        "-" * 70
    )

    print(
        "ARTIFACTS"
    )

    print(
        "-" * 70
    )

    print(
        f"Forecast : "
        f"{config.forecast_output}"
    )

    print(
        f"Scenarios: "
        f"{config.scenario_output}"
    )

    print(
        f"Summary  : "
        f"{config.summary_output}"
    )

    # =========================================================
    # COMPLETE
    # =========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "IMPACT FORECASTING COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
