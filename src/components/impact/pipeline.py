from pathlib import Path

from .config import ImpactConfig

from .loader import (
    load_phase2_events,
)

from .impact import (
    build_impact_forecast,
    build_impact_scenarios,
    build_summary,
    save_summary,
)


class ImpactPipeline:

    def __init__(
        self,
        config: ImpactConfig | None = None,
    ):

        self.config = (
            config
            if config is not None
            else ImpactConfig()
        )

    # =========================================================
    # RUN
    # =========================================================

    def run(self):

        self.config.create_output_dir()

        # -----------------------------------------------------
        # LOAD EVENTS
        # -----------------------------------------------------

        events = load_phase2_events(
            self.config.phase2_events_path
        )

        if events.empty:

            raise ValueError(
                "Phase 2 event artifact is empty."
            )

        # -----------------------------------------------------
        # FORECAST
        # -----------------------------------------------------

        (
            features,
            forecast,
        ) = build_impact_forecast(
            events,
            self.config,
        )

        # -----------------------------------------------------
        # SCENARIOS
        # -----------------------------------------------------

        scenarios = (
            build_impact_scenarios(
                forecast,
                self.config,
            )
        )

        # -----------------------------------------------------
        # SUMMARY
        # -----------------------------------------------------

        summary = build_summary(
            features,
            forecast,
            scenarios,
            self.config,
        )

        # -----------------------------------------------------
        # SAVE FORECAST
        # -----------------------------------------------------

        forecast_path = Path(
            self.config.forecast_output
        )

        forecast_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        forecast.to_csv(
            forecast_path,
            index=False,
        )

        # -----------------------------------------------------
        # SAVE SCENARIOS
        # -----------------------------------------------------

        scenario_path = Path(
            self.config.scenario_output
        )

        scenario_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        scenarios.to_csv(
            scenario_path,
            index=False,
        )

        # -----------------------------------------------------
        # SAVE SUMMARY
        # -----------------------------------------------------

        save_summary(
            summary,
            self.config.summary_output,
        )

        return {
            "features":
                features,

            "forecast":
                forecast,

            "scenarios":
                scenarios,

            "summary":
                summary,
        }
