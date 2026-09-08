from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImpactConfig:

    # =========================================================
    # INPUT ARTIFACTS
    # =========================================================

    phase2_events_path: str = (
        "artifacts/phase2/phase2_events.csv"
    )

    phase2_windows_path: str = (
        "artifacts/phase2/phase2_windows.csv"
    )

    fusion_path: str = (
        "artifacts/fusion/fraudsentinel_unified_risk.csv"
    )

    # =========================================================
    # OUTPUT ARTIFACTS
    # =========================================================

    output_dir: str = (
        "artifacts/impact"
    )

    forecast_output: str = (
        "artifacts/impact/impact_forecast.csv"
    )

    scenario_output: str = (
        "artifacts/impact/impact_scenarios.csv"
    )

    summary_output: str = (
        "artifacts/impact/impact_summary.json"
    )

    # =========================================================
    # FORECAST HORIZONS
    # =========================================================

    horizons_minutes: tuple = (
        15,
        30,
        60,
    )

    # =========================================================
    # FORECAST PARAMETERS
    # =========================================================

    min_exposure: float = 1e-6

    max_growth_multiplier: float = 10.0

    min_history_windows: int = 6

    velocity_alpha: float = 0.35

    # =========================================================
    # SCENARIO PARAMETERS
    # =========================================================

    moderate_loss_retention: float = 0.60

    aggressive_loss_retention: float = 0.30

    # =========================================================
    # GOVERNANCE
    # =========================================================

    model_version: str = (
        "impact_forecasting_v1"
    )

    random_seed: int = 42

    def create_output_dir(self) -> Path:

        path = Path(
            self.output_dir
        )

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path
