from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ValidationConfig:

    project_name: str = "FraudSentinel AI"

    project_root: Path = Path(
        "C:/FraudSentinel-AI"
    )

    output_dir: Path = Path(
        "C:/FraudSentinel-AI/artifacts/validation"
    )

    required_artifacts: dict[str, Path] = field(
        default_factory=lambda: {

            "phase1": Path(
                "artifacts/phase1/risk_output/"
                "transaction_risk.csv"
            ),

            "phase2_windows": Path(
                "artifacts/phase2/"
                "phase2_windows.csv"
            ),

            "phase2_events": Path(
                "artifacts/phase2/"
                "phase2_events.csv"
            ),

            "fusion": Path(
                "artifacts/fusion/"
                "fraudsentinel_unified_risk.csv"
            ),

            "xai_transaction": Path(
                "artifacts/xai/"
                "transaction_explanations.csv"
            ),

            "xai_spike": Path(
                "artifacts/xai/"
                "spike_explanations.csv"
            ),

            "xai_unified": Path(
                "artifacts/xai/"
                "unified_explanations.csv"
            ),

            "impact_forecast": Path(
                "artifacts/impact/"
                "impact_forecast.csv"
            ),

            "impact_scenarios": Path(
                "artifacts/impact/"
                "impact_scenarios.csv"
            ),

            "response_recommendations": Path(
                "artifacts/response/"
                "response_recommendations.csv"
            ),

            "response_actions": Path(
                "artifacts/response/"
                "response_actions.csv"
            ),
        }
    )

    min_rows: int = 1

    probability_columns: tuple[str, ...] = (
        "fraud_probability",
        "fusion_confidence",
        "phase2_confidence",
    )

    risk_score_columns: tuple[str, ...] = (
        "risk_score",
        "unified_risk_score",
        "spike_score",
    )