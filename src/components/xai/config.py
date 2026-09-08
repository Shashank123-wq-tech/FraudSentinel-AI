from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class XAIConfig:
    """
    Configuration for the FraudSentinel AI XAI layer.

    The XAI layer is deterministic. These thresholds control
    explanation labels, not fraud predictions.
    """

    input_path: Path = Path(
        "artifacts/fusion/fraudsentinel_unified_risk.csv"
    )

    output_dir: Path = Path(
        "artifacts/xai"
    )

    transaction_output: str = "transaction_explanations.csv"
    spike_output: str = "spike_explanations.csv"
    unified_output: str = "unified_explanations.csv"
    summary_output: str = "xai_summary.json"

    # Transaction-level explanation thresholds
    high_probability_threshold: float = 0.70
    very_high_probability_threshold: float = 0.90

    high_risk_score_threshold: float = 70.0
    critical_risk_score_threshold: float = 85.0

    # Phase-2 explanation thresholds
    high_spike_score_threshold: float = 57.15
    critical_spike_score_threshold: float = 75.15

    high_evidence_threshold: float = 0.50
    strong_evidence_threshold: float = 0.70

    # Financial materiality
    materiality_threshold: float = 0.50

    # Top contributors
    max_reasons: int = 5

    # Numeric precision for generated explanations
    probability_decimals: int = 4
    score_decimals: int = 2
    amount_decimals: int = 2

    # Explanation version
    explanation_version: str = "xai_v1"

    # State priority
    state_priority: Dict[str, int] = field(
        default_factory=lambda: {
            "CRITICAL_ACTIVE_SPIKE": 4,
            "VERIFIED_FRAUD_SPIKE": 3,
            "EARLY_WARNING": 2,
            "NORMAL": 1,
        }
    )

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @property
    def transaction_output_path(self) -> Path:
        return self.output_dir / self.transaction_output

    @property
    def spike_output_path(self) -> Path:
        return self.output_dir / self.spike_output

    @property
    def unified_output_path(self) -> Path:
        return self.output_dir / self.unified_output

    @property
    def summary_output_path(self) -> Path:
        return self.output_dir / self.summary_output
