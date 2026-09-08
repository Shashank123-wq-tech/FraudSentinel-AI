from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FusionConfig:
    """
    Configuration for FraudSentinel AI Fusion Layer.
    """

    # ---------------------------------------------------------
    # Versions
    # ---------------------------------------------------------

    fusion_version: str = "fusion_v1"
    phase1_version: str = "phase1_xgboost_v1"
    phase2_version: str = "phase2_temporal_v1"

    random_seed: int = 42

    # ---------------------------------------------------------
    # Input artifacts
    # ---------------------------------------------------------

    phase1_path: Path = Path(
        "artifacts/phase1/risk_output/transaction_risk.csv"
    )

    phase2_windows_path: Path = Path(
        "artifacts/phase2/phase2_windows.csv"
    )

    phase2_events_path: Path = Path(
        "artifacts/phase2/phase2_events.csv"
    )

    # ---------------------------------------------------------
    # Output artifacts
    # ---------------------------------------------------------

    output_dir: Path = Path(
        "artifacts/fusion"
    )

    unified_risk_filename: str = "unified_risk.csv"
    events_filename: str = "fusion_events.csv"
    summary_filename: str = "fusion_summary.json"

    # ---------------------------------------------------------
    # Phase 1 / Phase 2 fusion weights
    # ---------------------------------------------------------

    phase1_weight: float = 0.55
    phase2_weight: float = 0.45

    # ---------------------------------------------------------
    # Phase 2 state multipliers
    # ---------------------------------------------------------

    state_multiplier_normal: float = 1.00
    state_multiplier_early_warning: float = 1.05
    state_multiplier_verified: float = 1.15
    state_multiplier_critical: float = 1.30

    # ---------------------------------------------------------
    # Unified risk bands
    # ---------------------------------------------------------

    low_threshold: float = 25.0
    medium_threshold: float = 50.0
    high_threshold: float = 75.0
    very_high_threshold: float = 90.0

    # ---------------------------------------------------------
    # Response thresholds
    # ---------------------------------------------------------

    monitor_threshold: float = 40.0
    investigate_threshold: float = 60.0
    enhanced_control_threshold: float = 75.0
    immediate_response_threshold: float = 90.0

    # ---------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------

    minimum_confidence: float = 0.05
    maximum_confidence: float = 0.99

    # ---------------------------------------------------------
    # Temporal alignment
    # ---------------------------------------------------------

    # Maximum age allowed for a Phase 2 window relative
    # to the transaction timestamp.
    max_window_age_minutes: int = 15

    # A transaction must not be matched to a Phase 2
    # window that starts after the transaction.
    allow_future_window: bool = False

    # ---------------------------------------------------------
    # Numeric safety
    # ---------------------------------------------------------

    expected_loss_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.phase1_weight <= 1.0:
            raise ValueError("phase1_weight must be between 0 and 1.")

        if not 0.0 <= self.phase2_weight <= 1.0:
            raise ValueError("phase2_weight must be between 0 and 1.")

        if abs(
            (self.phase1_weight + self.phase2_weight) - 1.0
        ) > 1e-9:
            raise ValueError(
                "phase1_weight + phase2_weight must equal 1."
            )

        if self.max_window_age_minutes < 0:
            raise ValueError(
                "max_window_age_minutes cannot be negative."
            )

        if self.minimum_confidence < 0:
            raise ValueError(
                "minimum_confidence cannot be negative."
            )

        if self.maximum_confidence > 1:
            raise ValueError(
                "maximum_confidence cannot exceed 1."
            )

        if self.minimum_confidence > self.maximum_confidence:
            raise ValueError(
                "minimum_confidence cannot exceed maximum_confidence."
            )

        if self.expected_loss_multiplier < 0:
            raise ValueError(
                "expected_loss_multiplier cannot be negative."
            )

