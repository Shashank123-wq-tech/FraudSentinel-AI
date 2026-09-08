from dataclasses import dataclass


@dataclass(frozen=True)
class Phase2Config:
    """
    Configuration for FraudSentinel AI Phase 2.

    Phase 2 objective:
        Detect statistically and temporally abnormal increases
        in merchant-level expected fraud exposure.

    The primary decision horizon is 15 minutes.
    5-minute and 60-minute horizons provide corroborating evidence.
    """

    # ============================================================
    # TIME WINDOWS
    # ============================================================

    FAST_WINDOW_MINUTES: int = 5
    PRIMARY_WINDOW_MINUTES: int = 15
    SLOW_WINDOW_MINUTES: int = 60

    # ============================================================
    # BASELINE
    # ============================================================

    # 24 hours of 15-minute windows.
    BASELINE_WINDOWS: int = 96

    # Minimum historical windows before merchant-specific
    # statistics become reliable.
    MIN_HISTORY_WINDOWS: int = 24

    # Robust scale floor.
    SCALE_FLOOR: float = 1e-8

    # Probability smoothing.
    #
    # This prevents zero expected-fraud exposure from causing
    # infinite ratios.
    RATE_SMOOTHING: float = 0.25

    # ============================================================
    # ROBUST ANOMALY TRANSFORMATION
    # ============================================================

    # z -> bounded [0,1] anomaly.
    ANOMALY_K: float = 1.25

    # ============================================================
    # STATISTICAL SCORE WEIGHTS
    # ============================================================

    WEIGHT_RISK_RATE: float = 0.40
    WEIGHT_FRAUD_COUNT: float = 0.25
    WEIGHT_FRAUD_AMOUNT: float = 0.25
    WEIGHT_VOLUME: float = 0.10

    # ============================================================
    # TEMPORAL SCORE WEIGHTS
    # ============================================================

    WEIGHT_EWMA: float = 0.40
    WEIGHT_CUSUM: float = 0.30
    WEIGHT_ACCELERATION: float = 0.30

    # ============================================================
    # BREADTH
    # ============================================================

    # Saturation constants.
    #
    # They do NOT create fraud evidence.
    # They only describe how broad the activity is.
    CARD_SATURATION: float = 3.0
    NEW_CARD_SATURATION: float = 2.0

    # ============================================================
    # TAS
    # ============================================================

    WEIGHT_STATISTICAL: float = 0.50
    WEIGHT_TEMPORAL: float = 0.30
    WEIGHT_BREADTH: float = 0.20

    # ============================================================
    # FAS
    # ============================================================

    WEIGHT_TAS: float = 0.60
    WEIGHT_MATERIALITY: float = 0.20
    WEIGHT_COORDINATION: float = 0.20

    # ============================================================
    # DETECTION THRESHOLDS
    # ============================================================

    # These are deliberately separated from Phase-1 threshold.
    #
    # Phase 1 threshold = transaction classification.
    # Phase 2 threshold = temporal spike state.
    CANDIDATE_TAS: float = 35.0
    VERIFIED_TAS: float = 50.0
    CRITICAL_TAS: float = 70.0

    CANDIDATE_FAS: float = 35.0
    VERIFIED_FAS: float = 50.0
    CRITICAL_FAS: float = 70.0

    # ============================================================
    # ACTIVITY REQUIREMENTS
    # ============================================================

    MIN_TRANSACTIONS_CANDIDATE: int = 2
    MIN_UNIQUE_CARDS_COORDINATED: int = 2

    # ============================================================
    # EWMA
    # ============================================================

    EWMA_ALPHA: float = 0.25

    # ============================================================
    # CUSUM
    # ============================================================

    CUSUM_DRIFT: float = 0.50
    CUSUM_CAP: float = 5.0

    # ============================================================
    # GAP HANDLING
    # ============================================================

    MAX_GAP_MINUTES: int = 60

    # ============================================================
    # EVENT BUILDING
    # ============================================================

    EVENT_MAX_GAP_MINUTES: int = 60

    # ============================================================
    # VERSION
    # ============================================================

    DETECTOR_VERSION: str = "phase2_fraud_spike_v2.0"
    
    # ============================================================
# PHASE 2 — STATE CALIBRATION
# ============================================================

PHASE2_CANDIDATE_SCORE_THRESHOLD = 55.0
PHASE2_VERIFIED_SCORE_THRESHOLD = 65.0
PHASE2_CRITICAL_SCORE_THRESHOLD = 75.0

# Activity requirements
PHASE2_CANDIDATE_MIN_TRANSACTIONS = 1
PHASE2_VERIFIED_MIN_TRANSACTIONS = 2
PHASE2_CRITICAL_MIN_TRANSACTIONS = 2

# Persistence requirements
PHASE2_VERIFIED_PERSISTENCE_WINDOWS = 2
PHASE2_CRITICAL_PERSISTENCE_WINDOWS = 2

PHASE2_VERIFIED_PERSISTENCE_MINUTES = 30.0
PHASE2_CRITICAL_PERSISTENCE_MINUTES = 45.0

# Supporting evidence
PHASE2_VERIFIED_MIN_STATISTICAL_EVIDENCE = 0.35
PHASE2_VERIFIED_MIN_TEMPORAL_EVIDENCE = 0.20

PHASE2_CRITICAL_MIN_STATISTICAL_EVIDENCE = 0.55
PHASE2_CRITICAL_MIN_TEMPORAL_EVIDENCE = 0.30

# Breadth / coordination
PHASE2_CRITICAL_MIN_UNIQUE_CARDS = 2
PHASE2_CRITICAL_MIN_COORDINATION_SCORE = 0.45