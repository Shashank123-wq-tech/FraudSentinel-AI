from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# ARTIFACT ROOT
# ============================================================

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"


# ============================================================
# INPUT ARTIFACTS
# ============================================================

PHASE1_RISK = (
    ARTIFACT_ROOT
    / "phase1"
    / "risk_output"
    / "transaction_risk.csv"
)

PHASE2_WINDOWS = (
    ARTIFACT_ROOT
    / "phase2"
    / "phase2_windows.csv"
)

PHASE2_EVENTS = (
    ARTIFACT_ROOT
    / "phase2"
    / "phase2_events.csv"
)

FUSION = (
    ARTIFACT_ROOT
    / "fusion"
    / "fraudsentinel_unified_risk.csv"
)

XAI_TRANSACTION = (
    ARTIFACT_ROOT
    / "xai"
    / "transaction_explanations.csv"
)

XAI_SPIKE = (
    ARTIFACT_ROOT
    / "xai"
    / "spike_explanations.csv"
)

XAI_UNIFIED = (
    ARTIFACT_ROOT
    / "xai"
    / "unified_explanations.csv"
)

IMPACT_FORECAST = (
    ARTIFACT_ROOT
    / "impact"
    / "impact_forecast.csv"
)

IMPACT_SCENARIOS = (
    ARTIFACT_ROOT
    / "impact"
    / "impact_scenarios.csv"
)

RESPONSE_RECOMMENDATIONS = (
    ARTIFACT_ROOT
    / "response"
    / "response_recommendations.csv"
)

RESPONSE_ACTIONS = (
    ARTIFACT_ROOT
    / "response"
    / "response_actions.csv"
)


# ============================================================
# DASHBOARD SETTINGS
# ============================================================

FUSION_CHUNK_SIZE = 50_000
PHASE1_CHUNK_SIZE = 50_000
XAI_CHUNK_SIZE = 50_000

TOP_N = 15

TIME_SERIES_FREQ = "15min"


# ============================================================
# COLUMN DEFINITIONS
# ============================================================

FUSION_REQUIRED = [
    "transaction_id",
    "timestamp",
    "merchant_id",
    "card_id",
    "amount",
    "fraud_probability",
    "phase1_score",
    "phase2_score",
    "spike_state",
    "tas",
    "fas",
    "unified_risk_score",
    "expected_fraud_amount",
    "event_id",
    "response_action",
    "response_priority",
    "alert_flag",
]


# ============================================================
# RISK ORDER
# ============================================================

RISK_BAND_ORDER = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
]

SPIKE_STATE_ORDER = [
    "NORMAL",
    "EARLY_WARNING",
    "VERIFIED_FRAUD_SPIKE",
    "CRITICAL_ACTIVE_SPIKE",
]

RESPONSE_PRIORITY_ORDER = [
    "P3_LOW",
    "P2_MEDIUM",
    "P1_HIGH",
    "P0_CRITICAL",
]