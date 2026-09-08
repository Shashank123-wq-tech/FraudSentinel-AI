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
# ARTIFACT PATHS
# ============================================================

FUSION_FILE = (
    ARTIFACT_ROOT
    / "fusion"
    / "fraudsentinel_unified_risk.csv"
)

FUSION_EVENTS_FILE = (
    ARTIFACT_ROOT
    / "fusion"
    / "fusion_events.csv"
)

PHASE2_WINDOWS_FILE = (
    ARTIFACT_ROOT
    / "phase2"
    / "phase2_windows.csv"
)

PHASE2_EVENTS_FILE = (
    ARTIFACT_ROOT
    / "phase2"
    / "phase2_events.csv"
)

XAI_FILE = (
    ARTIFACT_ROOT
    / "xai"
    / "unified_explanations.csv"
)

IMPACT_FORECAST_FILE = (
    ARTIFACT_ROOT
    / "impact"
    / "impact_forecast.csv"
)

IMPACT_SCENARIOS_FILE = (
    ARTIFACT_ROOT
    / "impact"
    / "impact_scenarios.csv"
)

RESPONSE_RECOMMENDATIONS_FILE = (
    ARTIFACT_ROOT
    / "response"
    / "response_recommendations.csv"
)

RESPONSE_ACTIONS_FILE = (
    ARTIFACT_ROOT
    / "response"
    / "response_actions.csv"
)


# ============================================================
# API CONFIG
# ============================================================

API_TITLE = "FraudSentinel AI Intelligence API"

API_DESCRIPTION = (
    "Defense-only intelligence API exposing transaction risk, "
    "fraud spikes, XAI evidence, financial impact, and response "
    "recommendations."
)

API_VERSION = "1.0.0"

FUSION_CHUNK_SIZE = 50_000

MAX_TRANSACTION_RESULTS = 100

MODEL_VERSION = "fraudsentinel_v1"