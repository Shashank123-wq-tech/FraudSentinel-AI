from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# INPUT ARTIFACTS
# ============================================================

FUSION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "fusion"
    / "fraudsentinel_unified_risk.csv"
)

IMPACT_FORECAST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "impact"
    / "impact_forecast.csv"
)

IMPACT_SCENARIO_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "impact"
    / "impact_scenarios.csv"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "response"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RECOMMENDATION_PATH = (
    OUTPUT_DIR
    / "response_recommendations.csv"
)

ACTION_PATH = (
    OUTPUT_DIR
    / "response_actions.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "response_summary.json"
)


# ============================================================
# MODEL / POLICY VERSION
# ============================================================

RESPONSE_MODEL_VERSION = "response_policy_v1"


# ============================================================
# THRESHOLDS
# ============================================================

UNIFIED_RISK_HIGH = 70.0
UNIFIED_RISK_MEDIUM = 45.0

SPIKE_CRITICAL = "CRITICAL_ACTIVE_SPIKE"
SPIKE_VERIFIED = "VERIFIED_FRAUD_SPIKE"
SPIKE_EARLY = "EARLY_WARNING"
SPIKE_NORMAL = "NORMAL"


# ============================================================
# IMPACT THRESHOLDS
# ============================================================

HIGH_IMPACT_30M = 2500.0
HIGH_IMPACT_60M = 4000.0

MEDIUM_IMPACT_30M = 1000.0
MEDIUM_IMPACT_60M = 2000.0


# ============================================================
# RESPONSE PRIORITY
# ============================================================

PRIORITY_CRITICAL = "P0_CRITICAL"
PRIORITY_HIGH = "P1_HIGH"
PRIORITY_MEDIUM = "P2_MEDIUM"
PRIORITY_LOW = "P3_LOW"


# ============================================================
# RESPONSE ACTIONS
# ============================================================

ACTION_CONTAIN = "AGGRESSIVE_INTERVENTION"
ACTION_RESTRICT = "MODERATE_INTERVENTION"
ACTION_MONITOR = "MONITOR_AND_REASSESS"
ACTION_NO_ACTION = "NO_ACTION"


# ============================================================
# CONTROL TYPES
# ============================================================

CONTROL_CRITICAL = (
    "Temporarily tighten transaction controls and route "
    "high-risk transactions for additional verification"
)

CONTROL_MODERATE = (
    "Increase transaction monitoring and apply additional "
    "risk-based verification to suspicious transactions"
)

CONTROL_MONITOR = (
    "Continue enhanced monitoring and reassess the event "
    "on the next intelligence cycle"
)

CONTROL_NONE = (
    "Maintain normal transaction controls"
)
