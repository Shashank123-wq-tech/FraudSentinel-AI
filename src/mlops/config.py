from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# CORE DIRECTORIES
# ============================================================

SRC_DIR = PROJECT_ROOT / "src"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


# ============================================================
# MLOPS DIRECTORIES
# ============================================================

MLOPS_DIR = PROJECT_ROOT / "src" / "mlops"

REGISTRY_DIR = ARTIFACTS_DIR / "registry"
MONITORING_DIR = ARTIFACTS_DIR / "monitoring"
RUNS_DIR = ARTIFACTS_DIR / "runs"


# ============================================================
# ARTIFACT DIRECTORIES
# ============================================================

PHASE1_DIR = ARTIFACTS_DIR / "phase1"
PHASE2_DIR = ARTIFACTS_DIR / "phase2"
FUSION_DIR = ARTIFACTS_DIR / "fusion"
XAI_DIR = ARTIFACTS_DIR / "xai"
IMPACT_DIR = ARTIFACTS_DIR / "impact"
RESPONSE_DIR = ARTIFACTS_DIR / "response"
VALIDATION_DIR = ARTIFACTS_DIR / "validation"


# ============================================================
# MODEL DIRECTORIES
# ============================================================

PHASE1_MODEL_DIR = MODELS_DIR / "phase1"
PHASE2_MODEL_DIR = MODELS_DIR / "phase2"
MODEL_REGISTRY_DIR = MODELS_DIR / "registry"


# ============================================================
# VERSION INFORMATION
# ============================================================

PROJECT_NAME = "FraudSentinel AI"

PIPELINE_VERSION = "fraudsentinel_pipeline_v1"

COMPONENT_VERSIONS = {
    "phase1": "phase1_xgboost_v1",
    "phase2": "phase2_temporal_v1",
    "fusion": "fusion_v1",
    "xai": "xai_v1",
    "impact": "impact_forecasting_v1",
    "response": "response_policy_v1",
    "copilot": "copilot_v1",
}


# ============================================================
# RUNTIME
# ============================================================

RANDOM_SEED = 42


# ============================================================
# MONITORING
# ============================================================

MONITORING_VERSION = "mlops_monitoring_v1"

DRIFT_THRESHOLD = 0.20

MAX_ERROR_RATE = 0.05

MIN_ARTIFACT_SIZE_BYTES = 1


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for directory in [
    REGISTRY_DIR,
    MONITORING_DIR,
    RUNS_DIR,
    LOGS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )