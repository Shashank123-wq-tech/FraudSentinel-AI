import os
from pathlib import Path


# ============================================================
# CREATE FRAUDSENTINEL-AI REPOSITORY OUTSIDE EXISTING PROJECT
# ============================================================

# Current directory
current_dir = Path.cwd()

# IMPORTANT:
# This creates a completely separate folder next to the existing
# FraudSentinel-AI project.
#
# Example:
#
# Current/
# ├── FraudSentinel-AI/
# └── FraudSentinel-AI-Full-Repository/
#
# The existing FraudSentinel-AI folder is NOT modified.

project_root = current_dir / "FraudSentinel-AI-Full-Repository"


# ============================================================
# ALL FILES TO CREATE
# ============================================================

list_of_files = [

    # ========================================================
    # SRC ROOT
    # ========================================================

    "src/__init__.py",

    # ========================================================
    # PHASE 1 — TRANSACTION FRAUD INTELLIGENCE
    # ========================================================

    "src/components/__init__.py",

    "src/components/phase1/__init__.py",
    "src/components/phase1/data_ingestion.py",
    "src/components/phase1/data_validation.py",
    "src/components/phase1/feature_engineering.py",
    "src/components/phase1/feature_selection.py",
    "src/components/phase1/model_trainer.py",
    "src/components/phase1/model_calibration.py",
    "src/components/phase1/model_evaluation.py",
    "src/components/phase1/threshold_optimizer.py",
    "src/components/phase1/prediction.py",
    "src/components/phase1/model_registry.py",

    # ========================================================
    # PHASE 2 — TEMPORAL FRAUD-SPIKE INTELLIGENCE
    # ========================================================

    "src/components/phase2/__init__.py",

    # Phase 1 → Phase 2 interface
    "src/components/phase2/risk_output_builder.py",

    # Merchant / time aggregation
    "src/components/phase2/temporal_aggregation.py",
    "src/components/phase2/temporal_features.py",

    # Merchant baseline
    "src/components/phase2/merchant_baseline.py",

    # Anomaly detection
    "src/components/phase2/rolling_anomaly.py",
    "src/components/phase2/ewma_detector.py",
    "src/components/phase2/cusum_detector.py",

    # Multi-scale detection
    "src/components/phase2/multi_window_detector.py",

    # Spike intelligence
    "src/components/phase2/spike_scoring.py",
    "src/components/phase2/persistence_detector.py",
    "src/components/phase2/acceleration_detector.py",

    # Verification + event creation
    "src/components/phase2/spike_verification.py",
    "src/components/phase2/event_builder.py",

    # Evaluation
    "src/components/phase2/spike_evaluation.py",

    # ========================================================
    # PHASE 3 — LINK / GRAPH ANALYSIS
    # ========================================================

    "src/components/phase3/__init__.py",
    "src/components/phase3/graph_builder.py",
    "src/components/phase3/entity_resolution.py",
    "src/components/phase3/link_features.py",
    "src/components/phase3/network_anomaly.py",
    "src/components/phase3/coordinated_attack.py",

    # ========================================================
    # PHASE 4 — FRAUD DNA / EMERGING PATTERN INTELLIGENCE
    # ========================================================

    "src/components/phase4/__init__.py",
    "src/components/phase4/fraud_dna.py",
    "src/components/phase4/pattern_clustering.py",
    "src/components/phase4/pattern_evolution.py",
    "src/components/phase4/emerging_pattern.py",

    # ========================================================
    # COMMON COMPONENTS
    # ========================================================

    "src/common/__init__.py",
    "src/common/data_quality.py",
    "src/common/feature_utils.py",
    "src/common/time_utils.py",
    "src/common/entity_utils.py",
    "src/common/metrics.py",
    "src/common/serialization.py",

    # ========================================================
    # CONFIGURATION
    # ========================================================

    "src/configuration/__init__.py",
    "src/configuration/database.py",
    "src/configuration/aws_connection.py",
    "src/configuration/settings.py",

    # ========================================================
    # CLOUD STORAGE
    # ========================================================

    "src/cloud_storage/__init__.py",
    "src/cloud_storage/aws_storage.py",

    # ========================================================
    # DATA ACCESS
    # ========================================================

    "src/data_access/__init__.py",
    "src/data_access/transaction_repository.py",
    "src/data_access/risk_repository.py",
    "src/data_access/event_repository.py",

    # ========================================================
    # ENTITIES
    # ========================================================

    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "src/entity/artifact_entity.py",
    "src/entity/transaction_entity.py",
    "src/entity/risk_entity.py",
    "src/entity/merchant_entity.py",
    "src/entity/spike_event_entity.py",

    # ========================================================
    # EXCEPTION
    # ========================================================

    "src/exception/__init__.py",
    "src/exception/exception.py",

    # ========================================================
    # LOGGER
    # ========================================================

    "src/logger/__init__.py",
    "src/logger/logger.py",

    # ========================================================
    # PIPELINES
    # ========================================================

    "src/pipeline/__init__.py",
    "src/pipeline/phase1_training_pipeline.py",
    "src/pipeline/phase1_prediction_pipeline.py",
    "src/pipeline/phase2_spike_pipeline.py",
    "src/pipeline/end_to_end_pipeline.py",

    # ========================================================
    # API
    # ========================================================

    "src/api/__init__.py",
    "src/api/routes.py",
    "src/api/schemas.py",
    "src/api/app.py",

    # ========================================================
    # UTILITIES
    # ========================================================

    "src/utils/__init__.py",
    "src/utils/main_utils.py",
    "src/utils/file_utils.py",
    "src/utils/config_utils.py",

    # ========================================================
    # CONFIG FILES
    # ========================================================

    "config/model.yaml",
    "config/phase1.yaml",
    "config/phase2.yaml",
    "config/thresholds.yaml",
    "config/schema.yaml",

    # ========================================================
    # ARTIFACT DIRECTORIES
    # ========================================================

    "artifacts/phase1/models/.gitkeep",
    "artifacts/phase1/calibration/.gitkeep",
    "artifacts/phase1/predictions/.gitkeep",
    "artifacts/phase1/evaluations/.gitkeep",

    "artifacts/phase2/temporal_features/.gitkeep",
    "artifacts/phase2/baselines/.gitkeep",
    "artifacts/phase2/anomaly_scores/.gitkeep",
    "artifacts/phase2/spike_events/.gitkeep",

    # ========================================================
    # DATA DIRECTORIES
    # ========================================================

    "data/raw/.gitkeep",
    "data/interim/.gitkeep",
    "data/processed/phase1/.gitkeep",
    "data/processed/phase2/.gitkeep",
    "data/external/.gitkeep",

    # ========================================================
    # NOTEBOOK DIRECTORIES
    # ========================================================

    "notebooks/phase1/.gitkeep",
    "notebooks/phase2/.gitkeep",

    # ========================================================
    # TESTS
    # ========================================================

    "tests/unit/phase1/.gitkeep",
    "tests/unit/phase2/.gitkeep",

    "tests/integration/test_phase1_pipeline.py",
    "tests/integration/test_phase2_pipeline.py",

    "tests/test_end_to_end.py",

    # ========================================================
    # DEPLOYMENT
    # ========================================================

    "deployment/docker/Dockerfile",
    "deployment/docker/.dockerignore",

    "deployment/kubernetes/deployment.yaml",
    "deployment/kubernetes/service.yaml",
    "deployment/kubernetes/configmap.yaml",

    "deployment/monitoring/prometheus.yaml",

    # ========================================================
    # DASHBOARD
    # ========================================================

    "dashboard/app.py",
    "dashboard/components/.gitkeep",
    "dashboard/assets/.gitkeep",

    # ========================================================
    # SCRIPTS
    # ========================================================

    "scripts/train_phase1.py",
    "scripts/run_phase1_prediction.py",
    "scripts/run_phase2.py",
    "scripts/evaluate_system.py",

    # ========================================================
    # CI/CD
    # ========================================================

    ".github/workflows/ci.yml",
    ".github/workflows/cd.yml",

    # ========================================================
    # DOCUMENTATION
    # ========================================================

    "docs/architecture.md",
    "docs/phase1.md",
    "docs/phase2.md",
    "docs/metrics.md",
    "docs/model_card.md",

    # ========================================================
    # ROOT FILES
    # ========================================================

    "app.py",
    "demo.py",

    "requirements.txt",
    "requirements-dev.txt",

    "Dockerfile",
    ".dockerignore",
    ".gitignore",

    "setup.py",
    "pyproject.toml",

    "README.md",
]


# ============================================================
# CREATE ROOT DIRECTORY
# ============================================================

project_root.mkdir(parents=True, exist_ok=True)


# ============================================================
# CREATE ALL FILES
# ============================================================

created_files = 0
existing_files = 0

for relative_path in list_of_files:

    filepath = project_root / relative_path

    # Create parent directories
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Create empty file only if it does not exist
    if not filepath.exists():

        filepath.touch()

        created_files += 1

    else:

        existing_files += 1


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("FraudSentinel-AI Repository Created Successfully")
print("=" * 70)

print()
print(f"Repository location:")
print(project_root)

print()
print(f"Total files requested : {len(list_of_files)}")
print(f"Files created         : {created_files}")
print(f"Files already present : {existing_files}")

print()
print("IMPORTANT:")
print("This repository was created as a separate folder.")
print("The existing 'FraudSentinel-AI' project was NOT modified.")

print()
print("Repository root:")
print(project_root)

print()
print("=" * 70)