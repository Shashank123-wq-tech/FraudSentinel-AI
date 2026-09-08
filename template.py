import os
from pathlib import Path


project_name = "src"


list_of_files = [

    # ============================================================
    # SRC ROOT
    # ============================================================

    f"{project_name}/__init__.py",

    # ============================================================
    # PHASE 1 — TRANSACTION FRAUD INTELLIGENCE
    # ============================================================

    f"{project_name}/components/__init__.py",

    f"{project_name}/components/phase1/__init__.py",
    f"{project_name}/components/phase1/data_ingestion.py",
    f"{project_name}/components/phase1/data_validation.py",
    f"{project_name}/components/phase1/feature_engineering.py",
    f"{project_name}/components/phase1/feature_selection.py",
    f"{project_name}/components/phase1/model_trainer.py",
    f"{project_name}/components/phase1/model_calibration.py",
    f"{project_name}/components/phase1/model_evaluation.py",
    f"{project_name}/components/phase1/threshold_optimizer.py",
    f"{project_name}/components/phase1/prediction.py",
    f"{project_name}/components/phase1/model_registry.py",

    # ============================================================
    # PHASE 2 — TEMPORAL FRAUD-SPIKE INTELLIGENCE
    # ============================================================

    f"{project_name}/components/phase2/__init__.py",

    # Phase-1 → Phase-2 interface
    f"{project_name}/components/phase2/risk_output_builder.py",

    # Merchant/time aggregation
    f"{project_name}/components/phase2/temporal_aggregation.py",
    f"{project_name}/components/phase2/temporal_features.py",

    # Merchant baseline
    f"{project_name}/components/phase2/merchant_baseline.py",

    # Anomaly detection
    f"{project_name}/components/phase2/rolling_anomaly.py",
    f"{project_name}/components/phase2/ewma_detector.py",
    f"{project_name}/components/phase2/cusum_detector.py",

    # Multi-scale detection
    f"{project_name}/components/phase2/multi_window_detector.py",

    # Spike intelligence
    f"{project_name}/components/phase2/spike_scoring.py",
    f"{project_name}/components/phase2/persistence_detector.py",
    f"{project_name}/components/phase2/acceleration_detector.py",

    # Verification + event creation
    f"{project_name}/components/phase2/spike_verification.py",
    f"{project_name}/components/phase2/event_builder.py",

    # Evaluation
    f"{project_name}/components/phase2/spike_evaluation.py",

    # ============================================================
    # PHASE 3 — LINK / GRAPH ANALYSIS
    # ============================================================

    f"{project_name}/components/phase3/__init__.py",
    f"{project_name}/components/phase3/graph_builder.py",
    f"{project_name}/components/phase3/entity_resolution.py",
    f"{project_name}/components/phase3/link_features.py",
    f"{project_name}/components/phase3/network_anomaly.py",
    f"{project_name}/components/phase3/coordinated_attack.py",

    # ============================================================
    # PHASE 4 — FRAUD DNA / EMERGING PATTERN INTELLIGENCE
    # ============================================================

    f"{project_name}/components/phase4/__init__.py",
    f"{project_name}/components/phase4/fraud_dna.py",
    f"{project_name}/components/phase4/pattern_clustering.py",
    f"{project_name}/components/phase4/pattern_evolution.py",
    f"{project_name}/components/phase4/emerging_pattern.py",

    # ============================================================
    # COMMON COMPONENTS
    # ============================================================

    f"{project_name}/common/__init__.py",
    f"{project_name}/common/data_quality.py",
    f"{project_name}/common/feature_utils.py",
    f"{project_name}/common/time_utils.py",
    f"{project_name}/common/entity_utils.py",
    f"{project_name}/common/metrics.py",
    f"{project_name}/common/serialization.py",

    # ============================================================
    # CONFIGURATION
    # ============================================================

    f"{project_name}/configuration/__init__.py",
    f"{project_name}/configuration/database.py",
    f"{project_name}/configuration/aws_connection.py",
    f"{project_name}/configuration/settings.py",

    # ============================================================
    # CLOUD STORAGE
    # ============================================================

    f"{project_name}/cloud_storage/__init__.py",
    f"{project_name}/cloud_storage/aws_storage.py",

    # ============================================================
    # DATA ACCESS
    # ============================================================

    f"{project_name}/data_access/__init__.py",
    f"{project_name}/data_access/transaction_repository.py",
    f"{project_name}/data_access/risk_repository.py",
    f"{project_name}/data_access/event_repository.py",

    # ============================================================
    # ENTITIES
    # ============================================================

    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/entity/transaction_entity.py",
    f"{project_name}/entity/risk_entity.py",
    f"{project_name}/entity/merchant_entity.py",
    f"{project_name}/entity/spike_event_entity.py",

    # ============================================================
    # EXCEPTION
    # ============================================================

    f"{project_name}/exception/__init__.py",
    f"{project_name}/exception/exception.py",

    # ============================================================
    # LOGGER
    # ============================================================

    f"{project_name}/logger/__init__.py",
    f"{project_name}/logger/logger.py",

    # ============================================================
    # PIPELINES
    # ============================================================

    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/phase1_training_pipeline.py",
    f"{project_name}/pipeline/phase1_prediction_pipeline.py",
    f"{project_name}/pipeline/phase2_spike_pipeline.py",
    f"{project_name}/pipeline/end_to_end_pipeline.py",

    # ============================================================
    # API
    # ============================================================

    f"{project_name}/api/__init__.py",
    f"{project_name}/api/routes.py",
    f"{project_name}/api/schemas.py",
    f"{project_name}/api/app.py",

    # ============================================================
    # UTILITIES
    # ============================================================

    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",
    f"{project_name}/utils/file_utils.py",
    f"{project_name}/utils/config_utils.py",

    # ============================================================
    # CONFIG FILES
    # ============================================================

    "config/model.yaml",
    "config/phase1.yaml",
    "config/phase2.yaml",
    "config/thresholds.yaml",
    "config/schema.yaml",

    # ============================================================
    # ARTIFACT DIRECTORIES
    # ============================================================

    "artifacts/phase1/models/.gitkeep",
    "artifacts/phase1/calibration/.gitkeep",
    "artifacts/phase1/predictions/.gitkeep",
    "artifacts/phase1/evaluations/.gitkeep",

    "artifacts/phase2/temporal_features/.gitkeep",
    "artifacts/phase2/baselines/.gitkeep",
    "artifacts/phase2/anomaly_scores/.gitkeep",
    "artifacts/phase2/spike_events/.gitkeep",

    # ============================================================
    # DATA DIRECTORIES
    # ============================================================

    "data/raw/.gitkeep",
    "data/interim/.gitkeep",
    "data/processed/phase1/.gitkeep",
    "data/processed/phase2/.gitkeep",
    "data/external/.gitkeep",

    # ============================================================
    # NOTEBOOKS
    # ============================================================

    "notebooks/phase1/.gitkeep",
    "notebooks/phase2/.gitkeep",

    # ============================================================
    # TESTS
    # ============================================================

    "tests/unit/phase1/.gitkeep",
    "tests/unit/phase2/.gitkeep",
    "tests/integration/test_phase1_pipeline.py",
    "tests/integration/test_phase2_pipeline.py",
    "tests/test_end_to_end.py",

    # ============================================================
    # DEPLOYMENT
    # ============================================================

    "deployment/docker/Dockerfile",
    "deployment/docker/.dockerignore",

    "deployment/kubernetes/deployment.yaml",
    "deployment/kubernetes/service.yaml",
    "deployment/kubernetes/configmap.yaml",

    "deployment/monitoring/prometheus.yaml",

    # ============================================================
    # DASHBOARD
    # ============================================================

    "dashboard/app.py",
    "dashboard/components/.gitkeep",
    "dashboard/assets/.gitkeep",

    # ============================================================
    # SCRIPTS
    # ============================================================

    "scripts/train_phase1.py",
    "scripts/run_phase1_prediction.py",
    "scripts/run_phase2.py",
    "scripts/evaluate_system.py",

    # ============================================================
    # CI/CD
    # ============================================================

    ".github/workflows/ci.yml",
    ".github/workflows/cd.yml",

    # ============================================================
    # DOCUMENTATION
    # ============================================================

    "docs/architecture.md",
    "docs/phase1.md",
    "docs/phase2.md",
    "docs/metrics.md",
    "docs/model_card.md",

    # ============================================================
    # ROOT FILES
    # ============================================================

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


for filepath in list_of_files:

    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir:
        os.makedirs(filedir, exist_ok=True)

    if (
        not os.path.exists(filepath)
        or os.path.getsize(filepath) == 0
    ):
        with open(filepath, "w", encoding="utf-8") as f:
            pass

    else:
        print(f"File already present at: {filepath}")

print("\nFraudSentinel AI repository structure created successfully.")





#Now I want the full - fledged code in python files for phase - 1, in vs-code repository, I already have with best parameters.max_depth                : 6 min_child_weight         : 1 learning_rate            : 0.07091340754075995 subsample                : 0.8881945843174721 colsample_bytree         : 0.8445573399695375 gamma                    : 0.25073469812212335 reg_alpha                : 1.0560705268797807e-06 reg_lambda               : 0.19158576751393638  , just give me the end to end model training , evaluation , test pipeline. with this output table.