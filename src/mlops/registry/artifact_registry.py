import json
from datetime import datetime, timezone

from src.mlops.config import (
    ARTIFACTS_DIR,
    REGISTRY_DIR,
)


ARTIFACTS = {
    "phase1_transaction_risk":
        ARTIFACTS_DIR
        / "phase1"
        / "risk_output"
        / "transaction_risk.csv",

    "phase2_windows":
        ARTIFACTS_DIR
        / "phase2"
        / "phase2_windows.csv",

    "phase2_events":
        ARTIFACTS_DIR
        / "phase2"
        / "phase2_events.csv",

    "fusion_risk":
        ARTIFACTS_DIR
        / "fusion"
        / "fraudsentinel_unified_risk.csv",

    "xai_transactions":
        ARTIFACTS_DIR
        / "xai"
        / "transaction_explanations.csv",

    "xai_spikes":
        ARTIFACTS_DIR
        / "xai"
        / "spike_explanations.csv",

    "xai_unified":
        ARTIFACTS_DIR
        / "xai"
        / "unified_explanations.csv",

    "impact_forecast":
        ARTIFACTS_DIR
        / "impact"
        / "impact_forecast.csv",

    "impact_scenarios":
        ARTIFACTS_DIR
        / "impact"
        / "impact_scenarios.csv",

    "response_recommendations":
        ARTIFACTS_DIR
        / "response"
        / "response_recommendations.csv",

    "response_actions":
        ARTIFACTS_DIR
        / "response"
        / "response_actions.csv",
}


def inspect_artifacts():

    results = {}

    for name, path in ARTIFACTS.items():

        exists = path.exists()

        size = (
            path.stat().st_size
            if exists
            else 0
        )

        results[name] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": size,
            "status": (
                "PASS"
                if exists and size > 0
                else "FAIL"
            ),
        }

    return results


def save_artifact_registry():

    results = inspect_artifacts()

    registry = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "artifacts": results,
    }

    output = (
        REGISTRY_DIR
        / "artifact_registry.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            registry,
            file,
            indent=2,
        )

    return registry


if __name__ == "__main__":

    registry = save_artifact_registry()

    for name, info in registry["artifacts"].items():

        print(
            f"{name:30s} "
            f"{info['status']}"
        )