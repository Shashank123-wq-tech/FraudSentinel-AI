import json
from datetime import datetime, timezone

from src.mlops.config import (
    MONITORING_DIR,
)
from src.mlops.logging.logger import get_logger

from src.mlops.registry.artifact_registry import (
    inspect_artifacts,
)

from src.mlops.monitoring.data_monitor import (
    monitor_fusion_data,
)

from src.mlops.monitoring.model_monitor import (
    monitor_model,
)


logger = get_logger("SystemMonitor")


def run_system_monitor():

    logger.info(
        "Starting complete MLOps monitoring."
    )

    artifacts = inspect_artifacts()

    artifact_failures = [
        name
        for name, info
        in artifacts.items()
        if info["status"] != "PASS"
    ]

    data_result = monitor_fusion_data()

    model_result = monitor_model()

    overall_status = "PASS"

    if artifact_failures:
        overall_status = "FAIL"

    if data_result.get("status") != "PASS":
        overall_status = "FAIL"

    if model_result.get("status") != "PASS":
        overall_status = "FAIL"

    result = {
        "project":
            "FraudSentinel AI",

        "monitoring_version":
            "mlops_monitoring_v1",

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "overall_status":
            overall_status,

        "artifact_failures":
            artifact_failures,

        "data_monitoring":
            data_result,

        "model_monitoring":
            model_result,
    }

    output = (
        MONITORING_DIR
        / "system_monitoring.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    logger.info(
        "MLOps monitoring completed: %s",
        overall_status,
    )

    return result


if __name__ == "__main__":

    result = run_system_monitor()

    print(
        json.dumps(
            result,
            indent=2,
        )
    )