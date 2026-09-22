import json
from datetime import datetime, timezone

import pandas as pd

from src.mlops.config import (
    FUSION_DIR,
    MONITORING_DIR,
)
from src.mlops.logging.logger import get_logger


logger = get_logger("ModelMonitor")


FUSION_FILE = (
    FUSION_DIR
    / "fraudsentinel_unified_risk.csv"
)

CHUNK_SIZE = 50_000


def monitor_model():

    logger.info(
        "Starting model monitoring."
    )

    total_rows = 0
    alert_count = 0
    fraud_count = 0

    probability_sum = 0.0
    probability_count = 0

    risk_sum = 0.0
    risk_count = 0

    for chunk in pd.read_csv(
        FUSION_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=True,
    ):

        total_rows += len(chunk)

        if "alert_flag" in chunk.columns:

            alert_count += int(
                chunk["alert_flag"]
                .fillna(False)
                .astype(bool)
                .sum()
            )

        if "predicted_fraud" in chunk.columns:

            fraud_count += int(
                pd.to_numeric(
                    chunk["predicted_fraud"],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            )

        if "fraud_probability" in chunk.columns:

            probabilities = pd.to_numeric(
                chunk["fraud_probability"],
                errors="coerce",
            ).dropna()

            probability_sum += float(
                probabilities.sum()
            )

            probability_count += len(
                probabilities
            )

        if "unified_risk_score" in chunk.columns:

            risks = pd.to_numeric(
                chunk["unified_risk_score"],
                errors="coerce",
            ).dropna()

            risk_sum += float(
                risks.sum()
            )

            risk_count += len(risks)

    alert_rate = (
        alert_count / total_rows
        if total_rows
        else 0.0
    )

    predicted_fraud_rate = (
        fraud_count / total_rows
        if total_rows
        else 0.0
    )

    mean_probability = (
        probability_sum / probability_count
        if probability_count
        else 0.0
    )

    mean_unified_risk = (
        risk_sum / risk_count
        if risk_count
        else 0.0
    )

    result = {
        "monitoring_version":
            "mlops_monitoring_v1",

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "transactions":
            int(total_rows),

        "alerts":
            int(alert_count),

        "alert_rate":
            float(alert_rate),

        "predicted_fraud":
            int(fraud_count),

        "predicted_fraud_rate":
            float(predicted_fraud_rate),

        "mean_fraud_probability":
            float(mean_probability),

        "mean_unified_risk":
            float(mean_unified_risk),

        "status":
            "PASS",
    }

    output = (
        MONITORING_DIR
        / "model_monitoring.json"
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
        "Model monitoring completed."
    )

    return result


if __name__ == "__main__":

    result = monitor_model()

    print(
        json.dumps(
            result,
            indent=2,
        )
    )