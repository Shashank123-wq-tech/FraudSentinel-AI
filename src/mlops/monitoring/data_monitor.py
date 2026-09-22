import json
from datetime import datetime, timezone

import pandas as pd

from src.mlops.config import (
    FUSION_DIR,
    MONITORING_DIR,
)
from src.mlops.logging.logger import get_logger


logger = get_logger("DataMonitor")


FUSION_FILE = (
    FUSION_DIR
    / "fraudsentinel_unified_risk.csv"
)

CHUNK_SIZE = 50_000


def monitor_fusion_data():

    logger.info(
        "Starting Fusion data monitoring."
    )

    if not FUSION_FILE.exists():

        return {
            "status": "FAIL",
            "reason": "Fusion artifact not found.",
        }

    total_rows = 0
    missing_values = 0
    invalid_amounts = 0

    required_columns = [
        "transaction_id",
        "timestamp",
        "merchant_id",
        "amount",
        "fraud_probability",
        "unified_risk_score",
    ]

    first_chunk = True

    for chunk in pd.read_csv(
        FUSION_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=True,
    ):

        total_rows += len(chunk)

        if first_chunk:

            missing_columns = [
                col
                for col in required_columns
                if col not in chunk.columns
            ]

            first_chunk = False

        else:

            missing_columns = []

        for column in required_columns:

            if column in chunk.columns:

                missing_values += int(
                    chunk[column]
                    .isna()
                    .sum()
                )

        if "amount" in chunk.columns:

            amounts = pd.to_numeric(
                chunk["amount"],
                errors="coerce",
            )

            invalid_amounts += int(
                (
                    amounts < 0
                )
                .fillna(False)
                .sum()
            )

    status = "PASS"

    if first_chunk:

        status = "FAIL"

    elif missing_columns:

        status = "FAIL"

    elif invalid_amounts > 0:

        status = "FAIL"

    result = {
        "monitoring_version":
            "mlops_monitoring_v1",

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "file":
            str(FUSION_FILE),

        "rows":
            int(total_rows),

        "missing_values":
            int(missing_values),

        "invalid_amounts":
            int(invalid_amounts),

        "missing_columns":
            missing_columns,

        "status":
            status,
    }

    output = (
        MONITORING_DIR
        / "data_monitoring.json"
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
        "Fusion data monitoring completed: %s",
        status,
    )

    return result


if __name__ == "__main__":

    result = monitor_fusion_data()

    print(
        json.dumps(
            result,
            indent=2,
        )
    )