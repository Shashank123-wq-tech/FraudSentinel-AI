from __future__ import annotations

import pandas as pd

from .result import ValidationResult


class SchemaValidator:

    REQUIRED_COLUMNS = {

        "phase1": [
            "transaction_id",
            "timestamp",
            "merchant_id",
            "amount",
            "fraud_probability",
        ],

        "phase2_windows": [
            "merchant_id",
            "window_start",
            "window_end",
            "spike_score",
            "spike_state",
        ],

        "phase2_events": [
            "merchant_id",
            "start_time",
            "end_time",
        ],

        "fusion": [
            "transaction_id",
            "merchant_id",
            "fraud_probability",
            "unified_risk_score",
        ],
    }

    def validate(
        self,
        artifact_name: str,
        df: pd.DataFrame,
    ) -> ValidationResult:

        required = (
            self.REQUIRED_COLUMNS
            .get(
                artifact_name,
                [],
            )
        )

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:

            return ValidationResult(
                validator="SchemaValidator",
                check=artifact_name,
                status="FAIL",
                message=(
                    "Required columns missing."
                ),
                details={
                    "missing_columns": missing,
                    "available_columns": (
                        df.columns.tolist()
                    ),
                },
            )

        return ValidationResult(
            validator="SchemaValidator",
            check=artifact_name,
            status="PASS",
            message=(
                "Schema validation passed."
            ),
            details={
                "column_count": int(
                    len(df.columns)
                ),
            },
        )