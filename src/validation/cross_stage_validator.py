from __future__ import annotations

import pandas as pd

from .result import ValidationResult


class CrossStageValidator:

    def validate_phase1_to_fusion(
        self,
        phase1: pd.DataFrame,
        fusion: pd.DataFrame,
    ) -> ValidationResult:

        if (
            "fraud_probability"
            not in phase1.columns
            or "fraud_probability"
            not in fusion.columns
        ):

            return ValidationResult(
                validator="CrossStageValidator",
                check="phase1_to_fusion_probability",
                status="FAIL",
                message=(
                    "fraud_probability missing."
                ),
                details={},
            )

        merged = (
            phase1[
                [
                    "transaction_id",
                    "fraud_probability",
                ]
            ]
            .merge(
                fusion[
                    [
                        "transaction_id",
                        "fraud_probability",
                    ]
                ],
                on="transaction_id",
                suffixes=(
                    "_phase1",
                    "_fusion",
                ),
                how="inner",
            )
        )

        mismatches = int(
            (
                merged[
                    "fraud_probability_phase1"
                ]
                !=
                merged[
                    "fraud_probability_fusion"
                ]
            ).sum()
        )

        status = (
            "PASS"
            if mismatches == 0
            else "FAIL"
        )

        return ValidationResult(
            validator="CrossStageValidator",
            check="phase1_to_fusion_probability",
            status=status,
            message=(
                "Phase 1 fraud probability "
                "preserved through Fusion."
            ),
            details={
                "matched_rows": int(
                    len(merged)
                ),
                "mismatches": mismatches,
            },
        )

    def validate_unified_risk(
        self,
        fusion: pd.DataFrame,
    ) -> ValidationResult:

        if (
            "unified_risk_score"
            not in fusion.columns
        ):

            return ValidationResult(
                validator="CrossStageValidator",
                check="unified_risk",
                status="FAIL",
                message=(
                    "unified_risk_score missing."
                ),
                details={},
            )

        values = pd.to_numeric(
            fusion[
                "unified_risk_score"
            ],
            errors="coerce",
        )

        invalid = int(
            (
                values.isna()
                | (values < 0)
                | (values > 100)
            ).sum()
        )

        return ValidationResult(
            validator="CrossStageValidator",
            check="unified_risk",
            status=(
                "PASS"
                if invalid == 0
                else "FAIL"
            ),
            message=(
                "Unified risk range validation."
            ),
            details={
                "invalid_values": invalid,
                "min": float(values.min()),
                "max": float(values.max()),
            },
        )