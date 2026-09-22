from __future__ import annotations

import pandas as pd

from .result import ValidationResult


class LineageValidator:

    def validate_transaction_lineage(
        self,
        phase1: pd.DataFrame,
        fusion: pd.DataFrame,
    ) -> ValidationResult:

        if (
            "transaction_id"
            not in phase1.columns
        ):

            return ValidationResult(
                validator="LineageValidator",
                check="transaction_lineage",
                status="FAIL",
                message=(
                    "Phase 1 transaction_id missing."
                ),
                details={},
            )

        if (
            "transaction_id"
            not in fusion.columns
        ):

            return ValidationResult(
                validator="LineageValidator",
                check="transaction_lineage",
                status="FAIL",
                message=(
                    "Fusion transaction_id missing."
                ),
                details={},
            )

        phase1_ids = set(
            phase1["transaction_id"]
        )

        fusion_ids = set(
            fusion["transaction_id"]
        )

        missing_in_fusion = (
            phase1_ids
            - fusion_ids
        )

        extra_in_fusion = (
            fusion_ids
            - phase1_ids
        )

        status = (
            "PASS"
            if (
                len(missing_in_fusion) == 0
                and len(extra_in_fusion) == 0
            )
            else "FAIL"
        )

        return ValidationResult(
            validator="LineageValidator",
            check="transaction_lineage",
            status=status,
            message=(
                "Transaction lineage validation."
            ),
            details={
                "phase1_transactions": (
                    len(phase1_ids)
                ),
                "fusion_transactions": (
                    len(fusion_ids)
                ),
                "missing_in_fusion": (
                    len(missing_in_fusion)
                ),
                "extra_in_fusion": (
                    len(extra_in_fusion)
                ),
            },
        )