from __future__ import annotations

import pandas as pd

from .result import ValidationResult


class QualityValidator:

    def validate_dataframe(
        self,
        artifact_name: str,
        df: pd.DataFrame,
        min_rows: int,
    ) -> list[ValidationResult]:

        results = []

        # ========================================================
        # ROW COUNT
        # ========================================================

        row_count = len(df)

        if row_count < min_rows:

            results.append(
                ValidationResult(
                    validator="QualityValidator",
                    check=(
                        f"{artifact_name}.row_count"
                    ),
                    status="FAIL",
                    message=(
                        "Artifact contains insufficient rows."
                    ),
                    details={
                        "rows": int(row_count),
                    },
                )
            )

        else:

            results.append(
                ValidationResult(
                    validator="QualityValidator",
                    check=(
                        f"{artifact_name}.row_count"
                    ),
                    status="PASS",
                    message=(
                        "Row count validation passed."
                    ),
                    details={
                        "rows": int(row_count),
                    },
                )
            )

        # ========================================================
        # DUPLICATE TRANSACTION IDS
        # ========================================================

        if "transaction_id" in df.columns:

            duplicates = int(
                df["transaction_id"]
                .duplicated()
                .sum()
            )

            status = (
                "PASS"
                if duplicates == 0
                else "FAIL"
            )

            results.append(
                ValidationResult(
                    validator="QualityValidator",
                    check=(
                        f"{artifact_name}."
                        "duplicate_transaction_ids"
                    ),
                    status=status,
                    message=(
                        "Transaction ID uniqueness check."
                    ),
                    details={
                        "duplicate_count": duplicates,
                    },
                )
            )

        # ========================================================
        # NULL SUMMARY
        # ========================================================

        null_count = int(
            df.isna().sum().sum()
        )

        results.append(
            ValidationResult(
                validator="QualityValidator",
                check=(
                    f"{artifact_name}.null_summary"
                ),
                status="PASS",
                message=(
                    "Null-value profile collected."
                ),
                details={
                    "total_null_values": null_count,
                    "columns_with_nulls": {
                        column: int(count)
                        for column, count
                        in df.isna().sum().items()
                        if count > 0
                    },
                },
            )
        )

        return results

    def validate_probability(
        self,
        artifact_name: str,
        df: pd.DataFrame,
        column: str,
    ) -> ValidationResult | None:

        if column not in df.columns:

            return None

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid = int(
            (
                values.isna()
                | (values < 0)
                | (values > 1)
            ).sum()
        )

        status = (
            "PASS"
            if invalid == 0
            else "FAIL"
        )

        return ValidationResult(
            validator="QualityValidator",
            check=(
                f"{artifact_name}.{column}"
            ),
            status=status,
            message=(
                "Probability range validation."
            ),
            details={
                "invalid_values": invalid,
                "min": (
                    float(values.min())
                    if not values.empty
                    else None
                ),
                "max": (
                    float(values.max())
                    if not values.empty
                    else None
                ),
            },
        )

    def validate_risk_score(
        self,
        artifact_name: str,
        df: pd.DataFrame,
        column: str,
    ) -> ValidationResult | None:

        if column not in df.columns:

            return None

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid = int(
            (
                values.isna()
                | (values < 0)
                | (values > 100)
            ).sum()
        )

        status = (
            "PASS"
            if invalid == 0
            else "FAIL"
        )

        return ValidationResult(
            validator="QualityValidator",
            check=(
                f"{artifact_name}.{column}"
            ),
            status=status,
            message=(
                "Risk-score range validation."
            ),
            details={
                "invalid_values": invalid,
                "min": (
                    float(values.min())
                    if not values.empty
                    else None
                ),
                "max": (
                    float(values.max())
                    if not values.empty
                    else None
                ),
            },
        )