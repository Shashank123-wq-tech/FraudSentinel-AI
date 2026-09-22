from __future__ import annotations

from pathlib import Path

import pandas as pd

from .artifact_validator import (
    ArtifactValidator,
)

from .config import (
    ValidationConfig,
)

from .cross_stage_validator import (
    CrossStageValidator,
)

from .lineage_validator import (
    LineageValidator,
)

from .quality_validator import (
    QualityValidator,
)

from .report import (
    write_validation_report,
)

from .result import (
    ValidationResult,
)

from .schema_validator import (
    SchemaValidator,
)


class SystemValidator:

    def __init__(
        self,
        config: ValidationConfig | None = None,
    ):

        self.config = (
            config
            or ValidationConfig()
        )

        self.artifact_validator = (
            ArtifactValidator(
                self.config.project_root
            )
        )

        self.schema_validator = (
            SchemaValidator()
        )

        self.quality_validator = (
            QualityValidator()
        )

        self.lineage_validator = (
            LineageValidator()
        )

        self.cross_stage_validator = (
            CrossStageValidator()
        )

    # ============================================================
    # LOAD ARTIFACT
    # ============================================================

    def _load_artifact(
        self,
        relative_path: Path,
    ) -> pd.DataFrame:

        path = (
            self.config.project_root
            / relative_path
        )

        return pd.read_csv(
            path,
            low_memory=False,
        )

    # ============================================================
    # RUN VALIDATION
    # ============================================================

    def validate(
        self,
    ) -> dict:

        results: list[
            ValidationResult
        ] = []

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FraudSentinel AI "
            "SYSTEM VALIDATION"
        )

        print(
            "=" * 70
        )

        # ========================================================
        # 1. ARTIFACT VALIDATION
        # ========================================================

        print(
            "\n[1/5] Validating artifacts..."
        )

        artifact_results = (
            self.artifact_validator
            .validate(
                self.config.required_artifacts
            )
        )

        results.extend(
            artifact_results
        )

        artifact_failure = any(
            result.status == "FAIL"
            for result in artifact_results
        )

        if artifact_failure:

            print(
                "\nArtifact validation FAILED."
            )

            return self._finalize(
                results
            )

        # ========================================================
        # 2. LOAD DATA
        # ========================================================

        print(
            "\n[2/5] Loading artifacts..."
        )

        dataframes = {}

        for name, path in (
            self.config
            .required_artifacts
            .items()
        ):

            try:

                df = (
                    self._load_artifact(
                        path
                    )
                )

                dataframes[name] = df

                print(
                    f"Loaded {name}: "
                    f"{len(df):,} rows"
                )

            except Exception as exc:

                results.append(
                    ValidationResult(
                        validator=(
                            "SystemValidator"
                        ),
                        check=(
                            f"{name}.load"
                        ),
                        status="FAIL",
                        message=(
                            "Failed to load artifact."
                        ),
                        details={
                            "error": str(exc),
                        },
                    )
                )

        # ========================================================
        # 3. SCHEMA VALIDATION
        # ========================================================

        print(
            "\n[3/5] Validating schemas..."
        )

        for name, df in (
            dataframes.items()
        ):

            result = (
                self.schema_validator
                .validate(
                    name,
                    df,
                )
            )

            results.append(
                result
            )

        # ========================================================
        # 4. DATA QUALITY
        # ========================================================

        print(
            "\n[4/5] Validating data quality..."
        )

        for name, df in (
            dataframes.items()
        ):

            quality_results = (
                self.quality_validator
                .validate_dataframe(
                    artifact_name=name,
                    df=df,
                    min_rows=(
                        self.config
                        .min_rows
                    ),
                )
            )

            results.extend(
                quality_results
            )

            for column in (
                self.config
                .probability_columns
            ):

                result = (
                    self.quality_validator
                    .validate_probability(
                        artifact_name=name,
                        df=df,
                        column=column,
                    )
                )

                if result is not None:

                    results.append(
                        result
                    )

            for column in (
                self.config
                .risk_score_columns
            ):

                result = (
                    self.quality_validator
                    .validate_risk_score(
                        artifact_name=name,
                        df=df,
                        column=column,
                    )
                )

                if result is not None:

                    results.append(
                        result
                    )

        # ========================================================
        # 5. CROSS-STAGE VALIDATION
        # ========================================================

        print(
            "\n[5/5] Validating cross-stage lineage..."
        )

        if (
            "phase1"
            in dataframes
            and "fusion"
            in dataframes
        ):

            results.append(
                self.lineage_validator
                .validate_transaction_lineage(
                    phase1=(
                        dataframes[
                            "phase1"
                        ]
                    ),
                    fusion=(
                        dataframes[
                            "fusion"
                        ]
                    ),
                )
            )

            results.append(
                self.cross_stage_validator
                .validate_phase1_to_fusion(
                    phase1=(
                        dataframes[
                            "phase1"
                        ]
                    ),
                    fusion=(
                        dataframes[
                            "fusion"
                        ]
                    ),
                )
            )

            results.append(
                self.cross_stage_validator
                .validate_unified_risk(
                    fusion=(
                        dataframes[
                            "fusion"
                        ]
                    ),
                )
            )

        return self._finalize(
            results
        )

    # ============================================================
    # FINALIZE
    # ============================================================

    def _finalize(
        self,
        results: list[ValidationResult],
    ) -> dict:

        json_path, csv_path = (
            write_validation_report(
                results=results,
                output_dir=(
                    self.config
                    .output_dir
                ),
            )
        )

        passed = sum(
            result.status == "PASS"
            for result in results
        )

        failed = sum(
            result.status == "FAIL"
            for result in results
        )

        total = len(results)

        status = (
            "PASS"
            if failed == 0
            else "FAIL"
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "SYSTEM VALIDATION COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            f"Total checks : "
            f"{total}"
        )

        print(
            f"Passed       : "
            f"{passed}"
        )

        print(
            f"Failed       : "
            f"{failed}"
        )

        print(
            f"Status       : "
            f"{status}"
        )

        print(
            "\nReports:"
        )

        print(
            json_path
        )

        print(
            csv_path
        )

        return {
            "status": status,
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "report": str(
                json_path
            ),
            "summary": str(
                csv_path
            ),
            "results": [
                result.to_dict()
                for result in results
            ],
        }