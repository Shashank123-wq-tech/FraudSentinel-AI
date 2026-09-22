from __future__ import annotations

from pathlib import Path

from .result import ValidationResult


class ArtifactValidator:

    def __init__(
        self,
        project_root: Path,
    ):

        self.project_root = project_root

    def validate(
        self,
        artifacts: dict[str, Path],
    ) -> list[ValidationResult]:

        results = []

        for artifact_name, relative_path in artifacts.items():

            path = (
                self.project_root
                / relative_path
            )

            if not path.exists():

                results.append(
                    ValidationResult(
                        validator="ArtifactValidator",
                        check=artifact_name,
                        status="FAIL",
                        message=(
                            "Artifact does not exist."
                        ),
                        details={
                            "path": str(path),
                        },
                    )
                )

                continue

            if not path.is_file():

                results.append(
                    ValidationResult(
                        validator="ArtifactValidator",
                        check=artifact_name,
                        status="FAIL",
                        message=(
                            "Artifact is not a file."
                        ),
                        details={
                            "path": str(path),
                        },
                    )
                )

                continue

            size_bytes = path.stat().st_size

            if size_bytes == 0:

                results.append(
                    ValidationResult(
                        validator="ArtifactValidator",
                        check=artifact_name,
                        status="FAIL",
                        message=(
                            "Artifact is empty."
                        ),
                        details={
                            "path": str(path),
                            "size_bytes": size_bytes,
                        },
                    )
                )

                continue

            results.append(
                ValidationResult(
                    validator="ArtifactValidator",
                    check=artifact_name,
                    status="PASS",
                    message=(
                        "Artifact exists and is non-empty."
                    ),
                    details={
                        "path": str(path),
                        "size_bytes": size_bytes,
                    },
                )
            )

        return results