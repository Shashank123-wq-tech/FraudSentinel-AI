from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .result import ValidationResult


def write_validation_report(
    results: list[ValidationResult],
    output_dir: Path,
) -> tuple[Path, Path]:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_data = [
        result.to_dict()
        for result in results
    ]

    json_path = (
        output_dir
        / "validation_report.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results_data,
            file,
            indent=2,
        )

    summary_rows = []

    for result in results:

        summary_rows.append(
            {
                "validator": (
                    result.validator
                ),
                "check": result.check,
                "status": result.status,
                "message": result.message,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    csv_path = (
        output_dir
        / "validation_summary.csv"
    )

    summary.to_csv(
        csv_path,
        index=False,
    )

    return (
        json_path,
        csv_path,
    )