from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.entity.spike_event import (
    SeverityPolicy,
    SpikeScoreWeights,
)
from src.exception import ConfigurationError


REQUIRED_RESOLUTIONS = {
    "5m",
    "15m",
    "60m",
}


def load_phase2_config(
    path: str = "config/phase2.yaml",
) -> dict[str, Any]:

    config_path = Path(path)

    if not config_path.exists():
        raise ConfigurationError(
            f"Phase-2 configuration not found: {config_path}"
        )

    try:
        config = yaml.safe_load(
            config_path.read_text(
                encoding="utf-8"
            )
        ) or {}

    except Exception as exc:
        raise ConfigurationError(
            "Unable to parse Phase-2 YAML.",
            cause=exc,
        ) from exc

    try:

        # -----------------------------------------------------
        # Windows
        # -----------------------------------------------------

        window_sizes = config.get(
            "window_sizes",
            {},
        )

        if set(window_sizes) != REQUIRED_RESOLUTIONS:
            raise ConfigurationError(
                "window_sizes must contain exactly "
                "5m, 15m and 60m."
            )

        # -----------------------------------------------------
        # Baseline
        # -----------------------------------------------------

        baseline = config.get(
            "baseline",
            {},
        )

        history_hours = baseline.get(
            "history_hours",
            {},
        )

        min_history = baseline.get(
            "min_history_windows",
            {},
        )

        for resolution in REQUIRED_RESOLUTIONS:

            if (
                int(
                    history_hours.get(
                        resolution,
                        0,
                    )
                )
                <= 0
            ):
                raise ConfigurationError(
                    f"Invalid baseline history for {resolution}."
                )

            if (
                int(
                    min_history.get(
                        resolution,
                        0,
                    )
                )
                < 1
            ):
                raise ConfigurationError(
                    f"Invalid minimum history for {resolution}."
                )

        # -----------------------------------------------------
        # Weights
        # -----------------------------------------------------

        weight_config = config[
            "spike_score_weights"
        ]

        weights = SpikeScoreWeights(
            **{
                key: weight_config[key]
                for key
                in SpikeScoreWeights.__dataclass_fields__
            }
        )

        try:
            weights.validate()
        except ValueError as exc:
            raise ConfigurationError(
                "Invalid spike-score weights.",
                cause=exc,
            ) from exc

        # -----------------------------------------------------
        # Severity
        # -----------------------------------------------------

        severity_config = config[
            "severity_policy"
        ]

        severity = SeverityPolicy(
            **{
                key: severity_config[key]
                for key
                in SeverityPolicy.__dataclass_fields__
            }
        )

        try:
            severity.validate()
        except ValueError as exc:
            raise ConfigurationError(
                "Invalid severity policy.",
                cause=exc,
            ) from exc

        # -----------------------------------------------------
        # EWMA
        # -----------------------------------------------------

        if int(
            config["ewma"]["span"]
        ) <= 0:
            raise ConfigurationError(
                "EWMA span must be > 0."
            )

        # -----------------------------------------------------
        # CUSUM
        # -----------------------------------------------------

        if float(
            config["cusum"]["k"]
        ) < 0:
            raise ConfigurationError(
                "CUSUM k must be >= 0."
            )

        return config

    except ConfigurationError:
        raise

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:

        raise ConfigurationError(
            "Invalid Phase-2 configuration.",
            cause=exc,
        ) from exc