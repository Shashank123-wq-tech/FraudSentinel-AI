from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FusionConfig


def probability_to_score(
    probability: pd.Series,
) -> pd.Series:
    """
    Convert Phase 1 probability [0,1] to [0,100].
    """

    probability = pd.to_numeric(
        probability,
        errors="coerce",
    )

    return (
        probability
        .clip(lower=0.0, upper=1.0)
        * 100.0
    )


def normalize_phase2_score(
    spike_score: pd.Series,
) -> pd.Series:
    """
    Phase 2 spike_score is already defined on [0,100].
    """

    score = pd.to_numeric(
        spike_score,
        errors="coerce",
    )

    return score.clip(
        lower=0.0,
        upper=100.0,
    )


def state_multiplier(
    state: pd.Series,
    config: FusionConfig,
) -> pd.Series:
    """
    Convert Phase 2 state into a controlled fusion multiplier.
    """

    normalized = (
        state
        .astype("string")
        .str.strip()
        .str.upper()
        .replace({
            "EARLY_WARNING": "EARLY_WARNING",
            "EARLY WARNING": "EARLY_WARNING",
            "CANDIDATE_SPIKE": "EARLY_WARNING",
            "VERIFIED_SPIKE": "VERIFIED_FRAUD_SPIKE",
            "CRITICAL_SPIKE": "CRITICAL_ACTIVE_SPIKE",
        })
    )

    mapping = {
        "NORMAL": config.state_multiplier_normal,
        "EARLY_WARNING": config.state_multiplier_early_warning,
        "VERIFIED_FRAUD_SPIKE": config.state_multiplier_verified,
        "CRITICAL_ACTIVE_SPIKE": config.state_multiplier_critical,
    }

    return (
        normalized
        .map(mapping)
        .fillna(config.state_multiplier_normal)
        .astype(float)
    )


def phase2_confidence(
    windows: pd.DataFrame,
    config: FusionConfig,
) -> pd.Series:
    """
    Use Phase 2 history confidence as the principal confidence
    measure.

    Cold-start windows therefore receive low confidence.
    """

    if "history_confidence" not in windows.columns:
        return pd.Series(
            config.minimum_confidence,
            index=windows.index,
            dtype=float,
        )

    confidence = pd.to_numeric(
        windows["history_confidence"],
        errors="coerce",
    ).fillna(0.0)

    confidence = confidence.clip(
        lower=0.0,
        upper=1.0,
    )

    return confidence.clip(
        lower=config.minimum_confidence,
        upper=config.maximum_confidence,
    )
