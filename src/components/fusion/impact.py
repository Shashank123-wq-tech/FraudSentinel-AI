from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FusionConfig


def calculate_transaction_expected_loss(
    amount: pd.Series,
    fraud_probability: pd.Series,
    config: FusionConfig,
) -> pd.Series:
    """
    Expected fraud loss at transaction level.

    E[L_transaction] =
        transaction_amount × fraud_probability
    """

    amount = pd.to_numeric(
        amount,
        errors="coerce",
    ).fillna(0.0)

    probability = pd.to_numeric(
        fraud_probability,
        errors="coerce",
    ).fillna(0.0)

    expected_loss = (
        amount.clip(lower=0.0)
        * probability.clip(lower=0.0, upper=1.0)
        * config.expected_loss_multiplier
    )

    return expected_loss.replace(
        [np.inf, -np.inf],
        0.0,
    ).fillna(0.0)


def calculate_fused_expected_exposure(
    transaction_expected_loss: pd.Series,
    phase2_expected_fraud_amount: pd.Series,
) -> pd.Series:
    """
    Conservative exposure estimate.

    We do not add the two values because the Phase 2 amount is
    derived from transactions already represented by Phase 1.

    Instead, use the larger available estimate.
    """

    transaction_loss = pd.to_numeric(
        transaction_expected_loss,
        errors="coerce",
    ).fillna(0.0)

    phase2_loss = pd.to_numeric(
        phase2_expected_fraud_amount,
        errors="coerce",
    ).fillna(0.0)

    return pd.concat(
        [
            transaction_loss.rename("transaction"),
            phase2_loss.rename("phase2"),
        ],
        axis=1,
    ).max(axis=1)


def calculate_event_expected_exposure(
    event_expected_fraud_amount: pd.Series,
) -> pd.Series:
    """
    Preserve event-level expected fraud amount as contextual
    information without adding it to transaction exposure.
    """

    return pd.to_numeric(
        event_expected_fraud_amount,
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)

