"""
FraudSentinel AI — Phase 1 Feature Selection

Feature Selection Strategy
--------------------------
1. Calculate univariate signal for analysis:
   - Absolute correlation with the fraud target.
   - Mutual Information with the fraud target.

2. Perform actual feature selection using ONLY:
   - Feature-to-feature absolute correlation threshold.

If:
    |corr(feature_i, feature_j)| > 0.85

then the two features are considered redundant.
The feature with stronger absolute correlation to the
target is retained.

IMPORTANT:
-----------
- Mutual Information is NOT used to remove features.
- MI is retained only as an analytical signal.
- Feature selection is performed using TRAINING DATA ONLY.
- Test data is never used to select or remove features.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


# ============================================================
# CONFIGURATION
# ============================================================

CORRELATION_THRESHOLD = 0.85


# ============================================================
# UNIVARIATE SIGNAL ANALYSIS
# ============================================================

def univariate_signal(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """
    Calculate feature-level signal statistics using training data only.

    Returns:
        DataFrame containing:
        - abs_correlation
        - mutual_info

    NOTE:
        Mutual Information is calculated for analysis/reporting only.
        It does NOT participate in feature removal.
    """

    if X_train.empty:
        raise ValueError("X_train is empty.")

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must contain the same number of rows."
        )

    # --------------------------------------------------------
    # Absolute linear correlation with target
    # --------------------------------------------------------

    corr_with_target = (
        X_train
        .corrwith(y_train)
        .abs()
    )

    # --------------------------------------------------------
    # Mutual Information
    # --------------------------------------------------------

    mi_scores = mutual_info_classif(
        X_train,
        y_train,
        discrete_features=False,
        random_state=42,
    )

    mi_series = pd.Series(
        mi_scores,
        index=X_train.columns,
        name="mutual_info",
    )

    # --------------------------------------------------------
    # Combined analytical table
    # --------------------------------------------------------

    summary = pd.DataFrame({
        "abs_correlation": corr_with_target,
        "mutual_info": mi_series,
    })

    summary = summary.sort_values(
        "mutual_info",
        ascending=False,
    )

    return summary


# ============================================================
# CORRELATION-BASED FEATURE FILTER
# ============================================================

def correlation_filter(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: float = CORRELATION_THRESHOLD,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove redundant features using feature-to-feature correlation.

    Rule
    ----
    If:

        abs(corr(feature_i, feature_j)) > threshold

    then one feature is removed.

    The feature with stronger absolute correlation
    with the fraud target is retained.

    Parameters
    ----------
    X_train:
        Training feature matrix.

    y_train:
        Training fraud labels.

    threshold:
        Feature-to-feature absolute correlation threshold.

    Returns
    -------
    X_filtered:
        Feature matrix after redundancy filtering.

    removed_features:
        List of removed redundant features.
    """

    if X_train.empty:
        raise ValueError("X_train is empty.")

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must contain the same number of rows."
        )

    if not 0 < threshold <= 1:
        raise ValueError(
            f"Correlation threshold must be between 0 and 1. "
            f"Received: {threshold}"
        )

    # --------------------------------------------------------
    # Feature-to-feature absolute correlation
    # --------------------------------------------------------

    corr_matrix = X_train.corr().abs()

    # --------------------------------------------------------
    # Feature-to-target absolute correlation
    #
    # Used only to decide which member of a highly correlated
    # pair should be retained.
    # --------------------------------------------------------

    target_corr = (
        X_train
        .corrwith(y_train)
        .abs()
    )

    features = list(X_train.columns)

    removed_features = set()

    # --------------------------------------------------------
    # Pairwise redundancy filtering
    # --------------------------------------------------------

    for i in range(len(features)):

        feature_i = features[i]

        if feature_i in removed_features:
            continue

        for j in range(i + 1, len(features)):

            feature_j = features[j]

            if feature_j in removed_features:
                continue

            correlation_value = corr_matrix.loc[
                feature_i,
                feature_j,
            ]

            # ------------------------------------------------
            # Highly correlated pair
            # ------------------------------------------------

            if correlation_value > threshold:

                corr_i_target = target_corr.get(
                    feature_i,
                    0.0,
                )

                corr_j_target = target_corr.get(
                    feature_j,
                    0.0,
                )

                # ------------------------------------------------
                # Keep the feature with stronger target signal
                # ------------------------------------------------

                if corr_i_target >= corr_j_target:
                    removed_features.add(feature_j)

                else:
                    removed_features.add(feature_i)
                    break

    # --------------------------------------------------------
    # Final feature list
    # --------------------------------------------------------

    selected_features = [
        feature
        for feature in features
        if feature not in removed_features
    ]

    if not selected_features:
        raise ValueError(
            "Correlation filtering removed all features."
        )

    X_filtered = X_train[selected_features]

    return (
        X_filtered,
        sorted(removed_features),
    )


# ============================================================
# COMPLETE FEATURE-SELECTION PIPELINE
# ============================================================

def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    correlation_threshold: float = CORRELATION_THRESHOLD,
) -> Tuple[List[str], List[str]]:
    """
    Complete Phase-1 feature-selection pipeline.

    Actual selection:
        correlation filtering ONLY.

    Mutual Information:
        NOT used for removing features.

    Returns
    -------
    selected_features:
        Final retained features.

    correlation_removed:
        Features removed due to high feature-to-feature correlation.
    """

    X_corr, correlation_removed = correlation_filter(
        X_train=X_train,
        y_train=y_train,
        threshold=correlation_threshold,
    )

    selected_features = list(
        X_corr.columns
    )

    return (
        selected_features,
        correlation_removed,
    )