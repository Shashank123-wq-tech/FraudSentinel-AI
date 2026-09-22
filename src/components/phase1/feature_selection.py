from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class FeatureSelectionConfig:

    # --------------------------------------------------------
    # Layer 1 — Data quality
    # --------------------------------------------------------

    max_missing_rate: float = 0.01

    min_unique_values: int = 2

    # --------------------------------------------------------
    # Layer 2 — Redundancy
    # --------------------------------------------------------

    correlation_threshold: float = 0.90

    # --------------------------------------------------------
    # Layer 3 — Mutual Information
    # --------------------------------------------------------

    mi_random_state: int = 42

    mi_minimum: float = 0.001

    # --------------------------------------------------------
    # Layer 4 — Model importance
    # --------------------------------------------------------

    model_random_state: int = 42

    n_estimators: int = 500

    max_depth: int = 6

    learning_rate: float = 0.05

    subsample: float = 0.85

    colsample_bytree: float = 0.85

    min_child_weight: int = 5

    reg_alpha: float = 0.1

    reg_lambda: float = 1.0

    cumulative_importance: float = 0.98

    min_features_to_keep: int = 18

    max_features_to_keep: 24


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class FeatureSelectionResult:

    selected_features: List[str]

    layer1_removed: List[str]

    layer2_removed: List[str]

    layer3_weak_signal: List[str]

    layer4_removed: List[str]

    quality_report: pd.DataFrame

    correlation_report: pd.DataFrame

    mi_report: pd.DataFrame

    model_importance_report: pd.DataFrame

    final_report: pd.DataFrame


# ============================================================
# MAIN SELECTOR
# ============================================================

class FraudFeatureSelector:

    """
    Four-layer feature selection for FraudSentinel AI.

    Layer 1:
        Data quality

    Layer 2:
        Correlation / redundancy

    Layer 3:
        Mutual information

    Layer 4:
        XGBoost gain importance

    IMPORTANT:
        Selection is performed on TRAINING DATA ONLY.

    The XGBoost used in Layer 4 is a temporary screening
    model. It is NOT the final production fraud model.
    """

    def __init__(
        self,
        config: FeatureSelectionConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else FeatureSelectionConfig()
        )

        self.fitted_ = False

        self.result_: (
            FeatureSelectionResult | None
        ) = None

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_input(
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:

        if X_train is None:
            raise ValueError(
                "X_train is None."
            )

        if X_train.empty:
            raise ValueError(
                "X_train is empty."
            )

        if y_train is None:
            raise ValueError(
                "y_train is None."
            )

        if len(X_train) != len(y_train):
            raise ValueError(
                "X_train and y_train must have "
                "the same number of rows."
            )

        if y_train.isna().any():
            raise ValueError(
                "y_train contains missing values."
            )

        unique_target = set(
            y_train.unique()
        )

        if not unique_target.issubset(
            {0, 1}
        ):

            raise ValueError(
                "y_train must contain only 0/1 values."
            )

        non_numeric = list(
            X_train.select_dtypes(
                exclude=[np.number]
            ).columns
        )

        if non_numeric:

            raise ValueError(
                "Feature matrix contains non-numeric "
                f"columns: {non_numeric}"
            )

    # ========================================================
    # LAYER 1
    # DATA QUALITY
    # ========================================================

    def _layer1_data_quality(
        self,
        X_train: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        List[str],
        pd.DataFrame,
    ]:

        rows = []

        removed = []

        for feature in X_train.columns:

            series = X_train[
                feature
            ]

            missing_rate = float(
                series.isna().mean()
            )

            values = series.to_numpy(
                dtype=np.float64
            )

            infinite_count = int(
                np.isinf(
                    values
                ).sum()
            )

            unique_count = int(
                series.nunique(
                    dropna=True
                )
            )

            variance = float(
                series.var(
                    skipna=True
                )
            )

            remove = False

            reasons = []

            # ------------------------------------------------
            # Missing
            # ------------------------------------------------

            if (
                missing_rate
                >
                self.config.max_missing_rate
            ):

                remove = True

                reasons.append(
                    "high_missing_rate"
                )

            # ------------------------------------------------
            # Infinite
            # ------------------------------------------------

            if infinite_count > 0:

                remove = True

                reasons.append(
                    "infinite_values"
                )

            # ------------------------------------------------
            # Constant
            # ------------------------------------------------

            if (
                unique_count
                <
                self.config.min_unique_values
            ):

                remove = True

                reasons.append(
                    "constant_feature"
                )

            rows.append(
                {
                    "feature": feature,
                    "missing_rate": missing_rate,
                    "infinite_count": infinite_count,
                    "unique_count": unique_count,
                    "variance": variance,
                    "removed": remove,
                    "reason": "|".join(
                        reasons
                    ),
                }
            )

            if remove:

                removed.append(
                    feature
                )

        report = pd.DataFrame(
            rows
        )

        selected = [
            feature
            for feature in X_train.columns
            if feature not in removed
        ]

        if not selected:

            raise ValueError(
                "Layer 1 removed all features."
            )

        return (
            X_train[
                selected
            ].copy(),
            removed,
            report,
        )

    # ========================================================
    # LAYER 2
    # CORRELATION / REDUNDANCY
    # ========================================================

    def _layer2_correlation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[
        pd.DataFrame,
        List[str],
        pd.DataFrame,
    ]:

        correlation_matrix = (
            X_train
            .corr()
            .abs()
        )

        target_correlation = (
            X_train
            .corrwith(
                y_train
            )
            .abs()
            .fillna(0.0)
        )

        features = list(
            X_train.columns
        )

        removed = set()

        report_rows = []

        for i in range(
            len(features)
        ):

            feature_i = features[i]

            if feature_i in removed:
                continue

            for j in range(
                i + 1,
                len(features),
            ):

                feature_j = features[j]

                if feature_j in removed:
                    continue

                corr_value = float(
                    correlation_matrix.loc[
                        feature_i,
                        feature_j,
                    ]
                )

                if (
                    corr_value
                    >
                    self.config.correlation_threshold
                ):

                    target_i = float(
                        target_correlation.get(
                            feature_i,
                            0.0,
                        )
                    )

                    target_j = float(
                        target_correlation.get(
                            feature_j,
                            0.0,
                        )
                    )

                    # Keep the feature with
                    # stronger target signal.

                    if target_i >= target_j:

                        keep = feature_i

                        remove = feature_j

                    else:

                        keep = feature_j

                        remove = feature_i

                    removed.add(
                        remove
                    )

                    report_rows.append(
                        {
                            "feature_a": feature_i,
                            "feature_b": feature_j,
                            "correlation": corr_value,
                            "target_corr_a": target_i,
                            "target_corr_b": target_j,
                            "kept": keep,
                            "removed": remove,
                        }
                    )

                    if (
                        remove
                        ==
                        feature_i
                    ):

                        break

        selected = [
            feature
            for feature in features
            if feature not in removed
        ]

        if not selected:

            raise ValueError(
                "Layer 2 removed all features."
            )

        report = pd.DataFrame(
            report_rows
        )

        return (
            X_train[
                selected
            ].copy(),
            sorted(
                removed
            ),
            report,
        )

    # ========================================================
    # LAYER 3
    # MUTUAL INFORMATION
    # ========================================================

    def _layer3_mutual_information(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[
        pd.DataFrame,
        List[str],
        pd.DataFrame,
    ]:

        X_mi = (
            X_train
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(0.0)
        )

        mi_values = (
            mutual_info_classif(
                X_mi,
                y_train,
                discrete_features=False,
                random_state=(
                    self.config.mi_random_state
                ),
            )
        )

        mi_report = pd.DataFrame(
            {
                "feature": X_train.columns,
                "mutual_info": mi_values,
            }
        )

        mi_report[
            "weak_signal"
        ] = (
            mi_report[
                "mutual_info"
            ]
            <
            self.config.mi_minimum
        )

        mi_report = (
            mi_report
            .sort_values(
                "mutual_info",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        weak_features = (
            mi_report.loc[
                mi_report[
                    "weak_signal"
                ],
                "feature",
            ]
            .tolist()
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Weak MI does NOT automatically remove a feature.
        #
        # XGBoost can exploit conditional interactions.
        # ----------------------------------------------------

        return (
            X_train.copy(),
            weak_features,
            mi_report,
        )

    # ========================================================
    # LAYER 4
    # MODEL-BASED IMPORTANCE
    # ========================================================

    def _layer4_model_importance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[
        pd.DataFrame,
        List[str],
        pd.DataFrame,
    ]:

        positive = int(
            (
                y_train == 1
            ).sum()
        )

        negative = int(
            (
                y_train == 0
            ).sum()
        )

        if positive == 0:

            raise ValueError(
                "No positive fraud examples "
                "available for model-based "
                "feature selection."
            )

        scale_pos_weight = (
            negative
            /
            positive
        )

        # ----------------------------------------------------
        # TEMPORARY SCREENING MODEL
        # ----------------------------------------------------

        screening_model = XGBClassifier(

            n_estimators=(
                self.config.n_estimators
            ),

            max_depth=(
                self.config.max_depth
            ),

            learning_rate=(
                self.config.learning_rate
            ),

            subsample=(
                self.config.subsample
            ),

            colsample_bytree=(
                self.config.colsample_bytree
            ),

            min_child_weight=(
                self.config.min_child_weight
            ),

            reg_alpha=(
                self.config.reg_alpha
            ),

            reg_lambda=(
                self.config.reg_lambda
            ),

            objective="binary:logistic",

            eval_metric="aucpr",

            tree_method="hist",

            scale_pos_weight=(
                scale_pos_weight
            ),

            random_state=(
                self.config.model_random_state
            ),

            n_jobs=-1,
        )

        screening_model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # GAIN IMPORTANCE
        # ----------------------------------------------------

        gain_importance = (
            screening_model
            .get_booster()
            .get_score(
                importance_type="gain"
            )
        )

        report = pd.DataFrame(
            {
                "feature": X_train.columns,

                "gain": [
                    float(
                        gain_importance.get(
                            feature,
                            0.0,
                        )
                    )
                    for feature
                    in X_train.columns
                ],
            }
        )

        # ----------------------------------------------------
        # NORMALIZED GAIN
        # ----------------------------------------------------

        total_gain = float(
            report["gain"].sum()
        )

        if total_gain > 0:

            report[
                "gain_normalized"
            ] = (
                report["gain"]
                /
                total_gain
            )

        else:

            report[
                "gain_normalized"
            ] = 0.0

        # ----------------------------------------------------
        # SORT
        # ----------------------------------------------------

        report = (
            report
            .sort_values(
                "gain_normalized",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        # ----------------------------------------------------
        # CUMULATIVE GAIN
        # ----------------------------------------------------

        report[
            "cumulative_gain"
        ] = (
            report[
                "gain_normalized"
            ]
            .cumsum()
        )

        # ----------------------------------------------------
        # SELECT UP TO CUMULATIVE IMPORTANCE
        # ----------------------------------------------------

        selected_features = (
            report.loc[
                report[
                    "cumulative_gain"
                ]
                <=
                self.config.cumulative_importance,
                "feature",
            ]
            .tolist()
        )

        # ----------------------------------------------------
        # Always include first feature crossing threshold.
        # ----------------------------------------------------

        if (
            len(selected_features)
            <
            len(report)
        ):

            crossing_index = (
                len(selected_features)
            )

            selected_features.append(
                report.iloc[
                    crossing_index
                ]["feature"]
            )

        # ----------------------------------------------------
        # Minimum feature protection
        # ----------------------------------------------------

        if (
            len(selected_features)
            <
            self.config.min_features_to_keep
        ):

            selected_features = (
                report
                .head(
                    self.config.min_features_to_keep
                )["feature"]
                .tolist()
            )

        # ----------------------------------------------------
        # Maximum feature protection
        # ----------------------------------------------------

        if (
            self.config.max_features_to_keep
            is not None
        ):

            selected_features = (
                selected_features[
                    :self.config.max_features_to_keep
                ]
            )

        selected_set = set(
            selected_features
        )

        removed = [
            feature
            for feature in X_train.columns
            if feature not in selected_set
        ]

        report[
            "selected"
        ] = (
            report[
                "feature"
            ].isin(
                selected_set
            )
        )

        return (
            X_train[
                selected_features
            ].copy(),
            removed,
            report,
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    @staticmethod
    def _build_final_report(
        X_train: pd.DataFrame,
        selected_features: List[str],
        quality_report: pd.DataFrame,
        correlation_report: pd.DataFrame,
        mi_report: pd.DataFrame,
        model_importance_report: pd.DataFrame,
        layer1_removed: List[str],
        layer2_removed: List[str],
        layer4_removed: List[str],
    ) -> pd.DataFrame:

        report = pd.DataFrame(
            {
                "feature": X_train.columns
            }
        )

        # ----------------------------------------------------
        # Quality
        # ----------------------------------------------------

        if not quality_report.empty:

            quality = quality_report[
                [
                    "feature",
                    "missing_rate",
                    "infinite_count",
                    "unique_count",
                    "variance",
                    "removed",
                    "reason",
                ]
            ].copy()

            report = report.merge(
                quality,
                on="feature",
                how="left",
            )

        # ----------------------------------------------------
        # Mutual information
        # ----------------------------------------------------

        if not mi_report.empty:

            mi = mi_report[
                [
                    "feature",
                    "mutual_info",
                    "weak_signal",
                ]
            ].copy()

            report = report.merge(
                mi,
                on="feature",
                how="left",
            )

        # ----------------------------------------------------
        # Model importance
        # ----------------------------------------------------

        if not model_importance_report.empty:

            model = model_importance_report[
                [
                    "feature",
                    "gain",
                    "gain_normalized",
                    "cumulative_gain",
                    "selected",
                ]
            ].copy()

            report = report.merge(
                model,
                on="feature",
                how="left",
            )

        selected_set = set(
            selected_features
        )

        layer1_set = set(
            layer1_removed
        )

        layer2_set = set(
            layer2_removed
        )

        layer4_set = set(
            layer4_removed
        )

        weak_set = set(
            mi_report.loc[
                mi_report[
                    "weak_signal"
                ],
                "feature",
            ]
            .tolist()
        )

        # ----------------------------------------------------
        # Final selection
        # ----------------------------------------------------

        report[
            "final_selected"
        ] = (
            report[
                "feature"
            ]
            .isin(
                selected_set
            )
        )

        # ----------------------------------------------------
        # Selection stage
        # ----------------------------------------------------

        def determine_stage(
            feature: str,
        ) -> str:

            if feature in selected_set:

                return "FINAL"

            if feature in layer4_set:

                return "MODEL"

            if feature in layer2_set:

                return "CORRELATION"

            if feature in layer1_set:

                return "QUALITY"

            if feature in weak_set:

                return "MI_WEAK"

            return "OTHER"

        report[
            "selection_stage"
        ] = (
            report[
                "feature"
            ]
            .apply(
                determine_stage
            )
        )

        sort_columns = [
            "final_selected"
        ]

        ascending = [
            False
        ]

        if (
            "gain_normalized"
            in report.columns
        ):

            sort_columns.append(
                "gain_normalized"
            )

            ascending.append(
                False
            )

        if (
            "mutual_info"
            in report.columns
        ):

            sort_columns.append(
                "mutual_info"
            )

            ascending.append(
                False
            )

        return (
            report
            .sort_values(
                sort_columns,
                ascending=ascending,
            )
            .reset_index(
                drop=True
            )
        )

    # ========================================================
    # COMPLETE FIT
    # ========================================================

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> FeatureSelectionResult:

        self._validate_input(
            X_train,
            y_train,
        )

        # ====================================================
        # LAYER 1
        # ====================================================

        (
            X_layer1,
            layer1_removed,
            quality_report,
        ) = self._layer1_data_quality(
            X_train
        )

        # ====================================================
        # LAYER 2
        # ====================================================

        (
            X_layer2,
            layer2_removed,
            correlation_report,
        ) = self._layer2_correlation(
            X_layer1,
            y_train,
        )

        # ====================================================
        # LAYER 3
        # ====================================================

        (
            X_layer3,
            layer3_weak_signal,
            mi_report,
        ) = self._layer3_mutual_information(
            X_layer2,
            y_train,
        )

        # ====================================================
        # LAYER 4
        # ====================================================

        (
            X_final,
            layer4_removed,
            model_importance_report,
        ) = self._layer4_model_importance(
            X_layer3,
            y_train,
        )

        selected_features = list(
            X_final.columns
        )

        # ====================================================
        # FINAL REPORT
        # ====================================================

        final_report = (
            self._build_final_report(
                X_train=X_train,
                selected_features=selected_features,
                quality_report=quality_report,
                correlation_report=correlation_report,
                mi_report=mi_report,
                model_importance_report=(
                    model_importance_report
                ),
                layer1_removed=(
                    layer1_removed
                ),
                layer2_removed=(
                    layer2_removed
                ),
                layer4_removed=(
                    layer4_removed
                ),
            )
        )

        result = FeatureSelectionResult(

            selected_features=(
                selected_features
            ),

            layer1_removed=(
                layer1_removed
            ),

            layer2_removed=(
                layer2_removed
            ),

            layer3_weak_signal=(
                layer3_weak_signal
            ),

            layer4_removed=(
                layer4_removed
            ),

            quality_report=(
                quality_report
            ),

            correlation_report=(
                correlation_report
            ),

            mi_report=(
                mi_report
            ),

            model_importance_report=(
                model_importance_report
            ),

            final_report=(
                final_report
            ),
        )

        self.result_ = result

        self.fitted_ = True

        return result

    # ========================================================
    # TRANSFORM
    # ========================================================

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:

        if not self.fitted_:

            raise RuntimeError(
                "Feature selector has not been fitted."
            )

        selected = (
            self.result_
            .selected_features
        )

        missing = [
            feature
            for feature in selected
            if feature not in X.columns
        ]

        if missing:

            raise ValueError(
                "Input matrix is missing final "
                f"features: {missing}"
            )

        return X[
            selected
        ].copy()

    # ========================================================
    # FIT + TRANSFORM
    # ========================================================

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> pd.DataFrame:

        self.fit(
            X_train,
            y_train,
        )

        return self.transform(
            X_train
        )

    # ========================================================
    # GET FINAL FEATURES
    # ========================================================

    def get_selected_features(
        self,
    ) -> List[str]:

        if not self.fitted_:

            raise RuntimeError(
                "Feature selector has not been fitted."
            )

        return list(
            self.result_
            .selected_features
        )

    # ========================================================
    # SAVE FEATURE CONTRACT
    # ========================================================

    def save_feature_contract(
        self,
        path: str | Path,
    ) -> None:

        if not self.fitted_:

            raise RuntimeError(
                "Feature selector has not been fitted."
            )

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result = self.result_

        contract = {

            "selected_features": (
                result.selected_features
            ),

            "layer1_removed": (
                result.layer1_removed
            ),

            "layer2_removed": (
                result.layer2_removed
            ),

            "layer3_weak_signal": (
                result.layer3_weak_signal
            ),

            "layer4_removed": (
                result.layer4_removed
            ),

            "n_selected_features": (
                len(
                    result.selected_features
                )
            ),

            "correlation_threshold": (
                self.config.correlation_threshold
            ),

            "cumulative_importance": (
                self.config.cumulative_importance
            ),
        }

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                contract,
                file,
                indent=4,
            )


# ============================================================
# PRIMARY PROGRAMMATIC API
# ============================================================

def select_phase1_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: FeatureSelectionConfig | None = None,
) -> FeatureSelectionResult:

    selector = FraudFeatureSelector(
        config=config
    )

    return selector.fit(
        X_train,
        y_train,
    )


# ============================================================
# BACKWARD-COMPATIBLE PIPELINE API
# ============================================================

def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    correlation_threshold: float = 0.85,
) -> Tuple[
    List[str],
    List[str],
]:

    """
    Compatibility function for scripts/run_phase1.py.

    Existing pipeline expects:

        selected_features, removed_features = (
            select_features(
                X_train,
                y_train,
                correlation_threshold=0.85,
            )
        )

    The complete four-layer selector is executed internally.

    Returns
    -------
    selected_features:
        Final features selected by all four layers.

    removed_features:
        All features that are not present in the final
        feature contract.
    """

    config = FeatureSelectionConfig(
        correlation_threshold=(
            correlation_threshold
        )
    )

    selector = FraudFeatureSelector(
        config=config
    )

    result = selector.fit(
        X_train,
        y_train,
    )

    selected_features = list(
        result.selected_features
    )

    removed_features = [
        feature
        for feature in X_train.columns
        if feature not in selected_features
    ]

    return (
        selected_features,
        removed_features,
    )