from pathlib import Path

import pandas as pd

from src.logger.logger import get_logger

from src.components.phase1.feature_engineering import (
    engineer_phase1_features
)

from src.components.phase1.model_registry import (
    Phase1ModelRegistry
)

from src.components.phase1.prediction import (
    Phase1Predictor
)


logger = get_logger(__name__)


class Phase1PredictionPipeline:

    def __init__(
        self,
        model_path,
        threshold=0.84
    ):

        self.model_path = model_path

        self.threshold = threshold

    def run(
        self,
        input_path,
        output_path
    ):

        logger.info(
            "Starting Phase 1 prediction pipeline"
        )

        df = pd.read_csv(
            input_path
        )

        df, X, _ = (
            engineer_phase1_features(
                df
            )
        )

        registry = (
            Phase1ModelRegistry(
                self.model_path
            )
        )

        model = registry.load()

        predictor = Phase1Predictor(
            model,
            self.threshold
        )

        probabilities, predictions = (
            predictor.predict(X)
        )

        output = predictor.build_output(
            df,
            probabilities,
            predictions
        )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        output.to_csv(
            output_path,
            index=False
        )

        logger.info(
            "Predictions saved to %s",
            output_path
        )

        return output