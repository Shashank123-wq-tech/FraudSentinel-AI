from pathlib import Path

import xgboost as xgb

from src.logger.logger import get_logger
from src.exception.exception import FraudSentinelException


logger = get_logger(__name__)


class Phase1ModelRegistry:

    def __init__(self, model_path):

        self.model_path = Path(model_path)

    def save(self, model):

        try:

            self.model_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            model.save_model(
                str(self.model_path)
            )

            logger.info(
                "Model saved: %s",
                self.model_path
            )

        except Exception as e:

            logger.exception(
                "Failed to save XGBoost model"
            )

            raise FraudSentinelException(
                "Model saving failed",
                e
            )

    def load(self):

        try:

            model = xgb.XGBClassifier()

            model.load_model(
                str(self.model_path)
            )

            logger.info(
                "Model loaded: %s",
                self.model_path
            )

            return model

        except Exception as e:

            logger.exception(
                "Failed to load XGBoost model"
            )

            raise FraudSentinelException(
                "Model loading failed",
                e
            )