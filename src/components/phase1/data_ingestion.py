import pandas as pd

from src.logger.logger import get_logger
from src.exception.exception import FraudSentinelException


logger = get_logger(__name__)


class Phase1DataIngestion:

    def __init__(self, data_path):

        self.data_path = data_path

    def load_data(self):

        try:

            logger.info(
                "Loading Phase 1 transaction dataset: %s",
                self.data_path
            )

            df = pd.read_csv(self.data_path)

            logger.info(
                "Dataset loaded successfully. Shape=%s",
                df.shape
            )

            return df

        except Exception as e:

            logger.exception("Failed to load Phase 1 dataset")

            raise FraudSentinelException(
                "Phase 1 data ingestion failed",
                e
            )