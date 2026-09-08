from dataclasses import dataclass
from pathlib import Path


@dataclass
class Phase1Config:

    raw_data_path: str

    model_dir: str = "artifacts/phase1/models"
    prediction_dir: str = "artifacts/phase1/predictions"
    evaluation_dir: str = "artifacts/phase1/evaluations"

    model_name: str = "fraudsentinel_phase1_xgboost.json"

    threshold: float = 0.84

    random_state: int = 42

    test_quantile: float = 0.80

    n_estimators: int = 2000

    early_stopping_rounds: int = 30

    @property
    def model_path(self):

        return str(
            Path(self.model_dir) / self.model_name
        )