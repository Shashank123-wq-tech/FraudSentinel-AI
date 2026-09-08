from dataclasses import dataclass


@dataclass
class Phase1TrainingArtifacts:

    model_path: str
    feature_list_path: str
    metrics_path: str
    predictions_path: str

    threshold: float

    train_rows: int
    validation_rows: int
    test_rows: int

    best_iteration: int