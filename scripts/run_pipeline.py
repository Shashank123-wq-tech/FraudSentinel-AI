"""
FraudSentinel AI
MLOps Pipeline Runner

CLI entry point for the complete inference pipeline.
"""

from src.mlops.pipelines.inference_pipeline import (
    main,
)


if __name__ == "__main__":

    raise SystemExit(
        main()
    )