from pathlib import Path

import pandas as pd

from src.pipeline.phase2_spike_pipeline import (
    run_phase2_pipeline,
)


def test_phase2_end_to_end(
    tmp_path: Path,
):

    rows = []

    base = pd.Timestamp(
        "2026-01-01T00:00:00Z"
    )

    # ---------------------------------------------------------
    # Long enough history to initialize the baseline
    # ---------------------------------------------------------

    for i in range(120):

        timestamp = (
            base
            + pd.Timedelta(
                seconds=30 * i
            )
        )

        if i < 100:
            probability = 0.01
            score = 1.0
            predicted = 0

        else:
            probability = 0.90
            score = 90.0
            predicted = 1

        rows.append(
            {
                "transaction_id":
                    f"T{i}",

                "timestamp":
                    timestamp.isoformat(),

                "merchant_id":
                    "M1",

                "card_id":
                    f"C{i % 20}",

                "amount":
                    100.0,

                "fraud_probability":
                    probability,

                "risk_score":
                    score,

                "risk_band":
                    (
                        "LOW"
                        if predicted == 0
                        else "CRITICAL"
                    ),

                "predicted_fraud":
                    predicted,

                "model_version":
                    "fraud_model_v1",

                "threshold":
                    0.5,
            }
        )

    input_file = (
        tmp_path
        / "transaction_risk.csv"
    )

    output_dir = (
        tmp_path
        / "phase2"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        input_file,
        index=False,
    )

    result = run_phase2_pipeline(
        phase1_risk_output=str(
            input_file
        ),
        output_dir=str(
            output_dir
        ),
        config_path="config/phase2.yaml",
    )

    assert "transactions" in result
    assert "windows" in result
    assert "fused" in result
    assert "events" in result
    assert "metadata" in result

    assert (
        output_dir
        / "merchant_spike_events.parquet"
    ).exists()

    assert (
        output_dir
        / "run_metadata.json"
    ).exists()