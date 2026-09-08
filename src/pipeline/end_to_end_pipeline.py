"""End-to-end Phase-1 -> Phase-2 coordinator.

1) Phase-1 model is trained once on historical/raw data.
2) Phase-1 produces transaction-level fraud probabilities/risk bands.
3) The exact Phase-1 risk output table is passed to the existing Phase-2 pipeline.
4) Phase-2 uses fraud_probability as its primary transaction-level signal and
   keeps fraud_label only for evaluation, matching the uploaded Phase-2 contract.
"""
from __future__ import annotations

import os

from src.pipeline.phase1_training_pipeline import train_phase1
from src.pipeline.phase2_spike_pipeline import run_phase2_pipeline


def run_end_to_end(raw_df, phase1_artifact_dir="artifacts/phase1", phase2_artifact_dir="artifacts/phase2"):
    phase1 = train_phase1(raw_df, artifact_dir=phase1_artifact_dir)
    phase2 = run_phase2_pipeline(
        raw_df=raw_df,
        risk_output_table=phase1["risk_output"],
        output_dir=phase2_artifact_dir,
    )
    return {"phase1": phase1, "phase2": phase2}
