import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import XAIConfig
from .fusion_explainer import explain_fusion
from .loader import load_fusion_chunks
from .spike_explainer import explain_spikes
from .transaction_explainer import explain_transactions


def _append_csv(
    df: pd.DataFrame,
    path: Path,
    write_header: bool,
) -> None:
    """
    Append a DataFrame to CSV without retaining previous chunks.
    """

    df.to_csv(
        path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
    )


def _build_unified_xai(
    df: pd.DataFrame,
    transaction_xai: pd.DataFrame,
    spike_xai: pd.DataFrame,
    fusion_xai: pd.DataFrame,
    config: XAIConfig,
) -> pd.DataFrame:
    """
    Build unified XAI output for one chunk.
    """

    transaction_text = (
        transaction_xai[
            "transaction_explanation"
        ]
        .astype(str)
    )

    spike_text = (
        spike_xai[
            "spike_explanation"
        ]
        .astype(str)
    )

    fusion_text = (
        fusion_xai[
            "fusion_explanation"
        ]
        .astype(str)
    )

    amount = pd.to_numeric(
        df["amount"],
        errors="coerce",
    ).fillna(0)

    expected_loss = pd.to_numeric(
        df["transaction_expected_loss"],
        errors="coerce",
    ).fillna(0)

    expected_exposure = pd.to_numeric(
        df["expected_fraud_exposure"],
        errors="coerce",
    ).fillna(0)

    event_active = (
        df["event_active"]
        .fillna(False)
        .astype(bool)
    )

    event_id = (
        df["event_id"]
        .fillna("")
        .astype(str)
    )

    response_action = (
        df["response_action"]
        .fillna("MONITOR")
        .astype(str)
    )

    response_priority = (
        df["response_priority"]
        .fillna("LOW")
        .astype(str)
    )

    risk_band = (
        df["risk_band"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    spike_state = (
        df["spike_state"]
        .fillna("NORMAL")
        .astype(str)
    )

    unified_risk = pd.to_numeric(
        df["unified_risk_score"],
        errors="coerce",
    ).fillna(0)

    fraud_probability = pd.to_numeric(
        df["fraud_probability"],
        errors="coerce",
    ).fillna(0)

    # --------------------------------------------------------
    # Event explanation
    # --------------------------------------------------------

    event_explanation = pd.Series(
        np.where(
            event_active,
            "The transaction is associated with active event "
            + event_id
            + ".",
            "The transaction is not currently associated with "
            "an active fraud event.",
        ),
        index=df.index,
    )

    # --------------------------------------------------------
    # Financial explanation
    # --------------------------------------------------------

    financial_explanation = (
        "The transaction amount is "
        + amount.map(lambda x: f"{x:.2f}")
        + ", with transaction-level expected loss of "
        + expected_loss.map(lambda x: f"{x:.2f}")
        + " and combined expected fraud exposure of "
        + expected_exposure.map(lambda x: f"{x:.2f}")
        + "."
    )

    # --------------------------------------------------------
    # Response explanation
    # --------------------------------------------------------

    response_explanation = (
        "Recommended response: "
        + response_action
        + " with priority "
        + response_priority
        + "."
    )

    # --------------------------------------------------------
    # Primary reason
    # --------------------------------------------------------

    primary_reason = (
        fusion_xai[
            "fusion_primary_reason"
        ]
        .astype(str)
    )

    # --------------------------------------------------------
    # Full explanation
    # --------------------------------------------------------

    full_explanation = (
        "Transaction "
        + df["transaction_id"].astype(str)
        + " for merchant "
        + df["merchant_id"].astype(str)
        + " has a unified risk score of "
        + unified_risk.map(lambda x: f"{x:.2f}")
        + " and is classified as "
        + risk_band
        + " risk. "
        + transaction_text
        + " "
        + spike_text
        + " "
        + fusion_text
        + " "
        + financial_explanation
        + " "
        + event_explanation
        + " "
        + response_explanation
    )

    # --------------------------------------------------------
    # Unified dataframe
    # --------------------------------------------------------

    unified_xai = pd.DataFrame(
        {
            "xai_version":
                config.explanation_version,

            "xai_transaction_id":
                df["transaction_id"],

            "xai_merchant_id":
                df["merchant_id"],

            "xai_timestamp":
                df["timestamp"],

            "xai_risk_band":
                risk_band,

            "xai_spike_state":
                spike_state,

            "xai_unified_risk_score":
                unified_risk,

            "xai_fraud_probability":
                fraud_probability,

            "xai_primary_reason":
                primary_reason,

            "xai_transaction_reason":
                transaction_xai[
                    "transaction_primary_reason"
                ],

            "xai_spike_reason":
                spike_xai[
                    "spike_primary_reason"
                ],

            "xai_fusion_reason":
                fusion_xai[
                    "fusion_primary_reason"
                ],

            "xai_transaction_explanation":
                transaction_text,

            "xai_spike_explanation":
                spike_text,

            "xai_fusion_explanation":
                fusion_text,

            "xai_financial_explanation":
                financial_explanation,

            "xai_event_explanation":
                event_explanation,

            "xai_response_explanation":
                response_explanation,

            "xai_full_explanation":
                full_explanation,

            "xai_event_id":
                event_id,

            "xai_event_active":
                event_active,

            "xai_expected_exposure":
                expected_exposure,

            "xai_response_action":
                response_action,

            "xai_response_priority":
                response_priority,
        }
    )

    # --------------------------------------------------------
    # Preserve useful Fusion columns
    # --------------------------------------------------------

    key_columns = [
        "transaction_id",
        "timestamp",
        "merchant_id",
        "card_id",
        "amount",
        "fraud_probability",
        "risk_score",
        "risk_band",
        "phase1_score",
        "phase2_score",
        "spike_state",
        "unified_risk_score",
        "response_action",
        "response_priority",
        "alert_flag",
        "event_id",
        "event_active",
    ]

    existing_keys = [
        column
        for column in key_columns
        if column in df.columns
    ]

    unified_xai = pd.concat(
        [
            df[existing_keys].reset_index(drop=True),
            unified_xai.reset_index(drop=True),
        ],
        axis=1,
    )

    return unified_xai


def run_xai_pipeline(
    input_path: Path,
    output_dir: Path,
    config: Optional[XAIConfig] = None,
) -> pd.DataFrame:

    config = config or XAIConfig()

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 72)
    print("FRAUDSENTINEL AI — XAI PIPELINE")
    print("=" * 72)

    print(f"[INPUT] {input_path}")

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    transaction_path = (
        output_dir
        / config.transaction_output
    )

    spike_path = (
        output_dir
        / config.spike_output
    )

    unified_path = (
        output_dir
        / config.unified_output
    )

    summary_path = (
        output_dir
        / config.summary_output
    )

    # --------------------------------------------------------
    # Remove stale outputs
    # --------------------------------------------------------

    for path in [
        transaction_path,
        spike_path,
        unified_path,
    ]:
        if path.exists():
            path.unlink()

    # --------------------------------------------------------
    # Aggregated statistics
    # --------------------------------------------------------

    rows_processed = 0

    transaction_rows = 0
    spike_rows = 0
    unified_rows = 0

    alert_count = 0
    active_event_rows = 0

    risk_band_counts = {}
    spike_state_counts = {}

    unified_risk_sum = 0.0
    unified_risk_count = 0
    max_unified_risk = 0.0

    first_transaction_write = True
    first_spike_write = True
    first_unified_write = True

    chunk_number = 0

    # --------------------------------------------------------
    # Stream Fusion CSV
    # --------------------------------------------------------

    print("\n[STREAM] Processing Fusion output in chunks...")

    for df in load_fusion_chunks(
        input_path,
        strict_schema=False,
        chunk_size=50_000,
    ):

        chunk_number += 1

        chunk_rows = len(df)

        rows_processed += chunk_rows

        print(
            f"\n[CHUNK {chunk_number}] "
            f"{chunk_rows:,} rows"
            f" | Total processed: "
            f"{rows_processed:,}"
        )

        # ====================================================
        # TRANSACTION XAI
        # ====================================================

        transaction_xai = explain_transactions(
            df,
            config,
        )

        _append_csv(
            transaction_xai,
            transaction_path,
            write_header=first_transaction_write,
        )

        first_transaction_write = False

        transaction_rows += len(
            transaction_xai
        )

        # ====================================================
        # SPIKE XAI
        # ====================================================

        spike_xai = explain_spikes(
            df,
            config,
        )

        _append_csv(
            spike_xai,
            spike_path,
            write_header=first_spike_write,
        )

        first_spike_write = False

        spike_rows += len(
            spike_xai
        )

        # ====================================================
        # FUSION XAI
        # ====================================================

        fusion_xai = explain_fusion(
            df,
            config,
        )

        # ====================================================
        # UNIFIED XAI
        # ====================================================

        unified_xai = _build_unified_xai(
            df=df,
            transaction_xai=transaction_xai,
            spike_xai=spike_xai,
            fusion_xai=fusion_xai,
            config=config,
        )

        _append_csv(
            unified_xai,
            unified_path,
            write_header=first_unified_write,
        )

        first_unified_write = False

        unified_rows += len(
            unified_xai
        )

        # ====================================================
        # AGGREGATE SUMMARY
        # ====================================================

        state_series = (
            df["spike_state"]
            .fillna("UNKNOWN")
            .astype(str)
        )

        for state, count in (
            state_series.value_counts()
            .to_dict()
            .items()
        ):
            spike_state_counts[state] = (
                spike_state_counts.get(
                    state,
                    0,
                )
                + int(count)
            )

        band_series = (
            df["risk_band"]
            .fillna("UNKNOWN")
            .astype(str)
        )

        for band, count in (
            band_series.value_counts()
            .to_dict()
            .items()
        ):
            risk_band_counts[band] = (
                risk_band_counts.get(
                    band,
                    0,
                )
                + int(count)
            )

        alert_count += int(
            df["alert_flag"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        active_event_rows += int(
            df["event_active"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        unified_values = pd.to_numeric(
            df["unified_risk_score"],
            errors="coerce",
        ).dropna()

        if len(unified_values) > 0:

            unified_risk_sum += float(
                unified_values.sum()
            )

            unified_risk_count += int(
                len(unified_values)
            )

            chunk_max = float(
                unified_values.max()
            )

            max_unified_risk = max(
                max_unified_risk,
                chunk_max,
            )

        # ----------------------------------------------------
        # Explicitly release chunk memory
        # ----------------------------------------------------

        del transaction_xai
        del spike_xai
        del fusion_xai
        del unified_xai
        del df

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    if unified_risk_count > 0:
        mean_unified_risk = (
            unified_risk_sum
            / unified_risk_count
        )
    else:
        mean_unified_risk = 0.0

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "project": "FraudSentinel AI",
        "layer": "Explainable AI",
        "xai_version":
            config.explanation_version,

        "input": str(input_path),

        "processing_mode":
            "chunked_streaming",

        "chunk_size":
            50_000,

        "chunks_processed":
            chunk_number,

        "rows_processed":
            rows_processed,

        "transaction_explanations":
            transaction_rows,

        "spike_explanations":
            spike_rows,

        "unified_explanations":
            unified_rows,

        "alert_count":
            alert_count,

        "active_event_rows":
            active_event_rows,

        "risk_band_distribution":
            risk_band_counts,

        "spike_state_distribution":
            spike_state_counts,

        "mean_unified_risk":
            mean_unified_risk,

        "max_unified_risk":
            max_unified_risk,

        "output_files": {
            "transaction_explanations":
                str(transaction_path),

            "spike_explanations":
                str(spike_path),

            "unified_explanations":
                str(unified_path),
        },
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    print(
        f"\n[OK] {transaction_path}"
    )

    print(
        f"[OK] {spike_path}"
    )

    print(
        f"[OK] {unified_path}"
    )

    print(
        f"[OK] {summary_path}"
    )

    print("\n" + "=" * 72)
    print("XAI PIPELINE COMPLETE")
    print("=" * 72)

    print(
        f"Rows processed       : "
        f"{rows_processed:,}"
    )

    print(
        f"Chunks processed     : "
        f"{chunk_number:,}"
    )

    print(
        f"Mean unified risk    : "
        f"{mean_unified_risk:.4f}"
    )

    print(
        f"Max unified risk     : "
        f"{max_unified_risk:.4f}"
    )

    # --------------------------------------------------------
    # Return compact result rather than 1.3M rows.
    #
    # This is intentional:
    # returning the entire unified XAI DataFrame would recreate
    # the same memory problem we just eliminated.
    # --------------------------------------------------------

    return pd.DataFrame(
        {
            "rows_processed": [
                rows_processed
            ],

            "unified_risk_score": [
                mean_unified_risk
            ],

            "max_unified_risk": [
                max_unified_risk
            ],

            "alert_count": [
                alert_count
            ],

            "active_event_rows": [
                active_event_rows
            ],
        }
    )

