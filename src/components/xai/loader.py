from pathlib import Path
from typing import Iterator

import pandas as pd

from .schema import validate_schema


DEFAULT_CHUNK_SIZE = 50_000


NUMERIC_COLUMNS = [
    "amount",
    "fraud_probability",
    "risk_score",
    "phase1_score",
    "phase2_score",
    "phase2_confidence",
    "phase2_state_multiplier",
    "statistical_evidence",
    "financial_evidence",
    "temporal_evidence",
    "breadth_score",
    "coordination_score",
    "tas",
    "fas",
    "materiality",
    "expected_fraud_count",
    "expected_fraud_amount",
    "risk_rate",
    "transaction_count",
    "total_amount",
    "unique_cards",
    "high_risk_count",
    "new_cards",
    "new_card_rate",
    "history_count",
    "ewma_value",
    "ewma_residual",
    "cusum_value",
    "acceleration",
    "peak_spike_score",
    "duration_minutes",
    "transaction_count_event",
    "total_amount_event",
    "expected_fraud_count_event",
    "expected_fraud_amount_event",
    "unique_cards_event",
    "max_coordination_score",
    "max_tas",
    "max_fas",
    "base_fused_score",
    "confidence_adjusted_score",
    "fusion_confidence",
    "unified_risk_score",
    "transaction_expected_loss",
    "phase2_expected_fraud_amount",
    "expected_fraud_exposure",
    "threshold",
    "_phase2_window_age_minutes",
]


def _normalize_fusion_chunk(
    df: pd.DataFrame,
    strict_schema: bool = False,
) -> pd.DataFrame:
    """
    Normalize one Fusion CSV chunk.

    Important:
    Numeric validation is intentionally NOT performed here.

    A chunk may legitimately contain an entire column of empty/NaN
    values even though the column contains valid values elsewhere
    in the complete Fusion dataset.

    Numeric values are therefore coerced safely with pandas.
    """

    # --------------------------------------------------------
    # Schema validation
    # --------------------------------------------------------

    validate_schema(
        df,
        strict=strict_schema,
    )

    # --------------------------------------------------------
    # Timestamp normalization
    # --------------------------------------------------------

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True,
        )

    for column in [
        "window_start",
        "window_end",
    ]:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
                utc=True,
            )

    # --------------------------------------------------------
    # Numeric normalization
    #
    # IMPORTANT:
    # Do this BEFORE any numeric validation.
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Boolean normalization
    # --------------------------------------------------------

    boolean_columns = [
        "predicted_fraud",
        "event_active",
        "alert_flag",
        "baseline_ready",
        "cusum_signal",
        "_phase2_window_matched",
    ]

    for column in boolean_columns:

        if column not in df.columns:
            continue

        if pd.api.types.is_bool_dtype(
            df[column]
        ):
            continue

        normalized = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df[column] = normalized.map(
            {
                "true": True,
                "1": True,
                "yes": True,
                "y": True,
                "false": False,
                "0": False,
                "no": False,
                "n": False,
            }
        )

    return df


def load_fusion_output(
    path: Path,
    strict_schema: bool = False,
) -> pd.DataFrame:
    """
    Load the complete unified FraudSentinel Fusion output.

    This function is retained for compatibility.

    For the large 1.3M-row Fusion artifact, the XAI pipeline
    should use load_fusion_chunks().
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Fusion output not found:\n{path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected CSV input, received: {path.suffix}"
        )

    df = pd.read_csv(
        path,
        low_memory=True,
    )

    return _normalize_fusion_chunk(
        df,
        strict_schema=strict_schema,
    )


def load_fusion_chunks(
    path: Path,
    strict_schema: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """
    Stream the Fusion CSV in memory-safe chunks.

    Parameters
    ----------
    path:
        Path to fraudsentinel_unified_risk.csv.

    strict_schema:
        Whether schema validation should be strict.

    chunk_size:
        Number of rows loaded into memory at once.

    Yields
    ------
    pd.DataFrame
        One normalized Fusion chunk at a time.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Fusion output not found:\n{path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected CSV input, received: {path.suffix}"
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    # --------------------------------------------------------
    # Stream CSV
    # --------------------------------------------------------

    for chunk in pd.read_csv(
        path,
        chunksize=chunk_size,
        low_memory=True,
    ):

        yield _normalize_fusion_chunk(
            chunk,
            strict_schema=strict_schema,
        )

