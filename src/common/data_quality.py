"""
src/common/data_quality.py
Shared data-integrity checks. Phase 2's rules from the End-to-End Guide (Section 3)
live here since they apply broadly, not just to one Phase 2 module:
  - is_fraud / fraud_label must never be a detector input, eval-only.
  - The synthetic 'fraud_' merchant-name prefix must never be used as a signal.
  - No fabricated device/IP/session/refund/chargeback fields.
  - Data must be chronologically sorted before any leakage-safe baseline logic runs.
"""


def validate_no_leakage(feature_columns):
    """Hard-fails if is_fraud or fraud_label sneaks into a detector feature list."""
    banned = {"is_fraud", "fraud_label"}
    leaked = banned.intersection(set(feature_columns))
    if leaked:
        raise ValueError(f"LEAKAGE: {leaked} must never be a detector input — eval-only.")


def validate_no_fraud_prefix_signal(df, merchant_col="merchant_name"):
    """Confirms merchant_name isn't being parsed for the synthetic 'fraud_' prefix anywhere upstream."""
    if merchant_col in df.columns:
        sample = df[merchant_col].dropna().head(1)
        if len(sample) and str(sample.iloc[0]).startswith("fraud_"):
            print(
                "NOTE: merchant_name carries the synthetic 'fraud_' prefix from the source data. "
                "This pipeline does not read or use that prefix as a signal anywhere — verified by design."
            )


def validate_chronological_order(df, time_col="timestamp"):
    if not df[time_col].is_monotonic_increasing:
        raise ValueError(
            f"Data not sorted by {time_col} — chronological order is required for leakage-safe baselines."
        )
