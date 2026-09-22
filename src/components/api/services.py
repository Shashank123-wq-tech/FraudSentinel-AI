from typing import Optional, Any

import math
import pandas as pd

from .loaders import (
    get_artifact_status,
    find_fusion_transactions,
    get_top_risk_transactions,
    get_merchant_transactions,
    iter_fusion,
    load_phase2_events,
    load_phase2_windows,
    load_xai,
    load_impact_forecast,
    load_impact_scenarios,
    load_response_recommendations,
    load_response_actions,
)


# ============================================================
# JSON-SAFE HELPERS
# ============================================================

def _sanitize_value(value: Any) -> Any:
    """
    Convert a single Python / NumPy / pandas value into a
    JSON-safe native Python value.

    JSON does NOT allow:
        NaN
        +Infinity
        -Infinity

    These values are converted to None.

    Also converts NumPy scalar values into native Python types.
    """

    # --------------------------------------------------------
    # None
    # --------------------------------------------------------
    if value is None:
        return None

    # --------------------------------------------------------
    # pandas missing values
    # --------------------------------------------------------
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    # --------------------------------------------------------
    # NumPy scalar -> Python scalar
    # --------------------------------------------------------
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass

    # --------------------------------------------------------
    # Python float
    # --------------------------------------------------------
    if isinstance(value, float):

        if math.isnan(value):
            return None

        if math.isinf(value):
            return None

        return float(value)

    # --------------------------------------------------------
    # Integer
    # --------------------------------------------------------
    if isinstance(value, int):
        return int(value)

    # --------------------------------------------------------
    # Boolean
    # --------------------------------------------------------
    if isinstance(value, bool):
        return bool(value)

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------
    if isinstance(value, dict):
        return {
            str(key): _sanitize_value(item)
            for key, item in value.items()
        }

    # --------------------------------------------------------
    # List / Tuple
    # --------------------------------------------------------
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_value(item)
            for item in value
        ]

    # --------------------------------------------------------
    # Everything else
    # --------------------------------------------------------
    return value


def clean_records(
    df: Optional[pd.DataFrame],
):
    """
    Convert a pandas DataFrame into JSON-safe records.

    This function is the central serialization boundary for
    the FraudSentinel API.

    It protects FastAPI responses from:
        - NaN
        - +Infinity
        - -Infinity
        - NumPy scalar types
        - pandas missing values
        - problematic floating-point values

    Returns:
        list[dict]
    """

    # --------------------------------------------------------
    # Empty / None DataFrame
    # --------------------------------------------------------
    if df is None or df.empty:
        return []

    result = df.copy()

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------
    result = result.replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )

    # --------------------------------------------------------
    # Convert to object dtype.
    #
    # This is important because assigning None into a normal
    # float column can cause pandas to convert it back into
    # NaN.
    # --------------------------------------------------------
    result = result.astype(object)

    # --------------------------------------------------------
    # Replace all pandas missing values with Python None.
    # --------------------------------------------------------
    result = result.where(
        pd.notna(result),
        None,
    )

    # --------------------------------------------------------
    # Convert DataFrame to dictionaries.
    # --------------------------------------------------------
    records = result.to_dict(
        orient="records",
    )

    # --------------------------------------------------------
    # Final recursive safety pass.
    #
    # This catches values that survived pandas conversion.
    # --------------------------------------------------------
    safe_records = []

    for record in records:

        safe_record = {}

        for key, value in record.items():

            safe_record[str(key)] = _sanitize_value(
                value
            )

        safe_records.append(
            safe_record
        )

    return safe_records


# ============================================================
# OVERVIEW
# ============================================================

def get_overview():

    total_transactions = 0

    total_amount = 0.0

    expected_fraud_amount = 0.0

    risk_sum = 0.0

    risk_count = 0

    maximum_risk = 0.0

    merchants = set()

    alerts = 0

    # ---------------------------------------------------------
    # PROCESS FUSION DATA
    # ---------------------------------------------------------

    for chunk in iter_fusion():

        # -----------------------------------------------------
        # Transactions
        # -----------------------------------------------------

        total_transactions += int(
            len(chunk)
        )

        # -----------------------------------------------------
        # Merchants
        # -----------------------------------------------------

        if "merchant_id" in chunk.columns:

            merchant_values = (
                chunk["merchant_id"]
                .dropna()
                .astype(str)
                .unique()
            )

            merchants.update(
                merchant_values
            )

        # -----------------------------------------------------
        # Transaction amount
        # -----------------------------------------------------

        if "amount" in chunk.columns:

            amount = pd.to_numeric(
                chunk["amount"],
                errors="coerce",
            ).fillna(0)

            total_amount += float(
                amount.sum()
            )

        # -----------------------------------------------------
        # Expected fraud exposure
        # -----------------------------------------------------

        if "expected_fraud_exposure" in chunk.columns:

            exposure = pd.to_numeric(
                chunk["expected_fraud_exposure"],
                errors="coerce",
            ).fillna(0)

            expected_fraud_amount += float(
                exposure.sum()
            )

        # -----------------------------------------------------
        # Unified risk
        # -----------------------------------------------------

        if "unified_risk_score" in chunk.columns:

            risk = pd.to_numeric(
                chunk["unified_risk_score"],
                errors="coerce",
            ).dropna()

            if not risk.empty:

                risk_sum += float(
                    risk.sum()
                )

                risk_count += int(
                    len(risk)
                )

                maximum_risk = max(
                    maximum_risk,
                    float(risk.max()),
                )

        # -----------------------------------------------------
        # Alerts
        # -----------------------------------------------------

        if "alert_flag" in chunk.columns:

            alert_values = chunk[
                "alert_flag"
            ]

            # Handle string boolean values.
            if alert_values.dtype == object:

                alert_values = (
                    alert_values
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(
                        [
                            "true",
                            "1",
                            "yes",
                        ]
                    )
                )

            else:

                alert_values = (
                    alert_values
                    .fillna(False)
                    .astype(bool)
                )

            alerts += int(
                alert_values.sum()
            )

    # =========================================================
    # PHASE 2 SPIKE STATES
    # =========================================================

    events = load_phase2_events()

    early = 0

    verified = 0

    critical = 0

    if (
        not events.empty
        and "spike_state" in events.columns
    ):

        states = (
            events["spike_state"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        early = int(
            states.eq(
                "EARLY_WARNING"
            ).sum()
        )

        verified = int(
            states.eq(
                "VERIFIED_FRAUD_SPIKE"
            ).sum()
        )

        critical = int(
            states.eq(
                "CRITICAL_ACTIVE_SPIKE"
            ).sum()
        )

    # =========================================================
    # MEAN RISK
    # =========================================================

    if risk_count > 0:

        mean_unified_risk = float(
            risk_sum / risk_count
        )

    else:

        mean_unified_risk = 0.0

    # =========================================================
    # FINAL API RESPONSE
    # =========================================================

    response = {
        "transactions": int(
            total_transactions
        ),

        "merchants": int(
            len(merchants)
        ),

        "active_spikes": int(
            early
            + verified
            + critical
        ),

        "alerts": int(
            alerts
        ),

        "total_amount": float(
            total_amount
        ),

        "expected_fraud_amount": float(
            expected_fraud_amount
        ),

        "mean_unified_risk": float(
            mean_unified_risk
        ),

        "maximum_unified_risk": float(
            maximum_risk
        ),

        "critical_events": int(
            critical
        ),

        "verified_events": int(
            verified
        ),

        "early_warning_events": int(
            early
        ),
    }

    # Final safety pass.
    return _sanitize_value(
        response
    )


# ============================================================
# TRANSACTIONS
# ============================================================

def get_transactions(
    transaction_id: Optional[str] = None,
    merchant_id: Optional[str] = None,
    limit: int = 100,
):

    df = find_fusion_transactions(
        transaction_id=transaction_id,
        merchant_id=merchant_id,
        limit=limit,
    )

    return clean_records(
        df
    )


# ============================================================
# TOP-RISK TRANSACTIONS
# ============================================================

def get_top_risk(
    limit: int = 50,
):

    df = get_top_risk_transactions(
        limit=limit
    )

    return clean_records(
        df
    )


# ============================================================
# SPIKES
# ============================================================

def get_spikes():

    df = load_phase2_events()

    return clean_records(
        df
    )


def get_spike(
    event_id: str,
):

    df = load_phase2_events()

    if (
        df.empty
        or "event_id" not in df.columns
    ):
        return None

    result = df[
        df["event_id"]
        .astype(str)
        .eq(
            str(event_id)
        )
    ]

    if result.empty:
        return None

    records = clean_records(
        result.head(1)
    )

    if not records:
        return None

    return records[0]


# ============================================================
# XAI
# ============================================================

def get_xai(
    event_id: Optional[str] = None,
):

    df = load_xai()

    if df.empty:
        return []

    # ---------------------------------------------------------
    # Event filtering
    # ---------------------------------------------------------

    if (
        event_id is not None
        and "event_id" in df.columns
    ):

        df = df[
            df["event_id"]
            .astype(str)
            .eq(
                str(event_id)
            )
        ]

    # ---------------------------------------------------------
    # Limit API payload
    # ---------------------------------------------------------

    df = df.head(100)

    return clean_records(
        df
    )


# ============================================================
# IMPACT FORECAST
# ============================================================

def get_impact(
    event_id: str,
):

    df = load_impact_forecast()

    if df.empty:
        return []

    # ---------------------------------------------------------
    # Event filtering
    # ---------------------------------------------------------

    if "event_id" in df.columns:

        df = df[
            df["event_id"]
            .astype(str)
            .eq(
                str(event_id)
            )
        ]

    return clean_records(
        df
    )


# ============================================================
# IMPACT SCENARIOS
# ============================================================

def get_scenarios(
    event_id: str,
):

    df = load_impact_scenarios()

    if df.empty:
        return []

    # ---------------------------------------------------------
    # Event filtering
    # ---------------------------------------------------------

    if "event_id" in df.columns:

        df = df[
            df["event_id"]
            .astype(str)
            .eq(
                str(event_id)
            )
        ]

    return clean_records(
        df
    )


# ============================================================
# RESPONSE RECOMMENDATIONS
# ============================================================

def get_response(
    event_id: Optional[str] = None,
):

    df = load_response_recommendations()

    if df.empty:
        return []

    # ---------------------------------------------------------
    # Event filtering
    # ---------------------------------------------------------

    if (
        event_id is not None
        and "event_id" in df.columns
    ):

        df = df[
            df["event_id"]
            .astype(str)
            .eq(
                str(event_id)
            )
        ]

    # ---------------------------------------------------------
    # Limit payload
    # ---------------------------------------------------------

    df = df.head(100)

    return clean_records(
        df
    )


# ============================================================
# RESPONSE ACTIONS
# ============================================================

def get_response_actions(
    event_id: Optional[str] = None,
):

    df = load_response_actions()

    if df.empty:
        return []

    # ---------------------------------------------------------
    # Event filtering
    # ---------------------------------------------------------

    if (
        event_id is not None
        and "event_id" in df.columns
    ):

        df = df[
            df["event_id"]
            .astype(str)
            .eq(
                str(event_id)
            )
        ]

    # ---------------------------------------------------------
    # Limit payload
    # ---------------------------------------------------------

    df = df.head(100)

    return clean_records(
        df
    )


# ============================================================
# MERCHANT
# ============================================================

def get_merchant(
    merchant_id: str,
):

    # ---------------------------------------------------------
    # Merchant transactions
    # ---------------------------------------------------------

    transactions = get_merchant_transactions(
        merchant_id=merchant_id,
        limit=100,
    )

    # ---------------------------------------------------------
    # Phase 2 events
    # ---------------------------------------------------------

    events = load_phase2_events()

    merchant_events = pd.DataFrame()

    if (
        not events.empty
        and "merchant_id" in events.columns
    ):

        merchant_events = events[
            events["merchant_id"]
            .astype(str)
            .eq(
                str(merchant_id)
            )
        ]

    # ---------------------------------------------------------
    # Final response
    # ---------------------------------------------------------

    response = {
        "merchant_id": str(
            merchant_id
        ),

        "transactions": clean_records(
            transactions
        ),

        "spikes": clean_records(
            merchant_events
        ),
    }

    return _sanitize_value(
        response
    )