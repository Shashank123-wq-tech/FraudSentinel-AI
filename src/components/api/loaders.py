from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import (
    FUSION_FILE,
    FUSION_EVENTS_FILE,
    PHASE2_WINDOWS_FILE,
    PHASE2_EVENTS_FILE,
    XAI_FILE,
    IMPACT_FORECAST_FILE,
    IMPACT_SCENARIOS_FILE,
    RESPONSE_RECOMMENDATIONS_FILE,
    RESPONSE_ACTIONS_FILE,
    FUSION_CHUNK_SIZE,
)


# ============================================================
# GENERIC CSV
# ============================================================

def load_csv(path: Path) -> pd.DataFrame:

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


# ============================================================
# ARTIFACT STATUS
# ============================================================

def get_artifact_status() -> Dict[str, bool]:

    return {
        "fusion": FUSION_FILE.exists(),
        "phase2_events": PHASE2_EVENTS_FILE.exists(),
        "phase2_windows": PHASE2_WINDOWS_FILE.exists(),
        "xai": XAI_FILE.exists(),
        "impact_forecast": IMPACT_FORECAST_FILE.exists(),
        "impact_scenarios": IMPACT_SCENARIOS_FILE.exists(),
        "response_recommendations": RESPONSE_RECOMMENDATIONS_FILE.exists(),
        "response_actions": RESPONSE_ACTIONS_FILE.exists(),
    }


# ============================================================
# SMALL ARTIFACTS
# ============================================================

def load_phase2_events() -> pd.DataFrame:
    return load_csv(PHASE2_EVENTS_FILE)


def load_phase2_windows() -> pd.DataFrame:
    return load_csv(PHASE2_WINDOWS_FILE)


def load_xai() -> pd.DataFrame:
    return load_csv(XAI_FILE)


def load_impact_forecast() -> pd.DataFrame:
    return load_csv(IMPACT_FORECAST_FILE)


def load_impact_scenarios() -> pd.DataFrame:
    return load_csv(IMPACT_SCENARIOS_FILE)


def load_response_recommendations() -> pd.DataFrame:
    return load_csv(RESPONSE_RECOMMENDATIONS_FILE)


def load_response_actions() -> pd.DataFrame:
    return load_csv(RESPONSE_ACTIONS_FILE)


def load_fusion_events() -> pd.DataFrame:
    return load_csv(FUSION_EVENTS_FILE)


# ============================================================
# FUSION CHUNK ITERATOR
# ============================================================

def iter_fusion():

    if not FUSION_FILE.exists():
        return

    for chunk in pd.read_csv(
        FUSION_FILE,
        chunksize=FUSION_CHUNK_SIZE,
        low_memory=True,
    ):
        yield chunk


# ============================================================
# FUSION FILTER
# ============================================================

def find_fusion_transactions(
    transaction_id: Optional[str] = None,
    merchant_id: Optional[str] = None,
    limit: int = 100,
) -> pd.DataFrame:

    matches = []

    remaining = max(1, limit)

    for chunk in iter_fusion():

        if transaction_id is not None:
            chunk = chunk[
                chunk["transaction_id"]
                .astype(str)
                .eq(str(transaction_id))
            ]

        if merchant_id is not None:
            chunk = chunk[
                chunk["merchant_id"]
                .astype(str)
                .eq(str(merchant_id))
            ]

        if not chunk.empty:

            matches.append(
                chunk.head(remaining)
            )

            remaining -= len(matches[-1])

            if remaining <= 0:
                break

    if not matches:
        return pd.DataFrame()

    return pd.concat(
        matches,
        ignore_index=True,
    ).head(limit)


# ============================================================
# TOP RISK TRANSACTIONS
# ============================================================

def get_top_risk_transactions(
    limit: int = 50,
) -> pd.DataFrame:

    candidates = []

    for chunk in iter_fusion():

        if "unified_risk_score" not in chunk.columns:
            continue

        chunk["unified_risk_score"] = pd.to_numeric(
            chunk["unified_risk_score"],
            errors="coerce",
        )

        top = (
            chunk
            .nlargest(limit, "unified_risk_score")
        )

        candidates.append(top)

    if not candidates:
        return pd.DataFrame()

    result = pd.concat(
        candidates,
        ignore_index=True,
    )

    return (
        result
        .nlargest(limit, "unified_risk_score")
        .reset_index(drop=True)
    )


# ============================================================
# MERCHANT SEARCH
# ============================================================

def get_merchant_transactions(
    merchant_id: str,
    limit: int = 100,
) -> pd.DataFrame:

    return find_fusion_transactions(
        merchant_id=merchant_id,
        limit=limit,
    )