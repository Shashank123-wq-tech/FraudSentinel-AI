from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from .config import API_VERSION
from .loaders import get_artifact_status
from .services import (
    get_overview,
    get_transactions,
    get_top_risk,
    get_spikes,
    get_spike,
    get_xai,
    get_impact,
    get_scenarios,
    get_response,
    get_merchant,
)


router = APIRouter()


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "FraudSentinel AI Intelligence API",
        "version": API_VERSION,
    }


# ============================================================
# ARTIFACT STATUS
# ============================================================

@router.get("/status")
def status():

    return get_artifact_status()


# ============================================================
# OVERVIEW
# ============================================================

@router.get("/risk/overview")
def risk_overview():

    return get_overview()


# ============================================================
# TRANSACTIONS
# ============================================================

@router.get("/risk/transactions")
def risk_transactions(
    transaction_id: Optional[str] = None,
    merchant_id: Optional[str] = None,
    limit: int = Query(
        100,
        ge=1,
        le=500,
    ),
):

    return {
        "count": len(
            get_transactions(
                transaction_id=transaction_id,
                merchant_id=merchant_id,
                limit=limit,
            )
        ),
        "data": get_transactions(
            transaction_id=transaction_id,
            merchant_id=merchant_id,
            limit=limit,
        ),
    }


@router.get("/risk/top")
def top_risk(
    limit: int = Query(
        50,
        ge=1,
        le=500,
    )
):

    data = get_top_risk(limit)

    return {
        "count": len(data),
        "data": data,
    }


# ============================================================
# SPIKES
# ============================================================

@router.get("/spikes")
def spikes():

    data = get_spikes()

    return {
        "count": len(data),
        "data": data,
    }


@router.get("/spikes/{event_id}")
def spike(event_id: str):

    result = get_spike(event_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Spike event '{event_id}' not found.",
        )

    return result


# ============================================================
# XAI
# ============================================================

@router.get("/xai")
def xai(
    event_id: Optional[str] = None,
):

    data = get_xai(
        event_id=event_id
    )

    return {
        "count": len(data),
        "data": data,
    }


# ============================================================
# IMPACT
# ============================================================

@router.get("/impact/{event_id}")
def impact(event_id: str):

    return {
        "event_id": event_id,
        "forecast": get_impact(event_id),
        "scenarios": get_scenarios(event_id),
    }


# ============================================================
# RESPONSE
# ============================================================

@router.get("/response")
def response(
    event_id: Optional[str] = None,
):

    data = get_response(
        event_id=event_id
    )

    return {
        "count": len(data),
        "data": data,
    }


# ============================================================
# MERCHANT
# ============================================================

@router.get("/merchant/{merchant_id}")
def merchant(
    merchant_id: str,
):

    return get_merchant(
        merchant_id
    )