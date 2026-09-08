from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ============================================================
# HEALTH
# ============================================================

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# ============================================================
# ARTIFACT STATUS
# ============================================================

class ArtifactStatusResponse(BaseModel):
    fusion: bool
    phase2_events: bool
    phase2_windows: bool
    xai: bool
    impact_forecast: bool
    impact_scenarios: bool
    response_recommendations: bool
    response_actions: bool


# ============================================================
# GENERIC DATA RESPONSE
# ============================================================

class DataResponse(BaseModel):
    count: int
    data: List[Dict[str, Any]]


# ============================================================
# OVERVIEW
# ============================================================

class OverviewResponse(BaseModel):
    transactions: int
    merchants: int
    active_spikes: int
    alerts: int

    total_amount: float
    expected_fraud_amount: float

    mean_unified_risk: float
    maximum_unified_risk: float

    critical_events: int
    verified_events: int
    early_warning_events: int