"""
src/entity/risk_entity.py
Contract for the Phase-1 -> Phase-2 handoff (Section 4 of the Phase 2 spec).
"""

PHASE1_JOIN_COLUMNS = [
    "transaction_id", "fraud_probability", "risk_score", "risk_band",
    "predicted_fraud", "model_version", "threshold",
]
