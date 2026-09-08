"""
Schemas and required columns for the Response Recommendation Layer.
"""

FUSION_REQUIRED_COLUMNS = [
    "transaction_id",
    "timestamp",
    "merchant_id",
    "card_id",
    "amount",
    "fraud_probability",
    "unified_risk_score",
    "spike_state",
    "fusion_confidence",
    "response_priority",
    "alert_flag",
    "phase1_primary_signal",
    "phase2_primary_signal",
    "temporal_spike_signal",
    "early_warning_signal",
    "verified_spike_signal",
    "critical_spike_signal",
    "active_event_signal",
    "event_id",
    "event_active",
    "peak_spike_score",
    "duration_minutes",
    "expected_fraud_amount",
    "expected_fraud_count",
]


IMPACT_FORECAST_REQUIRED_COLUMNS = [
    "event_id",
    "horizon_minutes",
    "projected_loss",
    "lower_bound",
    "upper_bound",
    "projected_fraud_transactions",
    "projected_transactions",
    "forecast_confidence",
]


IMPACT_SCENARIO_REQUIRED_COLUMNS = [
    "event_id",
    "horizon_minutes",
    "scenario",
    "projected_loss",
    "loss_prevented",
]


OUTPUT_COLUMNS = [
    "recommendation_id",
    "event_id",
    "merchant_id",
    "timestamp",
    "spike_state",
    "unified_risk_score",
    "fraud_probability",
    "fusion_confidence",
    "peak_spike_score",
    "duration_minutes",
    "expected_fraud_amount",
    "expected_fraud_count",
    "forecast_confidence",
    "projected_loss_15m",
    "projected_loss_30m",
    "projected_loss_60m",
    "no_action_loss_60m",
    "moderate_loss_60m",
    "aggressive_loss_60m",
    "moderate_loss_prevented_60m",
    "aggressive_loss_prevented_60m",
    "threat_score",
    "financial_urgency",
    "response_score",
    "response_priority",
    "response_action",
    "recommended_control",
    "reason",
    "expected_loss_if_action",
    "estimated_loss_prevented",
    "recommendation_confidence",
    "policy_version",
]


ACTION_COLUMNS = [
    "action_id",
    "recommendation_id",
    "event_id",
    "merchant_id",
    "response_priority",
    "response_action",
    "recommended_control",
    "estimated_loss_prevented",
    "recommendation_confidence",
    "timestamp",
]
