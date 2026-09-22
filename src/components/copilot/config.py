import os

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# INTELLIGENCE API
# ============================================================

API_BASE_URL = os.getenv(
    "FRAUDSENTINEL_API_URL",
    "http://127.0.0.1:8000",
)

REQUEST_TIMEOUT = int(
    os.getenv(
        "FRAUDSENTINEL_API_TIMEOUT",
        "30",
    )
)


# ============================================================
# GROQ
# ============================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)


# ============================================================
# COPILOT CONTEXT MANAGEMENT
# ============================================================

# IMPORTANT:
#
# This is the maximum amount of serialized FraudSentinel
# evidence that should be passed into the LLM prompt.
#
# The API may contain much more information, but the LLM
# receives only a controlled subset.
#
# 12,000 characters is intentionally conservative for an
# 8,000-token-per-minute Groq limit.

MAX_CONTEXT_CHARS = int(
    os.getenv(
        "COPILOT_MAX_CONTEXT_CHARS",
        "12000",
    )
)


# ============================================================
# CONVERSATION CONTROL
# ============================================================

# Number of previous messages included in the LLM context.

MAX_CONVERSATION_MESSAGES = int(
    os.getenv(
        "COPILOT_MAX_CONVERSATION_MESSAGES",
        "4",
    )
)


# Maximum characters from one previous conversation message.

MAX_CONVERSATION_MESSAGE_CHARS = int(
    os.getenv(
        "COPILOT_MAX_CONVERSATION_MESSAGE_CHARS",
        "1500",
    )
)


# ============================================================
# EVIDENCE LIMITS
# ============================================================

# Maximum transaction records passed to the LLM.

MAX_TRANSACTION_EVIDENCE = int(
    os.getenv(
        "COPILOT_MAX_TRANSACTION_EVIDENCE",
        "10",
    )
)


# Maximum generic records passed to the LLM.

MAX_GENERAL_EVIDENCE = int(
    os.getenv(
        "COPILOT_MAX_GENERAL_EVIDENCE",
        "8",
    )
)


# Maximum number of fields from a generic evidence record.

MAX_GENERIC_FIELDS_PER_RECORD = int(
    os.getenv(
        "COPILOT_MAX_GENERIC_FIELDS_PER_RECORD",
        "18",
    )
)


# Maximum characters allowed for the user's current question.

MAX_QUESTION_CHARS = int(
    os.getenv(
        "COPILOT_MAX_QUESTION_CHARS",
        "2000",
    )
)


# ============================================================
# TRANSACTION FIELDS
# ============================================================

# These are the fields that actually matter when the user
# asks about transaction risk.
#
# We intentionally do NOT send the entire Fusion schema.

TRANSACTION_EVIDENCE_FIELDS = (
    "transaction_id",
    "timestamp",
    "merchant_id",
    "card_id",
    "amount",
    "fraud_probability",
    "risk_score",
    "unified_risk_score",
    "risk_band",
    "predicted_fraud",
    "alert_flag",
    "response_action",
    "response_priority",
    "spike_state",
    "event_id",
)


# ============================================================
# SPIKE FIELDS
# ============================================================

SPIKE_EVIDENCE_FIELDS = (
    "event_id",
    "merchant_id",
    "start_time",
    "end_time",
    "duration_minutes",
    "windows",
    "transaction_count",
    "total_amount",
    "expected_fraud_count",
    "expected_fraud_amount",
    "unique_cards",
    "max_coordination_score",
    "max_tas",
    "max_fas",
    "spike_score",
    "spike_state",
)


# ============================================================
# IMPACT FIELDS
# ============================================================

IMPACT_EVIDENCE_FIELDS = (
    "event_id",
    "horizon_minutes",
    "projected_loss",
    "lower_bound",
    "upper_bound",
    "forecast_confidence",
    "current_expected_loss",
    "expected_fraud_count",
)


# ============================================================
# RESPONSE FIELDS
# ============================================================

RESPONSE_EVIDENCE_FIELDS = (
    "event_id",
    "response_action",
    "response_priority",
    "control",
    "reason",
    "recommendation_confidence",
    "expected_loss_prevented",
    "expected_action_loss",
)


# ============================================================
# GLOBAL SAFETY
# ============================================================

# Absolute upper bound for evidence passed to the LLM.
#
# This prevents a future retriever change from accidentally
# creating a massive prompt.

HARD_MAX_CONTEXT_CHARS = int(
    os.getenv(
        "COPILOT_HARD_MAX_CONTEXT_CHARS",
        "14000",
    )
)