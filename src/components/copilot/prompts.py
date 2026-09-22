import json
import math
from typing import Any, Dict, List, Optional

from .config import (
    MAX_CONTEXT_CHARS,
    MAX_CONVERSATION_MESSAGES,
    MAX_CONVERSATION_MESSAGE_CHARS,
    MAX_TRANSACTION_EVIDENCE,
    MAX_GENERAL_EVIDENCE,
    MAX_GENERIC_FIELDS_PER_RECORD,
    MAX_QUESTION_CHARS,
    HARD_MAX_CONTEXT_CHARS,
    TRANSACTION_EVIDENCE_FIELDS,
    SPIKE_EVIDENCE_FIELDS,
    IMPACT_EVIDENCE_FIELDS,
    RESPONSE_EVIDENCE_FIELDS,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are FraudSentinel AI Copilot.

You are a defense-only fraud intelligence assistant for analysts,
risk teams, and fraud operations.

Your job is to interpret evidence produced by the FraudSentinel AI
system and explain it clearly.

============================================================
CORE RULES
============================================================

1. Never invent fraud scores, transaction values, event IDs,
   merchant IDs, model outputs, forecasts, or recommendations.

2. Treat the supplied FraudSentinel API evidence as the
   authoritative source for system-specific facts.

3. Do not independently act as a fraud detection model.

4. Do not invent blocking, restricting, monitoring, or
   containment decisions.

5. When discussing actions, explain the recommendation already
   produced by the FraudSentinel Response layer.

6. Clearly distinguish:

   - observed transaction evidence
   - transaction model risk
   - temporal spike intelligence
   - XAI evidence
   - financial forecast
   - response recommendation

7. If evidence is missing, explicitly say that the available
   evidence is insufficient.

8. If an event_id, merchant_id, or transaction_id was not
   supplied, do not invent one.

9. Be concise but analytically useful.

10. For risk questions, prioritize the strongest available
    risk signals.

11. For spike questions, prioritize:

    - spike state
    - spike score
    - duration
    - transaction volume
    - expected fraud
    - expected financial impact
    - breadth
    - coordination

12. For financial questions, distinguish:

    - current expected loss
    - forecasted loss
    - lower/upper forecast bounds
    - scenario estimates

13. For response questions, distinguish:

    - recommended action
    - priority
    - control
    - expected effect

14. Never claim that an action was actually executed unless
    explicit execution evidence is supplied.

15. If data appears contradictory, report the contradiction
    instead of silently correcting it.

16. Never provide offensive cybersecurity guidance.

17. Never fabricate missing evidence.

============================================================
ANSWER STYLE
============================================================

Answer the user's question directly.

Prefer a structure such as:

Situation:
...

Evidence:
...

Risk interpretation:
...

Financial impact:
...

Recommended defensive response:
...

Only include sections that are relevant to the question.

Do not reproduce large raw API payloads.

Do not repeat the same evidence unnecessarily.
"""


# ============================================================
# VALUE NORMALIZATION
# ============================================================

def _safe_value(
    value: Any,
) -> Any:
    """
    Convert values into compact JSON-safe representations.
    """

    if value is None:
        return None

    # NumPy scalar.
    if hasattr(value, "item"):

        try:
            value = value.item()

        except (
            ValueError,
            TypeError,
        ):
            pass

    if isinstance(value, float):

        if math.isnan(value):
            return None

        if math.isinf(value):
            return None

        return value

    if isinstance(
        value,
        (
            str,
            int,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _safe_value(item)
            for item in value
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
        }

    return str(value)


# ============================================================
# TRUNCATE
# ============================================================

def _truncate(
    text: str,
    max_chars: int,
) -> str:

    text = str(
        text or ""
    )

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + "\n...[truncated]"
    )


# ============================================================
# RECORD EXTRACTION
# ============================================================

def _extract_records(
    evidence: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Extract records from common retriever evidence structures.
    """

    if not isinstance(
        evidence,
        dict,
    ):
        return []

    records = []

    candidate_keys = (
        "records",
        "transactions",
        "top_risk",
        "data",
        "results",
        "items",
    )

    for key in candidate_keys:

        value = evidence.get(
            key
        )

        if not isinstance(
            value,
            list,
        ):
            continue

        for item in value:

            if isinstance(
                item,
                dict,
            ):

                records.append(
                    item
                )

    return records


# ============================================================
# DEDUPLICATION
# ============================================================

def _deduplicate_records(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    if not records:
        return []

    output = []

    seen = set()

    for record in records:

        transaction_id = record.get(
            "transaction_id"
        )

        event_id = record.get(
            "event_id"
        )

        merchant_id = record.get(
            "merchant_id"
        )

        timestamp = record.get(
            "timestamp"
        )

        key = (
            transaction_id,
            event_id,
            merchant_id,
            timestamp,
        )

        # If there is no meaningful identity,
        # preserve the record.
        if all(
            item is None
            for item in key
        ):

            output.append(
                record
            )

            continue

        if key in seen:
            continue

        seen.add(
            key
        )

        output.append(
            record
        )

    return output


# ============================================================
# TRANSACTION FORMATTER
# ============================================================

def _format_transaction_records(
    records: List[Dict[str, Any]],
) -> str:

    records = _deduplicate_records(
        records
    )

    if not records:
        return (
            "No transaction evidence available."
        )

    output = []

    for index, record in enumerate(
        records[
            :MAX_TRANSACTION_EVIDENCE
        ],
        start=1,
    ):

        output.append(
            f"Transaction {index}:"
        )

        for field in (
            TRANSACTION_EVIDENCE_FIELDS
        ):

            if field not in record:
                continue

            value = _safe_value(
                record.get(field)
            )

            if value is None:
                continue

            output.append(
                f"  {field}: {value}"
            )

        output.append("")

    return "\n".join(
        output
    )


# ============================================================
# SPIKE FORMATTER
# ============================================================

def _format_spike_records(
    records: List[Dict[str, Any]],
) -> str:

    if not records:
        return (
            "No spike evidence available."
        )

    output = []

    for index, record in enumerate(
        records[
            :MAX_GENERAL_EVIDENCE
        ],
        start=1,
    ):

        output.append(
            f"Spike {index}:"
        )

        for field in (
            SPIKE_EVIDENCE_FIELDS
        ):

            if field not in record:
                continue

            value = _safe_value(
                record.get(field)
            )

            if value is None:
                continue

            output.append(
                f"  {field}: {value}"
            )

        output.append("")

    return "\n".join(
        output
    )


# ============================================================
# IMPACT FORMATTER
# ============================================================

def _format_impact_records(
    records: List[Dict[str, Any]],
) -> str:

    if not records:
        return (
            "No financial impact evidence available."
        )

    output = []

    for index, record in enumerate(
        records[
            :MAX_GENERAL_EVIDENCE
        ],
        start=1,
    ):

        output.append(
            f"Impact record {index}:"
        )

        for field in (
            IMPACT_EVIDENCE_FIELDS
        ):

            if field not in record:
                continue

            value = _safe_value(
                record.get(field)
            )

            if value is None:
                continue

            output.append(
                f"  {field}: {value}"
            )

        output.append("")

    return "\n".join(
        output
    )


# ============================================================
# RESPONSE FORMATTER
# ============================================================

def _format_response_records(
    records: List[Dict[str, Any]],
) -> str:

    if not records:
        return (
            "No response recommendation evidence available."
        )

    output = []

    for index, record in enumerate(
        records[
            :MAX_GENERAL_EVIDENCE
        ],
        start=1,
    ):

        output.append(
            f"Response recommendation {index}:"
        )

        for field in (
            RESPONSE_EVIDENCE_FIELDS
        ):

            if field not in record:
                continue

            value = _safe_value(
                record.get(field)
            )

            if value is None:
                continue

            output.append(
                f"  {field}: {value}"
            )

        output.append("")

    return "\n".join(
        output
    )


# ============================================================
# GENERIC FORMATTER
# ============================================================

def _format_generic_records(
    records: List[Dict[str, Any]],
) -> str:

    if not records:
        return (
            "No evidence records available."
        )

    output = []

    for index, record in enumerate(
        records[
            :MAX_GENERAL_EVIDENCE
        ],
        start=1,
    ):

        output.append(
            f"Record {index}:"
        )

        field_count = 0

        for key, value in record.items():

            if field_count >= (
                MAX_GENERIC_FIELDS_PER_RECORD
            ):
                break

            safe_value = _safe_value(
                value
            )

            if safe_value is None:
                continue

            output.append(
                f"  {key}: {safe_value}"
            )

            field_count += 1

        output.append("")

    return "\n".join(
        output
    )


# ============================================================
# INTENT-AWARE EVIDENCE FORMATTER
# ============================================================

def _format_evidence(
    intent: str,
    records: List[Dict[str, Any]],
) -> str:

    normalized_intent = (
        str(intent or "general")
        .strip()
        .lower()
    )

    if normalized_intent in {
        "transaction",
        "transactions",
        "top_risk",
        "risk",
    }:

        return _format_transaction_records(
            records
        )

    if normalized_intent in {
        "spike",
        "spikes",
        "temporal",
    }:

        return _format_spike_records(
            records
        )

    if normalized_intent in {
        "impact",
        "forecast",
        "financial",
    }:

        return _format_impact_records(
            records
        )

    if normalized_intent in {
        "response",
        "action",
        "recommendation",
    }:

        return _format_response_records(
            records
        )

    return _format_generic_records(
        records
    )


# ============================================================
# ENTITY FORMATTER
# ============================================================

def _format_entities(
    entities: Any,
) -> str:

    if not isinstance(
        entities,
        dict,
    ):
        return "{}"

    safe_entities = {
        str(key): _safe_value(value)
        for key, value in entities.items()
    }

    return json.dumps(
        safe_entities,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# CONVERSATION FORMATTER
# ============================================================

def _format_conversation(
    conversation: Optional[
        List[Dict[str, str]]
    ],
) -> str:

    if not conversation:
        return ""

    lines = []

    for message in conversation[
        -MAX_CONVERSATION_MESSAGES:
    ]:

        role = message.get(
            "role",
            "",
        )

        content = message.get(
            "content",
            "",
        )

        if role not in {
            "user",
            "assistant",
        }:
            continue

        content = _truncate(
            content,
            MAX_CONVERSATION_MESSAGE_CHARS,
        )

        lines.append(
            f"{role.upper()}: {content}"
        )

    if not lines:
        return ""

    return (
        "RECENT CONVERSATION:\n"
        + "\n".join(lines)
    )


# ============================================================
# FINAL CONTEXT BUDGET
# ============================================================

def _apply_context_budget(
    text: str,
) -> str:

    # Use the lower of the configurable context and hard limit.
    max_chars = min(
        MAX_CONTEXT_CHARS,
        HARD_MAX_CONTEXT_CHARS,
    )

    return _truncate(
        text,
        max_chars,
    )


# ============================================================
# EVIDENCE PROMPT
# ============================================================

def build_evidence_prompt(
    question: str,
    retrieval: Dict[str, Any],
    conversation: Optional[
        List[Dict[str, str]]
    ] = None,
) -> str:

    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    question = _truncate(
        question,
        MAX_QUESTION_CHARS,
    )

    # --------------------------------------------------------
    # Retrieval metadata
    # --------------------------------------------------------

    intent = retrieval.get(
        "intent",
        "general",
    )

    route_confidence = retrieval.get(
        "route_confidence",
        0.0,
    )

    route_reason = retrieval.get(
        "route_reason",
        "",
    )

    entities = retrieval.get(
        "entities",
        {},
    )

    evidence = retrieval.get(
        "evidence",
        {},
    )

    if not isinstance(
        evidence,
        dict,
    ):
        evidence = {}

    # --------------------------------------------------------
    # Records
    # --------------------------------------------------------

    records = _extract_records(
        evidence
    )

    evidence_text = _format_evidence(
        intent=intent,
        records=records,
    )

    # --------------------------------------------------------
    # Tools
    # --------------------------------------------------------

    tools = evidence.get(
        "tools",
        [],
    )

    # --------------------------------------------------------
    # Character count
    # --------------------------------------------------------

    raw_evidence_character_count = (
        evidence.get(
            "character_count",
            0,
        )
    )

    # --------------------------------------------------------
    # Conversation
    # --------------------------------------------------------

    history_text = _format_conversation(
        conversation
    )

    # --------------------------------------------------------
    # Prompt sections
    # --------------------------------------------------------

    sections = []

    sections.append(
        "USER QUESTION:\n"
        + question
    )

    sections.append(
        "DETECTED INTENT:\n"
        + str(intent)
    )

    sections.append(
        "ROUTE CONFIDENCE:\n"
        + str(route_confidence)
    )

    if route_reason:

        sections.append(
            "ROUTE REASON:\n"
            + _truncate(
                route_reason,
                1000,
            )
        )

    sections.append(
        "DETECTED ENTITIES:\n"
        + _format_entities(
            entities
        )
    )

    sections.append(
        "TOOLS USED:\n"
        + str(tools)
    )

    sections.append(
        "EVIDENCE CHARACTER COUNT BEFORE "
        "LLM COMPRESSION:\n"
        + str(
            raw_evidence_character_count
        )
    )

    sections.append(
        "COMPACT FRAUDSENTINEL EVIDENCE:\n"
        + evidence_text
    )

    if history_text:

        sections.append(
            history_text
        )

    sections.append(
        """
ANSWER REQUIREMENTS:

- Answer the user's question directly.
- Use only the supplied FraudSentinel evidence.
- Do not fabricate missing values.
- Do not fabricate identifiers.
- Do not reproduce the complete raw API payload.
- Explain the most important evidence.
- Distinguish transaction risk from temporal spike risk.
- Distinguish expected loss from forecasted loss.
- Treat response actions as recommendations unless execution
  evidence is explicitly supplied.
- If evidence is insufficient, say so clearly.
"""
    )

    prompt = "\n\n".join(
        sections
    )

    # --------------------------------------------------------
    # Final context safety limit
    # --------------------------------------------------------

    return _apply_context_budget(
        prompt
    )