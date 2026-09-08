import json
from typing import Any, Dict, List


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are FraudSentinel AI Copilot.

You are a defense-only fraud intelligence assistant for analysts,
risk teams, and fraud operations.

Your job is to interpret evidence produced by the FraudSentinel AI
system and explain it clearly.

IMPORTANT RULES:

1. Never invent fraud scores, transaction values, event IDs,
   merchant IDs, model outputs, forecasts, or recommendations.

2. Treat the supplied API evidence as the authoritative source
   for system-specific facts.

3. Do not independently act as a fraud detection model.

4. Do not invent blocking, restricting, monitoring, or containment
   decisions.

5. When discussing actions, explain the recommendation already
   produced by the FraudSentinel Response layer.

6. Clearly distinguish:
   - observed evidence
   - model-generated risk
   - spike intelligence
   - XAI evidence
   - financial forecast
   - response recommendation

7. If evidence is missing, say that the available evidence is
   insufficient.

8. If an event_id, merchant_id, or transaction_id was not supplied,
   do not invent one.

9. Be concise but analytically useful.

10. When appropriate, structure the response as:
    - Situation
    - Evidence
    - Risk interpretation
    - Financial impact
    - Recommended defensive response

11. Never provide offensive cybersecurity guidance.

12. Never claim that an action was actually executed. The system
    provides recommendations unless explicit execution evidence
    is present.

13. If data appears contradictory, report the contradiction instead
    of silently correcting it.

You are an explanation and decision-support layer above the
FraudSentinel intelligence pipeline.
"""


# ============================================================
# EVIDENCE PROMPT
# ============================================================

def build_evidence_prompt(
    question: str,
    retrieval: Dict[str, Any],
    conversation: List[Dict[str, str]] | None = None,
) -> str:

    conversation = conversation or []

    history = ""

    if conversation:

        history_lines = []

        for message in conversation[-6:]:

            role = message.get(
                "role",
                "unknown",
            )

            content = message.get(
                "content",
                "",
            )

            history_lines.append(
                f"{role.upper()}: {content}"
            )

        history = (
            "\n\nRECENT CONVERSATION:\n"
            + "\n".join(history_lines)
        )

    evidence = retrieval.get(
        "evidence",
        {},
    )

    serialized = evidence.get(
        "serialized",
        "{}",
    )

    intent = retrieval.get(
        "intent",
        "general",
    )

    entities = retrieval.get(
        "entities",
        {},
    )

    return f"""
USER QUESTION:
{question}

DETECTED INTENT:
{intent}

DETECTED ENTITIES:
{json.dumps(entities, indent=2)}

FRAUDSENTINEL API EVIDENCE:
{serialized}

{history}

ANSWER REQUIREMENTS:

- Answer the user's question directly.
- Use only the evidence supplied above for FraudSentinel-specific facts.
- Do not fabricate missing values.
- If an identifier was not supplied, say so.
- Explain important numbers rather than merely listing them.
- If discussing risk, distinguish transaction risk from spike risk.
- If discussing a spike, include state, score, duration, breadth,
  and relevant evidence when available.
- If discussing financial impact, distinguish current expected loss
  from forecasted loss and scenario estimates.
- If discussing response, clearly state that it is a system
  recommendation and not an already-executed action.
- Mention uncertainty or missing evidence when relevant.
"""