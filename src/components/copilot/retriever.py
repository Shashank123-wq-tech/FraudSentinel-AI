from typing import Any, Dict

from .entities import entity_extractor
from .evidence import evidence_builder
from .router import router
from .tools import CopilotTools


# ============================================================
# RETRIEVER
# ============================================================

class CopilotRetriever:

    def __init__(
        self,
        tool_registry: CopilotTools | None = None,
    ):

        self.tools = tool_registry or CopilotTools()

    # ========================================================
    # SAFE TOOL EXECUTION
    # ========================================================

    def _execute_tool(
        self,
        tool_name: str,
        entities,
    ) -> Any:

        try:

            # ------------------------------------------------
            # SYSTEM
            # ------------------------------------------------

            if tool_name == "health":
                return self.tools.health()

            if tool_name == "status":
                return self.tools.status()

            # ------------------------------------------------
            # OVERVIEW
            # ------------------------------------------------

            if tool_name == "overview":
                return self.tools.overview()

            # ------------------------------------------------
            # TRANSACTION
            # ------------------------------------------------

            if tool_name == "transaction":

                return self.tools.transaction(
                    transaction_id=entities.transaction_id,
                    merchant_id=entities.merchant_id,
                    limit=20,
                )

            # ------------------------------------------------
            # TOP RISK
            # ------------------------------------------------

            if tool_name == "top_risk":
                return self.tools.top_risk(
                    limit=20,
                )

            # ------------------------------------------------
            # SPIKES
            # ------------------------------------------------

            if tool_name == "spikes":
                return self.tools.spikes()

            if tool_name == "spike":

                if not entities.event_id:

                    return {
                        "status": "not_executed",
                        "reason": (
                            "No event_id was explicitly provided "
                            "in the question."
                        ),
                    }

                return self.tools.spike(
                    entities.event_id,
                )

            # ------------------------------------------------
            # XAI
            # ------------------------------------------------

            if tool_name == "xai":

                return self.tools.xai(
                    event_id=entities.event_id,
                )

            # ------------------------------------------------
            # IMPACT
            # ------------------------------------------------

            if tool_name == "impact":

                if not entities.event_id:

                    return {
                        "status": "not_executed",
                        "reason": (
                            "Impact requires an event_id."
                        ),
                    }

                return self.tools.impact(
                    entities.event_id,
                )

            # ------------------------------------------------
            # RESPONSE
            # ------------------------------------------------

            if tool_name == "response":

                return self.tools.response(
                    event_id=entities.event_id,
                )

            # ------------------------------------------------
            # MERCHANT
            # ------------------------------------------------

            if tool_name == "merchant":

                if not entities.merchant_id:

                    return {
                        "status": "not_executed",
                        "reason": (
                            "Merchant investigation requires "
                            "a merchant_id."
                        ),
                    }

                return self.tools.merchant(
                    entities.merchant_id,
                )

            return {
                "status": "not_executed",
                "reason": f"Unknown tool: {tool_name}",
            }

        except Exception as exc:

            return {
                "status": "error",
                "tool": tool_name,
                "error": str(exc),
            }

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        question: str,
    ) -> Dict[str, Any]:

        route = router.route(question)

        entities = entity_extractor.extract(
            question,
        )

        evidence = {}

        # ----------------------------------------------------
        # Execute routed tools
        # ----------------------------------------------------

        for tool_name in route.tools:

            # ------------------------------------------------
            # Conditional tools
            # ------------------------------------------------

            if tool_name in {
                "spike",
                "impact",
            } and not entities.event_id:

                evidence[tool_name] = {
                    "status": "not_executed",
                    "reason": (
                        "An explicit event_id is required "
                        "for this tool."
                    ),
                }

                continue

            if tool_name == "merchant" and not entities.merchant_id:

                evidence[tool_name] = {
                    "status": "not_executed",
                    "reason": (
                        "An explicit merchant_id is required "
                        "for this tool."
                    ),
                }

                continue

            result = self._execute_tool(
                tool_name,
                entities,
            )

            evidence[tool_name] = result

        # ----------------------------------------------------
        # Build compact evidence package
        # ----------------------------------------------------

        package = evidence_builder.build(
            evidence,
        )

        return {
            "question": question,
            "intent": route.intent.value,
            "route_confidence": route.confidence,
            "route_reason": route.reason,
            "entities": {
                "event_id": entities.event_id,
                "merchant_id": entities.merchant_id,
                "transaction_id": entities.transaction_id,
            },
            "evidence": package,
        }


# ============================================================
# DEFAULT RETRIEVER
# ============================================================

retriever = CopilotRetriever()