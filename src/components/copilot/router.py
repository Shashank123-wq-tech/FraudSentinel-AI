from dataclasses import dataclass
from enum import Enum
from typing import List


# ============================================================
# INTENTS
# ============================================================

class CopilotIntent(str, Enum):
    SYSTEM_STATUS = "system_status"
    OVERVIEW = "overview"
    TRANSACTION = "transaction"
    TOP_RISK = "top_risk"
    SPIKES = "spikes"
    SPIKE_INVESTIGATION = "spike_investigation"
    XAI = "xai"
    IMPACT = "impact"
    RESPONSE = "response"
    MERCHANT = "merchant"
    GENERAL = "general"


# ============================================================
# ROUTE
# ============================================================

@dataclass
class CopilotRoute:
    intent: CopilotIntent
    tools: List[str]
    confidence: float
    reason: str


# ============================================================
# ROUTER
# ============================================================

class CopilotRouter:
    """
    Lightweight deterministic router.

    The router does NOT make fraud decisions.
    It only decides which existing Intelligence API
    endpoints should be queried.
    """

    def route(self, question: str) -> CopilotRoute:

        q = question.lower().strip()

        # --------------------------------------------------------
        # SYSTEM
        # --------------------------------------------------------

        if any(
            term in q
            for term in [
                "health",
                "healthy",
                "status",
                "system status",
                "api status",
                "working",
                "online",
            ]
        ):
            return CopilotRoute(
                intent=CopilotIntent.SYSTEM_STATUS,
                tools=[
                    "health",
                    "status",
                ],
                confidence=0.98,
                reason="Question asks about system/API health or status.",
            )

        # --------------------------------------------------------
        # SPIKE INVESTIGATION
        # --------------------------------------------------------

        if any(
            term in q
            for term in [
                "spike",
                "fraud spike",
                "attack",
                "burst",
                "surge",
                "incident",
                "event",
                "why is this event",
                "why is the event",
            ]
        ):
            return CopilotRoute(
                intent=CopilotIntent.SPIKE_INVESTIGATION,
                tools=[
                    "spikes",
                    "spike",
                    "xai",
                    "impact",
                    "response",
                ],
                confidence=0.95,
                reason="Question concerns a fraud-spike event or incident.",
            )

        # --------------------------------------------------------
        # IMPACT
        # --------------------------------------------------------

        if any(
            term in q
            for term in [
                "impact",
                "loss",
                "financial loss",
                "expected loss",
                "forecast",
                "forecasted",
                "exposure",
                "prevented loss",
                "scenario",
                "scenarios",
            ]
        ):
            return CopilotRoute(
                intent=CopilotIntent.IMPACT,
                tools=[
                    "overview",
                    "impact",
                    "spikes",
                    "response",
                ],
                confidence=0.93,
                reason="Question concerns financial impact or loss forecasting.",
            )

        # --------------------------------------------------------
        # RESPONSE
        # --------------------------------------------------------

        if any(
            term in q
            for term in [
                "what should we do",
                "what action",
                "recommended action",
                "recommendation",
                "recommend",
                "response",
                "intervention",
                "contain",
                "restrict",
                "monitor",
                "block",
                "defensive action",
                "mitigation",
            ]
        ):
            return CopilotRoute(
                intent=CopilotIntent.RESPONSE,
                tools=[
                    "overview",
                    "spikes",
                    "impact",
                    "response",
                    "xai",
                ],
                confidence=0.94,
                reason="Question asks for defensive response or intervention guidance.",
            )

        # --------------------------------------------------------
        # XAI
        # --------------------------------------------------------

        if any(
            term in q
            for term in [
                "why",
                "explain",
                "explanation",
                "reason",
                "because",
                "driver",
                "drivers",
                "feature",
                "features",
                "evidence",
            ]
        ):
            return CopilotRoute(
                intent=CopilotIntent.XAI,
                tools=[
                    "xai",
                    "overview",
                    "spikes",
                ],
                confidence=0.91,
                reason="Question asks for explanation or supporting evidence.",
            )

        # --------------------------------------------------------
        # MERCHANT
        # --------------------------------------------------------

        if any(
            term in q
            for term in [
                "merchant",
                "merchant risk",
                "merchant activity",
                "merchant transactions",
                "merchant fraud",
            ]
        ):
            return CopilotRoute(
                intent=CopilotIntent.MERCHANT,
                tools=[
                    "merchant",
                    "spikes",
                    "response",
                    "xai",
                ],
                confidence=0.92,
                reason="Question concerns merchant-level investigation.",
            )

        # --------------------------------------------------------
        # TRANSACTION
        # --------------------------------------------------------

        if any(
            term in q
            for term in [
                "transaction",
                "transaction risk",
                "fraud probability",
                "risk score",
                "fraudulent transaction",
                "fraud transaction",
            ]
        ):
            return CopilotRoute(
                intent=CopilotIntent.TRANSACTION,
                tools=[
                    "transaction",
                    "top_risk",
                ],
                confidence=0.92,
                reason="Question concerns transaction-level risk.",
            )

        # --------------------------------------------------------
        # TOP RISK
        # --------------------------------------------------------

        if any(
            term in q
            for term in [
                "highest risk",
                "top risk",
                "riskiest",
                "most risky",
                "largest risk",
                "dangerous transactions",
                "high risk transactions",
            ]
        ):
            return CopilotRoute(
                intent=CopilotIntent.TOP_RISK,
                tools=[
                    "top_risk",
                ],
                confidence=0.95,
                reason="Question asks for highest-risk transactions.",
            )

        # --------------------------------------------------------
        # SPIKE LIST
        # --------------------------------------------------------

        if any(
            term in q
            for term in [
                "all spikes",
                "recent spikes",
                "fraud events",
                "active spikes",
                "active events",
                "fraud incidents",
            ]
        ):
            return CopilotRoute(
                intent=CopilotIntent.SPIKES,
                tools=[
                    "spikes",
                ],
                confidence=0.94,
                reason="Question asks for fraud-spike events.",
            )

        # --------------------------------------------------------
        # OVERVIEW
        # --------------------------------------------------------

        if any(
            term in q
            for term in [
                "overview",
                "summary",
                "overall",
                "overall risk",
                "how are we doing",
                "what is happening",
                "current risk",
                "current situation",
                "dashboard",
            ]
        ):
            return CopilotRoute(
                intent=CopilotIntent.OVERVIEW,
                tools=[
                    "overview",
                    "spikes",
                    "top_risk",
                    "response",
                ],
                confidence=0.95,
                reason="Question asks for an overall fraud-risk summary.",
            )

        # --------------------------------------------------------
        # GENERAL
        # --------------------------------------------------------

        return CopilotRoute(
            intent=CopilotIntent.GENERAL,
            tools=[
                "overview",
                "status",
            ],
            confidence=0.60,
            reason="No specialized intent matched; using general system context.",
        )


# ============================================================
# DEFAULT ROUTER
# ============================================================

router = CopilotRouter()