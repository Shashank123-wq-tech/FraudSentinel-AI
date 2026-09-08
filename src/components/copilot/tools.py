from typing import Any, Dict, Optional

from .api_client import FraudSentinelAPIClient


# ============================================================
# TOOL REGISTRY
# ============================================================

class CopilotTools:

    def __init__(
        self,
        api_client: Optional[FraudSentinelAPIClient] = None,
    ):
        self.api = api_client or FraudSentinelAPIClient()

    # ========================================================
    # SYSTEM
    # ========================================================

    def health(self) -> Dict[str, Any]:
        return self.api.get("/api/health")

    def status(self) -> Dict[str, Any]:
        return self.api.get("/api/status")

    # ========================================================
    # OVERVIEW
    # ========================================================

    def overview(self) -> Dict[str, Any]:
        return self.api.get("/api/risk/overview")

    # ========================================================
    # TRANSACTIONS
    # ========================================================

    def transaction(
        self,
        transaction_id: Optional[str] = None,
        merchant_id: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:

        params = {
            "limit": limit,
        }

        if transaction_id:
            params["transaction_id"] = transaction_id

        if merchant_id:
            params["merchant_id"] = merchant_id

        return self.api.get(
            "/api/risk/transactions",
            params=params,
        )

    # ========================================================
    # TOP RISK
    # ========================================================

    def top_risk(
        self,
        limit: int = 20,
    ) -> Dict[str, Any]:

        return self.api.get(
            "/api/risk/top",
            params={
                "limit": limit,
            },
        )

    # ========================================================
    # SPIKES
    # ========================================================

    def spikes(self) -> Dict[str, Any]:

        return self.api.get(
            "/api/spikes",
        )

    def spike(
        self,
        event_id: str,
    ) -> Dict[str, Any]:

        return self.api.get(
            f"/api/spikes/{event_id}",
        )

    # ========================================================
    # XAI
    # ========================================================

    def xai(
        self,
        event_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        params = {}

        if event_id:
            params["event_id"] = event_id

        return self.api.get(
            "/api/xai",
            params=params,
        )

    # ========================================================
    # IMPACT
    # ========================================================

    def impact(
        self,
        event_id: str,
    ) -> Dict[str, Any]:

        return self.api.get(
            f"/api/impact/{event_id}",
        )

    # ========================================================
    # RESPONSE
    # ========================================================

    def response(
        self,
        event_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        params = {}

        if event_id:
            params["event_id"] = event_id

        return self.api.get(
            "/api/response",
            params=params,
        )

    # ========================================================
    # MERCHANT
    # ========================================================

    def merchant(
        self,
        merchant_id: str,
    ) -> Dict[str, Any]:

        return self.api.get(
            f"/api/merchant/{merchant_id}",
        )


# ============================================================
# TOOL INSTANCE
# ============================================================

tools = CopilotTools()