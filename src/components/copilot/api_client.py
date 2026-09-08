from typing import Any, Optional

import requests

from .config import API_BASE_URL, REQUEST_TIMEOUT


class FraudSentinelAPIClient:

    def __init__(
        self,
        base_url: str = API_BASE_URL,
        timeout: int = REQUEST_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(
        self,
        path: str,
        params: Optional[dict] = None,
    ) -> dict[str, Any]:

        url = f"{self.base_url}{path}"

        response = requests.get(
            url,
            params=params,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    def health(self) -> dict[str, Any]:
        return self.get("/api/health")

    def status(self) -> dict[str, Any]:
        return self.get("/api/status")

    def risk_overview(self) -> dict[str, Any]:
        return self.get("/api/risk/overview")

    def risk_top(self) -> dict[str, Any]:
        return self.get("/api/risk/top")

    def spikes(self) -> dict[str, Any]:
        return self.get("/api/spikes")

    def spike(self, event_id: str) -> dict[str, Any]:
        return self.get(
            f"/api/spikes/{event_id}"
        )

    def xai(self) -> dict[str, Any]:
        return self.get("/api/xai")

    def response(self) -> dict[str, Any]:
        return self.get("/api/response")

    def impact(self, event_id: str) -> dict[str, Any]:
        return self.get(
            f"/api/impact/{event_id}"
        )

    def merchant(
        self,
        merchant_id: str,
    ) -> dict[str, Any]:

        return self.get(
            f"/api/merchant/{merchant_id}"
        )