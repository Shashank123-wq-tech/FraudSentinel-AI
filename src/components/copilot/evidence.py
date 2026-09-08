import json
from typing import Any, Dict


# ============================================================
# EVIDENCE BUILDER
# ============================================================

class EvidenceBuilder:

    def __init__(
        self,
        max_chars: int = 30000,
    ):
        self.max_chars = max_chars

    # ========================================================
    # NORMALIZE
    # ========================================================

    @staticmethod
    def normalize(
        value: Any,
    ) -> Any:

        if isinstance(value, dict):

            return {
                str(k): EvidenceBuilder.normalize(v)
                for k, v in value.items()
            }

        if isinstance(value, list):

            return [
                EvidenceBuilder.normalize(v)
                for v in value
            ]

        return value

    # ========================================================
    # LIMIT LIST
    # ========================================================

    @staticmethod
    def limit_records(
        value: Any,
        limit: int = 20,
    ) -> Any:

        if isinstance(value, dict):

            result = {}

            for key, item in value.items():

                if isinstance(item, list):

                    result[key] = item[:limit]

                else:

                    result[key] = item

            return result

        if isinstance(value, list):
            return value[:limit]

        return value

    # ========================================================
    # BUILD
    # ========================================================

    def build(
        self,
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:

        normalized = self.normalize(evidence)

        # Limit potentially large endpoint results.
        limited = {}

        for tool_name, result in normalized.items():

            limited[tool_name] = self.limit_records(
                result,
                limit=20,
            )

        # ----------------------------------------------------
        # Convert to JSON
        # ----------------------------------------------------

        serialized = json.dumps(
            limited,
            indent=2,
            default=str,
        )

        # ----------------------------------------------------
        # Hard context limit
        # ----------------------------------------------------

        if len(serialized) > self.max_chars:

            serialized = (
                serialized[: self.max_chars]
                + "\n\n[Evidence truncated for context safety.]"
            )

        return {
            "tools": list(evidence.keys()),
            "data": limited,
            "serialized": serialized,
            "character_count": len(serialized),
        }


# ============================================================
# DEFAULT BUILDER
# ============================================================

evidence_builder = EvidenceBuilder(
    max_chars=30000,
)