import re
from dataclasses import dataclass
from typing import Optional


# ============================================================
# ENTITY CONTAINER
# ============================================================

@dataclass
class CopilotEntities:
    event_id: Optional[str] = None
    merchant_id: Optional[str] = None
    transaction_id: Optional[str] = None


# ============================================================
# ENTITY EXTRACTOR
# ============================================================

class EntityExtractor:
    """
    Extracts explicit entity identifiers from analyst questions.

    This module does not infer hidden identifiers.
    It only extracts identifiers that appear in the question.
    """

    EVENT_PATTERNS = [
        r"\bevent[_\s-]?id[\s:=#-]*([A-Za-z0-9_.:-]+)",
        r"\bevent[\s:=#-]+([A-Za-z0-9_.:-]+)",
        r"\b(EVT[_-][A-Za-z0-9_.:-]+)",
    ]

    MERCHANT_PATTERNS = [
        r"\bmerchant[_\s-]?id[\s:=#-]*([A-Za-z0-9_.:-]+)",
        r"\bmerchant[\s:=#-]+([A-Za-z0-9_.:-]+)",
        r"\b(MER[_-][A-Za-z0-9_.:-]+)",
    ]

    TRANSACTION_PATTERNS = [
        r"\btransaction[_\s-]?id[\s:=#-]*([A-Za-z0-9_.:-]+)",
        r"\btransaction[\s:=#-]+([A-Za-z0-9_.:-]+)",
        r"\b(TXN[_-][A-Za-z0-9_.:-]+)",
    ]

    @staticmethod
    def _extract(
        question: str,
        patterns: list[str],
    ) -> Optional[str]:

        for pattern in patterns:

            match = re.search(
                pattern,
                question,
                flags=re.IGNORECASE,
            )

            if match:
                return match.group(1)

        return None

    def extract(self, question: str) -> CopilotEntities:

        return CopilotEntities(
            event_id=self._extract(
                question,
                self.EVENT_PATTERNS,
            ),
            merchant_id=self._extract(
                question,
                self.MERCHANT_PATTERNS,
            ),
            transaction_id=self._extract(
                question,
                self.TRANSACTION_PATTERNS,
            ),
        )


# ============================================================
# DEFAULT EXTRACTOR
# ============================================================

entity_extractor = EntityExtractor()