from typing import Dict, List, Optional

from .llm import FraudSentinelLLM
from .prompts import (
    SYSTEM_PROMPT,
    build_evidence_prompt,
)
from .retriever import CopilotRetriever


# ============================================================
# COPILOT
# ============================================================

class FraudSentinelCopilot:

    def __init__(
        self,
        retriever: Optional[CopilotRetriever] = None,
        llm: Optional[FraudSentinelLLM] = None,
    ):

        self.retriever = (
            retriever
            or CopilotRetriever()
        )

        self.llm = (
            llm
            or FraudSentinelLLM()
        )

    # ========================================================
    # BUILD MESSAGES
    # ========================================================

    def _build_messages(
        self,
        question: str,
        retrieval: Dict,
        conversation: Optional[
            List[Dict[str, str]]
        ] = None,
    ) -> List[Dict[str, str]]:

        prompt = build_evidence_prompt(
            question=question,
            retrieval=retrieval,
            conversation=conversation,
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        # ----------------------------------------------------
        # Preserve recent conversation
        # ----------------------------------------------------

        if conversation:

            for message in conversation[-6:]:

                role = message.get(
                    "role",
                )

                content = message.get(
                    "content",
                    "",
                )

                if role in {
                    "user",
                    "assistant",
                }:

                    messages.append(
                        {
                            "role": role,
                            "content": content,
                        }
                    )

        # ----------------------------------------------------
        # Current grounded question
        # ----------------------------------------------------

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return messages

    # ========================================================
    # ASK
    # ========================================================

    def ask(
        self,
        question: str,
        conversation: Optional[
            List[Dict[str, str]]
        ] = None,
    ) -> Dict:

        question = question.strip()

        if not question:

            return {
                "answer": (
                    "Please provide a question."
                ),
                "intent": "general",
                "evidence": {},
            }

        # ----------------------------------------------------
        # Retrieve system evidence
        # ----------------------------------------------------

        retrieval = self.retriever.retrieve(
            question,
        )

        # ----------------------------------------------------
        # Build grounded prompt
        # ----------------------------------------------------

        messages = self._build_messages(
            question=question,
            retrieval=retrieval,
            conversation=conversation,
        )

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        answer = self.llm.generate(
            messages,
        )

        return {
            "answer": answer,
            "intent": retrieval.get(
                "intent",
                "general",
            ),
            "route_confidence": retrieval.get(
                "route_confidence",
                0.0,
            ),
            "route_reason": retrieval.get(
                "route_reason",
                "",
            ),
            "entities": retrieval.get(
                "entities",
                {},
            ),
            "tools_used": retrieval.get(
                "evidence",
                {},
            ).get(
                "tools",
                [],
            ),
            "evidence_character_count": retrieval.get(
                "evidence",
                {},
            ).get(
                "character_count",
                0,
            ),
        }