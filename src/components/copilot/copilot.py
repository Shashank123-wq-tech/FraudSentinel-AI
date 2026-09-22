from typing import Dict, List, Optional

from .llm import FraudSentinelLLM

from .prompts import (
    SYSTEM_PROMPT,
    build_evidence_prompt,
)

from .retriever import (
    CopilotRetriever,
)

from .config import (
    MAX_CONVERSATION_MESSAGES,
    MAX_CONVERSATION_MESSAGE_CHARS,
)


# ============================================================
# FRAUDSENTINEL COPILOT
# ============================================================

class FraudSentinelCopilot:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        retriever: Optional[
            CopilotRetriever
        ] = None,
        llm: Optional[
            FraudSentinelLLM
        ] = None,
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
    # CONVERSATION NORMALIZATION
    # ========================================================

    def _normalize_conversation(
        self,
        conversation: Optional[
            List[Dict[str, str]]
        ],
    ) -> List[Dict[str, str]]:

        if not conversation:
            return []

        normalized = []

        for message in conversation[
            -MAX_CONVERSATION_MESSAGES:
        ]:

            if not isinstance(
                message,
                dict,
            ):
                continue

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

            content = str(
                content or ""
            )

            content = content[
                :MAX_CONVERSATION_MESSAGE_CHARS
            ]

            normalized.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return normalized

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

        normalized_conversation = (
            self._normalize_conversation(
                conversation
            )
        )

        prompt = build_evidence_prompt(
            question=question,
            retrieval=retrieval,
            conversation=normalized_conversation,
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        # ----------------------------------------------------
        # Previous conversation
        # ----------------------------------------------------

        for message in normalized_conversation:

            messages.append(
                {
                    "role": message[
                        "role"
                    ],
                    "content": message[
                        "content"
                    ],
                }
            )

        # ----------------------------------------------------
        # Current grounded prompt
        # ----------------------------------------------------

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return messages

    # ========================================================
    # MODEL NAME
    # ========================================================

    def _get_model_name(
        self,
    ) -> str:

        model_name = getattr(
            self.llm,
            "model",
            None,
        )

        if model_name:
            return str(
                model_name
            )

        model_name = getattr(
            self.llm,
            "model_name",
            None,
        )

        if model_name:
            return str(
                model_name
            )

        return "unknown"

    # ========================================================
    # EMPTY RESPONSE
    # ========================================================

    def _empty_response(
        self,
    ) -> Dict:

        return {
            "answer": (
                "Please provide a question."
            ),

            "intent": "general",

            "route_confidence": 0.0,

            "route_reason": "",

            "entities": {},

            "tools_used": [],

            "evidence_character_count": 0,

            "evidence_context": {},

            "model": self._get_model_name(),
        }

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

        # ----------------------------------------------------
        # Normalize question
        # ----------------------------------------------------

        question = str(
            question or ""
        ).strip()

        # ----------------------------------------------------
        # Empty question
        # ----------------------------------------------------

        if not question:
            return self._empty_response()

        # ----------------------------------------------------
        # Retrieve FraudSentinel evidence
        # ----------------------------------------------------

        retrieval = (
            self.retriever.retrieve(
                question,
            )
        )

        if not isinstance(
            retrieval,
            dict,
        ):
            retrieval = {
                "intent": "general",
                "route_confidence": 0.0,
                "route_reason": (
                    "Retriever returned "
                    "no structured evidence."
                ),
                "entities": {},
                "evidence": {},
            }

        # ----------------------------------------------------
        # Build grounded messages
        # ----------------------------------------------------

        messages = (
            self._build_messages(
                question=question,
                retrieval=retrieval,
                conversation=conversation,
            )
        )

        # ----------------------------------------------------
        # Generate LLM answer
        # ----------------------------------------------------

        answer = self.llm.generate(
            messages
        )

        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        evidence = retrieval.get(
            "evidence",
            {},
        )

        if not isinstance(
            evidence,
            dict,
        ):
            evidence = {}

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model_name = (
            self._get_model_name()
        )

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

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

            "tools_used": evidence.get(
                "tools",
                [],
            ),

            "evidence_character_count": evidence.get(
                "character_count",
                0,
            ),

            "evidence_context": evidence,

            "model": model_name,
        }