from groq import Groq

from .config import (
    GROQ_API_KEY,
    GROQ_MODEL,
)


class FraudSentinelLLM:
    """
    Groq LLM client for FraudSentinel AI Copilot.

    The LLM is responsible only for interpreting
    and explaining evidence retrieved from the
    FraudSentinel Intelligence API.
    """

    def __init__(
        self,
        api_key: str | None = GROQ_API_KEY,
        model: str = GROQ_MODEL,
    ):
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured. "
                "Add it to the project .env file."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model = model

    def generate(
        self,
        messages: list[dict],
    ) -> str:
        """
        Generate a response from the Groq LLM.
        """

        response = (
            self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=1200,
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )