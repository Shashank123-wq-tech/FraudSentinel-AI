import os

from dotenv import load_dotenv


load_dotenv()


# ============================================================
# INTELLIGENCE API
# ============================================================

API_BASE_URL = os.getenv(
    "FRAUDSENTINEL_API_URL",
    "http://127.0.0.1:8000",
)

REQUEST_TIMEOUT = int(
    os.getenv(
        "FRAUDSENTINEL_API_TIMEOUT",
        "30",
    )
)


# ============================================================
# GROQ
# ============================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)


# ============================================================
# COPILOT
# ============================================================

MAX_CONTEXT_CHARS = int(
    os.getenv(
        "COPILOT_MAX_CONTEXT_CHARS",
        "30000",
    )
)