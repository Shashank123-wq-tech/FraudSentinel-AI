from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
)

from .endpoints import router


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================

app.include_router(
    router,
    prefix="/api",
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "FraudSentinel AI",
        "service": "Intelligence API",
        "status": "online",
        "version": API_VERSION,
    }