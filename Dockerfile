FROM python:3.11-slim

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Application directory
# ------------------------------------------------------------

WORKDIR /app

# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# ------------------------------------------------------------
# Application source
# ------------------------------------------------------------

COPY src ./src
COPY scripts ./scripts
COPY dashboard ./dashboard
COPY models ./models
COPY configs ./configs

# Artifacts and data are intentionally copied for the
# production-like local image.
COPY artifacts ./artifacts
COPY data ./data

# ------------------------------------------------------------
# Runtime directories
# ------------------------------------------------------------

RUN mkdir -p /app/logs

# ------------------------------------------------------------
# Python path
# ------------------------------------------------------------

ENV PYTHONPATH=/app

# ------------------------------------------------------------
# Default command
# ------------------------------------------------------------

CMD ["python", "-m", "scripts.run_pipeline"]