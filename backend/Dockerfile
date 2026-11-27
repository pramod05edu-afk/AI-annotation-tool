# ---- Stage 1: install dependencies with uv (build tools stay out of the final image) ----
FROM python:3.11-slim AS builder

# git: one dependency (the CLIP tokenizer used by YOLOE) is installed from a pinned Git commit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first: this layer is cached until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Then the application code.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ---- Stage 2: small runtime image, running as a non-root user ----
FROM python:3.11-slim

# OpenCV (used by Ultralytics) needs these two system libraries.
RUN apt-get update     && apt-get install -y --no-install-recommends libgl1 libglib2.0-0     && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/weights \
    && chown app:app /app/weights

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=4)"

# Apply database migrations, then start the API.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
