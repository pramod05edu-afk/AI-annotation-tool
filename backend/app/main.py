"""FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI

from app import __version__
from app.routers import annotations, health, images, logs, model_versions, projects
from app.storage import get_storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Runs once at startup (before `yield`) and once at shutdown (after `yield`)."""
    try:
        get_storage().ensure_bucket()
        logger.info("Storage bucket ready")
    except (BotoCoreError, ClientError):
        # Keep running: /api/health will report storage as unavailable.
        logger.exception("Could not prepare the storage bucket")
    # Phase 4: load the AI models here, once, instead of on every request.
    yield


app = FastAPI(
    title="AI-Assisted Annotation API",
    version=__version__,
    lifespan=lifespan,
    # Serve the docs under /api so they also work through the nginx proxy.
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

for router in (
    health.router,
    projects.router,
    images.router,
    annotations.router,
    model_versions.router,
    logs.router,
):
    app.include_router(router, prefix="/api")
