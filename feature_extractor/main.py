import logging
import uvicorn
from typing import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware

from common.config import SETTINGS
from common.constants import SERVICE_NAME, SERVICE_VERSION
from feature_extractor.extractor import build_extractor
from feature_extractor.middleware import log_requests_middleware
from feature_extractor.routes import router
from feature_extractor.utils import initialize_logger, shutdown_logs_executor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    extractor = await run_in_threadpool(build_extractor, SETTINGS)
    if SETTINGS.model.warmup:
        await run_in_threadpool(extractor.warmup)

    app.state.extractor = extractor
    try:
        yield
    finally:
        app.state.extractor = None
        extractor.close()
        shutdown_logs_executor()


app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description="Extracts a DINOv3 feature vector and per-layer feature maps from one image resized to 512x512.",
    lifespan=lifespan,
)
app.middleware("http")(log_requests_middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

if __name__ == "__main__":
    initialize_logger("vfm_feature_extractor.log")
    logger.info(f"Starting {SERVICE_NAME} on port {SETTINGS.server_port}")

    uvicorn.run(app, host="0.0.0.0", port=SETTINGS.server_port)
