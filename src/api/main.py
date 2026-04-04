"""FastAPI inference service for CatanRL."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request

from .model_loader import ModelManager
from .routes.feedback import router as feedback_router
from .routes.game import router as game_router
from .routes.health import router as health_router
from .routes.predict import router as predict_router

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    model_manager = ModelManager()
    model_manager.load()
    app.state.model = model_manager
    app.state.start_time = time.time()
    logger.info("server_started", model_version=model_manager.version)
    yield
    logger.info("server_shutdown")


app = FastAPI(
    title="CatanRL",
    description="Inference API for the CatanRL reinforcement learning agent.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Inject a unique request ID and log every request with latency."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.time()
    response = await call_next(request)
    latency = (time.time() - start) * 1000
    logger.info(
        "request_complete",
        request_id=request_id,
        path=request.url.path,
        latency_ms=round(latency, 1),
        status=response.status_code,
    )
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(predict_router, tags=["inference"])
app.include_router(health_router, tags=["health"])
app.include_router(feedback_router, tags=["feedback"])
app.include_router(game_router)
