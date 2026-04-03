"""Health and model version routes for the CatanRL inference API."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from ..schemas import HealthResponse, ModelVersionResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return service health status."""
    model_manager = request.app.state.model
    start_time = request.app.state.start_time
    uptime = time.time() - start_time

    return HealthResponse(
        status="ok" if model_manager.policy is not None else "degraded",
        model_loaded=model_manager.policy is not None,
        uptime_s=round(uptime, 2),
        model_version=model_manager.version if model_manager.policy else None,
    )


@router.get("/model/version", response_model=ModelVersionResponse)
async def model_version(request: Request) -> ModelVersionResponse:
    """Return details about the loaded model version."""
    model_manager = request.app.state.model
    metadata = model_manager.metadata

    return ModelVersionResponse(
        version=model_manager.version,
        trained_at=metadata.get("trained_at"),
        win_rate_vs_heuristic=metadata.get("win_rate_vs_heuristic"),
    )
