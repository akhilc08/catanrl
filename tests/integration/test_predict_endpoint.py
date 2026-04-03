"""Integration tests for the CatanRL FastAPI inference API."""

from __future__ import annotations

import time

import httpx
import pytest
import pytest_asyncio
from src.api.main import app
from src.api.model_loader import ModelManager


@pytest_asyncio.fixture
async def client():
    """Create an async test client with app state initialized."""
    from httpx import ASGITransport

    # Manually initialize app state (lifespan doesn't run with ASGITransport)
    model_manager = ModelManager()
    model_manager.load()
    app.state.model = model_manager
    app.state.start_time = time.time()

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


def _make_valid_predict_body() -> dict:
    """Create a valid predict request body."""
    hexes = [{"type": "wood", "number": 5, "position": i} for i in range(19)]
    vertices = [{"building": None, "player": None} for _ in range(54)]
    edges = [{"road": False, "player": None} for _ in range(72)]
    ports = [{"type": "3:1", "vertices": [0, 1]}]
    return {
        "board_state": {
            "hexes": hexes,
            "vertices": vertices,
            "edges": edges,
            "robber": 0,
            "ports": ports,
        },
        "player_index": 0,
        "player_resources": {
            "wood": 3,
            "brick": 2,
            "sheep": 1,
            "wheat": 0,
            "ore": 0,
        },
        "game_phase": "MAIN",
    }


@pytest.mark.asyncio
async def test_predict_valid_request(client: httpx.AsyncClient):
    """POST /predict with valid request returns 200 with correct schema."""
    body = _make_valid_predict_body()
    response = await client.post("/predict", json=body)
    assert response.status_code == 200

    data = response.json()
    assert "moves" in data
    assert "strategy_summary" in data
    assert "win_probability" in data
    assert "model_version" in data
    assert "inference_latency_ms" in data
    assert isinstance(data["moves"], list)
    assert len(data["moves"]) > 0

    move = data["moves"][0]
    assert "action" in move
    assert "action_id" in move
    assert "score" in move
    assert "explanation" in move


@pytest.mark.asyncio
async def test_predict_invalid_request(client: httpx.AsyncClient):
    """POST /predict with invalid request returns 422."""
    body = {"invalid": "data"}
    response = await client.post("/predict", json=body)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_invalid_player_index(client: httpx.AsyncClient):
    """POST /predict with out-of-range player_index returns 422."""
    body = _make_valid_predict_body()
    body["player_index"] = 10
    response = await client.post("/predict", json=body)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_health_endpoint(client: httpx.AsyncClient):
    """GET /health returns 200 with health info."""
    response = await client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "model_loaded" in data
    assert "uptime_s" in data
    # With the untrained fallback, model should be loaded
    assert data["model_loaded"] is True


@pytest.mark.asyncio
async def test_model_version_endpoint(client: httpx.AsyncClient):
    """GET /model/version returns version info."""
    response = await client.get("/model/version")
    assert response.status_code == 200

    data = response.json()
    assert "version" in data
    assert isinstance(data["version"], str)


@pytest.mark.asyncio
async def test_feedback_endpoint(client: httpx.AsyncClient):
    """POST /feedback stores and returns success."""
    body = {
        "request_id": "test-request-001",
        "was_move_good": True,
        "comment": "Great move!",
    }
    response = await client.post("/feedback", json=body)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "stored"
    assert data["request_id"] == "test-request-001"


@pytest.mark.asyncio
async def test_feedback_without_comment(client: httpx.AsyncClient):
    """POST /feedback without comment still works."""
    body = {
        "request_id": "test-request-002",
        "was_move_good": False,
    }
    response = await client.post("/feedback", json=body)
    assert response.status_code == 200
    assert response.json()["status"] == "stored"


@pytest.mark.asyncio
async def test_predict_different_phases(client: httpx.AsyncClient):
    """POST /predict works for different game phases."""
    for phase in ["MAIN", "ROLL", "SETUP_FIRST", "DISCARD", "ROBBER_PLACE"]:
        body = _make_valid_predict_body()
        body["game_phase"] = phase
        response = await client.post("/predict", json=body)
        assert response.status_code == 200, f"Failed for phase={phase}"
