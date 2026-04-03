"""Tests for the API Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.api.schemas import (
    BoardState,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MoveRecommendation,
    PlayerResources,
    PredictRequest,
    PredictResponse,
)


def _make_board_state() -> dict:
    """Create a minimal valid board state dict."""
    hexes = [
        {"type": "wood", "number": 5, "position": i} for i in range(19)
    ]
    vertices = [{"building": None, "player": None} for _ in range(54)]
    edges = [{"road": False, "player": None} for _ in range(72)]
    ports = [{"type": "3:1", "vertices": [0, 1]}]
    return {
        "hexes": hexes,
        "vertices": vertices,
        "edges": edges,
        "robber": 0,
        "ports": ports,
    }


class TestPredictRequest:

    def test_valid_request(self):
        req = PredictRequest(
            board_state=BoardState(**_make_board_state()),
            player_index=0,
            player_resources=PlayerResources(wood=3, brick=2, sheep=1, wheat=0, ore=0),
            game_phase="MAIN",
        )
        assert req.player_index == 0
        assert req.player_resources.wood == 3

    def test_invalid_player_index_too_high(self):
        with pytest.raises(ValidationError):
            PredictRequest(
                board_state=BoardState(**_make_board_state()),
                player_index=5,
                player_resources=PlayerResources(),
                game_phase="MAIN",
            )

    def test_invalid_player_index_negative(self):
        with pytest.raises(ValidationError):
            PredictRequest(
                board_state=BoardState(**_make_board_state()),
                player_index=-1,
                player_resources=PlayerResources(),
                game_phase="MAIN",
            )

    def test_all_player_indices_valid(self):
        for idx in range(4):
            req = PredictRequest(
                board_state=BoardState(**_make_board_state()),
                player_index=idx,
                player_resources=PlayerResources(),
                game_phase="ROLL",
            )
            assert req.player_index == idx


class TestPredictResponse:

    def test_serialization(self):
        resp = PredictResponse(
            moves=[
                MoveRecommendation(
                    action="BUILD_SETTLEMENT:10",
                    action_id=84,
                    score=0.95,
                    explanation="Build a settlement on vertex 10.",
                )
            ],
            strategy_summary="Expansion-focused.",
            win_probability=0.62,
            model_version="v1.0",
            inference_latency_ms=15,
        )
        data = resp.model_dump()
        assert data["win_probability"] == 0.62
        assert len(data["moves"]) == 1
        assert data["moves"][0]["action_id"] == 84
        assert data["model_version"] == "v1.0"


class TestHealthResponse:

    def test_serialization(self):
        resp = HealthResponse(
            status="ok",
            model_loaded=True,
            uptime_s=120.5,
            model_version="v1.0",
        )
        data = resp.model_dump()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["uptime_s"] == 120.5

    def test_without_model_version(self):
        resp = HealthResponse(
            status="degraded",
            model_loaded=False,
            uptime_s=0.1,
        )
        data = resp.model_dump()
        assert data["model_version"] is None


class TestFeedbackRequest:

    def test_valid_feedback(self):
        req = FeedbackRequest(
            request_id="abc-123",
            was_move_good=True,
            comment="Great recommendation!",
        )
        assert req.request_id == "abc-123"
        assert req.was_move_good is True

    def test_feedback_without_comment(self):
        req = FeedbackRequest(
            request_id="def-456",
            was_move_good=False,
        )
        assert req.comment is None

    def test_feedback_missing_required_field(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(was_move_good=True)  # missing request_id


class TestFeedbackResponse:

    def test_serialization(self):
        resp = FeedbackResponse(status="stored", request_id="abc-123")
        data = resp.model_dump()
        assert data["status"] == "stored"
        assert data["request_id"] == "abc-123"
