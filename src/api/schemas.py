"""Pydantic models for the CatanRL inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HexState(BaseModel):
    type: str = Field(..., description="Resource type: wood, brick, sheep, wheat, ore, desert")
    number: int = Field(..., ge=2, le=12, description="Number token")
    position: int = Field(..., ge=0, le=18)


class VertexState(BaseModel):
    building: str | None = Field(None, description="none, settlement, city")
    player: int | None = Field(None, ge=0, le=3)


class EdgeState(BaseModel):
    road: bool = False
    player: int | None = None


class PortState(BaseModel):
    type: str  # "3:1" or resource name for 2:1
    vertices: list[int]


class BoardState(BaseModel):
    hexes: list[HexState]
    vertices: list[VertexState]
    edges: list[EdgeState]
    robber: int = Field(..., ge=0, le=18)
    ports: list[PortState]


class PlayerResources(BaseModel):
    wood: int = 0
    brick: int = 0
    sheep: int = 0
    wheat: int = 0
    ore: int = 0


class PredictRequest(BaseModel):
    board_state: BoardState
    player_index: int = Field(..., ge=0, le=3)
    player_resources: PlayerResources
    game_phase: str


class MoveRecommendation(BaseModel):
    action: str
    action_id: int
    score: float
    explanation: str
    attention_highlights: list[int] = []


class PredictResponse(BaseModel):
    moves: list[MoveRecommendation]
    strategy_summary: str
    win_probability: float
    model_version: str
    inference_latency_ms: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_s: float
    model_version: str | None = None


class FeedbackRequest(BaseModel):
    request_id: str
    was_move_good: bool
    comment: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    request_id: str


class ModelVersionResponse(BaseModel):
    version: str
    trained_at: str | None = None
    win_rate_vs_heuristic: float | None = None
