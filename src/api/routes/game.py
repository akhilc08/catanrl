"""Human-vs-AI game session endpoints.

POST /game/start          — create a new game, returns game_id + initial state
POST /game/{id}/step      — human submits their action; AI auto-plays for other seats
GET  /game/{id}/state     — current board/obs state
GET  /game/{id}/legal     — list of legal action IDs for the current player
DELETE /game/{id}         — end/clean up a session
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...rl.env.catan_env import CatanEnv

router = APIRouter(prefix="/game", tags=["human-vs-ai"])

# In-memory session store  {game_id: {"env": CatanEnv, "human_seat": int, "obs": dict, "done": bool}}
_sessions: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    human_seat: int = 0          # Which player seat the human controls (0-3)


class StartResponse(BaseModel):
    game_id: str
    human_seat: int
    current_player: int
    legal_actions: list[int]
    is_your_turn: bool


class StepRequest(BaseModel):
    action_id: int


class StepResponse(BaseModel):
    game_id: str
    current_player: int
    legal_actions: list[int]
    reward: float
    done: bool
    winner: int                   # -1 if game not over
    is_your_turn: bool
    ai_actions_taken: list[int]   # AI moves executed since last human turn


class StateResponse(BaseModel):
    game_id: str
    current_player: int
    legal_actions: list[int]
    done: bool
    winner: int
    is_your_turn: bool
    turn: int
    vp: list[int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session(game_id: str) -> dict[str, Any]:
    if game_id not in _sessions:
        raise HTTPException(status_code=404, detail="Game session not found")
    return _sessions[game_id]


def _legal_actions(obs: dict) -> list[int]:
    return [int(i) for i in np.where(obs["action_mask"])[0]]


def _policy_act(policy, obs: dict, device: torch.device) -> int:
    def _t(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.from_numpy(arr).to(dtype).unsqueeze(0).to(device)

    obs_dict = {
        "hex_features": _t(obs["hex_features"]),
        "vertex_features": _t(obs["vertex_features"]),
        "edge_features": _t(obs["edge_features"]),
        "player_features": _t(obs["player_features"]),
        "current_player": torch.tensor([obs["current_player"]], dtype=torch.long, device=device),
    }
    mask = _t(obs["action_mask"])
    policy.eval()
    with torch.no_grad():
        action, _, _, _ = policy.get_action_and_value(obs_dict, mask)
    return int(action.item())


def _advance_ai(session: dict, model_manager) -> tuple[list[int], float, bool, dict]:
    """Step the AI until it's the human's turn (or game ends). Returns (ai_actions, total_reward, done, info)."""
    env: CatanEnv = session["env"]
    obs = session["obs"]
    human_seat: int = session["human_seat"]
    ai_actions: list[int] = []
    total_reward = 0.0
    done = False
    info: dict = {}

    while not done and int(obs["current_player"]) != human_seat:
        action = _policy_act(model_manager.policy, obs, model_manager.device)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        ai_actions.append(action)

    session["obs"] = obs
    session["done"] = done
    return ai_actions, total_reward, done, info


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/start", response_model=StartResponse)
async def start_game(body: StartRequest, request: Request):
    model_manager = request.app.state.model
    if not model_manager.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    game_id = str(uuid.uuid4())
    env = CatanEnv(num_players=4)
    obs, _ = env.reset()

    session = {"env": env, "human_seat": body.human_seat, "obs": obs, "done": False}
    _sessions[game_id] = session

    # If it's not the human's turn first, advance AI
    ai_actions, _, done, _ = _advance_ai(session, model_manager)
    obs = session["obs"]

    return StartResponse(
        game_id=game_id,
        human_seat=body.human_seat,
        current_player=int(obs["current_player"]),
        legal_actions=_legal_actions(obs),
        is_your_turn=int(obs["current_player"]) == body.human_seat,
    )


@router.post("/{game_id}/step", response_model=StepResponse)
async def step_game(game_id: str, body: StepRequest, request: Request):
    session = _get_session(game_id)
    model_manager = request.app.state.model

    if session["done"]:
        raise HTTPException(status_code=400, detail="Game is already over")

    obs = session["obs"]
    human_seat: int = session["human_seat"]

    if int(obs["current_player"]) != human_seat:
        raise HTTPException(status_code=400, detail="Not your turn")

    mask = obs["action_mask"].astype(bool)
    if not mask[body.action_id]:
        raise HTTPException(status_code=400, detail=f"Action {body.action_id} is not legal")

    env: CatanEnv = session["env"]
    obs, reward, terminated, truncated, info = env.step(body.action_id)
    done = terminated or truncated
    session["obs"] = obs
    session["done"] = done

    ai_actions: list[int] = []
    if not done:
        ai_actions, _, done, info = _advance_ai(session, model_manager)

    obs = session["obs"]
    return StepResponse(
        game_id=game_id,
        current_player=int(obs["current_player"]),
        legal_actions=_legal_actions(obs) if not done else [],
        reward=float(reward),
        done=done,
        winner=int(info.get("winner", -1)),
        is_your_turn=not done and int(obs["current_player"]) == human_seat,
        ai_actions_taken=ai_actions,
    )


@router.get("/{game_id}/state", response_model=StateResponse)
async def get_state(game_id: str):
    session = _get_session(game_id)
    obs = session["obs"]
    env: CatanEnv = session["env"]
    info = env._get_info() if hasattr(env, "_get_info") else {}
    vp = info.get("vp", [0, 0, 0, 0])

    return StateResponse(
        game_id=game_id,
        current_player=int(obs["current_player"]),
        legal_actions=_legal_actions(obs) if not session["done"] else [],
        done=session["done"],
        winner=int(info.get("winner", -1)),
        is_your_turn=not session["done"] and int(obs["current_player"]) == session["human_seat"],
        turn=int(info.get("turn", 0)),
        vp=[int(v) for v in vp] if hasattr(vp, "__iter__") else [0, 0, 0, 0],
    )


@router.get("/{game_id}/legal")
async def get_legal_actions(game_id: str):
    session = _get_session(game_id)
    return {"legal_actions": _legal_actions(session["obs"])}


@router.delete("/{game_id}")
async def end_game(game_id: str):
    _get_session(game_id)
    del _sessions[game_id]
    return {"status": "deleted", "game_id": game_id}
