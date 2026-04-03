"""Prediction route for the CatanRL inference API."""

from __future__ import annotations

import time

import numpy as np
import structlog
from fastapi import APIRouter, Request

from ...rl.env.action_space import ActionSpace, decode_bank_trade
from ..schemas import MoveRecommendation, PredictRequest, PredictResponse

logger = structlog.get_logger()

router = APIRouter()

# Maps resource index to human-readable name
_RESOURCE_NAMES = {0: "wood", 1: "brick", 2: "sheep", 3: "wheat", 4: "ore"}

# Maps hex resource type string to one-hot index
_HEX_TYPE_INDEX = {
    "wood": 0,
    "brick": 1,
    "sheep": 2,
    "wheat": 3,
    "ore": 4,
    "desert": 5,
}

# Building type to feature index
_BUILDING_INDEX = {"settlement": 1, "city": 2}


def _explain_action(action_type: str, param: int) -> str:
    """Generate a human-readable explanation for a decoded action."""
    if action_type == "ROLL_DICE":
        return "Roll the dice to start your turn."
    if action_type == "END_TURN":
        return "End your turn."
    if action_type == "BUILD_ROAD":
        return f"Build a road on edge {param}."
    if action_type == "BUILD_SETTLEMENT":
        return f"Build a settlement on vertex {param}."
    if action_type == "BUILD_CITY":
        return f"Upgrade settlement to city on vertex {param}."
    if action_type == "BUY_DEV_CARD":
        return "Buy a development card."
    if action_type == "PLAY_KNIGHT":
        return f"Play knight and move robber to hex {param}."
    if action_type == "PLAY_ROAD_BUILDING":
        return "Play road building development card."
    if action_type == "PLAY_YEAR_OF_PLENTY":
        return f"Play year of plenty for {_RESOURCE_NAMES.get(param, param)}."
    if action_type == "PLAY_MONOPOLY":
        return f"Play monopoly on {_RESOURCE_NAMES.get(param, param)}."
    if action_type == "BANK_TRADE":
        give_res, get_res = decode_bank_trade(param)
        return f"Trade {_RESOURCE_NAMES[give_res]} for {_RESOURCE_NAMES[get_res]} at the bank."
    if action_type == "PLACE_ROBBER":
        return f"Place robber on hex {param}."
    if action_type == "STEAL_FROM_PLAYER":
        return f"Steal a resource from player {param}."
    if action_type == "DISCARD_RESOURCE":
        return f"Discard {_RESOURCE_NAMES.get(param, param)}."
    return f"{action_type} (param={param})"


def _request_to_obs(req: PredictRequest) -> tuple[dict, np.ndarray]:
    """Convert a PredictRequest into an observation dict and action mask.

    Builds numpy arrays matching the shapes expected by CatanPolicy._encode:
        hex_features:    (19, 9)   — 6 type one-hot + desert + number_scaled + robber
        vertex_features: (54, 7)   — 3 building one-hot + 4 owner one-hot
        edge_features:   (72, 5)   — 1 has_road + 4 owner one-hot
        player_features: (4, 14)   — resources(5) + dev_cards(5) + vp + pieces_left(3)
        current_player:  scalar int
    """
    board = req.board_state

    # --- Hex features (19, 9) ---
    hex_feat = np.zeros((19, 9), dtype=np.float32)
    for h in board.hexes:
        idx = h.position
        type_idx = _HEX_TYPE_INDEX.get(h.type, 5)
        hex_feat[idx, type_idx] = 1.0  # one-hot for resource type
        if h.type == "desert":
            hex_feat[idx, 6] = 1.0  # desert flag
        hex_feat[idx, 7] = h.number / 12.0  # scaled number token
        if idx == board.robber:
            hex_feat[idx, 8] = 1.0  # robber present

    # --- Vertex features (54, 7) ---
    vertex_feat = np.zeros((54, 7), dtype=np.float32)
    for i, v in enumerate(board.vertices):
        if v.building and v.building != "none":
            btype = _BUILDING_INDEX.get(v.building, 0)
            vertex_feat[i, btype] = 1.0  # building type one-hot (idx 0=none,1=settle,2=city)
        else:
            vertex_feat[i, 0] = 1.0  # no building
        if v.player is not None:
            vertex_feat[i, 3 + v.player] = 1.0  # owner one-hot

    # --- Edge features (72, 5) ---
    edge_feat = np.zeros((72, 5), dtype=np.float32)
    for i, e in enumerate(board.edges):
        if e.road:
            edge_feat[i, 0] = 1.0
        if e.player is not None:
            edge_feat[i, 1 + e.player] = 1.0

    # --- Player features (4, 14) ---
    # We only have the requesting player's resources; zero out the rest.
    player_feat = np.zeros((4, 14), dtype=np.float32)
    p = req.player_index
    player_feat[p, 0] = req.player_resources.wood
    player_feat[p, 1] = req.player_resources.brick
    player_feat[p, 2] = req.player_resources.sheep
    player_feat[p, 3] = req.player_resources.wheat
    player_feat[p, 4] = req.player_resources.ore

    current_player = np.array(req.player_index, dtype=np.int64)

    obs_dict = {
        "hex_features": hex_feat,
        "vertex_features": vertex_feat,
        "edge_features": edge_feat,
        "player_features": player_feat,
        "current_player": current_player,
    }

    # --- Action mask ---
    # Without the full env we can't compute a precise mask, so we enable all
    # actions in the relevant phase category. The model handles soft masking.
    action_mask = np.zeros(ActionSpace.TOTAL_ACTIONS, dtype=np.bool_)
    phase = req.game_phase.upper()

    if phase in ("SETUP_FIRST", "SETUP_SECOND"):
        action_mask[ActionSpace.BUILD_SETTLEMENT_OFFSET:
                    ActionSpace.BUILD_SETTLEMENT_OFFSET + 54] = True
        action_mask[ActionSpace.BUILD_ROAD_OFFSET:
                    ActionSpace.BUILD_ROAD_OFFSET + 72] = True
    elif phase == "ROLL":
        action_mask[ActionSpace.ROLL_DICE_OFFSET] = True
        action_mask[ActionSpace.PLAY_KNIGHT_OFFSET:
                    ActionSpace.PLAY_KNIGHT_OFFSET + 19] = True
    elif phase == "DISCARD":
        action_mask[ActionSpace.DISCARD_RESOURCE_OFFSET:
                    ActionSpace.DISCARD_RESOURCE_OFFSET + 5] = True
    elif phase == "ROBBER_PLACE":
        action_mask[ActionSpace.PLACE_ROBBER_OFFSET:
                    ActionSpace.PLACE_ROBBER_OFFSET + 19] = True
    elif phase == "ROBBER_STEAL":
        action_mask[ActionSpace.STEAL_FROM_PLAYER_OFFSET:
                    ActionSpace.STEAL_FROM_PLAYER_OFFSET + 4] = True
    elif phase in ("MAIN", ""):
        # Enable all main-phase actions
        action_mask[ActionSpace.END_TURN_OFFSET] = True
        action_mask[ActionSpace.BUILD_ROAD_OFFSET:
                    ActionSpace.BUILD_ROAD_OFFSET + 72] = True
        action_mask[ActionSpace.BUILD_SETTLEMENT_OFFSET:
                    ActionSpace.BUILD_SETTLEMENT_OFFSET + 54] = True
        action_mask[ActionSpace.BUILD_CITY_OFFSET:
                    ActionSpace.BUILD_CITY_OFFSET + 54] = True
        action_mask[ActionSpace.BUY_DEV_CARD_OFFSET] = True
        action_mask[ActionSpace.PLAY_KNIGHT_OFFSET:
                    ActionSpace.PLAY_KNIGHT_OFFSET + 19] = True
        action_mask[ActionSpace.PLAY_ROAD_BUILDING_OFFSET] = True
        action_mask[ActionSpace.PLAY_YEAR_OF_PLENTY_OFFSET:
                    ActionSpace.PLAY_YEAR_OF_PLENTY_OFFSET + 5] = True
        action_mask[ActionSpace.PLAY_MONOPOLY_OFFSET:
                    ActionSpace.PLAY_MONOPOLY_OFFSET + 5] = True
        action_mask[ActionSpace.BANK_TRADE_OFFSET:
                    ActionSpace.BANK_TRADE_OFFSET + 20] = True
    elif phase == "ROAD_BUILDING":
        action_mask[ActionSpace.BUILD_ROAD_OFFSET:
                    ActionSpace.BUILD_ROAD_OFFSET + 72] = True
        action_mask[ActionSpace.END_TURN_OFFSET] = True
    elif phase == "YEAR_OF_PLENTY":
        action_mask[ActionSpace.PLAY_YEAR_OF_PLENTY_OFFSET:
                    ActionSpace.PLAY_YEAR_OF_PLENTY_OFFSET + 5] = True
    else:
        # Unknown phase — enable everything as a safe fallback
        action_mask[:] = True

    return obs_dict, action_mask


def _generate_strategy_summary(moves: list[MoveRecommendation]) -> str:
    """Generate a brief strategy summary from the top moves."""
    if not moves:
        return "No legal moves available."
    top = moves[0]
    action_type, _ = ActionSpace.decode_action(top.action_id)
    if "BUILD" in action_type:
        return "Expansion-focused: the model recommends building infrastructure."
    if "TRADE" in action_type:
        return "Resource management: the model recommends trading to optimize resources."
    if "KNIGHT" in action_type or "ROBBER" in action_type:
        return "Aggressive: the model recommends disrupting opponents."
    if action_type == "END_TURN":
        return "Conservative: no strong plays available, end the turn."
    if action_type == "ROLL_DICE":
        return "Roll phase: roll the dice to proceed."
    return f"Recommended action: {action_type.lower().replace('_', ' ')}."


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    """Run model inference on the given board state and return top move recommendations."""
    start = time.time()
    model_manager = request.app.state.model
    request_id = getattr(request.state, "request_id", "unknown")

    obs_dict, action_mask = _request_to_obs(body)
    actions_with_scores, win_prob = model_manager.predict(obs_dict, action_mask)

    moves: list[MoveRecommendation] = []
    for action_id, score in actions_with_scores:
        action_type, param = ActionSpace.decode_action(action_id)
        explanation = _explain_action(action_type, param)
        action_name = f"{action_type}"
        if param > 0:
            action_name += f":{param}"
        moves.append(
            MoveRecommendation(
                action=action_name,
                action_id=action_id,
                score=round(score, 4),
                explanation=explanation,
            )
        )

    latency_ms = int((time.time() - start) * 1000)

    logger.info(
        "inference_complete",
        request_id=request_id,
        top_action=moves[0].action if moves else "none",
        win_prob=round(win_prob, 4),
        latency_ms=latency_ms,
        phase=body.game_phase,
    )

    return PredictResponse(
        moves=moves,
        strategy_summary=_generate_strategy_summary(moves),
        win_probability=round(win_prob, 4),
        model_version=model_manager.version,
        inference_latency_ms=latency_ms,
    )
