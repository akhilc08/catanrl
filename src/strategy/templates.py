"""Natural language explanation templates for Catan action recommendations."""

from __future__ import annotations

RESOURCE_NAMES = ["Wood", "Brick", "Sheep", "Wheat", "Ore"]

# Probability of rolling each number with 2d6
ROLL_PROBABILITIES = {
    2: 1 / 36,
    3: 2 / 36,
    4: 3 / 36,
    5: 4 / 36,
    6: 5 / 36,
    7: 6 / 36,
    8: 5 / 36,
    9: 4 / 36,
    10: 3 / 36,
    11: 2 / 36,
    12: 1 / 36,
}

TEMPLATES = {
    "BUILD_SETTLEMENT": (
        "Build a settlement at vertex {vertex_id} to gain access to {resources}. "
        "This location has {production_score:.1f} expected production per roll."
    ),
    "BUILD_CITY": (
        "Upgrade settlement at vertex {vertex_id} to a city, "
        "doubling production of {resources}."
    ),
    "BUILD_ROAD": (
        "Build a road on edge {edge_id} toward {target}, "
        "extending your network to {length} roads."
    ),
    "BUY_DEV_CARD": "Buy a development card ({reason}).",
    "BANK_TRADE": "Trade {ratio} {give} for 1 {get} at the bank ({reason}).",
    "PLAY_KNIGHT": (
        "Play Knight to move the robber to hex {hex_id} ({resource} production), "
        "blocking {target_player}."
    ),
    "ROLL_DICE": "Roll the dice to start your turn.",
    "END_TURN": "End your turn -- {reason}.",
    "PLACE_ROBBER": (
        "Place the robber on hex {hex_id} ({resource} with number {number}), "
        "blocking {target_player}."
    ),
    "STEAL_FROM_PLAYER": "Steal a random resource from Player {player_id}.",
    "DISCARD_RESOURCE": "Discard 1 {resource} to meet the discard requirement.",
    "PLAY_ROAD_BUILDING": "Play Road Building to place 2 free roads.",
    "PLAY_YEAR_OF_PLENTY": "Play Year of Plenty to take 1 free {resource} from the bank.",
    "PLAY_MONOPOLY": (
        "Play Monopoly on {resource} to collect all of that "
        "resource from other players."
    ),
}


def _vertex_resources(vertex_id: int, env) -> list[str]:
    """Return list of resource names produced by hexes adjacent to a vertex."""
    resources = []
    for hi in env.vert_hexes[vertex_id]:
        ht = int(env.hex_types[hi])
        if ht < 5:
            resources.append(RESOURCE_NAMES[ht])
    return resources


def _vertex_production_score(vertex_id: int, env) -> float:
    """Compute expected production per roll for a vertex."""
    score = 0.0
    for hi in env.vert_hexes[vertex_id]:
        num = int(env.hex_numbers[hi])
        if num > 0 and int(env.hex_types[hi]) < 5 and hi != env.robber_hex:
            score += ROLL_PROBABILITIES.get(num, 0.0)
    return score


def _hex_resource_name(hex_id: int, env) -> str:
    """Return the resource name for a hex."""
    ht = int(env.hex_types[hex_id])
    if ht < 5:
        return RESOURCE_NAMES[ht]
    return "Desert"


def _players_adjacent_to_hex(hex_id: int, env, exclude_player: int) -> list[int]:
    """Return list of players with buildings adjacent to a hex, excluding one."""
    players = set()
    for v in env.hex_vertices[hex_id]:
        owner = int(env.vertex_owner[v])
        if owner > 0 and (owner - 1) != exclude_player:
            players.add(owner - 1)
    return sorted(players)


def _count_player_roads(player: int, env) -> int:
    """Count total roads placed by a player."""
    from ..rl.env.catan_env import MAX_ROADS_PER_PLAYER

    return MAX_ROADS_PER_PLAYER - int(env.player_roads_left[player])


def _road_direction_target(edge_id: int, player: int, env) -> str:
    """Describe what the road is heading toward."""
    v0, v1 = int(env.edge_vertices[edge_id, 0]), int(env.edge_vertices[edge_id, 1])

    # Check if either endpoint has no building and good resources
    for v in (v0, v1):
        if env.vertex_building[v] == 0:
            resources = _vertex_resources(v, env)
            if resources:
                return f"a potential settlement producing {', '.join(resources)}"

    return "expanding your road network"


class ExplanationGenerator:
    """Generates natural language explanations for recommended actions."""

    def __init__(self):
        self.templates = TEMPLATES

    def generate(
        self,
        action_type: str,
        action_param: int,
        obs_dict: dict,
        score: float,
        env=None,
    ) -> str:
        """Generate explanation for an action given the game state.

        Parameters
        ----------
        action_type : str
            Action category name (e.g. "BUILD_SETTLEMENT").
        action_param : int
            Parameter within the action category.
        obs_dict : dict
            Observation dictionary from the environment.
        score : float
            Score / value estimate for this action.
        env : CatanEnv, optional
            If provided, used for richer explanations. Otherwise falls back to
            obs_dict-only info.
        """
        if action_type == "ROLL_DICE":
            return self.templates["ROLL_DICE"]

        if action_type == "END_TURN":
            reason = (
                "no better moves available"
                if score < 0.1
                else "saving resources for next turn"
            )
            return self.templates["END_TURN"].format(reason=reason)

        if action_type == "BUILD_SETTLEMENT":
            vertex_id = action_param
            if env is not None:
                resources = _vertex_resources(vertex_id, env)
                prod = _vertex_production_score(vertex_id, env)
            else:
                resources = self._resources_from_obs(vertex_id, obs_dict)
                prod = 0.0
            res_str = ", ".join(resources) if resources else "no resources"
            return self.templates["BUILD_SETTLEMENT"].format(
                vertex_id=vertex_id, resources=res_str, production_score=prod
            )

        if action_type == "BUILD_CITY":
            vertex_id = action_param
            if env is not None:
                resources = _vertex_resources(vertex_id, env)
            else:
                resources = self._resources_from_obs(vertex_id, obs_dict)
            res_str = ", ".join(resources) if resources else "no resources"
            return self.templates["BUILD_CITY"].format(
                vertex_id=vertex_id, resources=res_str
            )

        if action_type == "BUILD_ROAD":
            edge_id = action_param
            if env is not None:
                cur_player = int(obs_dict.get("current_player", 0))
                target = _road_direction_target(edge_id, cur_player, env)
                length = _count_player_roads(cur_player, env) + 1
            else:
                target = "expanding your road network"
                length = 0
            return self.templates["BUILD_ROAD"].format(
                edge_id=edge_id, target=target, length=length
            )

        if action_type == "BUY_DEV_CARD":
            cp = int(obs_dict.get("current_player", 0))
            player_feat = obs_dict.get("player_features")
            if player_feat is not None:
                vp = float(player_feat[cp][10]) * 10.0
                if vp >= 7:
                    reason = "close to winning, VP card could seal it"
                else:
                    reason = "building army or seeking VP cards"
            else:
                reason = "strategic investment"
            return self.templates["BUY_DEV_CARD"].format(reason=reason)

        if action_type == "BANK_TRADE":
            from ..rl.env.action_space import decode_bank_trade

            give_res, get_res = decode_bank_trade(action_param)
            if env is not None:
                ratio = env._get_trade_ratio(int(obs_dict.get("current_player", 0)), give_res)
            else:
                ratio = 4
            give_name = RESOURCE_NAMES[give_res]
            get_name = RESOURCE_NAMES[get_res]
            reason = f"need {get_name} for building"
            return self.templates["BANK_TRADE"].format(
                ratio=ratio, give=give_name, get=get_name, reason=reason
            )

        if action_type == "PLAY_KNIGHT":
            hex_id = action_param
            if env is not None:
                resource = _hex_resource_name(hex_id, env)
                cp = int(obs_dict.get("current_player", 0))
                targets = _players_adjacent_to_hex(hex_id, env, cp)
                target_player = f"Player {targets[0]}" if targets else "no one"
            else:
                resource = "unknown"
                target_player = "an opponent"
            return self.templates["PLAY_KNIGHT"].format(
                hex_id=hex_id, resource=resource, target_player=target_player
            )

        if action_type == "PLACE_ROBBER":
            hex_id = action_param
            if env is not None:
                resource = _hex_resource_name(hex_id, env)
                number = int(env.hex_numbers[hex_id])
                cp = int(obs_dict.get("current_player", 0))
                targets = _players_adjacent_to_hex(hex_id, env, cp)
                target_player = f"Player {targets[0]}" if targets else "no one"
            else:
                resource = "unknown"
                number = 0
                target_player = "an opponent"
            return self.templates["PLACE_ROBBER"].format(
                hex_id=hex_id, resource=resource, number=number,
                target_player=target_player,
            )

        if action_type == "STEAL_FROM_PLAYER":
            return self.templates["STEAL_FROM_PLAYER"].format(player_id=action_param)

        if action_type == "DISCARD_RESOURCE":
            return self.templates["DISCARD_RESOURCE"].format(
                resource=RESOURCE_NAMES[action_param]
            )

        if action_type == "PLAY_ROAD_BUILDING":
            return self.templates["PLAY_ROAD_BUILDING"]

        if action_type == "PLAY_YEAR_OF_PLENTY":
            return self.templates["PLAY_YEAR_OF_PLENTY"].format(
                resource=RESOURCE_NAMES[action_param]
            )

        if action_type == "PLAY_MONOPOLY":
            return self.templates["PLAY_MONOPOLY"].format(
                resource=RESOURCE_NAMES[action_param]
            )

        return f"Take action {action_type} (param={action_param})."

    def _resources_from_obs(self, vertex_id: int, obs_dict: dict) -> list[str]:
        """Fallback: infer resources from hex_features in obs_dict."""
        # Without the env, we can only tell resource types from hex features
        # This is a rough approximation
        return []

    def generate_strategy_summary(self, moves: list[dict], obs_dict: dict) -> str:
        """Generate an overall strategy summary for the current turn.

        Parameters
        ----------
        moves : list[dict]
            List of ranked move dicts, each with keys:
            action, action_type, action_param, mean_vp, win_rate, explanation.
        obs_dict : dict
            Current observation dictionary.

        Returns
        -------
        str
            Multi-line strategy summary.
        """
        if not moves:
            return "No candidate moves to evaluate."

        cp = int(obs_dict.get("current_player", 0))
        player_feat = obs_dict.get("player_features")
        if player_feat is not None:
            vp = float(player_feat[cp][10]) * 10.0
        else:
            vp = 0.0

        lines = [f"Strategy Summary (Player {cp}, {vp:.0f} VP):"]
        lines.append("")

        for i, move in enumerate(moves):
            rank = i + 1
            action_type = move.get("action_type", "unknown")
            explanation = move.get("explanation", "")
            win_rate = move.get("win_rate", 0.0)
            mean_vp = move.get("mean_vp", 0.0)

            lines.append(
                f"  {rank}. {action_type} -- "
                f"Win rate: {win_rate:.1%}, Expected VP gain: {mean_vp:.2f}"
            )
            if explanation:
                lines.append(f"     {explanation}")

        best = moves[0]
        lines.append("")
        lines.append(
            f"Recommended: {best.get('action_type', 'unknown')} "
            f"(win rate {best.get('win_rate', 0.0):.1%})"
        )

        return "\n".join(lines)
