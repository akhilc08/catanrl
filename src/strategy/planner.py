"""Monte Carlo rollout planner for multi-step Catan strategy."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import structlog
import torch

from ..rl.env.action_space import ActionSpace
from .templates import ExplanationGenerator

logger = structlog.get_logger()


class MonteCarloPlanner:
    """Uses Monte Carlo rollouts to evaluate multi-step strategies.

    Clones the environment state, takes a candidate action, then uses the
    policy to play out the remaining moves. Aggregates VP gain and win rate
    across multiple rollouts.

    Parameters
    ----------
    policy : CatanPolicy
        The trained policy for rollout simulation.
    env_class : type
        The environment class (CatanEnv) to instantiate for rollouts.
    num_rollouts : int
        Number of forward simulations per candidate action.
    max_depth : int
        Maximum number of steps per rollout.
    """

    def __init__(
        self,
        policy,
        env_class,
        num_rollouts: int = 50,
        max_depth: int = 20,
    ):
        self.policy = policy
        self.env_class = env_class
        self.num_rollouts = num_rollouts
        self.max_depth = max_depth
        self.generator = ExplanationGenerator()

    def evaluate_action(self, env_state: dict, action: int) -> dict:
        """Evaluate a single action by running num_rollouts forward simulations.

        Parameters
        ----------
        env_state : dict
            Serialized environment state (all attributes needed to reconstruct
            the game state).
        action : int
            The flat action ID to evaluate.

        Returns
        -------
        dict
            {mean_vp_gain, win_rate, mean_game_length, confidence}
        """
        vp_gains = []
        wins = 0
        game_lengths = []

        for _ in range(self.num_rollouts):
            env = self._restore_env(env_state)
            result = self._simulate_rollout(env, action)
            vp_gains.append(result["vp_gain"])
            if result["won"]:
                wins += 1
            game_lengths.append(result["steps"])

        vp_arr = np.array(vp_gains, dtype=np.float64)
        mean_vp = float(vp_arr.mean())
        std_vp = float(vp_arr.std())
        win_rate = wins / self.num_rollouts
        mean_length = float(np.mean(game_lengths))

        # Confidence based on standard error
        se = std_vp / np.sqrt(self.num_rollouts) if self.num_rollouts > 1 else 0.0
        confidence = max(0.0, 1.0 - se)

        return {
            "mean_vp_gain": mean_vp,
            "win_rate": win_rate,
            "mean_game_length": mean_length,
            "confidence": confidence,
        }

    def plan(
        self,
        env_state: dict,
        candidate_actions: list[int],
        top_k: int = 3,
    ) -> list[dict]:
        """Evaluate multiple candidate actions via rollouts.

        Parameters
        ----------
        env_state : dict
            Serialized environment state.
        candidate_actions : list[int]
            List of flat action IDs to evaluate.
        top_k : int
            Number of top actions to return.

        Returns
        -------
        list[dict]
            Ranked list of {action, action_type, action_param, mean_vp,
            win_rate, explanation} sorted by mean_vp descending.
        """
        if not candidate_actions:
            return []

        results = []
        for action in candidate_actions:
            action_type, action_param = ActionSpace.decode_action(action)

            logger.debug(
                "evaluating_action",
                action=action,
                action_type=action_type,
                action_param=action_param,
            )

            eval_result = self.evaluate_action(env_state, action)

            # Generate explanation using the env for context
            env = self._restore_env(env_state)
            obs = env._get_observation()
            explanation = self.generator.generate(
                action_type, action_param, obs, eval_result["mean_vp_gain"], env=env
            )

            results.append({
                "action": action,
                "action_type": action_type,
                "action_param": action_param,
                "mean_vp": eval_result["mean_vp_gain"],
                "win_rate": eval_result["win_rate"],
                "mean_game_length": eval_result["mean_game_length"],
                "confidence": eval_result["confidence"],
                "explanation": explanation,
            })

        # Sort by mean VP gain descending, then win rate as tiebreaker
        results.sort(key=lambda r: (r["mean_vp"], r["win_rate"]), reverse=True)

        return results[:top_k]

    def _simulate_rollout(self, env, action: int) -> dict:
        """Run one rollout from current state after taking the given action.

        Parameters
        ----------
        env : CatanEnv
            A fresh copy of the environment at the decision point.
        action : int
            The initial action to take.

        Returns
        -------
        dict
            {vp_gain, won, steps}
        """
        player = env.current_player
        initial_vp = int(env.player_vp[player])

        # Take the candidate action
        obs, reward, terminated, truncated, info = env.step(action)
        steps = 1

        # Continue playing using the policy until game ends or max_depth
        while not terminated and not truncated and steps < self.max_depth:
            action_mask = obs["action_mask"]

            # Use policy to select action
            with torch.no_grad():
                self.policy.eval()
                act_tensor, _, _, _ = self.policy.get_action_and_value(
                    obs, action_mask
                )
                next_action = int(act_tensor.item())

            obs, reward, terminated, truncated, info = env.step(next_action)
            steps += 1

        final_vp = int(env.player_vp[player])
        vp_gain = final_vp - initial_vp
        won = env.winner == player

        return {"vp_gain": vp_gain, "won": won, "steps": steps}

    def _restore_env(self, env_state: dict) -> Any:
        """Create a new env instance and restore state from a state dict.

        Parameters
        ----------
        env_state : dict
            State dictionary containing all env attributes.

        Returns
        -------
        CatanEnv
            A new environment with the given state.
        """
        env = self.env_class()
        env.reset()

        # Restore all state attributes
        for key, value in env_state.items():
            if isinstance(value, np.ndarray):
                setattr(env, key, value.copy())
            elif isinstance(value, list):
                setattr(env, key, copy.deepcopy(value))
            else:
                setattr(env, key, value)

        return env

    @staticmethod
    def capture_env_state(env) -> dict:
        """Capture the current state of a CatanEnv as a dict for restoration.

        Parameters
        ----------
        env : CatanEnv
            The environment to capture.

        Returns
        -------
        dict
            State dictionary with all mutable attributes.
        """
        state_keys = [
            "hex_types", "hex_numbers", "robber_hex", "port_info_current",
            "player_resources", "player_dev_cards", "player_knights_played",
            "player_vp", "player_roads_left", "player_settlements_left",
            "player_cities_left", "vertex_building", "vertex_owner",
            "edge_road", "edge_owner", "dev_card_deck", "dev_card_deck_index",
            "current_player", "game_phase", "dice_roll", "turn_counter",
            "setup_sub_phase", "setup_settlements_placed", "last_setup_settlement",
            "setup_order", "setup_index", "longest_road_player",
            "longest_road_length", "largest_army_player", "largest_army_size",
            "has_played_dev_card_this_turn", "dev_card_bought_this_turn_idx",
            "road_building_roads_left", "year_of_plenty_resources_left",
            "discard_remaining", "discard_player_index", "winner",
        ]
        state = {}
        for key in state_keys:
            value = getattr(env, key)
            if isinstance(value, np.ndarray):
                state[key] = value.copy()
            elif isinstance(value, list):
                state[key] = copy.deepcopy(value)  # type: ignore[arg-type]
            else:
                state[key] = value
        return state
