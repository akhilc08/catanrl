"""Arena: head-to-head evaluation of a trained policy vs a baseline agent.

The eval policy rotates seats across games so results aren't biased by
first-mover advantage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from ..env.catan_env import CatanEnv
from ..models.policy import CatanPolicy


class Agent(Protocol):
    def act(self, obs: dict) -> int: ...


@dataclass
class ArenaResult:
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    mean_game_length: float
    games_completed: int          # total episodes across all training so far (for sample efficiency)
    game_lengths: list[int] = field(default_factory=list)


class Arena:
    """Run N full 4-player Catan games: eval policy vs 3 opponent agents.

    Parameters
    ----------
    policy : CatanPolicy
        The trained policy to evaluate (set to eval mode internally).
    opponent : Agent
        Baseline agent (RandomAgent or HeuristicAgent) filling the other 3 seats.
    device : str
    num_games : int
        Number of games to play.
    rotate_seats : bool
        If True, the eval policy cycles through seats 0-3 for fairness.
    games_completed : int
        Running total of training episodes so far (for sample efficiency logging).
    """

    def __init__(
        self,
        policy: CatanPolicy,
        opponent: Agent,
        device: str = "cpu",
        num_games: int = 50,
        rotate_seats: bool = True,
        games_completed: int = 0,
    ) -> None:
        self.policy = policy
        self.opponent = opponent
        self.device = torch.device(device)
        self.num_games = num_games
        self.rotate_seats = rotate_seats
        self.games_completed = games_completed

    def _policy_act(self, obs: dict) -> int:
        def _t(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
            return torch.from_numpy(arr).to(dtype).unsqueeze(0).to(self.device)

        obs_dict = {
            "hex_features": _t(obs["hex_features"]),
            "vertex_features": _t(obs["vertex_features"]),
            "edge_features": _t(obs["edge_features"]),
            "player_features": _t(obs["player_features"]),
            "current_player": torch.tensor(
                [obs["current_player"]], dtype=torch.long, device=self.device
            ),
        }
        mask = _t(obs["action_mask"])
        with torch.no_grad():
            action, _, _, _ = self.policy.get_action_and_value(obs_dict, mask)
        return int(action.item())

    def run(self) -> ArenaResult:
        self.policy.eval()
        wins = losses = draws = 0
        game_lengths: list[int] = []

        for game_idx in range(self.num_games):
            eval_seat = game_idx % 4 if self.rotate_seats else 0
            env = CatanEnv(num_players=4)
            obs, _ = env.reset()
            done = False
            steps = 0

            while not done:
                current = int(obs["current_player"])
                if current == eval_seat:
                    action = self._policy_act(obs)
                else:
                    action = self.opponent.act(obs)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

            game_lengths.append(steps)
            winner = info.get("winner", -1)
            if winner == eval_seat:
                wins += 1
            elif winner == -1:
                draws += 1
            else:
                losses += 1

        total = wins + losses + draws
        return ArenaResult(
            games=total,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=wins / total if total > 0 else 0.0,
            mean_game_length=float(np.mean(game_lengths)) if game_lengths else 0.0,
            games_completed=self.games_completed,
            game_lengths=game_lengths,
        )
