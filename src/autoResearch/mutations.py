"""Mutation engine for autonomous hyperparameter search.

Defines the search space over MAPPOConfig and proposes mutations based on
experiment history using UCB1-style exploration.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from ..rl.training.mappo import MAPPOConfig


# ---------------------------------------------------------------------------
# Mutation definition
# ---------------------------------------------------------------------------

@dataclass
class Mutation:
    """A single proposed change to a MAPPOConfig field."""

    param: str
    old_value: Any
    new_value: Any
    description: str

    def apply(self, config: MAPPOConfig) -> MAPPOConfig:
        """Return a new config with this mutation applied (does not mutate in place)."""
        import copy
        new_cfg = copy.deepcopy(config)
        setattr(new_cfg, self.param, self.new_value)
        return new_cfg

    def __str__(self) -> str:
        return f"{self.param}: {self.old_value} -> {self.new_value}  ({self.description})"


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

# Each entry: (param, candidate_values, description_template)
_SEARCH_SPACE: list[tuple[str, list[Any], str]] = [
    ("learning_rate", [1e-4, 2e-4, 3e-4, 5e-4, 1e-3], "lr sweep"),
    ("gamma", [0.95, 0.97, 0.99, 0.995], "discount factor"),
    ("gae_lambda", [0.90, 0.92, 0.95, 0.97, 0.99], "GAE lambda"),
    ("clip_coef", [0.1, 0.15, 0.2, 0.3, 0.4], "PPO clip"),
    ("ent_coef", [0.002, 0.005, 0.01, 0.02, 0.05], "entropy bonus"),
    ("vf_coef", [0.25, 0.5, 0.75, 1.0], "value loss weight"),
    ("update_epochs", [2, 3, 4, 6, 8], "PPO epochs per rollout"),
    ("num_minibatches", [2, 4, 8], "minibatch count"),
    ("max_grad_norm", [0.3, 0.5, 0.8, 1.0], "gradient clipping"),
    ("num_steps", [64, 128, 256], "rollout length"),
]


# ---------------------------------------------------------------------------
# UCB1 arm state
# ---------------------------------------------------------------------------

@dataclass
class _ArmStats:
    param: str
    value: Any
    n_trials: int = 0
    total_delta: float = 0.0

    @property
    def mean_delta(self) -> float:
        return self.total_delta / self.n_trials if self.n_trials > 0 else 0.0

    def ucb1(self, total_trials: int, c: float = 1.0) -> float:
        if self.n_trials == 0:
            return float("inf")
        return self.mean_delta + c * math.sqrt(math.log(total_trials + 1) / self.n_trials)


# ---------------------------------------------------------------------------
# Mutation engine
# ---------------------------------------------------------------------------

class MutationEngine:
    """Proposes MAPPOConfig mutations using UCB1 over the search space.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    exploration_c : float
        UCB1 exploration constant. Higher = more exploration.
    """

    def __init__(self, seed: int = 42, exploration_c: float = 1.0) -> None:
        self._rng = random.Random(seed)
        self._c = exploration_c
        self._arms: dict[tuple[str, Any], _ArmStats] = {}

        for param, candidates, _desc in _SEARCH_SPACE:
            for val in candidates:
                key = (param, val)
                self._arms[key] = _ArmStats(param=param, value=val)

        self._total_trials = 0

    # ------------------------------------------------------------------

    def propose(self, base_config: MAPPOConfig) -> Mutation:
        """Pick the best UCB1 arm and return a Mutation relative to base_config.

        Arms that match the current base_config value are excluded (no-op mutations).
        Untried arms are always explored first (UCB1 = inf).
        """
        eligible = [
            arm for (param, val), arm in self._arms.items()
            if getattr(base_config, param) != val
        ]

        if not eligible:
            # Fallback: random arm from full space
            arm = self._rng.choice(list(self._arms.values()))
        else:
            arm = max(eligible, key=lambda a: a.ucb1(self._total_trials, self._c))

        old_val = getattr(base_config, arm.param)
        desc = f"UCB1 explore (n={arm.n_trials}, μΔ={arm.mean_delta:+.4f})"
        return Mutation(
            param=arm.param,
            old_value=old_val,
            new_value=arm.value,
            description=desc,
        )

    def record(self, mutation: Mutation, win_rate_delta: float) -> None:
        """Update UCB1 arm statistics after a trial."""
        key = (mutation.param, mutation.new_value)
        if key not in self._arms:
            return
        arm = self._arms[key]
        arm.n_trials += 1
        arm.total_delta += win_rate_delta
        self._total_trials += 1

    def summary(self) -> list[dict]:
        """Return arms sorted by mean_delta descending."""
        rows = []
        for arm in self._arms.values():
            if arm.n_trials > 0:
                rows.append(
                    {
                        "param": arm.param,
                        "value": arm.value,
                        "n_trials": arm.n_trials,
                        "mean_delta": arm.mean_delta,
                    }
                )
        return sorted(rows, key=lambda r: r["mean_delta"], reverse=True)
