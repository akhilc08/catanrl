"""Time-bounded trial evaluator for autonomous research.

Runs a MAPPO training trial until the wall-clock budget is exhausted, then
returns a TrialResult with the final win_rate and mean_reward.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..rl.env.action_space import ActionSpace
from ..rl.models.gnn_encoder import CatanGNNEncoder
from ..rl.models.policy import CatanPolicy
from ..rl.training.mappo import MAPPOConfig, MAPPOTrainer


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    """Outcome of a single timed experiment."""

    win_rate: float
    mean_reward: float
    mean_game_length: float
    elapsed_seconds: float
    num_updates: int
    global_step: int
    config_snapshot: dict[str, Any]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Time-bounded trainer
# ---------------------------------------------------------------------------

class _TimeBoundedTrainer(MAPPOTrainer):
    """MAPPO trainer that stops after ``time_budget_seconds`` of wall time."""

    def __init__(
        self,
        config: MAPPOConfig,
        policy: CatanPolicy,
        device: str,
        time_budget_seconds: float,
    ) -> None:
        super().__init__(config, policy, device)
        self._budget = time_budget_seconds

    def train(self) -> dict:
        cfg = self.config
        batch_size = cfg.num_steps * cfg.num_envs
        # Upper bound: more updates than we could ever finish in the budget
        max_updates = int(self._budget * 1000)

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        start_time = time.monotonic()
        last_metrics: dict = {}

        for update_idx in range(1, max_updates + 1):
            self.num_updates = update_idx

            rollout = self.collect_rollout()
            update_metrics = self.update(rollout)

            mean_reward = (
                float(np.mean(self.completed_episode_rewards[-100:]))
                if self.completed_episode_rewards
                else 0.0
            )
            mean_game_length = (
                float(np.mean(self.completed_episode_lengths[-100:]))
                if self.completed_episode_lengths
                else 0.0
            )
            recent_wins = self.completed_episode_wins[-100:]
            win_rate = (
                float(np.mean([1.0 if w >= 0 else 0.0 for w in recent_wins]))
                if recent_wins
                else 0.0
            )

            last_metrics = {
                **update_metrics,
                "mean_reward": mean_reward,
                "win_rate": win_rate,
                "mean_game_length": mean_game_length,
            }

            elapsed = time.monotonic() - start_time
            if update_idx % cfg.log_interval == 0:
                sps = int(self.global_step / elapsed) if elapsed > 0 else 0
                print(
                    f"  [trial] update={update_idx}  "
                    f"step={self.global_step}  sps={sps}  "
                    f"win_rate={win_rate:.3f}  "
                    f"reward={mean_reward:.3f}  "
                    f"elapsed={elapsed:.0f}s/{self._budget:.0f}s"
                )

            if elapsed >= self._budget:
                print(f"  [trial] time budget reached ({elapsed:.1f}s)")
                break

        return last_metrics


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Runs timed MAPPO trials and returns TrialResult.

    Parameters
    ----------
    device : str
        Torch device string.
    time_budget_seconds : float
        Wall-clock seconds per trial.
    num_envs : int
        Override num_envs for quick trials (default: use config value).
    checkpoint_dir : str
        Where to write trial checkpoints.
    """

    def __init__(
        self,
        device: str = "cpu",
        time_budget_seconds: float = 300.0,
        num_envs: int | None = None,
        checkpoint_dir: str = "models/autoresearch_trials",
    ) -> None:
        self.device = device
        self.time_budget_seconds = time_budget_seconds
        self._num_envs_override = num_envs
        self.checkpoint_dir = checkpoint_dir

    def run_trial(self, config: MAPPOConfig) -> TrialResult:
        """Train from scratch with *config* for the time budget.

        A fresh policy is created for each trial so results are comparable.
        """
        trial_cfg = _patch_config(config, self._num_envs_override, self.checkpoint_dir)
        config_snapshot = _config_to_dict(trial_cfg)

        t0 = time.monotonic()
        try:
            encoder = CatanGNNEncoder.from_env_defaults(
                num_players=trial_cfg.num_players,
            )
            policy = CatanPolicy(
                gnn_encoder=encoder,
                action_dim=ActionSpace.TOTAL_ACTIONS,
            )

            trainer = _TimeBoundedTrainer(
                config=trial_cfg,
                policy=policy,
                device=self.device,
                time_budget_seconds=self.time_budget_seconds,
            )
            metrics = trainer.train()

            elapsed = time.monotonic() - t0
            return TrialResult(
                win_rate=metrics.get("win_rate", 0.0),
                mean_reward=metrics.get("mean_reward", 0.0),
                mean_game_length=metrics.get("mean_game_length", 0.0),
                elapsed_seconds=elapsed,
                num_updates=trainer.num_updates,
                global_step=trainer.global_step,
                config_snapshot=config_snapshot,
            )

        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            return TrialResult(
                win_rate=0.0,
                mean_reward=0.0,
                mean_game_length=0.0,
                elapsed_seconds=elapsed,
                num_updates=0,
                global_step=0,
                config_snapshot=config_snapshot,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_config(
    base: MAPPOConfig,
    num_envs_override: int | None,
    checkpoint_dir: str,
) -> MAPPOConfig:
    import copy
    cfg = copy.deepcopy(base)
    # Trials don't log to MLflow by default
    cfg.mlflow_experiment = ""
    cfg.checkpoint_dir = checkpoint_dir
    cfg.save_interval = 999_999  # no intermediate saves during trial
    if num_envs_override is not None:
        cfg.num_envs = num_envs_override
    # Allow the time-bounded loop to run "forever" (it stops on its own)
    cfg.total_timesteps = 10_000_000_000
    return cfg


def _config_to_dict(cfg: MAPPOConfig) -> dict[str, Any]:
    return {
        "num_envs": cfg.num_envs,
        "learning_rate": cfg.learning_rate,
        "gamma": cfg.gamma,
        "gae_lambda": cfg.gae_lambda,
        "num_steps": cfg.num_steps,
        "num_minibatches": cfg.num_minibatches,
        "update_epochs": cfg.update_epochs,
        "clip_coef": cfg.clip_coef,
        "ent_coef": cfg.ent_coef,
        "vf_coef": cfg.vf_coef,
        "max_grad_norm": cfg.max_grad_norm,
    }
