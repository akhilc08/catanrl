"""Multi-Agent PPO (MAPPO) training loop for CatanRL.

All 4 agents share a single policy network (parameter sharing).
Supports parallel environment rollouts, GAE, clipped PPO updates,
MLflow logging, and periodic checkpoint saving.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..env.action_space import ActionSpace
from ..env.catan_env import CatanEnv
from ..models.gnn_encoder import CatanGNNEncoder
from ..models.policy import CatanPolicy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MAPPOConfig:
    """Hyperparameters for MAPPO training."""

    # Environment
    num_envs: int = 8
    num_players: int = 4

    # Training
    total_timesteps: int = 10_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_steps: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Logging
    mlflow_experiment: str = "catanrl"
    log_interval: int = 10
    save_interval: int = 100
    checkpoint_dir: str = "models/checkpoints"


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """Stores rollout data collected from parallel environments.

    All tensors have shape ``(num_steps, num_envs, ...)``.
    """

    hex_features: torch.Tensor       # (T, E, 19, 9)
    vertex_features: torch.Tensor    # (T, E, 54, 7)
    edge_features: torch.Tensor      # (T, E, 72, 5)
    player_features: torch.Tensor    # (T, E, 4, 14)
    current_player: torch.Tensor     # (T, E)
    action_masks: torch.Tensor       # (T, E, 261)
    actions: torch.Tensor            # (T, E)
    log_probs: torch.Tensor          # (T, E)
    rewards: torch.Tensor            # (T, E)
    dones: torch.Tensor              # (T, E)
    values: torch.Tensor             # (T, E)
    advantages: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    returns: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))


# ---------------------------------------------------------------------------
# GAE Computation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation.

    Parameters
    ----------
    rewards : Tensor (T, E)
    values : Tensor (T, E)
    dones : Tensor (T, E)
    next_value : Tensor (E,)
    gamma, gae_lambda : float

    Returns
    -------
    advantages : Tensor (T, E)
    returns : Tensor (T, E)
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(rewards[0])

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MAPPOTrainer:
    """Multi-Agent PPO trainer with parameter sharing for CatanRL.

    Parameters
    ----------
    config : MAPPOConfig
        Training configuration.
    policy : CatanPolicy
        Shared actor-critic policy network.
    device : str
        Torch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        config: MAPPOConfig,
        policy: CatanPolicy,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.policy = policy.to(device)
        self.device = torch.device(device)
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=config.learning_rate, eps=1e-5
        )

        # Create parallel environments
        self.envs: list[CatanEnv] = [
            CatanEnv(num_players=config.num_players) for _ in range(config.num_envs)
        ]

        # Tracking
        self.global_step = 0
        self.num_updates = 0

        # Per-env episode tracking
        self.episode_rewards = np.zeros(config.num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(config.num_envs, dtype=np.int64)
        self.completed_episode_rewards: list[float] = []
        self.completed_episode_lengths: list[int] = []
        self.completed_episode_wins: list[int] = []  # player index of winner

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------

    def _reset_envs(self) -> list[dict]:
        """Reset all environments and return initial observations."""
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        self.episode_rewards[:] = 0.0
        self.episode_lengths[:] = 0
        return obs_list

    def _obs_to_tensors(
        self, obs_list: list[dict]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack a list of observation dicts into batched tensors.

        Returns
        -------
        hex_features : (num_envs, 19, 9)
        vertex_features : (num_envs, 54, 7)
        edge_features : (num_envs, 72, 5)
        player_features : (num_envs, 4, 14)
        current_player : (num_envs,)
        action_masks : (num_envs, 261)
        """
        hex_f = np.stack([o["hex_features"] for o in obs_list])
        vert_f = np.stack([o["vertex_features"] for o in obs_list])
        edge_f = np.stack([o["edge_features"] for o in obs_list])
        player_f = np.stack([o["player_features"] for o in obs_list])
        cp = np.array([o["current_player"] for o in obs_list], dtype=np.int64)
        masks = np.stack([o["action_mask"] for o in obs_list])

        device = self.device
        return (
            torch.from_numpy(hex_f).float().to(device),
            torch.from_numpy(vert_f).float().to(device),
            torch.from_numpy(edge_f).float().to(device),
            torch.from_numpy(player_f).float().to(device),
            torch.from_numpy(cp).long().to(device),
            torch.from_numpy(masks).float().to(device),
        )

    # ------------------------------------------------------------------
    # Rollout Collection
    # ------------------------------------------------------------------

    def collect_rollout(self) -> RolloutBuffer:
        """Collect ``num_steps`` of experience from ``num_envs`` parallel environments.

        Returns a filled :class:`RolloutBuffer` with computed GAE advantages.
        """
        cfg = self.config
        n_steps, n_envs = cfg.num_steps, cfg.num_envs
        device = self.device

        # Pre-allocate storage
        all_hex = torch.zeros(n_steps, n_envs, 19, 9, device=device)
        all_vert = torch.zeros(n_steps, n_envs, 54, 3 + cfg.num_players, device=device)
        all_edge = torch.zeros(n_steps, n_envs, 72, 1 + cfg.num_players, device=device)
        all_player = torch.zeros(n_steps, n_envs, cfg.num_players, 14, device=device)
        all_cp = torch.zeros(n_steps, n_envs, dtype=torch.long, device=device)
        all_masks = torch.zeros(n_steps, n_envs, ActionSpace.TOTAL_ACTIONS, device=device)
        all_actions = torch.zeros(n_steps, n_envs, dtype=torch.long, device=device)
        all_logprobs = torch.zeros(n_steps, n_envs, device=device)
        all_rewards = torch.zeros(n_steps, n_envs, device=device)
        all_dones = torch.zeros(n_steps, n_envs, device=device)
        all_values = torch.zeros(n_steps, n_envs, device=device)

        # Initialize environments if needed (first rollout)
        if not hasattr(self, "_current_obs"):
            self._current_obs = self._reset_envs()

        obs_list = self._current_obs

        self.policy.eval()
        with torch.no_grad():
            for step in range(n_steps):
                hex_f, vert_f, edge_f, player_f, cp, masks = self._obs_to_tensors(obs_list)

                # Store observations
                all_hex[step] = hex_f
                all_vert[step] = vert_f
                all_edge[step] = edge_f
                all_player[step] = player_f
                all_cp[step] = cp
                all_masks[step] = masks

                # Get actions from policy
                obs_dict = {
                    "hex_features": hex_f,
                    "vertex_features": vert_f,
                    "edge_features": edge_f,
                    "player_features": player_f,
                    "current_player": cp,
                }
                actions, log_probs, _, values = self.policy.get_action_and_value(
                    obs_dict, masks
                )

                all_actions[step] = actions
                all_logprobs[step] = log_probs
                all_values[step] = values.squeeze(-1)

                # Step environments
                actions_np = actions.cpu().numpy()
                next_obs_list = []
                for i, env in enumerate(self.envs):
                    obs, reward, terminated, truncated, info = env.step(int(actions_np[i]))
                    done = terminated or truncated

                    all_rewards[step, i] = reward
                    all_dones[step, i] = float(done)

                    self.episode_rewards[i] += reward
                    self.episode_lengths[i] += 1

                    if done:
                        # Record completed episode stats
                        self.completed_episode_rewards.append(float(self.episode_rewards[i]))
                        self.completed_episode_lengths.append(int(self.episode_lengths[i]))
                        self.completed_episode_wins.append(info.get("winner", -1))

                        # Reset environment
                        obs, _ = env.reset()
                        self.episode_rewards[i] = 0.0
                        self.episode_lengths[i] = 0

                    next_obs_list.append(obs)

                obs_list = next_obs_list
                self.global_step += n_envs

            # Compute bootstrap value for GAE
            hex_f, vert_f, edge_f, player_f, cp, _ = self._obs_to_tensors(obs_list)
            obs_dict = {
                "hex_features": hex_f,
                "vertex_features": vert_f,
                "edge_features": edge_f,
                "player_features": player_f,
                "current_player": cp,
            }
            next_value = self.policy.get_value(obs_dict).squeeze(-1)  # (E,)

        # Store current obs for next rollout
        self._current_obs = obs_list

        # Compute GAE
        advantages, returns = compute_gae(
            all_rewards, all_values, all_dones, next_value,
            cfg.gamma, cfg.gae_lambda,
        )

        return RolloutBuffer(
            hex_features=all_hex,
            vertex_features=all_vert,
            edge_features=all_edge,
            player_features=all_player,
            current_player=all_cp,
            action_masks=all_masks,
            actions=all_actions,
            log_probs=all_logprobs,
            rewards=all_rewards,
            dones=all_dones,
            values=all_values,
            advantages=advantages,
            returns=returns,
        )

    # ------------------------------------------------------------------
    # PPO Update
    # ------------------------------------------------------------------

    def update(self, rollout: RolloutBuffer) -> dict:
        """Run PPO update epochs on the collected rollout.

        Parameters
        ----------
        rollout : RolloutBuffer
            Filled buffer from :meth:`collect_rollout`.

        Returns
        -------
        dict
            Loss and diagnostic metrics.
        """
        cfg = self.config
        n_steps, n_envs = cfg.num_steps, cfg.num_envs
        batch_size = n_steps * n_envs
        minibatch_size = batch_size // cfg.num_minibatches

        # Flatten (T, E, ...) -> (T*E, ...)
        b_hex = rollout.hex_features.reshape(batch_size, 19, -1)
        b_vert = rollout.vertex_features.reshape(batch_size, 54, -1)
        b_edge = rollout.edge_features.reshape(batch_size, 72, -1)
        b_player = rollout.player_features.reshape(batch_size, cfg.num_players, -1)
        b_cp = rollout.current_player.reshape(batch_size)
        b_masks = rollout.action_masks.reshape(batch_size, -1)
        b_actions = rollout.actions.reshape(batch_size)
        b_logprobs = rollout.log_probs.reshape(batch_size)
        b_advantages = rollout.advantages.reshape(batch_size)
        b_returns = rollout.returns.reshape(batch_size)
        b_values = rollout.values.reshape(batch_size)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Accumulators for metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        num_updates = 0

        self.policy.train()

        for _epoch in range(cfg.update_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                mb_obs = {
                    "hex_features": b_hex[mb_idx],
                    "vertex_features": b_vert[mb_idx],
                    "edge_features": b_edge[mb_idx],
                    "player_features": b_player[mb_idx],
                    "current_player": b_cp[mb_idx],
                }

                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    mb_obs, b_masks[mb_idx], action=b_actions[mb_idx]
                )
                new_values = new_values.squeeze(-1)

                # Ratio
                log_ratio = new_log_probs - b_logprobs[mb_idx]
                ratio = log_ratio.exp()

                # Approx KL for diagnostics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()

                mb_advantages = b_advantages[mb_idx]

                # Clipped surrogate loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                v_loss_unclipped = (new_values - b_returns[mb_idx]) ** 2
                v_clipped = b_values[mb_idx] + torch.clamp(
                    new_values - b_values[mb_idx],
                    -cfg.clip_coef,
                    cfg.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_idx]) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                # Total loss
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_approx_kl += approx_kl.item()
                total_clip_frac += clip_frac.item()
                num_updates += 1

        # Explained variance
        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = 1.0 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 1e-8 else 0.0

        metrics = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "approx_kl": total_approx_kl / num_updates,
            "clip_fraction": total_clip_frac / num_updates,
            "explained_variance": explained_var,
        }
        return metrics

    # ------------------------------------------------------------------
    # Main Training Loop
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """Run the full MAPPO training loop. Returns final metrics dict."""
        cfg = self.config
        batch_size = cfg.num_steps * cfg.num_envs
        num_total_updates = cfg.total_timesteps // batch_size

        # Ensure checkpoint directory exists
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        # MLflow setup
        try:
            import mlflow

            mlflow.set_experiment(cfg.mlflow_experiment)
            mlflow.start_run()
            mlflow.log_params(
                {
                    "num_envs": cfg.num_envs,
                    "num_players": cfg.num_players,
                    "total_timesteps": cfg.total_timesteps,
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
            )
            use_mlflow = True
        except ImportError:
            print("[WARN] mlflow not installed; skipping MLflow logging.")
            use_mlflow = False

        start_time = time.time()
        last_metrics: dict = {}

        try:
            for update_idx in range(1, num_total_updates + 1):
                self.num_updates = update_idx

                # Collect rollout
                rollout = self.collect_rollout()

                # PPO update
                update_metrics = self.update(rollout)

                # Episode statistics
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

                # Win rate: fraction of completed episodes where there was a winner (not -1)
                recent_wins = self.completed_episode_wins[-100:]
                win_rate = (
                    float(np.mean([1.0 if w >= 0 else 0.0 for w in recent_wins]))
                    if recent_wins
                    else 0.0
                )

                all_metrics = {
                    **update_metrics,
                    "mean_reward": mean_reward,
                    "win_rate": win_rate,
                    "mean_game_length": mean_game_length,
                }
                last_metrics = all_metrics

                # Logging
                if update_idx % cfg.log_interval == 0:
                    elapsed = time.time() - start_time
                    sps = int(self.global_step / elapsed) if elapsed > 0 else 0
                    print(
                        f"Update {update_idx}/{num_total_updates} | "
                        f"Step {self.global_step} | "
                        f"SPS {sps} | "
                        f"policy_loss={update_metrics['policy_loss']:.4f} | "
                        f"value_loss={update_metrics['value_loss']:.4f} | "
                        f"entropy={update_metrics['entropy']:.4f} | "
                        f"approx_kl={update_metrics['approx_kl']:.4f} | "
                        f"mean_reward={mean_reward:.3f} | "
                        f"win_rate={win_rate:.2f} | "
                        f"mean_game_len={mean_game_length:.0f}"
                    )

                    if use_mlflow:
                        mlflow.log_metrics(all_metrics, step=self.global_step)
                        mlflow.log_metric("sps", sps, step=self.global_step)

                # Checkpoint saving
                if update_idx % cfg.save_interval == 0:
                    ckpt_path = os.path.join(
                        cfg.checkpoint_dir, f"policy_update_{update_idx}.pt"
                    )
                    torch.save(self.policy.state_dict(), ckpt_path)
                    print(f"  -> Saved checkpoint: {ckpt_path}")

        finally:
            if use_mlflow:
                mlflow.end_run()

        # Save final model
        final_path = os.path.join(cfg.checkpoint_dir, "policy_final.pt")
        torch.save(self.policy.state_dict(), final_path)
        print(f"Training complete. Final model saved to {final_path}")

        return last_metrics


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments, build policy and trainer, and run training."""
    parser = argparse.ArgumentParser(description="MAPPO training for CatanRL")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    args = parser.parse_args()

    config = MAPPOConfig(
        total_timesteps=args.total_timesteps,
        learning_rate=args.lr,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )

    # Build encoder and policy
    encoder = CatanGNNEncoder.from_env_defaults(
        num_players=config.num_players,
    )
    policy = CatanPolicy(
        gnn_encoder=encoder,
        action_dim=ActionSpace.TOTAL_ACTIONS,
    )

    param_count = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {param_count:,}")
    print(f"Device: {args.device}")
    print(f"Batch size per update: {config.num_steps * config.num_envs}")
    print(f"Total updates: {config.total_timesteps // (config.num_steps * config.num_envs)}")

    trainer = MAPPOTrainer(config=config, policy=policy, device=args.device)
    final_metrics = trainer.train()
    print(f"Final metrics: {final_metrics}")


if __name__ == "__main__":
    main()
