"""Self-play training with opponent pool for CatanRL.

Maintains a pool of past policy checkpoints and periodically evaluates the
current policy against sampled opponents.  Strong checkpoints are added to
the pool; when the pool is full the weakest opponent is evicted.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from ..env.catan_env import CatanEnv
from ..models.gnn_encoder import CatanGNNEncoder
from ..models.policy import CatanPolicy
from .mappo import MAPPOConfig, MAPPOTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SelfPlayConfig:
    """Configuration for self-play opponent pool management.

    Parameters
    ----------
    pool_size : int
        Maximum number of opponent checkpoints kept in memory.
    save_interval : int
        Evaluate and potentially add to pool every *save_interval* MAPPO
        updates.
    sample_strategy : str
        How to pick an opponent from the pool:
        ``"uniform"`` — equal probability,
        ``"latest"`` — always the most recent checkpoint,
        ``"prioritized"`` — weight by inverse win-rate (harder opponents
        sampled more often).
    win_rate_threshold : float
        Current policy must achieve at least this win rate against the pool
        before its checkpoint is eligible for addition.
    checkpoint_dir : str
        Directory where opponent state dicts are persisted.
    """

    pool_size: int = 10
    save_interval: int = 50
    sample_strategy: str = "uniform"
    win_rate_threshold: float = 0.55
    checkpoint_dir: str = "models/opponents"


# ---------------------------------------------------------------------------
# Opponent entry
# ---------------------------------------------------------------------------

@dataclass
class OpponentEntry:
    """A single opponent checkpoint stored in the pool.

    Attributes
    ----------
    state_dict : dict
        ``policy.state_dict()`` snapshot.
    metadata : dict
        Arbitrary metadata — typically ``training_step``, ``win_rate``,
        ``timestamp``.
    games_played : int
        Number of evaluation games this opponent has participated in (as the
        opponent).
    games_won : int
        Number of games this opponent won when sampled as opponent.
    """

    state_dict: dict
    metadata: dict
    games_played: int = 0
    games_won: int = 0

    @property
    def opponent_win_rate(self) -> float:
        """Win rate of this checkpoint *when used as the opponent*."""
        if self.games_played == 0:
            return 0.5  # prior — assume average until proven otherwise
        return self.games_won / self.games_played


# ---------------------------------------------------------------------------
# Opponent Pool
# ---------------------------------------------------------------------------

class OpponentPool:
    """Manages a fixed-size pool of past policy checkpoints.

    Parameters
    ----------
    config : SelfPlayConfig
        Pool configuration.
    policy_factory : callable, optional
        A zero-argument callable that returns a fresh :class:`CatanPolicy`
        instance.  Required for :meth:`get_opponent_policy`.  When ``None``,
        calling that method raises ``RuntimeError``.
    """

    def __init__(
        self,
        config: SelfPlayConfig,
        policy_factory: Callable[[], CatanPolicy] | None = None,
    ) -> None:
        self.config = config
        self.pool: list[OpponentEntry] = []
        self._policy_factory = policy_factory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.pool)

    @property
    def is_empty(self) -> bool:
        return len(self.pool) == 0

    def add_checkpoint(self, policy_state_dict: dict, metadata: dict) -> None:
        """Add a new opponent to the pool.

        If the pool has reached ``pool_size``, the weakest entry (highest
        opponent win rate, i.e. the one *least useful* as a training opponent
        because the current policy already beats it easily) is evicted.

        Parameters
        ----------
        policy_state_dict : dict
            ``policy.state_dict()`` to store.
        metadata : dict
            Arbitrary metadata (``training_step``, ``win_rate``, etc.).
        """
        # Deep-copy state dict so mutations to the live policy don't affect us
        entry = OpponentEntry(
            state_dict=copy.deepcopy(policy_state_dict),
            metadata=dict(metadata),
        )

        if len(self.pool) >= self.config.pool_size:
            self._evict_weakest()

        self.pool.append(entry)

        # Persist to disk
        idx = len(self.pool) - 1
        step = metadata.get("training_step", idx)
        path = os.path.join(self.config.checkpoint_dir, f"opponent_{step}.pt")
        torch.save(policy_state_dict, path)
        logger.info(
            "Added opponent #%d to pool (step=%s). Pool size: %d/%d",
            idx, step, len(self.pool), self.config.pool_size,
        )

    def _evict_weakest(self) -> None:
        """Remove the weakest opponent from the pool.

        "Weakest" = highest opponent_win_rate.  An opponent that wins a lot
        against the current policy is *hard* and should be kept; an opponent
        that rarely wins is "weak" and can be replaced.
        """
        if not self.pool:
            return
        # Find entry with highest opponent_win_rate (wins the most as opponent
        # -> actually the *strongest* opponent).  We want to remove the one
        # that loses the most as opponent = lowest opponent_win_rate.
        weakest_idx = min(range(len(self.pool)), key=lambda i: self.pool[i].opponent_win_rate)
        removed = self.pool.pop(weakest_idx)
        logger.info(
            "Evicted opponent '%s' (opponent_win_rate=%.2f) from pool.",
            removed.metadata.get("training_step", "?"),
            removed.opponent_win_rate,
        )

    def sample_opponent(self) -> dict:
        """Sample an opponent ``state_dict`` from the pool.

        Returns
        -------
        dict
            A policy ``state_dict``.

        Raises
        ------
        RuntimeError
            If the pool is empty.
        """
        if self.is_empty:
            raise RuntimeError("Cannot sample from an empty opponent pool.")

        strategy = self.config.sample_strategy

        if strategy == "latest":
            entry = self.pool[-1]
        elif strategy == "prioritized":
            # Weight by inverse opponent win rate — opponents that beat us
            # more often are sampled more (harder practice).
            weights = np.array([e.opponent_win_rate for e in self.pool], dtype=np.float64)
            weights = weights + 1e-8  # avoid zero
            weights /= weights.sum()
            idx = int(np.random.choice(len(self.pool), p=weights))
            entry = self.pool[idx]
        else:  # uniform
            idx = int(np.random.randint(len(self.pool)))
            entry = self.pool[idx]

        return entry.state_dict

    def get_opponent_policy(self, device: str = "cpu") -> CatanPolicy:
        """Return a ready-to-use opponent policy loaded from the pool.

        Parameters
        ----------
        device : str
            Torch device string.

        Returns
        -------
        CatanPolicy
            A policy in eval mode, loaded with a sampled checkpoint.
        """
        if self._policy_factory is None:
            raise RuntimeError(
                "OpponentPool.get_opponent_policy requires a policy_factory."
            )
        state_dict = self.sample_opponent()
        policy = self._policy_factory()
        policy.load_state_dict(state_dict)
        policy = policy.to(device)
        policy.eval()
        return policy

    def evaluate_against_pool(
        self,
        policy: CatanPolicy,
        num_games: int = 50,
        num_players: int = 4,
        device: str = "cpu",
    ) -> float:
        """Play *num_games* against randomly sampled pool opponents.

        Parameters
        ----------
        policy : CatanPolicy
            The current training policy (used as player 0).
        num_games : int
            Number of evaluation games to play.
        num_players : int
            Number of players per game.
        device : str
            Torch device.

        Returns
        -------
        float
            Win rate of *policy* (player 0) across the games.
        """
        if self.is_empty:
            return 0.0

        wins = 0
        policy.eval()

        for _ in range(num_games):
            # Pick an opponent for this game
            opp_state_dict = self.sample_opponent()
            opp_entry = self._find_entry(opp_state_dict)

            assert self._policy_factory is not None
            opp_policy = self._policy_factory()
            opp_policy.load_state_dict(opp_state_dict)
            opp_policy = opp_policy.to(device)
            opp_policy.eval()

            result = _play_one_game(
                player0_policy=policy,
                opponent_policy=opp_policy,
                num_players=num_players,
                device=device,
            )

            if result["winner"] == 0:
                wins += 1

            # Update entry stats
            if opp_entry is not None:
                opp_entry.games_played += 1
                if result["winner"] != 0 and result["winner"] >= 0:
                    opp_entry.games_won += 1

        return wins / max(num_games, 1)

    def _find_entry(self, state_dict: dict) -> OpponentEntry | None:
        """Find the pool entry whose state_dict is *state_dict* (by identity)."""
        for entry in self.pool:
            if entry.state_dict is state_dict:
                return entry
        return None

    def get_metrics(self) -> dict:
        """Return loggable metrics about the pool."""
        if not self.pool:
            return {
                "self_play/pool_size": 0,
                "self_play/mean_pool_opponent_win_rate": 0.0,
            }
        rates = [e.opponent_win_rate for e in self.pool]
        return {
            "self_play/pool_size": len(self.pool),
            "self_play/mean_pool_opponent_win_rate": float(np.mean(rates)),
        }


# ---------------------------------------------------------------------------
# Self-Play Trainer
# ---------------------------------------------------------------------------

class SelfPlayTrainer:
    """Wraps :class:`MAPPOTrainer` with self-play opponent management.

    Parameters
    ----------
    config : SelfPlayConfig
        Self-play / opponent pool settings.
    mappo_config : MAPPOConfig
        MAPPO training hyper-parameters.
    policy : CatanPolicy
        The shared actor-critic policy to train.
    device : str
        Torch device.
    """

    def __init__(
        self,
        config: SelfPlayConfig,
        mappo_config: MAPPOConfig,
        policy: CatanPolicy,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.mappo_config = mappo_config
        self.policy = policy
        self.device = device

        # Build MAPPO trainer
        self.mappo_trainer = MAPPOTrainer(
            config=mappo_config, policy=policy, device=device,
        )

        # Build a factory that can create fresh policy instances (same arch)
        self._policy_factory = self._make_policy_factory(policy)

        # Build opponent pool
        self.opponent_pool = OpponentPool(
            config=config,
            policy_factory=self._policy_factory,
        )

    @staticmethod
    def _make_policy_factory(reference_policy: CatanPolicy) -> Callable[[], CatanPolicy]:
        """Create a zero-argument callable that returns a new CatanPolicy
        with the same architecture as *reference_policy*."""
        # Capture architecture parameters from the reference
        gnn = reference_policy.gnn_encoder
        action_dim = reference_policy.action_dim
        hidden_dim = reference_policy.hidden_dim

        # Extract GNN constructor params from the encoder's attributes
        hex_proj = gnn.input_projs["hex"]
        vertex_proj = gnn.input_projs["vertex"]
        edge_proj = gnn.input_projs["edge"]
        assert isinstance(hex_proj, torch.nn.Linear)
        assert isinstance(vertex_proj, torch.nn.Linear)
        assert isinstance(edge_proj, torch.nn.Linear)
        gnn_kwargs: dict[str, int] = {
            "hex_in_features": hex_proj.in_features,
            "vertex_in_features": vertex_proj.in_features,
            "edge_in_features": edge_proj.in_features,
            "player_in_features": gnn.player_in_features,
            "hidden_dim": gnn.hidden_dim,
            "num_heads": gnn.num_heads,
            "num_layers": gnn.num_layers,
            "output_dim": gnn.output_dim,
        }

        def _factory() -> CatanPolicy:
            enc = CatanGNNEncoder(**gnn_kwargs)
            return CatanPolicy(gnn_encoder=enc, action_dim=action_dim, hidden_dim=hidden_dim)

        return _factory

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, total_updates: int) -> dict:
        """Run the MAPPO training loop with periodic self-play evaluation.

        Parameters
        ----------
        total_updates : int
            Total number of MAPPO update steps to perform.

        Returns
        -------
        dict
            Final aggregated metrics.
        """
        cfg = self.config
        trainer = self.mappo_trainer

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        # MLflow setup
        try:
            import mlflow
            use_mlflow = True
        except ImportError:
            use_mlflow = False

        last_metrics: dict = {}
        start_time = time.time()

        for update_idx in range(1, total_updates + 1):
            trainer.num_updates = update_idx

            # Collect rollout and do PPO update
            rollout = trainer.collect_rollout()
            update_metrics = trainer.update(rollout)

            # Episode stats from the trainer
            mean_reward = (
                float(np.mean(trainer.completed_episode_rewards[-100:]))
                if trainer.completed_episode_rewards
                else 0.0
            )
            mean_game_length = (
                float(np.mean(trainer.completed_episode_lengths[-100:]))
                if trainer.completed_episode_lengths
                else 0.0
            )

            all_metrics = {
                **update_metrics,
                "mean_reward": mean_reward,
                "mean_game_length": mean_game_length,
            }

            # --- Self-play checkpoint evaluation ---
            if update_idx % cfg.save_interval == 0:
                eval_results = self.play_evaluation_games(num_games=50)
                eval_win_rate = eval_results["win_rate"]

                all_metrics["self_play/evaluation_win_rate"] = eval_win_rate
                all_metrics.update(self.opponent_pool.get_metrics())

                logger.info(
                    "Update %d — eval win rate vs pool: %.2f (threshold %.2f)",
                    update_idx, eval_win_rate, cfg.win_rate_threshold,
                )

                # Add to pool if strong enough (or pool is empty — seed it)
                if eval_win_rate >= cfg.win_rate_threshold or self.opponent_pool.is_empty:
                    self.opponent_pool.add_checkpoint(
                        policy_state_dict=self.policy.state_dict(),
                        metadata={
                            "training_step": trainer.global_step,
                            "win_rate": eval_win_rate,
                            "timestamp": time.time(),
                        },
                    )

                # Print summary
                elapsed = time.time() - start_time
                print(
                    f"[SelfPlay] Update {update_idx}/{total_updates} | "
                    f"elapsed={elapsed:.0f}s | "
                    f"eval_win_rate={eval_win_rate:.2f} | "
                    f"pool_size={self.opponent_pool.size}/{cfg.pool_size}"
                )

            # Periodic MAPPO logging
            if update_idx % self.mappo_config.log_interval == 0:
                elapsed = time.time() - start_time
                sps = int(trainer.global_step / elapsed) if elapsed > 0 else 0
                print(
                    f"Update {update_idx}/{total_updates} | "
                    f"Step {trainer.global_step} | "
                    f"SPS {sps} | "
                    f"policy_loss={update_metrics['policy_loss']:.4f} | "
                    f"value_loss={update_metrics['value_loss']:.4f} | "
                    f"entropy={update_metrics['entropy']:.4f} | "
                    f"mean_reward={mean_reward:.3f}"
                )

                if use_mlflow:
                    # Filter out non-numeric metrics for mlflow
                    numeric_metrics = {
                        k: v for k, v in all_metrics.items()
                        if isinstance(v, (int, float))
                    }
                    mlflow.log_metrics(numeric_metrics, step=trainer.global_step)

            last_metrics = all_metrics

        # Save final model
        final_path = os.path.join(cfg.checkpoint_dir, "policy_final.pt")
        torch.save(self.policy.state_dict(), final_path)
        print(f"Self-play training complete. Final model: {final_path}")

        return last_metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def play_evaluation_games(
        self,
        num_games: int = 50,
        num_players: int = 4,
    ) -> dict:
        """Evaluate the current policy against pool opponents.

        If the pool is empty, plays against a random-action baseline.

        Parameters
        ----------
        num_games : int
            Number of games to play.
        num_players : int
            Players per game.

        Returns
        -------
        dict
            ``win_rate``, ``mean_vp``, ``mean_game_length``.
        """
        device = self.device
        self.policy.eval()

        wins = 0
        total_vp = 0.0
        total_length = 0

        for _ in range(num_games):
            # Build opponent policy (or None for random play)
            if not self.opponent_pool.is_empty:
                opp_state_dict = self.opponent_pool.sample_opponent()
                opp_entry = self.opponent_pool._find_entry(opp_state_dict)

                assert self._policy_factory is not None
                opp_policy = self._policy_factory()
                opp_policy.load_state_dict(opp_state_dict)
                opp_policy = opp_policy.to(device)
                opp_policy.eval()
            else:
                opp_policy = None
                opp_entry = None

            result = _play_one_game(
                player0_policy=self.policy,
                opponent_policy=opp_policy,
                num_players=num_players,
                device=device,
            )

            if result["winner"] == 0:
                wins += 1
            total_vp += result["player0_vp"]
            total_length += result["game_length"]

            # Update opponent entry stats
            if opp_entry is not None:
                opp_entry.games_played += 1
                if result["winner"] != 0 and result["winner"] >= 0:
                    opp_entry.games_won += 1

        n = max(num_games, 1)
        return {
            "win_rate": wins / n,
            "mean_vp": total_vp / n,
            "mean_game_length": total_length / n,
        }


# ---------------------------------------------------------------------------
# Game simulation helper
# ---------------------------------------------------------------------------

def _select_action(
    policy: CatanPolicy | None,
    obs: dict,
    device: str,
) -> int:
    """Select an action using *policy*, or uniformly random if policy is None.

    Parameters
    ----------
    policy : CatanPolicy or None
        If ``None``, a random legal action is chosen.
    obs : dict
        Observation dictionary from :class:`CatanEnv`.
    device : str
        Torch device string.

    Returns
    -------
    int
        Selected action index.
    """
    mask = obs["action_mask"]

    if policy is None:
        # Random legal action
        legal = np.where(mask)[0]
        if len(legal) == 0:
            return 0  # fallback — should not happen in a well-formed env
        return int(np.random.choice(legal))

    with torch.no_grad():
        obs_dict = {
            "hex_features": obs["hex_features"],
            "vertex_features": obs["vertex_features"],
            "edge_features": obs["edge_features"],
            "player_features": obs["player_features"],
            "current_player": obs["current_player"],
        }
        action_mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(device)
        action, _, _, _ = policy.get_action_and_value(obs_dict, action_mask_t)
        return int(action.item())


def _play_one_game(
    player0_policy: CatanPolicy,
    opponent_policy: CatanPolicy | None,
    num_players: int = 4,
    device: str = "cpu",
    max_steps: int = 2000,
) -> dict:
    """Play a single game with player 0 using *player0_policy* and all other
    players using *opponent_policy* (or random if ``None``).

    Parameters
    ----------
    player0_policy : CatanPolicy
        Policy for the training agent (player 0).
    opponent_policy : CatanPolicy or None
        Policy for opponents (players 1..N-1).  If ``None``, opponents
        take uniformly random legal actions.
    num_players : int
        Number of players.
    device : str
        Torch device.
    max_steps : int
        Safety limit on environment steps.

    Returns
    -------
    dict
        ``winner`` (int, -1 if no winner), ``player0_vp`` (int),
        ``game_length`` (int).
    """
    env = CatanEnv(num_players=num_players)
    obs, info = env.reset()

    for _ in range(max_steps):
        current_player = obs["current_player"]

        if current_player == 0:
            action = _select_action(player0_policy, obs, device)
        else:
            action = _select_action(opponent_policy, obs, device)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    winner = info.get("winner", -1)
    vp_array = info.get("vp", np.zeros(num_players))
    player0_vp = int(vp_array[0]) if len(vp_array) > 0 else 0
    game_length = info.get("turn", 0)

    return {
        "winner": winner,
        "player0_vp": player0_vp,
        "game_length": game_length,
    }
