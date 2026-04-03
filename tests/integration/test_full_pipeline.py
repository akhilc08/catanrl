"""Integration tests for the full CatanRL pipeline: env + model."""

from __future__ import annotations

import numpy as np
import pytest
from src.rl.env.catan_env import CatanEnv


class TestEnvStepLoop:
    """Test stepping through the environment with random valid actions."""

    def test_reset_step_observe(self):
        """Create env, reset, take a few random actions, verify state changes."""
        env = CatanEnv(num_players=4)
        obs, info = env.reset(seed=42)

        # Initial state checks
        assert obs["game_phase"] == CatanEnv.PHASE_SETUP_FIRST
        assert obs["current_player"] == 0

        initial_mask = obs["action_mask"]
        initial_valid_count = initial_mask.sum()
        assert initial_valid_count > 0

        # Take 10 random valid steps
        np.random.seed(42)
        for i in range(10):
            mask = obs["action_mask"] if isinstance(obs, dict) else env.get_action_mask()
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            action = int(np.random.choice(valid))
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(obs, dict)
            assert isinstance(reward, float)
            if terminated or truncated:
                break

    def test_state_changes_after_actions(self):
        """Verify that taking actions actually changes the game state."""
        env = CatanEnv(num_players=4)
        obs1, _ = env.reset(seed=42)

        mask = obs1["action_mask"]
        valid = np.where(mask)[0]
        action = int(valid[0])

        obs2, _, _, _, _ = env.step(action)

        # Something should have changed: either board state or game phase
        changed = (
            not np.array_equal(obs1["hex_features"], obs2["hex_features"])
            or not np.array_equal(obs1["vertex_features"], obs2["vertex_features"])
            or not np.array_equal(obs1["edge_features"], obs2["edge_features"])
            or obs1["game_phase"] != obs2["game_phase"]
            or not np.array_equal(obs1["action_mask"], obs2["action_mask"])
        )
        assert changed, "State should change after taking an action"


class TestFullGameLoop:

    def test_random_game_until_termination(self):
        """Play a full game with random valid actions until termination."""
        env = CatanEnv(num_players=4)
        env.reset(seed=42)
        np.random.seed(42)

        steps = 0
        max_steps = 5000
        terminated = False
        truncated = False

        while steps < max_steps:
            mask = env.get_action_mask()
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            action = int(np.random.choice(valid))
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break

        assert terminated or truncated, f"Game did not end in {max_steps} steps"
        assert steps > 0

    def test_game_phases_progress(self):
        """Verify the game goes through setup into normal play."""
        env = CatanEnv(num_players=4)
        env.reset(seed=42)
        np.random.seed(42)

        phases_seen = set()
        phases_seen.add(env.game_phase)

        for _ in range(500):
            mask = env.get_action_mask()
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            action = int(np.random.choice(valid))
            _, _, terminated, truncated, _ = env.step(action)
            phases_seen.add(env.game_phase)
            if terminated or truncated:
                break

        # Should have seen at least setup and roll phases
        assert CatanEnv.PHASE_SETUP_FIRST in phases_seen
        assert CatanEnv.PHASE_ROLL in phases_seen or CatanEnv.PHASE_SETUP_SECOND in phases_seen


class TestEnvToGNNPipeline:
    """Test feeding environment observations to the GNN encoder."""

    def test_env_obs_to_encoder(self):
        """Environment observation can be fed to CatanGNNEncoder."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from src.rl.models.gnn_encoder import CatanGNNEncoder

        env = CatanEnv(num_players=4)
        obs, _ = env.reset(seed=42)

        encoder = CatanGNNEncoder.from_env_defaults(
            hidden_dim=32, num_heads=2, num_layers=1, output_dim=64
        )

        # Convert observation arrays to tensors with batch dimension
        hex_feat = torch.from_numpy(obs["hex_features"]).float().unsqueeze(0)
        vert_feat = torch.from_numpy(obs["vertex_features"]).float().unsqueeze(0)
        edge_feat = torch.from_numpy(obs["edge_features"]).float().unsqueeze(0)
        player_feat = torch.from_numpy(obs["player_features"]).float().unsqueeze(0)
        current_player = torch.tensor([obs["current_player"]], dtype=torch.long)

        with torch.no_grad():
            embedding = encoder(hex_feat, vert_feat, edge_feat, player_feat, current_player)

        assert embedding.shape == (1, 64)
        assert torch.isfinite(embedding).all()

    def test_env_obs_to_policy(self):
        """Environment observation can be fed through the full policy."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from src.rl.models.gnn_encoder import CatanGNNEncoder
        from src.rl.models.policy import CatanPolicy

        env = CatanEnv(num_players=4)
        obs, _ = env.reset(seed=42)

        encoder = CatanGNNEncoder.from_env_defaults(
            hidden_dim=32, num_heads=2, num_layers=1, output_dim=64
        )
        policy = CatanPolicy(gnn_encoder=encoder, action_dim=261, hidden_dim=64)

        # The env returns current_player as a plain int; wrap it for the policy
        obs["current_player"] = np.array(obs["current_player"], dtype=np.int64)
        mask = obs["action_mask"].astype(np.float32)

        with torch.no_grad():
            action, log_prob, entropy, value = policy.get_action_and_value(obs, mask)

        assert 0 <= action.item() < 261
        assert torch.isfinite(log_prob).all()
        assert torch.isfinite(value).all()

    def test_multiple_steps_through_policy(self):
        """Step through a few env steps using the policy to choose actions."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from src.rl.models.gnn_encoder import CatanGNNEncoder
        from src.rl.models.policy import CatanPolicy

        env = CatanEnv(num_players=4)
        obs, _ = env.reset(seed=42)

        encoder = CatanGNNEncoder.from_env_defaults(
            hidden_dim=32, num_heads=2, num_layers=1, output_dim=64
        )
        policy = CatanPolicy(gnn_encoder=encoder, action_dim=261, hidden_dim=64)

        for _ in range(20):
            obs["current_player"] = np.array(obs["current_player"], dtype=np.int64)
            mask = obs["action_mask"].astype(np.float32)
            with torch.no_grad():
                action, _, _, _ = policy.get_action_and_value(obs, mask)

            obs, reward, terminated, truncated, info = env.step(action.item())
            if terminated or truncated:
                break
