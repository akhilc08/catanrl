"""Tests for the CatanPolicy actor-critic."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.rl.models.gnn_encoder import CatanGNNEncoder  # noqa: E402
from src.rl.models.policy import CatanPolicy  # noqa: E402
from torch.distributions import Categorical  # noqa: E402


@pytest.fixture
def policy() -> CatanPolicy:
    """Create a small policy for testing."""
    encoder = CatanGNNEncoder.from_env_defaults(
        hidden_dim=32, num_heads=2, num_layers=1, output_dim=64
    )
    return CatanPolicy(gnn_encoder=encoder, action_dim=261, hidden_dim=64)


def _make_batch_inputs(batch_size: int = 1):
    """Generate random batched tensor inputs."""
    hex_feat = torch.randn(batch_size, 19, 9)
    vert_feat = torch.randn(batch_size, 54, 7)
    edge_feat = torch.randn(batch_size, 72, 5)
    player_feat = torch.randn(batch_size, 4, 14)
    current_player = torch.zeros(batch_size, dtype=torch.long)
    # Action mask: enable about half the actions
    action_mask = torch.ones(batch_size, 261, dtype=torch.bool)
    return hex_feat, vert_feat, edge_feat, player_feat, current_player, action_mask


def _make_obs_dict():
    """Generate a numpy observation dict like CatanEnv produces."""
    return {
        "hex_features": np.random.randn(19, 9).astype(np.float32),
        "vertex_features": np.random.randn(54, 7).astype(np.float32),
        "edge_features": np.random.randn(72, 5).astype(np.float32),
        "player_features": np.random.randn(4, 14).astype(np.float32),
        "current_player": np.array(0, dtype=np.int64),
    }


class TestPolicyCreation:

    def test_creates_without_error(self, policy: CatanPolicy):
        assert policy is not None
        assert policy.action_dim == 261

    def test_has_actor_and_critic(self, policy: CatanPolicy):
        assert hasattr(policy, "actor")
        assert hasattr(policy, "critic")
        assert hasattr(policy, "gnn_encoder")


class TestForward:

    def test_returns_categorical_and_value(self, policy: CatanPolicy):
        inputs = _make_batch_inputs(batch_size=1)
        with torch.no_grad():
            dist, value = policy(*inputs)
        assert isinstance(dist, Categorical)
        assert value.shape == (1, 1)

    def test_batch_forward(self, policy: CatanPolicy):
        inputs = _make_batch_inputs(batch_size=4)
        with torch.no_grad():
            dist, value = policy(*inputs)
        assert isinstance(dist, Categorical)
        assert value.shape == (4, 1)
        # Distribution should be over 261 actions
        assert dist.probs.shape == (4, 261)


class TestActionMasking:

    def test_masked_actions_have_zero_probability(self, policy: CatanPolicy):
        """Actions that are masked out should have ~0 probability."""
        hex_feat, vert_feat, edge_feat, player_feat, cp, _ = _make_batch_inputs(1)
        # Mask: only allow actions 0..9
        mask = torch.zeros(1, 261, dtype=torch.bool)
        mask[0, :10] = True

        with torch.no_grad():
            dist, _ = policy(hex_feat, vert_feat, edge_feat, player_feat, cp, mask)

        probs = dist.probs[0]
        # Masked actions should have negligible probability
        assert probs[10:].sum().item() < 1e-6
        # Unmasked actions should have positive probability
        assert probs[:10].sum().item() > 0.99


class TestGetActionAndValue:

    def test_returns_correct_tuple(self, policy: CatanPolicy):
        obs = _make_obs_dict()
        mask = np.ones(261, dtype=np.float32)

        with torch.no_grad():
            action, log_prob, entropy, value = policy.get_action_and_value(obs, mask)

        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert entropy.shape == (1,)
        assert value.shape == (1, 1)

    def test_action_in_valid_range(self, policy: CatanPolicy):
        obs = _make_obs_dict()
        mask = np.ones(261, dtype=np.float32)

        with torch.no_grad():
            action, _, _, _ = policy.get_action_and_value(obs, mask)

        assert 0 <= action.item() < 261

    def test_evaluate_given_action(self, policy: CatanPolicy):
        obs = _make_obs_dict()
        mask = np.ones(261, dtype=np.float32)
        given_action = torch.tensor([5])

        with torch.no_grad():
            action, log_prob, entropy, value = policy.get_action_and_value(
                obs, mask, action=given_action
            )

        assert action.item() == 5
        assert log_prob.shape == (1,)


class TestGetValue:

    def test_returns_scalar_value(self, policy: CatanPolicy):
        obs = _make_obs_dict()
        with torch.no_grad():
            value = policy.get_value(obs)
        assert value.shape == (1, 1)
        assert torch.isfinite(value).all()
