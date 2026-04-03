"""Tests for the CatanGNNEncoder."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.rl.models.gnn_encoder import CatanGNNEncoder  # noqa: E402


@pytest.fixture
def encoder() -> CatanGNNEncoder:
    """Create an encoder with default env dimensions."""
    return CatanGNNEncoder.from_env_defaults(
        hidden_dim=32, num_heads=2, num_layers=2, output_dim=64
    )


def _make_random_inputs(batch_size: int = 1):
    """Generate random input tensors matching CatanEnv observation shapes."""
    hex_features = torch.randn(batch_size, 19, 9)
    vertex_features = torch.randn(batch_size, 54, 7)
    edge_features = torch.randn(batch_size, 72, 5)
    player_features = torch.randn(batch_size, 4, 14)
    current_player = torch.zeros(batch_size, dtype=torch.long)
    return hex_features, vertex_features, edge_features, player_features, current_player


class TestEncoderCreation:

    def test_creates_without_error(self):
        encoder = CatanGNNEncoder.from_env_defaults()
        assert encoder is not None

    def test_from_env_defaults_dimensions(self):
        encoder = CatanGNNEncoder.from_env_defaults(
            hidden_dim=64, num_heads=4, num_layers=3, output_dim=128
        )
        assert encoder.hidden_dim == 64
        assert encoder.num_heads == 4
        assert encoder.num_layers == 3
        assert encoder.output_dim == 128

    def test_custom_dimensions(self):
        encoder = CatanGNNEncoder(
            hex_in_features=9,
            vertex_in_features=7,
            edge_in_features=5,
            player_in_features=14,
            hidden_dim=32,
            num_heads=2,
            num_layers=2,
            output_dim=48,
        )
        assert encoder.output_dim == 48


class TestForwardPass:

    def test_single_sample(self, encoder: CatanGNNEncoder):
        inputs = _make_random_inputs(batch_size=1)
        with torch.no_grad():
            output = encoder(*inputs)
        assert output.shape == (1, 64)

    def test_batch_of_4(self, encoder: CatanGNNEncoder):
        inputs = _make_random_inputs(batch_size=4)
        with torch.no_grad():
            output = encoder(*inputs)
        assert output.shape == (4, 64)

    def test_output_is_finite(self, encoder: CatanGNNEncoder):
        inputs = _make_random_inputs(batch_size=2)
        with torch.no_grad():
            output = encoder(*inputs)
        assert torch.isfinite(output).all()

    def test_output_shape_matches_output_dim(self):
        for out_dim in [32, 64, 128, 256]:
            enc = CatanGNNEncoder.from_env_defaults(
                hidden_dim=32, num_heads=2, num_layers=1, output_dim=out_dim
            )
            inputs = _make_random_inputs(batch_size=1)
            with torch.no_grad():
                output = enc(*inputs)
            assert output.shape == (1, out_dim)


class TestAttentionWeights:

    def test_attention_weights_non_empty(self, encoder: CatanGNNEncoder):
        inputs = _make_random_inputs(batch_size=1)
        with torch.no_grad():
            encoder(*inputs)
        weights = encoder.get_attention_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0
        # At least some relation types should have weights
        # This depends on torch_geometric version; just ensure dict is returned
        assert isinstance(weights, dict)

    def test_attention_weights_keys(self, encoder: CatanGNNEncoder):
        inputs = _make_random_inputs(batch_size=1)
        with torch.no_grad():
            encoder(*inputs)
        weights = encoder.get_attention_weights()
        expected_relations = {
            "hex_to_vertex",
            "vertex_to_hex",
            "vertex_to_edge",
            "edge_to_vertex",
            "vertex_to_vertex",
        }
        assert set(weights.keys()) == expected_relations
