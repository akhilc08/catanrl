"""Actor-critic policy heads for CatanRL MAPPO.

Combines a shared GNN encoder with separate actor (action logits) and critic
(state value) MLP heads.  The actor applies action masking so only legal
actions receive probability mass.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .gnn_encoder import CatanGNNEncoder


def _orthogonal_init(layer: nn.Module, gain: float = 1.0) -> None:
    """Apply orthogonal initialization to a linear layer."""
    if not isinstance(layer, nn.Linear):
        return
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class CatanPolicy(nn.Module):
    """Combined actor-critic policy for Catan MAPPO.

    The GNN encoder is shared between the actor and critic heads (parameter
    sharing).  The actor produces masked action logits and the critic produces
    a scalar state-value estimate.

    Parameters
    ----------
    gnn_encoder : CatanGNNEncoder
        Pre-constructed GNN encoder that maps board observations to a
        fixed-size embedding of shape ``(batch, encoder.output_dim)``.
    action_dim : int
        Number of discrete actions (default 261 from ``ActionSpace.TOTAL_ACTIONS``).
    hidden_dim : int
        Width of the actor and critic hidden layers.
    """

    def __init__(
        self,
        gnn_encoder: CatanGNNEncoder,
        action_dim: int = 261,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.gnn_encoder = gnn_encoder
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        encoder_out = gnn_encoder.output_dim

        # ---- Actor head ----
        self.actor = nn.Sequential(
            nn.Linear(encoder_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # ---- Critic head ----
        self.critic = nn.Sequential(
            nn.Linear(encoder_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Orthogonal init
        _orthogonal_init(self.actor[0], gain=np.sqrt(2))
        _orthogonal_init(self.actor[2], gain=0.01)   # small gain for actor output
        _orthogonal_init(self.critic[0], gain=np.sqrt(2))
        _orthogonal_init(self.critic[2], gain=1.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(
        x: torch.Tensor | np.ndarray, device: torch.device
    ) -> torch.Tensor:
        """Convert numpy arrays to float tensors on *device*."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(device)
        return x.to(device)

    @staticmethod
    def _to_long_tensor(
        x: torch.Tensor | np.ndarray, device: torch.device
    ) -> torch.Tensor:
        """Convert to long tensor on *device*."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).long().to(device)
        return x.long().to(device)

    def _encode(self, obs_dict: Mapping[str, torch.Tensor | np.ndarray]) -> torch.Tensor:
        """Run the shared GNN encoder on an observation dict.

        Handles numpy-to-tensor conversion and adds a batch dimension when the
        inputs are un-batched (rank-2 for node features, rank-1 for player features).
        """
        device = next(self.parameters()).device

        hex_feat = self._to_tensor(obs_dict["hex_features"], device)
        vertex_feat = self._to_tensor(obs_dict["vertex_features"], device)
        edge_feat = self._to_tensor(obs_dict["edge_features"], device)
        player_feat = self._to_tensor(obs_dict["player_features"], device)
        current_player = self._to_long_tensor(obs_dict["current_player"], device)

        # Add batch dim if needed  (e.g. hex_feat is (19, F) -> (1, 19, F))
        if hex_feat.dim() == 2:
            hex_feat = hex_feat.unsqueeze(0)
            vertex_feat = vertex_feat.unsqueeze(0)
            edge_feat = edge_feat.unsqueeze(0)
            player_feat = player_feat.unsqueeze(0)
            current_player = current_player.unsqueeze(0)

        return self.gnn_encoder(
            hex_feat, vertex_feat, edge_feat, player_feat, current_player
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        hex_features: torch.Tensor,
        vertex_features: torch.Tensor,
        edge_features: torch.Tensor,
        player_features: torch.Tensor,
        current_player: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        """Forward pass returning an action distribution and value estimate.

        Parameters
        ----------
        hex_features : Tensor (batch, 19, F_hex)
        vertex_features : Tensor (batch, 54, F_vert)
        edge_features : Tensor (batch, 72, F_edge)
        player_features : Tensor (batch, 4, F_player)
        current_player : Tensor (batch,) int
        action_mask : Tensor (batch, action_dim) boolean
            True where an action is legal.

        Returns
        -------
        dist : Categorical
            Action distribution over legal actions.
        value : Tensor (batch, 1)
            State-value estimate.
        """
        embedding = self.gnn_encoder(
            hex_features, vertex_features, edge_features, player_features, current_player
        )

        # Actor
        logits = self.actor(embedding)  # (batch, action_dim)
        logits = logits.masked_fill(~action_mask.bool(), -1e8)
        dist = Categorical(logits=logits)

        # Critic
        value = self.critic(embedding)  # (batch, 1)

        return dist, value

    def get_action_and_value(
        self,
        obs_dict: Mapping[str, torch.Tensor | np.ndarray],
        action_mask: torch.Tensor | np.ndarray,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) an action and return policy quantities.

        Parameters
        ----------
        obs_dict : dict
            Observation dictionary as returned by ``CatanEnv.reset()`` /
            ``CatanEnv.step()``.  Keys: ``hex_features``, ``vertex_features``,
            ``edge_features``, ``player_features``, ``current_player``.
        action_mask : array-like (batch, action_dim) or (action_dim,)
            Boolean mask of legal actions.
        action : Tensor, optional
            If provided, compute log-prob and entropy for this action instead of
            sampling a new one.

        Returns
        -------
        action : Tensor (batch,)
        log_prob : Tensor (batch,)
        entropy : Tensor (batch,)
        value : Tensor (batch, 1)
        """
        device = next(self.parameters()).device
        embedding = self._encode(obs_dict)  # (batch, encoder_out)

        # Ensure action_mask is a tensor with batch dim
        mask = self._to_tensor(action_mask, device)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        # Actor
        logits = self.actor(embedding)
        logits = logits.masked_fill(~mask.bool(), -1e8)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # Critic
        value = self.critic(embedding)

        return action, log_prob, entropy, value

    def get_value(
        self, obs_dict: Mapping[str, torch.Tensor | np.ndarray]
    ) -> torch.Tensor:
        """Return the critic's value estimate for the given observation.

        Parameters
        ----------
        obs_dict : dict
            Observation dictionary (same format as ``get_action_and_value``).

        Returns
        -------
        Tensor (batch, 1)
        """
        embedding = self._encode(obs_dict)
        return self.critic(embedding)
