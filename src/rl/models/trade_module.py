"""Trade evaluation module for CatanRL.

Scores discrete trade actions by combining the board embedding with
per-trade feature vectors through a small MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TradeModule(nn.Module):
    """Evaluates trade proposals.

    Used by the policy to score discrete trade actions.  Takes the board
    embedding (from the GNN encoder) concatenated with trade-specific features
    and produces a scalar score per trade.

    Parameters
    ----------
    input_dim : int
        Dimension of the board embedding.
    hidden_dim : int
        Width of the hidden layer.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 64) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        board_embedding: torch.Tensor,
        trade_features: torch.Tensor,
    ) -> torch.Tensor:
        """Score one or more trade proposals.

        Parameters
        ----------
        board_embedding : Tensor (batch, input_dim)
            Board-level embedding from the GNN encoder.
        trade_features : Tensor (batch, num_trades, input_dim)
            Feature vector for each candidate trade.

        Returns
        -------
        Tensor (batch, num_trades)
            Scalar score for each trade proposal.
        """
        # Expand board embedding to match trade dimension
        batch, num_trades, feat_dim = trade_features.shape
        board_exp = board_embedding.unsqueeze(1).expand(-1, num_trades, -1)

        # Concatenate and score
        combined = torch.cat([board_exp, trade_features], dim=-1)  # (B, T, 2*input_dim)
        scores = self.mlp(combined).squeeze(-1)  # (B, T)
        return scores
