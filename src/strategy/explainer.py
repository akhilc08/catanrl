"""SHAP-style and attention-based explainability for the GNN policy."""

from __future__ import annotations

import numpy as np
import structlog
import torch

from ..rl.env.action_space import ActionSpace
from .templates import (
    ExplanationGenerator,
    _hex_resource_name,
)

logger = structlog.get_logger()


class StrategyExplainer:
    """Extracts interpretable explanations from the GNN policy.

    Uses attention weight extraction and input-gradient feature importance
    to explain why the policy recommends specific actions.
    """

    def __init__(self, policy, env=None):
        """
        Parameters
        ----------
        policy : CatanPolicy
            The actor-critic policy with a GNN encoder.
        env : CatanEnv, optional
            If provided, used for richer explanations (resource names, etc.).
        """
        self.policy = policy
        self.env = env
        self.generator = ExplanationGenerator()

    def extract_attention_weights(self, obs_dict: dict) -> dict[str, list[torch.Tensor]]:
        """Run a forward pass and extract attention weights from each GAT layer.

        Parameters
        ----------
        obs_dict : dict
            Observation dictionary from the environment.

        Returns
        -------
        dict
            Mapping from relation name to list of attention weight tensors
            (one per GAT layer).
        """
        self.policy.eval()
        with torch.no_grad():
            # Run the encoder to populate attention weights
            self.policy._encode(obs_dict)

        return self.policy.gnn_encoder.get_attention_weights()

    def get_important_hexes(self, obs_dict: dict, top_k: int = 5) -> list[dict]:
        """Identify the most important hexes based on attention aggregation.

        Uses the hex_to_vertex attention weights, aggregating across heads and
        layers to compute a per-hex importance score.

        Parameters
        ----------
        obs_dict : dict
            Observation dictionary.
        top_k : int
            Number of top hexes to return.

        Returns
        -------
        list[dict]
            List of {hex_id, importance_score, resource_type} sorted by importance.
        """
        attention_weights = self.extract_attention_weights(obs_dict)

        # Aggregate hex_to_vertex attention across layers
        hex_importance = np.zeros(19, dtype=np.float64)

        h2v_weights = attention_weights.get("hex_to_vertex", [])
        if not h2v_weights:
            logger.warning("no_hex_to_vertex_attention")
            # Fallback: use hex features directly
            hex_feat = obs_dict.get("hex_features")
            if hex_feat is not None:
                if isinstance(hex_feat, torch.Tensor):
                    hex_feat = hex_feat.cpu().numpy()
                if hex_feat.ndim == 3:
                    hex_feat = hex_feat[0]
                # Use number token (column 7 scaled) as proxy for importance
                hex_importance = hex_feat[:, 7].astype(np.float64)
        else:
            for layer_alpha in h2v_weights:
                # alpha shape: (num_edges_in_batch, num_heads)
                alpha = layer_alpha.cpu().numpy()
                if alpha.ndim == 2:
                    alpha = alpha.mean(axis=1)  # average over heads

                # hex_to_vertex edges: each hex has 6 outgoing edges
                # Edge ordering: hex 0 -> 6 edges, hex 1 -> 6 edges, ...
                # For a single graph (batch=1), there are 19*6 = 114 edges
                edges_per_hex = 6
                num_hexes = 19
                single_graph_edges = num_hexes * edges_per_hex

                # Handle batched case: take first graph only
                if len(alpha) > single_graph_edges:
                    alpha = alpha[:single_graph_edges]

                for hi in range(min(num_hexes, len(alpha) // edges_per_hex)):
                    start = hi * edges_per_hex
                    end = start + edges_per_hex
                    hex_importance[hi] += float(np.mean(alpha[start:end]))

        # Normalize
        total = hex_importance.sum()
        if total > 0:
            hex_importance /= total

        # Get top-k
        top_indices = np.argsort(hex_importance)[::-1][:top_k]

        results = []
        for hi in top_indices:
            if self.env is not None:
                resource_type = _hex_resource_name(int(hi), self.env)
            else:
                resource_type = "unknown"
            results.append({
                "hex_id": int(hi),
                "importance_score": float(hex_importance[hi]),
                "resource_type": resource_type,
            })

        return results

    def get_feature_importance(
        self, obs_dict: dict, action_id: int
    ) -> dict[str, np.ndarray]:
        """Compute feature importance for a specific action using input gradients.

        Uses the gradient of the action logit with respect to each input feature
        as a fast approximation of SHAP values.

        Parameters
        ----------
        obs_dict : dict
            Observation dictionary.
        action_id : int
            The flat action ID to explain.

        Returns
        -------
        dict
            Mapping from feature group name to importance array:
            - "hex_features": (19, F_hex) importance
            - "vertex_features": (54, F_vert) importance
            - "edge_features": (72, F_edge) importance
            - "player_features": (num_players, F_player) importance
        """
        self.policy.eval()
        device = next(self.policy.parameters()).device

        # Prepare input tensors with gradients enabled
        hex_feat = self._to_tensor(obs_dict["hex_features"], device).requires_grad_(True)
        vertex_feat = self._to_tensor(obs_dict["vertex_features"], device).requires_grad_(True)
        edge_feat = self._to_tensor(obs_dict["edge_features"], device).requires_grad_(True)
        player_feat = self._to_tensor(obs_dict["player_features"], device).requires_grad_(True)
        current_player = self._to_long(obs_dict["current_player"], device)

        # Add batch dim if needed
        if hex_feat.dim() == 2:
            hex_feat = hex_feat.unsqueeze(0)
            vertex_feat = vertex_feat.unsqueeze(0)
            edge_feat = edge_feat.unsqueeze(0)
            player_feat = player_feat.unsqueeze(0)
            current_player = current_player.unsqueeze(0)

        # Forward pass through encoder
        embedding = self.policy.gnn_encoder(
            hex_feat, vertex_feat, edge_feat, player_feat, current_player
        )

        # Get actor logits
        logits = self.policy.actor(embedding)  # (1, action_dim)

        # Backprop from the target action logit
        target_logit = logits[0, action_id]
        target_logit.backward()

        # Collect gradients * input (saliency)
        importance = {}
        for name, tensor in [
            ("hex_features", hex_feat),
            ("vertex_features", vertex_feat),
            ("edge_features", edge_feat),
            ("player_features", player_feat),
        ]:
            if tensor.grad is not None:
                # Saliency = |grad * input|
                sal = (tensor.grad * tensor).abs().squeeze(0).detach().cpu().numpy()
                importance[name] = sal
            else:
                importance[name] = np.zeros(tensor.squeeze(0).shape, dtype=np.float32)

        return importance

    def explain_action(
        self, obs_dict: dict, action_id: int, action_type: str | None = None
    ) -> str:
        """Generate a human-readable explanation for why an action was recommended.

        Combines attention weights, feature importance, and template-based
        explanation generation.

        Parameters
        ----------
        obs_dict : dict
            Observation dictionary.
        action_id : int
            Flat action ID.
        action_type : str, optional
            If not provided, decoded from action_id.

        Returns
        -------
        str
            Human-readable explanation string.
        """
        if action_type is None:
            action_type, action_param = ActionSpace.decode_action(action_id)
        else:
            _, action_param = ActionSpace.decode_action(action_id)

        # Get feature importance
        try:
            importance = self.get_feature_importance(obs_dict, action_id)
        except Exception as e:
            logger.warning("feature_importance_failed", error=str(e))
            importance = {}

        # Get base explanation from template
        base_explanation = self.generator.generate(
            action_type, action_param, obs_dict, score=0.0, env=self.env
        )

        # Augment with feature importance insights
        extra_insights = []

        if "hex_features" in importance:
            hex_imp = importance["hex_features"]
            # Sum importance per hex
            per_hex = hex_imp.sum(axis=1)  # (19,)
            top_hex = int(np.argmax(per_hex))
            if per_hex[top_hex] > 0.01:
                if self.env is not None:
                    hex_name = _hex_resource_name(top_hex, self.env)
                else:
                    hex_name = f"hex {top_hex}"
                extra_insights.append(
                    f"Key factor: {hex_name} hex ({top_hex}) has "
                    f"highest influence on this decision."
                )

        if "player_features" in importance:
            player_imp = importance["player_features"]
            cp = int(obs_dict.get("current_player", 0))
            if cp < len(player_imp):
                feat_imp = player_imp[cp]
                # Feature names for player features
                feat_names = [
                    "Wood", "Brick", "Sheep", "Wheat", "Ore",
                    "Knights", "VP cards", "Road Building", "Year of Plenty", "Monopoly",
                    "Victory Points", "Roads left", "Settlements left", "Cities left",
                ]
                top_feat_idx = int(np.argmax(feat_imp))
                if top_feat_idx < len(feat_names) and feat_imp[top_feat_idx] > 0.01:
                    extra_insights.append(
                        f"Your {feat_names[top_feat_idx]} count strongly influenced this choice."
                    )

        if extra_insights:
            return base_explanation + " " + " ".join(extra_insights)
        return base_explanation

    @staticmethod
    def _to_tensor(x, device: torch.device) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(device)
        return x.float().to(device)

    @staticmethod
    def _to_long(x, device: torch.device) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(np.asarray(x)).long().to(device)
        if isinstance(x, (int, np.integer)):
            return torch.tensor(x, dtype=torch.long, device=device)
        return x.long().to(device)
