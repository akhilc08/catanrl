"""Model loading and inference for the CatanRL API."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import structlog
import torch

from ..rl.models.gnn_encoder import CatanGNNEncoder
from ..rl.models.policy import CatanPolicy

logger = structlog.get_logger()


class ModelManager:
    """Loads and manages the champion CatanRL policy model."""

    def __init__(self) -> None:
        self.policy: CatanPolicy | None = None
        self.version: str = "unknown"
        self.metadata: dict = {}
        self.device: str = "cpu"

    def load(self, path: str | None = None) -> None:
        """Load model from local path or S3.

        Resolution order:
        1. Explicit *path* argument
        2. MODEL_PATH environment variable
        3. S3_MODEL_BUCKET environment variable (downloads to /tmp)
        4. Create an untrained (random) policy for dev/testing
        """
        model_path = path or os.environ.get("MODEL_PATH")

        # --- Try local checkpoint ---
        if model_path and Path(model_path).exists():
            self._load_checkpoint(model_path)
            return

        # --- Try S3 ---
        s3_bucket = os.environ.get("S3_MODEL_BUCKET")
        s3_key = os.environ.get("S3_MODEL_KEY", "champion/policy.pt")
        if s3_bucket:
            local_path = Path("/tmp/catanrl_model.pt")
            try:
                import boto3

                logger.info("downloading_model_from_s3", bucket=s3_bucket, key=s3_key)
                s3 = boto3.client("s3")
                s3.download_file(s3_bucket, s3_key, str(local_path))
                self._load_checkpoint(str(local_path))
                return
            except Exception as exc:
                logger.warning("s3_download_failed", error=str(exc))

        # --- Fallback: untrained policy ---
        logger.warning("no_model_found", detail="Creating untrained policy for dev/testing")
        encoder = CatanGNNEncoder.from_env_defaults()
        self.policy = CatanPolicy(gnn_encoder=encoder)
        self.policy.eval()
        self.version = "untrained-dev"
        self.metadata = {"note": "Random untrained weights"}

    def _load_checkpoint(self, path: str) -> None:
        """Load a saved checkpoint from disk."""
        logger.info("loading_model", path=path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Support both raw state_dict and wrapped checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self.version = checkpoint.get("version", "unknown")
            self.metadata = {
                k: v for k, v in checkpoint.items() if k != "model_state_dict"
            }
        else:
            state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
            self.version = "unknown"
            self.metadata = {}

        encoder = CatanGNNEncoder.from_env_defaults()
        self.policy = CatanPolicy(gnn_encoder=encoder)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        logger.info("model_loaded", version=self.version)

    def predict(self, obs_dict: dict, action_mask: np.ndarray) -> tuple[list, float]:
        """Run inference and return top-3 actions with scores and win probability.

        Parameters
        ----------
        obs_dict : dict
            Observation dictionary with keys: hex_features, vertex_features,
            edge_features, player_features, current_player.
        action_mask : np.ndarray
            Boolean mask of shape (261,) indicating legal actions.

        Returns
        -------
        tuple[list[tuple[int, float]], float]
            (top_k list of (action_id, probability), win_probability)
        """
        if self.policy is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        with torch.no_grad():
            device = next(self.policy.parameters()).device
            mask_tensor = torch.from_numpy(action_mask).float().unsqueeze(0).to(device)

            # Encode observation and get logits
            embedding = self.policy._encode(obs_dict)

            # Actor head: masked softmax
            logits = self.policy.actor(embedding)  # (1, 261)
            logits = logits.masked_fill(~mask_tensor.bool(), -1e8)
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # (261,)

            # Critic head: value estimate -> approximate win probability via sigmoid
            value = self.policy.critic(embedding).squeeze()  # scalar
            win_prob = torch.sigmoid(value).item()

            # Top-3 actions
            top_k = min(3, int(action_mask.sum()))
            if top_k == 0:
                return [], win_prob

            top_probs, top_indices = torch.topk(probs, k=top_k)
            actions_with_scores = [
                (int(top_indices[i].item()), float(top_probs[i].item()))
                for i in range(top_k)
            ]

        return actions_with_scores, win_prob
