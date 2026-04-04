#!/usr/bin/env python3
"""Dispatch CatanRL training to Modal GPU or run locally.

Modal (GPU):
    modal run scripts/dispatch_training.py
    modal run scripts/dispatch_training.py --timesteps 5000000 --num-envs 16

Local (CPU, slow):
    python scripts/dispatch_training.py --local
    python scripts/dispatch_training.py --local --timesteps 200000
"""

from __future__ import annotations

import json
import os
import sys

# ---------------------------------------------------------------------------
# Modal app definition (module-level so Modal CLI can discover it)
# ---------------------------------------------------------------------------

try:
    import modal
    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False

if _MODAL_AVAILABLE:
    app = modal.App("catanrl-training")

    _TORCH_VERSION = "2.1.0"
    _CUDA_TAG = "cu118"

    # Build container image with PyTorch + PyG + training deps.
    # Source code is injected via add_local_dir (copied into the image at
    # build time), so the remote function can import from src.*
    _image = (
        modal.Image.debian_slim(python_version="3.10")
        .pip_install(
            f"torch=={_TORCH_VERSION}",
            index_url=f"https://download.pytorch.org/whl/{_CUDA_TAG}",
        )
        .pip_install(
            "torch-geometric",
            extra_index_url=(
                f"https://data.pyg.org/whl/torch-{_TORCH_VERSION}+{_CUDA_TAG}.html"
            ),
        )
        .pip_install(
            "numpy>=1.24.0,<2.0",  # torch 2.1.0 is not compatible with numpy 2.x
            "mlflow>=2.9.0",
            "structlog>=23.2.0",
            "scipy>=1.11.0",
            "gymnasium>=0.29.0",
        )
        .add_local_dir("src", remote_path="/app/src", copy=True)
    )

    _volume = modal.Volume.from_name("catanrl-models", create_if_missing=True)

    @app.function(
        image=_image,
        gpu="A10G",
        timeout=4 * 3600,
        volumes={"/models": _volume},
    )
    def train_remote(
        total_timesteps: int = 2_000_000,
        num_envs: int = 16,
        learning_rate: float = 3e-4,
        num_steps: int = 128,
        log_interval: int = 10,
        save_interval: int = 100,
        resume_from: str | None = None,
        league: bool = False,
    ) -> dict:
        """Training function that runs inside the Modal container on GPU."""
        import sys
        import torch

        sys.path.insert(0, "/app")

        from src.rl.env.action_space import ActionSpace
        from src.rl.models.gnn_encoder import CatanGNNEncoder
        from src.rl.models.policy import CatanPolicy
        from src.rl.training.mappo import MAPPOConfig, MAPPOTrainer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[modal] device={device}")
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[modal] gpu={gpu_name}  vram={vram_gb:.1f}GB")

        config = MAPPOConfig(
            total_timesteps=total_timesteps,
            num_envs=num_envs,
            learning_rate=learning_rate,
            num_steps=num_steps,
            checkpoint_dir="/models/checkpoints",
            log_interval=log_interval,
            save_interval=save_interval,
            resume_checkpoint=resume_from,
            elo_path="/models/elo.json",
            opponent_mix={"random": 0.4, "heuristic": 0.6} if league else None,
        )

        encoder = CatanGNNEncoder.from_env_defaults()
        policy = CatanPolicy(
            gnn_encoder=encoder,
            action_dim=ActionSpace.TOTAL_ACTIONS,
        )
        param_count = sum(p.numel() for p in policy.parameters())
        batch_size = config.num_steps * config.num_envs
        num_updates = config.total_timesteps // batch_size
        print(
            f"[modal] params={param_count:,}  "
            f"batch={batch_size}  updates={num_updates:,}"
        )

        trainer = MAPPOTrainer(config=config, policy=policy, device=device)
        metrics = trainer.train()

        # Persist champion checkpoint to volume
        os.makedirs("/models", exist_ok=True)
        torch.save(policy.state_dict(), "/models/champion.pt")
        print("[modal] saved /models/champion.pt")
        _volume.commit()

        return metrics

    @app.local_entrypoint()
    def modal_main(
        timesteps: int = 2_000_000,
        num_envs: int = 16,
        lr: float = 3e-4,
        resume_from: str = "",
        league: bool = False,
    ) -> None:
        """
        Entrypoint used by `modal run scripts/dispatch_training.py`.

        Override defaults:
            modal run scripts/dispatch_training.py --timesteps 5000000
            modal run scripts/dispatch_training.py --num-envs 32
            modal run scripts/dispatch_training.py --resume-from /models/checkpoints/policy_update_400.pt
        """
        resume = resume_from or None
        print(
            f"[dispatch] timesteps={timesteps:,}  "
            f"num_envs={num_envs}  lr={lr}  gpu=A10G"
            f"  league={league}"
            + (f"  resume={resume}" if resume else "")
        )
        metrics = train_remote.remote(
            total_timesteps=timesteps,
            num_envs=num_envs,
            learning_rate=lr,
            resume_from=resume,
            league=league,
        )
        print("\n[dispatch] Training complete.")
        print(json.dumps(metrics, indent=2, default=str))
        print("\n[dispatch] Retrieve model checkpoint:")
        print("  modal volume get catanrl-models champion.pt models/champion.pt")
        print("\n[dispatch] Then serve the API:")
        print("  MODEL_PATH=models/champion.pt uvicorn src.api.main:app")


# ---------------------------------------------------------------------------
# Local training (CPU, for smoke-testing only)
# ---------------------------------------------------------------------------

def run_local(
    total_timesteps: int = 100_000,
    num_envs: int = 4,
    learning_rate: float = 3e-4,
    num_steps: int = 128,
    checkpoint_dir: str = "models/checkpoints",
) -> dict:
    """Run training on the local machine (CPU). Slow — for smoke-testing only."""
    from src.rl.env.action_space import ActionSpace
    from src.rl.models.gnn_encoder import CatanGNNEncoder
    from src.rl.models.policy import CatanPolicy
    from src.rl.training.mappo import MAPPOConfig, MAPPOTrainer

    config = MAPPOConfig(
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        learning_rate=learning_rate,
        num_steps=num_steps,
        checkpoint_dir=checkpoint_dir,
        log_interval=5,
        save_interval=50,
    )
    encoder = CatanGNNEncoder.from_env_defaults()
    policy = CatanPolicy(gnn_encoder=encoder, action_dim=ActionSpace.TOTAL_ACTIONS)

    param_count = sum(p.numel() for p in policy.parameters())
    batch_size = config.num_steps * config.num_envs
    print(f"[local] params={param_count:,}  batch={batch_size}  device=cpu")

    trainer = MAPPOTrainer(config=config, policy=policy, device="cpu")
    metrics = trainer.train()
    print(json.dumps(metrics, indent=2, default=str))
    return metrics


# ---------------------------------------------------------------------------
# CLI (--local flag; Modal GPU uses its own entrypoint above)
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Dispatch CatanRL training.\n"
            "  GPU (Modal):  modal run scripts/dispatch_training.py\n"
            "  CPU (local):  python scripts/dispatch_training.py --local"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--local", action="store_true", help="Run on CPU locally")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints")
    args = parser.parse_args()

    if args.local:
        run_local(
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            learning_rate=args.lr,
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        if not _MODAL_AVAILABLE:
            print("[dispatch] modal not installed. Run: pip install modal")
            sys.exit(1)
        print(
            "[dispatch] To train on GPU, run:\n"
            "  modal run scripts/dispatch_training.py\n"
            "  modal run scripts/dispatch_training.py --timesteps 5000000\n\n"
            "To train locally:\n"
            "  python scripts/dispatch_training.py --local"
        )


if __name__ == "__main__":
    main()
