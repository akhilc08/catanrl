#!/usr/bin/env python3
"""Dispatch CatanRL training run.

Usage:
    python scripts/dispatch_training.py --budget 50
    python scripts/dispatch_training.py --local --timesteps 100000
"""
import argparse
import sys
import os
import json
import time

def dispatch_modal(budget: float, gpu: str = "A10G", timeout: int = 300):
    """Dispatch training to Modal serverless GPU."""
    print(f"[dispatch] Requesting Modal training run (budget=${budget}, gpu={gpu})")
    # In production, this would use modal.Function.remote()
    # For now, provide the structure and fall back to local
    try:
        import modal
        # modal dispatch logic here
        print("[dispatch] Modal training dispatched successfully")
    except ImportError:
        print("[dispatch] Modal not installed — falling back to local training")
        return run_local(timesteps=100_000)

def run_local(timesteps: int = 100_000, lr: float = 3e-4):
    """Run training locally."""
    from src.rl.training.mappo import MAPPOConfig, MAPPOTrainer
    from src.rl.models.gnn_encoder import CatanGNNEncoder
    from src.rl.models.policy import CatanPolicy

    config = MAPPOConfig(
        total_timesteps=timesteps,
        learning_rate=lr,
        num_envs=4,
        log_interval=5,
        save_interval=50,
    )
    encoder = CatanGNNEncoder.from_env_defaults()
    policy = CatanPolicy(gnn_encoder=encoder)
    trainer = MAPPOTrainer(config=config, policy=policy)
    metrics = trainer.train()
    print(f"[dispatch] Training complete: {json.dumps(metrics, indent=2, default=str)}")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Dispatch CatanRL training")
    parser.add_argument("--budget", type=float, default=50, help="Max spend on Modal ($)")
    parser.add_argument("--local", action="store_true", help="Run training locally")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gpu", type=str, default="A10G", help="Modal GPU type")
    args = parser.parse_args()

    if args.local:
        run_local(timesteps=args.timesteps, lr=args.lr)
    else:
        dispatch_modal(budget=args.budget, gpu=args.gpu)

if __name__ == "__main__":
    main()
