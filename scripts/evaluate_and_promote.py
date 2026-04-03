#!/usr/bin/env python3
"""Evaluate challenger model against champion and promote if significantly better.

Usage:
    python scripts/evaluate_and_promote.py --threshold 0.02
    python scripts/evaluate_and_promote.py --challenger models/challenger.pt --champion models/champion.pt
"""
import argparse
import os
import sys
import json
import shutil

def load_policy(path):
    """Load a CatanPolicy from checkpoint."""
    import torch
    from src.rl.models.gnn_encoder import CatanGNNEncoder
    from src.rl.models.policy import CatanPolicy

    encoder = CatanGNNEncoder.from_env_defaults()
    policy = CatanPolicy(gnn_encoder=encoder)
    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        policy.load_state_dict(state_dict)
    return policy

def evaluate_matchup(policy_a, policy_b, num_games: int = 200):
    """Play num_games between two policies. Returns policy_a win rate."""
    import numpy as np
    from src.rl.env.catan_env import CatanEnv

    wins = 0
    total_vp = 0

    for game_idx in range(num_games):
        env = CatanEnv(num_players=4)
        obs, info = env.reset(seed=game_idx)
        done = False

        while not done:
            mask = env.get_action_mask()
            current = env.current_player

            # Player 0 = policy_a, players 1-3 = policy_b
            if current == 0:
                policy = policy_a
            else:
                policy = policy_b

            action, _, _, _ = policy.get_action_and_value(obs, mask)
            action = action.item() if hasattr(action, 'item') else int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        winner = info.get("winner", -1)
        if winner == 0:
            wins += 1
        total_vp += env.players[0]["victory_points"]

    win_rate = wins / num_games
    mean_vp = total_vp / num_games
    return {"win_rate": win_rate, "mean_vp": mean_vp, "num_games": num_games}

def promote(challenger_path: str, champion_path: str, metadata: dict):
    """Promote challenger to champion."""
    # Archive current champion
    if os.path.exists(champion_path):
        import time
        archive_path = champion_path.replace(".pt", f"_archived_{int(time.time())}.pt")
        shutil.copy2(champion_path, archive_path)
        print(f"[promote] Archived current champion to {archive_path}")

    shutil.copy2(challenger_path, champion_path)

    # Save promotion metadata
    meta_path = champion_path.replace(".pt", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[promote] Challenger promoted to champion at {champion_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate and promote CatanRL models")
    parser.add_argument("--threshold", type=float, default=0.02, help="Min win rate delta to promote")
    parser.add_argument("--challenger", type=str, default="models/challenger.pt")
    parser.add_argument("--champion", type=str, default="models/champion.pt")
    parser.add_argument("--num-games", type=int, default=200, help="Evaluation games")
    args = parser.parse_args()

    print(f"[eval] Loading challenger from {args.challenger}")
    challenger = load_policy(args.challenger)

    print(f"[eval] Loading champion from {args.champion}")
    champion = load_policy(args.champion)

    print(f"[eval] Playing {args.num_games} evaluation games...")
    results = evaluate_matchup(challenger, champion, num_games=args.num_games)

    print(f"[eval] Challenger win rate: {results['win_rate']:.3f}")
    print(f"[eval] Challenger mean VP: {results['mean_vp']:.2f}")

    baseline = 0.25  # expected win rate in 4-player game
    delta = results["win_rate"] - baseline

    if delta >= args.threshold:
        print(f"[eval] Delta {delta:.3f} >= threshold {args.threshold} — PROMOTING")
        promote(args.challenger, args.champion, {
            "win_rate": results["win_rate"],
            "mean_vp": results["mean_vp"],
            "delta": delta,
            "num_eval_games": results["num_games"],
        })
    else:
        print(f"[eval] Delta {delta:.3f} < threshold {args.threshold} — keeping current champion")

if __name__ == "__main__":
    main()
