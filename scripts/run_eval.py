"""Run post-training evaluation: arena vs random + heuristic, print Elo + sample stats.

Usage:
    modal run scripts/run_eval.py
    modal run scripts/run_eval.py --checkpoint /models/checkpoints/policy_update_700.pt
    modal run scripts/run_eval.py --num-games 200
"""

from __future__ import annotations

import modal

app = modal.App("catanrl-eval")

_TORCH_VERSION = "2.1.0"
_CUDA_TAG = "cu118"

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
        "numpy>=1.24.0,<2.0",
        "scipy>=1.11.0",
        "gymnasium>=0.29.0",
    )
    .add_local_dir("src", remote_path="/app/src", copy=True)
)

_volume = modal.Volume.from_name("catanrl-models", create_if_missing=False)


@app.function(
    image=_image,
    cpu=4,
    timeout=3600,
    volumes={"/models": _volume},
)
def run_eval(
    checkpoint: str = "/models/checkpoints/policy_final.pt",
    num_games: int = 200,
) -> dict:
    import sys
    import json
    sys.path.insert(0, "/app")

    import torch
    from src.rl.env.action_space import ActionSpace
    from src.rl.models.gnn_encoder import CatanGNNEncoder
    from src.rl.models.policy import CatanPolicy
    from src.rl.eval.agents import RandomAgent, HeuristicAgent
    from src.rl.eval.arena import Arena
    from src.rl.eval.elo import EloTracker

    print(f"[eval] Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")

    encoder = CatanGNNEncoder.from_env_defaults()
    policy = CatanPolicy(gnn_encoder=encoder, action_dim=ActionSpace.TOTAL_ACTIONS)

    state_dict = ckpt["policy_state_dict"] if isinstance(ckpt, dict) and "policy_state_dict" in ckpt else ckpt
    policy.load_state_dict(state_dict)
    policy.eval()

    elo = EloTracker(save_path="/models/elo.json")
    elo.register("final")

    results = {}
    for name, opponent in [("random", RandomAgent()), ("heuristic", HeuristicAgent())]:
        print(f"\n[eval] Running {num_games} games vs {name}...")
        arena = Arena(policy=policy, opponent=opponent, device="cpu", num_games=num_games)
        r = arena.run()

        elo_rating, _ = elo.update(
            player_a="final", player_b=name,
            wins_a=r.wins, wins_b=r.losses, draws=r.draws,
        )
        results[name] = {
            "wins": r.wins, "losses": r.losses, "draws": r.draws,
            "win_rate": round(r.win_rate, 4),
            "mean_game_length": round(r.mean_game_length, 1),
            "elo": round(elo_rating, 1),
        }
        print(
            f"  W={r.wins} L={r.losses} D={r.draws}  "
            f"win_rate={r.win_rate:.3f}  elo={elo_rating:.1f}  "
            f"mean_game_len={r.mean_game_length:.0f}"
        )

    _volume.commit()
    print("\n" + elo.summary())
    print("\n[eval] Full results:")
    print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint()
def main(
    checkpoint: str = "/models/checkpoints/policy_final.pt",
    num_games: int = 200,
) -> None:
    results = run_eval.remote(checkpoint=checkpoint, num_games=num_games)
    import json
    print(json.dumps(results, indent=2))
