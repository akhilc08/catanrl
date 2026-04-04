"""Elo rating tracker for checkpoint-vs-checkpoint and checkpoint-vs-baseline results.

Ratings are persisted to a JSON file so they survive across training runs.
Fixed baseline ratings:
  random    = 800
  heuristic = 1000
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass


@dataclass
class EloRecord:
    name: str
    rating: float
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games: int = 0


_FIXED = {"random", "heuristic"}


class EloTracker:
    """Elo rating system for named checkpoints and baselines.

    K=32 (standard competitive Elo).
    Baseline agents have fixed ratings that never change.
    """

    K: float = 32.0
    _BASELINE_RATINGS = {"random": 800.0, "heuristic": 1000.0}

    def __init__(self, save_path: str | None = None) -> None:
        self.ratings: dict[str, EloRecord] = {}
        self.history: list[dict] = []
        self.save_path = save_path

        if save_path and os.path.exists(save_path):
            self._load(save_path)
        else:
            for name, r in self._BASELINE_RATINGS.items():
                self.ratings[name] = EloRecord(name=name, rating=r)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, name: str, initial_rating: float = 1200.0) -> None:
        """Register a new checkpoint with an initial Elo rating."""
        if name not in self.ratings:
            self.ratings[name] = EloRecord(name=name, rating=initial_rating)

    def update(
        self,
        player_a: str,
        player_b: str,
        wins_a: int,
        wins_b: int,
        draws: int,
        global_step: int = 0,
    ) -> tuple[float, float]:
        """Apply match results and return updated (rating_a, rating_b)."""
        rec_a = self.ratings[player_a]
        rec_b = self.ratings[player_b]
        total = wins_a + wins_b + draws
        if total == 0:
            return rec_a.rating, rec_b.rating

        score_a = (wins_a + 0.5 * draws) / total
        score_b = (wins_b + 0.5 * draws) / total
        exp_a = self._expected(rec_a.rating, rec_b.rating)
        exp_b = self._expected(rec_b.rating, rec_a.rating)

        if player_a not in _FIXED:
            rec_a.rating += self.K * total * (score_a - exp_a)
        if player_b not in _FIXED:
            rec_b.rating += self.K * total * (score_b - exp_b)

        rec_a.wins += wins_a; rec_a.losses += wins_b; rec_a.draws += draws; rec_a.games += total
        rec_b.wins += wins_b; rec_b.losses += wins_a; rec_b.draws += draws; rec_b.games += total

        self.history.append({
            "global_step": global_step,
            "player_a": player_a,
            "player_b": player_b,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "draws": draws,
            "rating_a": round(rec_a.rating, 1),
            "rating_b": round(rec_b.rating, 1),
        })

        if self.save_path:
            self._persist()

        return rec_a.rating, rec_b.rating

    def get_rating(self, name: str) -> float:
        return self.ratings[name].rating if name in self.ratings else 1200.0

    def summary(self) -> str:
        lines = ["Elo Ratings:"]
        for name, rec in sorted(self.ratings.items(), key=lambda x: -x[1].rating):
            lines.append(
                f"  {name:35s}  {rec.rating:7.1f}  "
                f"W={rec.wins} L={rec.losses} D={rec.draws}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def _persist(self) -> None:
        assert self.save_path
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(
                {"ratings": {k: asdict(v) for k, v in self.ratings.items()},
                 "history": self.history},
                f, indent=2,
            )

    def _load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.ratings = {k: EloRecord(**v) for k, v in data["ratings"].items()}
        self.history = data.get("history", [])
        # Ensure baselines are always present
        for name, r in self._BASELINE_RATINGS.items():
            if name not in self.ratings:
                self.ratings[name] = EloRecord(name=name, rating=r)
