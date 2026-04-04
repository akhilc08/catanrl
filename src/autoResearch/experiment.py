"""Experiment tracking for autonomous research.

Logs every trial to results.tsv (untracked by git, persists across sessions)
and maintains the best-known config.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..rl.training.mappo import MAPPOConfig
from .evaluator import TrialResult
from .mutations import Mutation


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRecord:
    trial_id: int
    timestamp: str
    mutation_param: str
    mutation_old: str
    mutation_new: str
    win_rate: float
    mean_reward: float
    elapsed_seconds: float
    num_updates: int
    global_step: int
    is_best: bool
    error: str

    # TSV columns in order
    _FIELDS = [
        "trial_id", "timestamp", "mutation_param", "mutation_old",
        "mutation_new", "win_rate", "mean_reward", "elapsed_seconds",
        "num_updates", "global_step", "is_best", "error",
    ]

    def as_row(self) -> list[str]:
        return [str(getattr(self, f)) for f in self._FIELDS]

    @classmethod
    def header(cls) -> list[str]:
        return list(cls._FIELDS)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """Loads, appends, and queries experiment history from a TSV file.

    Parameters
    ----------
    results_path : str
        Path to results.tsv (created if absent).
    best_config_path : str
        Path to JSON file storing the best known MAPPOConfig (created if absent).
    """

    def __init__(
        self,
        results_path: str = "models/autoresearch/results.tsv",
        best_config_path: str = "models/autoresearch/best_config.json",
    ) -> None:
        self.results_path = Path(results_path)
        self.best_config_path = Path(best_config_path)
        self._records: list[ExperimentRecord] = []

        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_config_path.parent.mkdir(parents=True, exist_ok=True)

        self._load()

    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.results_path.exists():
            return
        with self.results_path.open(newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    rec = ExperimentRecord(
                        trial_id=int(row["trial_id"]),
                        timestamp=row["timestamp"],
                        mutation_param=row["mutation_param"],
                        mutation_old=row["mutation_old"],
                        mutation_new=row["mutation_new"],
                        win_rate=float(row["win_rate"]),
                        mean_reward=float(row["mean_reward"]),
                        elapsed_seconds=float(row["elapsed_seconds"]),
                        num_updates=int(row["num_updates"]),
                        global_step=int(row["global_step"]),
                        is_best=row["is_best"].lower() == "true",
                        error=row.get("error", ""),
                    )
                    self._records.append(rec)
                except (KeyError, ValueError):
                    continue  # tolerate malformed rows

    def _write_header_if_needed(self) -> None:
        if not self.results_path.exists() or self.results_path.stat().st_size == 0:
            with self.results_path.open("w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(ExperimentRecord.header())

    def _append_row(self, rec: ExperimentRecord) -> None:
        self._write_header_if_needed()
        with self.results_path.open("a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(rec.as_row())

    # ------------------------------------------------------------------

    @property
    def next_trial_id(self) -> int:
        return len(self._records) + 1

    @property
    def best_win_rate(self) -> float:
        bests = [r.win_rate for r in self._records if r.is_best]
        return max(bests, default=0.0)

    def load_best_config(self, default: MAPPOConfig) -> MAPPOConfig:
        """Load best config from JSON, falling back to *default*."""
        if not self.best_config_path.exists():
            return default
        try:
            data = json.loads(self.best_config_path.read_text())
            cfg = MAPPOConfig()
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            return cfg
        except (json.JSONDecodeError, AttributeError):
            return default

    def save_best_config(self, config: MAPPOConfig) -> None:
        """Persist config to JSON."""
        data = {
            k: getattr(config, k)
            for k in vars(MAPPOConfig())
            if not k.startswith("_")
        }
        self.best_config_path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------

    def record(
        self,
        mutation: Mutation,
        result: TrialResult,
        is_best: bool,
    ) -> ExperimentRecord:
        """Append one trial result to the TSV and in-memory list."""
        rec = ExperimentRecord(
            trial_id=self.next_trial_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            mutation_param=mutation.param,
            mutation_old=str(mutation.old_value),
            mutation_new=str(mutation.new_value),
            win_rate=round(result.win_rate, 6),
            mean_reward=round(result.mean_reward, 6),
            elapsed_seconds=round(result.elapsed_seconds, 1),
            num_updates=result.num_updates,
            global_step=result.global_step,
            is_best=is_best,
            error=result.error or "",
        )
        self._records.append(rec)
        self._append_row(rec)
        return rec

    # ------------------------------------------------------------------

    def summary(self, n: int = 10) -> str:
        """Return a human-readable summary of the last *n* trials."""
        recent = self._records[-n:]
        lines = [
            f"{'ID':>4}  {'param':>18}  {'new_val':>10}  "
            f"{'win_rate':>9}  {'Δ':>8}  {'best?':>5}  {'error'}"
        ]
        lines.append("-" * 80)
        prev_wr = None
        for r in recent:
            delta = f"{r.win_rate - prev_wr:+.4f}" if prev_wr is not None else "  n/a"
            best_mark = "✓" if r.is_best else " "
            err = r.error[:20] if r.error else ""
            lines.append(
                f"{r.trial_id:>4}  {r.mutation_param:>18}  {r.mutation_new:>10}  "
                f"{r.win_rate:>9.4f}  {delta:>8}  {best_mark:>5}  {err}"
            )
            prev_wr = r.win_rate
        return "\n".join(lines)
