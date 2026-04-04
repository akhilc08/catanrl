"""AutoResearcher: autonomous hyperparameter search for CatanRL.

Mirrors Karpathy's autoresearch loop:
  1. Load best known config (or default)
  2. Propose a mutation (UCB1 over search space)
  3. Run a time-bounded training trial
  4. If win_rate improves → keep mutation, update best config
     Else           → discard mutation, keep best config
  5. Log result to results.tsv
  6. Repeat until max_experiments reached or interrupted

Usage
-----
Run indefinitely (Ctrl-C to stop):

    python -m src.autoResearch.researcher

Or from code:

    from src.autoResearch import AutoResearcher
    researcher = AutoResearcher()
    researcher.run(max_experiments=50)
"""

from __future__ import annotations

import signal
import sys
import time
from dataclasses import dataclass, field

from ..rl.training.mappo import MAPPOConfig
from .evaluator import Evaluator, TrialResult
from .experiment import ExperimentTracker
from .mutations import Mutation, MutationEngine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ResearchConfig:
    """Top-level configuration for the autonomous researcher.

    Parameters
    ----------
    time_budget_seconds : float
        Wall-clock seconds for each training trial.
    num_envs : int
        Number of parallel environments during trials (keep small for speed).
    max_experiments : int
        Stop after this many experiments (0 = run forever).
    device : str
        Torch device (``"cpu"`` or ``"cuda"``).
    results_dir : str
        Directory for results.tsv and best_config.json.
    exploration_c : float
        UCB1 exploration constant for the mutation engine.
    seed : int
        Random seed for reproducibility.
    min_improvement : float
        Minimum win_rate delta required to accept a mutation (guards against
        noise declaring marginal changes as improvements).
    """

    time_budget_seconds: float = 300.0   # 5 min per trial
    num_envs: int = 4
    max_experiments: int = 0             # 0 = unlimited
    device: str = "cpu"
    results_dir: str = "models/autoresearch"
    exploration_c: float = 1.0
    seed: int = 42
    min_improvement: float = 0.01        # require ≥1% win_rate gain


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------

class AutoResearcher:
    """Autonomous MAPPO hyperparameter search.

    Parameters
    ----------
    research_config : ResearchConfig | None
        Settings for the research loop.  Defaults to :class:`ResearchConfig`.
    base_mappo_config : MAPPOConfig | None
        Starting point for the search.  If ``None``, uses :class:`MAPPOConfig`
        defaults (which may be overridden by a previously saved best config).
    """

    def __init__(
        self,
        research_config: ResearchConfig | None = None,
        base_mappo_config: MAPPOConfig | None = None,
    ) -> None:
        self.rc = research_config or ResearchConfig()

        self.tracker = ExperimentTracker(
            results_path=f"{self.rc.results_dir}/results.tsv",
            best_config_path=f"{self.rc.results_dir}/best_config.json",
        )

        default_cfg = base_mappo_config or MAPPOConfig()
        self.best_config: MAPPOConfig = self.tracker.load_best_config(default_cfg)

        self.evaluator = Evaluator(
            device=self.rc.device,
            time_budget_seconds=self.rc.time_budget_seconds,
            num_envs=self.rc.num_envs,
            checkpoint_dir=f"{self.rc.results_dir}/trial_checkpoints",
        )

        self.mutation_engine = MutationEngine(
            seed=self.rc.seed,
            exploration_c=self.rc.exploration_c,
        )

        self._stop_requested = False
        self._experiment_count = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, max_experiments: int | None = None) -> None:
        """Run the autonomous research loop.

        Parameters
        ----------
        max_experiments : int | None
            Override :attr:`ResearchConfig.max_experiments` for this call.
            ``None`` uses the value from ``research_config``.
        """
        limit = max_experiments if max_experiments is not None else self.rc.max_experiments

        # Graceful Ctrl-C handling
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

        print("=" * 72)
        print("AutoResearcher starting")
        print(f"  device           = {self.rc.device}")
        print(f"  time_budget      = {self.rc.time_budget_seconds:.0f}s per trial")
        print(f"  num_envs         = {self.rc.num_envs}")
        print(f"  min_improvement  = {self.rc.min_improvement:.4f}")
        print(f"  max_experiments  = {limit if limit > 0 else '∞'}")
        print(f"  results_dir      = {self.rc.results_dir}")
        print(f"  baseline win_rate = {self.tracker.best_win_rate:.4f}")
        print("=" * 72)

        try:
            while not self._stop_requested:
                if limit > 0 and self._experiment_count >= limit:
                    print(f"\nReached max_experiments={limit}. Stopping.")
                    break

                self._run_one_experiment()

        finally:
            signal.signal(signal.SIGINT, original_sigint)
            self._print_final_summary()

    # ------------------------------------------------------------------

    def _run_one_experiment(self) -> None:
        self._experiment_count += 1
        exp_num = self._experiment_count
        best_before = self.tracker.best_win_rate

        # 1. Propose mutation
        mutation = self.mutation_engine.propose(self.best_config)
        candidate_config = mutation.apply(self.best_config)

        print(f"\n[Exp {exp_num}] {mutation}")
        print(f"  best_win_rate_so_far = {best_before:.4f}")

        # 2. Run trial
        t0 = time.monotonic()
        result = self.evaluator.run_trial(candidate_config)
        elapsed = time.monotonic() - t0

        if not result.ok:
            print(f"  [Exp {exp_num}] TRIAL ERROR: {result.error}")
            self.mutation_engine.record(mutation, win_rate_delta=-1.0)
            self.tracker.record(mutation, result, is_best=False)
            return

        delta = result.win_rate - best_before
        improved = delta >= self.rc.min_improvement

        print(
            f"  win_rate={result.win_rate:.4f}  "
            f"Δ={delta:+.4f}  "
            f"reward={result.mean_reward:.4f}  "
            f"updates={result.num_updates}  "
            f"steps={result.global_step}  "
            f"elapsed={elapsed:.0f}s"
        )

        # 3. Keep or discard
        if improved:
            print(f"  ✓ IMPROVEMENT — accepting mutation ({mutation.param}={mutation.new_value})")
            self.best_config = candidate_config
            self.tracker.save_best_config(self.best_config)
        else:
            action = "no improvement" if delta >= 0 else "regression"
            print(f"  ✗ {action} — discarding mutation, keeping best config")

        # 4. Update UCB1 stats
        self.mutation_engine.record(mutation, win_rate_delta=delta)

        # 5. Log to TSV
        self.tracker.record(mutation, result, is_best=improved)

    # ------------------------------------------------------------------

    def _handle_sigint(self, signum: int, frame: object) -> None:
        print("\n[AutoResearcher] Interrupt received — finishing current trial…")
        self._stop_requested = True

    def _print_final_summary(self) -> None:
        print("\n" + "=" * 72)
        print(f"AutoResearcher finished after {self._experiment_count} experiments.")
        print(f"Best win_rate: {self.tracker.best_win_rate:.4f}")
        print("\nTop mutations by mean Δwin_rate:")
        for row in self.mutation_engine.summary()[:10]:
            print(
                f"  {row['param']:>18} = {str(row['value']):>8}  "
                f"n={row['n_trials']}  μΔ={row['mean_delta']:+.4f}"
            )
        print("\nRecent trials:")
        print(self.tracker.summary(n=10))
        print(f"\nResults log: {self.tracker.results_path}")
        print(f"Best config: {self.tracker.best_config_path}")
        print("=" * 72)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous MAPPO hyperparameter search")
    parser.add_argument("--budget", type=float, default=300.0,
                        help="Seconds per trial (default: 300)")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Parallel envs per trial (default: 4)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-experiments", type=int, default=0,
                        help="Stop after N experiments; 0=unlimited (default: 0)")
    parser.add_argument("--results-dir", type=str, default="models/autoresearch")
    parser.add_argument("--exploration-c", type=float, default=1.0)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rc = ResearchConfig(
        time_budget_seconds=args.budget,
        num_envs=args.num_envs,
        device=args.device,
        max_experiments=args.max_experiments,
        results_dir=args.results_dir,
        exploration_c=args.exploration_c,
        min_improvement=args.min_improvement,
        seed=args.seed,
    )
    AutoResearcher(research_config=rc).run()


if __name__ == "__main__":
    main()
