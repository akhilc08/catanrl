"""Curriculum learning scheduler for CatanRL.

Gradually increases game complexity across training phases, from simplified
2-player games with restricted features to full 4-player games with all
mechanics enabled and self-play opponents.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase definition
# ---------------------------------------------------------------------------

@dataclass
class CurriculumPhase:
    """Defines a single curriculum stage with env and opponent settings.

    Parameters
    ----------
    name : str
        Short identifier (e.g. ``"basics"``).
    description : str
        Human-readable explanation of what the agent should learn.
    num_players : int
        Number of players in the environment.
    enable_trading : bool
        Whether inter-player and bank trading is allowed.
    enable_dev_cards : bool
        Whether development cards are available.
    enable_robber : bool
        Whether the robber mechanic is active.
    max_turns : int
        Maximum turns before the game is truncated.
    opponent_type : str
        ``"random"``, ``"heuristic"``, or ``"self_play"``.
    reward_scale : float
        Multiplier applied to environment rewards.
    """

    name: str
    description: str
    num_players: int = 4
    enable_trading: bool = True
    enable_dev_cards: bool = True
    enable_robber: bool = True
    max_turns: int = 500
    opponent_type: str = "random"
    reward_scale: float = 1.0


# ---------------------------------------------------------------------------
# Curriculum config
# ---------------------------------------------------------------------------

@dataclass
class CurriculumConfig:
    """Top-level configuration for curriculum scheduling.

    Parameters
    ----------
    phase_advance_threshold : float
        Win rate (0-1) the agent must achieve against the current difficulty
        before advancing to the next phase.
    eval_window : int
        Number of completed games used to compute the rolling win rate.
    phases : list[CurriculumPhase]
        Ordered list of curriculum phases.
    """

    phase_advance_threshold: float = 0.6
    eval_window: int = 100
    phases: list[CurriculumPhase] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """Manages progression through curriculum phases based on win rate.

    Parameters
    ----------
    config : CurriculumConfig
        Configuration with phase definitions and advancement criteria.
    """

    def __init__(self, config: CurriculumConfig) -> None:
        if not config.phases:
            raise ValueError("CurriculumConfig.phases must contain at least one phase.")
        self.config = config
        self.current_phase_idx: int = 0
        # Rolling window of game results (True = win, False = loss)
        self._results: deque[bool] = deque(maxlen=config.eval_window)
        # Lifetime stats per phase
        self.phase_stats: list[dict] = [
            {"games": 0, "wins": 0} for _ in config.phases
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> CurriculumPhase:
        """Return the active curriculum phase."""
        return self.config.phases[self.current_phase_idx]

    def get_current_phase(self) -> CurriculumPhase:
        """Return the active curriculum phase."""
        return self.current_phase

    @property
    def is_final_phase(self) -> bool:
        """True if the scheduler is on the last phase."""
        return self.current_phase_idx >= len(self.config.phases) - 1

    @property
    def win_rate(self) -> float:
        """Rolling win rate over the evaluation window."""
        if not self._results:
            return 0.0
        return sum(self._results) / len(self._results)

    def report_result(self, win: bool) -> bool:
        """Report the outcome of a completed game.

        Parameters
        ----------
        win : bool
            Whether the training agent won the game.

        Returns
        -------
        bool
            True if the phase advanced as a result of this report.
        """
        self._results.append(win)
        stats = self.phase_stats[self.current_phase_idx]
        stats["games"] += 1
        if win:
            stats["wins"] += 1

        if self.should_advance():
            return self.advance() is not None
        return False

    def should_advance(self) -> bool:
        """Check whether the advancement criteria are met.

        Requires at least ``eval_window`` games completed and a rolling
        win rate at or above ``phase_advance_threshold``.
        """
        if self.is_final_phase:
            return False
        if len(self._results) < self.config.eval_window:
            return False
        return self.win_rate >= self.config.phase_advance_threshold

    def advance(self) -> CurriculumPhase | None:
        """Advance to the next phase if one exists.

        Returns
        -------
        CurriculumPhase or None
            The new phase, or ``None`` if already at the final phase.
        """
        if self.is_final_phase:
            return None

        self.current_phase_idx += 1
        # Clear the rolling window so the agent must prove itself anew
        self._results.clear()

        new_phase = self.current_phase
        logger.info(
            "Curriculum advanced to phase %d/%d: '%s' — %s",
            self.current_phase_idx + 1,
            len(self.config.phases),
            new_phase.name,
            new_phase.description,
        )
        return new_phase

    def get_env_kwargs(self) -> dict:
        """Build keyword arguments for :class:`CatanEnv` from the current phase.

        The env may not support all keys yet — callers should pass them via
        ``**kwargs`` and let the env ignore unsupported ones.
        """
        phase = self.current_phase
        return {
            "num_players": phase.num_players,
            "enable_trading": phase.enable_trading,
            "enable_dev_cards": phase.enable_dev_cards,
            "enable_robber": phase.enable_robber,
            "max_turns": phase.max_turns,
        }

    def get_metrics(self) -> dict:
        """Return a dict of loggable metrics about curriculum state."""
        phase = self.current_phase
        return {
            "curriculum/phase_idx": self.current_phase_idx,
            "curriculum/phase_name": phase.name,
            "curriculum/win_rate": self.win_rate,
            "curriculum/games_in_window": len(self._results),
            "curriculum/total_games_this_phase": self.phase_stats[self.current_phase_idx]["games"],
            "curriculum/opponent_type": phase.opponent_type,
        }

    def state_dict(self) -> dict:
        """Serialize scheduler state for checkpointing."""
        return {
            "current_phase_idx": self.current_phase_idx,
            "results": list(self._results),
            "phase_stats": self.phase_stats,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler state from a checkpoint."""
        self.current_phase_idx = state["current_phase_idx"]
        self._results = deque(state["results"], maxlen=self.config.eval_window)
        self.phase_stats = state["phase_stats"]


# ---------------------------------------------------------------------------
# Default curriculum factory
# ---------------------------------------------------------------------------

def default_curriculum() -> CurriculumConfig:
    """Return a sensible default 5-phase curriculum for CatanRL.

    Phases
    ------
    1. **basics** — 2 players, no trading/dev cards/robber, random opponents.
       Learn settlement placement and resource management.
    2. **robber** — 2 players, robber enabled, random opponents.
       Learn to cope with the robber and 7-rolls.
    3. **dev_cards** — 2 players, dev cards + robber, heuristic opponents.
       Learn development card strategy against a stronger opponent.
    4. **trading** — 4 players, all features, heuristic opponents.
       Learn trading and multi-player dynamics.
    5. **full_game** — 4 players, all features, self-play opponents.
       Competitive play against past versions of itself.
    """
    phases = [
        CurriculumPhase(
            name="basics",
            description="Learn building and resource management (2p, no extras, random opponent)",
            num_players=2,
            enable_trading=False,
            enable_dev_cards=False,
            enable_robber=False,
            max_turns=300,
            opponent_type="random",
            reward_scale=1.0,
        ),
        CurriculumPhase(
            name="robber",
            description="Learn robber mechanics (2p, robber enabled, random opponent)",
            num_players=2,
            enable_trading=False,
            enable_dev_cards=False,
            enable_robber=True,
            max_turns=400,
            opponent_type="random",
            reward_scale=1.0,
        ),
        CurriculumPhase(
            name="dev_cards",
            description="Learn dev card strategy (2p, dev cards + robber, heuristic opponent)",
            num_players=2,
            enable_trading=False,
            enable_dev_cards=True,
            enable_robber=True,
            max_turns=400,
            opponent_type="heuristic",
            reward_scale=1.0,
        ),
        CurriculumPhase(
            name="trading",
            description=(
                "Learn trading and multiplayer dynamics "
                "(4p, all features, heuristic opponents)"
            ),
            num_players=4,
            enable_trading=True,
            enable_dev_cards=True,
            enable_robber=True,
            max_turns=500,
            opponent_type="heuristic",
            reward_scale=1.0,
        ),
        CurriculumPhase(
            name="full_game",
            description="Competitive self-play (4p, all features, self-play opponents)",
            num_players=4,
            enable_trading=True,
            enable_dev_cards=True,
            enable_robber=True,
            max_turns=500,
            opponent_type="self_play",
            reward_scale=1.0,
        ),
    ]
    return CurriculumConfig(
        phase_advance_threshold=0.6,
        eval_window=100,
        phases=phases,
    )
