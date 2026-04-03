"""Tests for the CurriculumScheduler."""

from __future__ import annotations

import pytest
from src.rl.training.curriculum import (
    CurriculumConfig,
    CurriculumPhase,
    CurriculumScheduler,
    default_curriculum,
)


@pytest.fixture
def config() -> CurriculumConfig:
    """Return the default 5-phase curriculum config."""
    return default_curriculum()


@pytest.fixture
def scheduler(config: CurriculumConfig) -> CurriculumScheduler:
    """Return a scheduler with the default config."""
    return CurriculumScheduler(config)


class TestDefaultCurriculum:

    def test_returns_5_phases(self):
        cfg = default_curriculum()
        assert len(cfg.phases) == 5

    def test_phase_names(self):
        cfg = default_curriculum()
        names = [p.name for p in cfg.phases]
        assert names == ["basics", "robber", "dev_cards", "trading", "full_game"]

    def test_first_phase_is_2_players(self):
        cfg = default_curriculum()
        assert cfg.phases[0].num_players == 2

    def test_last_phase_is_self_play(self):
        cfg = default_curriculum()
        assert cfg.phases[-1].opponent_type == "self_play"

    def test_threshold_and_window(self):
        cfg = default_curriculum()
        assert cfg.phase_advance_threshold == 0.6
        assert cfg.eval_window == 100


class TestSchedulerInit:

    def test_starts_at_phase_0(self, scheduler: CurriculumScheduler):
        assert scheduler.current_phase_idx == 0

    def test_current_phase_is_basics(self, scheduler: CurriculumScheduler):
        assert scheduler.current_phase.name == "basics"

    def test_not_final_phase(self, scheduler: CurriculumScheduler):
        assert not scheduler.is_final_phase

    def test_initial_win_rate_is_zero(self, scheduler: CurriculumScheduler):
        assert scheduler.win_rate == 0.0

    def test_empty_phases_raises(self):
        with pytest.raises(ValueError, match="at least one phase"):
            CurriculumScheduler(CurriculumConfig(phases=[]))


class TestReportResult:

    def test_advances_when_threshold_met(self, scheduler: CurriculumScheduler):
        """Report enough wins to trigger advancement."""
        # Need eval_window (100) results with >= 60% win rate
        for _ in range(100):
            scheduler.report_result(win=True)

        # Should have advanced
        assert scheduler.current_phase_idx == 1

    def test_no_advance_below_threshold(self, scheduler: CurriculumScheduler):
        """Report results below threshold -- should not advance."""
        for _ in range(100):
            scheduler.report_result(win=False)

        assert scheduler.current_phase_idx == 0

    def test_no_advance_without_full_window(self, scheduler: CurriculumScheduler):
        """Even with all wins, need full eval_window games."""
        for _ in range(50):  # only 50, less than eval_window=100
            scheduler.report_result(win=True)

        assert scheduler.current_phase_idx == 0

    def test_does_not_advance_past_last_phase(self):
        """Advancing at the final phase should be a no-op."""
        cfg = CurriculumConfig(
            phase_advance_threshold=0.6,
            eval_window=10,
            phases=[
                CurriculumPhase(name="only_phase", description="The only phase"),
            ],
        )
        sched = CurriculumScheduler(cfg)
        assert sched.is_final_phase

        for _ in range(20):
            advanced = sched.report_result(win=True)
            assert not advanced

        assert sched.current_phase_idx == 0

    def test_report_result_returns_true_on_advance(self, scheduler: CurriculumScheduler):
        advanced_ever = False
        for _ in range(100):
            result = scheduler.report_result(win=True)
            if result:
                advanced_ever = True
        assert advanced_ever

    def test_multiple_advances(self, scheduler: CurriculumScheduler):
        """Can advance through multiple phases."""
        for phase_target in range(1, 5):
            for _ in range(100):
                scheduler.report_result(win=True)
            assert scheduler.current_phase_idx == phase_target


class TestGetEnvKwargs:

    def test_returns_dict(self, scheduler: CurriculumScheduler):
        kwargs = scheduler.get_env_kwargs()
        assert isinstance(kwargs, dict)

    def test_contains_expected_keys(self, scheduler: CurriculumScheduler):
        kwargs = scheduler.get_env_kwargs()
        expected = {
            "num_players", "enable_trading", "enable_dev_cards",
            "enable_robber", "max_turns",
        }
        assert set(kwargs.keys()) == expected

    def test_phase_0_values(self, scheduler: CurriculumScheduler):
        kwargs = scheduler.get_env_kwargs()
        assert kwargs["num_players"] == 2
        assert kwargs["enable_trading"] is False
        assert kwargs["enable_dev_cards"] is False
        assert kwargs["enable_robber"] is False
        assert kwargs["max_turns"] == 300


class TestStateDictRoundtrip:

    def test_save_load_state(self, scheduler: CurriculumScheduler):
        # Report some results
        for _ in range(50):
            scheduler.report_result(win=True)

        state = scheduler.state_dict()
        assert "current_phase_idx" in state
        assert "results" in state

        # Create a new scheduler and load state
        config = default_curriculum()
        sched2 = CurriculumScheduler(config)
        sched2.load_state_dict(state)

        assert sched2.current_phase_idx == scheduler.current_phase_idx
        assert sched2.win_rate == scheduler.win_rate
