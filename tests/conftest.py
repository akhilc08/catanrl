"""Shared pytest fixtures for CatanRL tests."""

from __future__ import annotations

import numpy as np
import pytest
from src.rl.env.action_space import ActionSpace
from src.rl.env.catan_env import CatanEnv


@pytest.fixture
def env() -> CatanEnv:
    """Return a fresh CatanEnv reset with a fixed seed."""
    e = CatanEnv(num_players=4)
    e.reset(seed=42)
    return e


@pytest.fixture
def env_unseeded() -> CatanEnv:
    """Return a CatanEnv without a fixed seed (for randomization tests)."""
    e = CatanEnv(num_players=4)
    e.reset()
    return e


@pytest.fixture
def action_space() -> ActionSpace:
    """Return an ActionSpace instance."""
    return ActionSpace()


def step_through_setup(env: CatanEnv) -> None:
    """Helper: play through the entire setup phase with random valid actions."""
    while env.game_phase in (env.PHASE_SETUP_FIRST, env.PHASE_SETUP_SECOND):
        mask = env.get_action_mask()
        valid_actions = np.where(mask)[0]
        if len(valid_actions) == 0:
            break
        action = int(np.random.choice(valid_actions))
        env.step(action)


def take_random_valid_action(env: CatanEnv) -> tuple[dict, float, bool, bool, dict]:
    """Take a single random valid action and return step result."""
    mask = env.get_action_mask()
    valid_actions = np.where(mask)[0]
    assert len(valid_actions) > 0, "No valid actions available"
    action = int(np.random.choice(valid_actions))
    return env.step(action)
