"""Tests for the CatanRL Gymnasium environment."""

from __future__ import annotations

import numpy as np
import pytest
from src.rl.env.action_space import ActionSpace
from src.rl.env.catan_env import MAX_TURNS, CatanEnv

from tests.conftest import step_through_setup


class TestReset:
    """Tests for env.reset()."""

    def test_returns_obs_and_info(self, env: CatanEnv):
        obs, info = env.reset(seed=123)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_observation_keys(self, env: CatanEnv):
        obs, _ = env.reset(seed=123)
        expected_keys = {
            "hex_features",
            "vertex_features",
            "edge_features",
            "player_features",
            "action_mask",
            "current_player",
            "game_phase",
        }
        assert set(obs.keys()) == expected_keys

    def test_hex_features_shape(self, env: CatanEnv):
        obs, _ = env.reset(seed=123)
        assert obs["hex_features"].shape == (19, 9)

    def test_vertex_features_shape(self, env: CatanEnv):
        obs, _ = env.reset(seed=123)
        assert obs["vertex_features"].shape == (54, 7)  # 3 + num_players

    def test_edge_features_shape(self, env: CatanEnv):
        obs, _ = env.reset(seed=123)
        assert obs["edge_features"].shape == (72, 5)  # 1 + num_players

    def test_player_features_shape(self, env: CatanEnv):
        obs, _ = env.reset(seed=123)
        assert obs["player_features"].shape == (4, 14)

    def test_action_mask_shape(self, env: CatanEnv):
        obs, _ = env.reset(seed=123)
        assert obs["action_mask"].shape == (ActionSpace.TOTAL_ACTIONS,)

    def test_initial_phase_is_setup_first(self, env: CatanEnv):
        obs, _ = env.reset(seed=123)
        assert obs["game_phase"] == CatanEnv.PHASE_SETUP_FIRST
        assert env.game_phase == CatanEnv.PHASE_SETUP_FIRST


class TestInitialActionMask:

    def test_has_at_least_one_valid_action(self, env: CatanEnv):
        obs, _ = env.reset(seed=42)
        mask = obs["action_mask"]
        assert mask.sum() > 0

    def test_initial_mask_only_settlements(self, env: CatanEnv):
        """In SETUP_FIRST with sub_phase=0, only settlement placements are legal."""
        obs, _ = env.reset(seed=42)
        mask = obs["action_mask"]
        # Should only have settlement actions
        for i in range(ActionSpace.TOTAL_ACTIONS):
            if mask[i]:
                action_type, _ = ActionSpace.decode_action(i)
                assert action_type == "BUILD_SETTLEMENT"


class TestSetupPhase:

    def test_place_settlement_then_road(self, env: CatanEnv):
        """Can place a settlement, then must place a road."""
        obs, _ = env.reset(seed=42)
        mask = obs["action_mask"]

        # Place a settlement
        settle_actions = np.where(mask)[0]
        assert len(settle_actions) > 0
        action = int(settle_actions[0])
        obs, reward, terminated, truncated, info = env.step(action)
        assert not terminated
        assert not truncated

        # Now should be in road sub-phase
        assert env.setup_sub_phase == 1
        new_mask = obs["action_mask"]
        for i in np.where(new_mask)[0]:
            action_type, _ = ActionSpace.decode_action(i)
            assert action_type == "BUILD_ROAD"

    def test_setup_progresses_through_phases(self, env: CatanEnv):
        """After completing all setup, should reach ROLL phase."""
        np.random.seed(42)
        step_through_setup(env)
        assert env.game_phase == CatanEnv.PHASE_ROLL


class TestStep:

    def test_step_with_valid_action_no_crash(self, env: CatanEnv):
        obs, _ = env.reset(seed=42)
        mask = obs["action_mask"]
        valid = np.where(mask)[0]
        obs2, reward, terminated, truncated, info = env.step(int(valid[0]))
        assert isinstance(obs2, dict)
        assert isinstance(reward, float)

    def test_illegal_action_penalty(self, env: CatanEnv):
        """Illegal actions should return -1.0 reward."""
        obs, _ = env.reset(seed=42)
        mask = obs["action_mask"]
        # Find an illegal action
        illegal = np.where(~mask.astype(bool))[0]
        assert len(illegal) > 0
        _, reward, _, _, _ = env.step(int(illegal[0]))
        assert reward == -1.0


class TestResourceGeneration:

    def test_dice_roll_generates_resources(self, env: CatanEnv):
        """After setup, rolling the dice should potentially generate resources."""
        np.random.seed(42)
        step_through_setup(env)
        assert env.game_phase == CatanEnv.PHASE_ROLL

        # Roll the dice
        roll_action = ActionSpace.encode_action("ROLL_DICE", 0)
        env.step(roll_action)

        # After roll, we should be in MAIN or ROBBER_PLACE/DISCARD
        assert env.game_phase in (
            CatanEnv.PHASE_MAIN,
            CatanEnv.PHASE_ROBBER_PLACE,
            CatanEnv.PHASE_DISCARD,
        )


class TestBuildingCosts:

    def test_road_cost_deducted(self, env: CatanEnv):
        """Building a road in MAIN phase should deduct resources."""
        np.random.seed(42)
        step_through_setup(env)

        # Give player 0 enough resources
        env.current_player = 0
        env.game_phase = CatanEnv.PHASE_MAIN
        env.player_resources[0] = np.array([10, 10, 10, 10, 10], dtype=np.int32)

        mask = env.get_action_mask()
        road_actions = []
        for i in range(ActionSpace.BUILD_ROAD_OFFSET, ActionSpace.BUILD_ROAD_OFFSET + 72):
            if mask[i]:
                road_actions.append(i)

        if road_actions:
            resources_before = env.player_resources[0].copy()
            env.step(road_actions[0])
            # Road costs 1 wood + 1 brick
            assert env.player_resources[0, 0] == resources_before[0] - 1  # wood
            assert env.player_resources[0, 1] == resources_before[1] - 1  # brick

    def test_settlement_cost_deducted(self, env: CatanEnv):
        """Building a settlement in MAIN phase should deduct resources."""
        np.random.seed(42)
        step_through_setup(env)

        env.current_player = 0
        env.game_phase = CatanEnv.PHASE_MAIN
        env.player_resources[0] = np.array([10, 10, 10, 10, 10], dtype=np.int32)

        mask = env.get_action_mask()
        settle_actions = []
        settle_end = ActionSpace.BUILD_SETTLEMENT_OFFSET + 54
        for i in range(ActionSpace.BUILD_SETTLEMENT_OFFSET, settle_end):
            if mask[i]:
                settle_actions.append(i)

        if settle_actions:
            resources_before = env.player_resources[0].copy()
            env.step(settle_actions[0])
            # Settlement costs 1 wood + 1 brick + 1 sheep + 1 wheat
            assert env.player_resources[0, 0] == resources_before[0] - 1  # wood
            assert env.player_resources[0, 1] == resources_before[1] - 1  # brick
            assert env.player_resources[0, 2] == resources_before[2] - 1  # sheep
            assert env.player_resources[0, 3] == resources_before[3] - 1  # wheat


class TestVictoryAndTruncation:

    def test_victory_at_10_vp(self, env: CatanEnv):
        """Game should terminate when a player reaches 10 VP."""
        np.random.seed(42)
        step_through_setup(env)

        # Force player 0 to 9 VP and let them gain 1 more
        env.current_player = 0
        env.game_phase = CatanEnv.PHASE_MAIN
        env.player_vp[0] = 9
        env.player_resources[0] = np.array([10, 10, 10, 10, 10], dtype=np.int32)

        # Find a settlement action to gain 1 VP
        mask = env.get_action_mask()
        settle_actions = []
        settle_end = ActionSpace.BUILD_SETTLEMENT_OFFSET + 54
        for i in range(ActionSpace.BUILD_SETTLEMENT_OFFSET, settle_end):
            if mask[i]:
                settle_actions.append(i)

        if settle_actions:
            _, reward, terminated, _, info = env.step(settle_actions[0])
            assert terminated
            assert env.winner == 0
            assert env.game_phase == CatanEnv.PHASE_GAME_OVER

    def test_max_turn_truncation(self, env: CatanEnv):
        """Game should truncate at MAX_TURNS."""
        np.random.seed(42)
        step_through_setup(env)

        # Force turn counter near max
        env.turn_counter = MAX_TURNS
        env.game_phase = CatanEnv.PHASE_MAIN
        env.current_player = 0

        # End turn should trigger truncation
        end_action = ActionSpace.encode_action("END_TURN", 0)
        _, _, terminated, truncated, _ = env.step(end_action)
        # After END_TURN the turn counter advances, so now >= MAX_TURNS
        assert truncated or terminated


class TestBoardRandomization:

    def test_two_resets_different_boards(self):
        """Resetting with different seeds should produce different board layouts."""
        env1 = CatanEnv(num_players=4)
        env1.reset(seed=1)
        hex_types_1 = env1.hex_types.copy()

        env2 = CatanEnv(num_players=4)
        env2.reset(seed=2)
        hex_types_2 = env2.hex_types.copy()

        # Very unlikely to be identical with different seeds
        assert not np.array_equal(hex_types_1, hex_types_2)


class TestFullGameLoop:
    """Run a full game loop to make sure nothing crashes."""

    def test_random_game_terminates(self):
        """A game with random valid actions should eventually end."""
        env = CatanEnv(num_players=4)
        env.reset(seed=42)
        np.random.seed(42)

        max_steps = 5000
        for _ in range(max_steps):
            mask = env.get_action_mask()
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            action = int(np.random.choice(valid))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        else:
            pytest.fail("Game did not terminate within 5000 steps")
