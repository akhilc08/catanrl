"""Action space encoding and masking for the Catan RL environment.

Flat integer action encoding with boolean masking for legal actions.

Action Categories and Offsets:
    ROLL_DICE:          1 action   (offset 0)
    END_TURN:           1 action   (offset 1)
    BUILD_ROAD:        72 actions  (offset 2,   one per edge)
    BUILD_SETTLEMENT:  54 actions  (offset 74,  one per vertex)
    BUILD_CITY:        54 actions  (offset 128, one per vertex)
    BUY_DEV_CARD:       1 action   (offset 182)
    PLAY_KNIGHT:       19 actions  (offset 183, one per hex)
    PLAY_ROAD_BUILDING: 1 action   (offset 202)
    PLAY_YEAR_OF_PLENTY: 5 actions (offset 203, one per resource)
    PLAY_MONOPOLY:      5 actions  (offset 208, one per resource)
    BANK_TRADE:        20 actions  (offset 213, give_res * 4 + get_res pairs)
    PLACE_ROBBER:      19 actions  (offset 233, one per hex)
    STEAL_FROM_PLAYER:  4 actions  (offset 252, one per player)
    DISCARD_RESOURCE:   5 actions  (offset 256, one per resource)

    Total: 261 actions
"""

from __future__ import annotations

import numpy as np


class ActionSpace:
    """Flat integer action space with masking for Catan."""

    # Category sizes
    _SIZES = {
        "ROLL_DICE": 1,
        "END_TURN": 1,
        "BUILD_ROAD": 72,
        "BUILD_SETTLEMENT": 54,
        "BUILD_CITY": 54,
        "BUY_DEV_CARD": 1,
        "PLAY_KNIGHT": 19,
        "PLAY_ROAD_BUILDING": 1,
        "PLAY_YEAR_OF_PLENTY": 5,
        "PLAY_MONOPOLY": 5,
        "BANK_TRADE": 20,
        "PLACE_ROBBER": 19,
        "STEAL_FROM_PLAYER": 4,
        "DISCARD_RESOURCE": 5,
    }

    # Computed offsets
    ROLL_DICE_OFFSET = 0
    END_TURN_OFFSET = 1
    BUILD_ROAD_OFFSET = 2
    BUILD_SETTLEMENT_OFFSET = 74
    BUILD_CITY_OFFSET = 128
    BUY_DEV_CARD_OFFSET = 182
    PLAY_KNIGHT_OFFSET = 183
    PLAY_ROAD_BUILDING_OFFSET = 202
    PLAY_YEAR_OF_PLENTY_OFFSET = 203
    PLAY_MONOPOLY_OFFSET = 208
    BANK_TRADE_OFFSET = 213
    PLACE_ROBBER_OFFSET = 233
    STEAL_FROM_PLAYER_OFFSET = 252
    DISCARD_RESOURCE_OFFSET = 256

    TOTAL_ACTIONS = 261  # 256 + 5

    # Ordered list of (name, offset, size) for decode
    _CATEGORIES: list[tuple[str, int, int]] = [
        ("ROLL_DICE", ROLL_DICE_OFFSET, 1),
        ("END_TURN", END_TURN_OFFSET, 1),
        ("BUILD_ROAD", BUILD_ROAD_OFFSET, 72),
        ("BUILD_SETTLEMENT", BUILD_SETTLEMENT_OFFSET, 54),
        ("BUILD_CITY", BUILD_CITY_OFFSET, 54),
        ("BUY_DEV_CARD", BUY_DEV_CARD_OFFSET, 1),
        ("PLAY_KNIGHT", PLAY_KNIGHT_OFFSET, 19),
        ("PLAY_ROAD_BUILDING", PLAY_ROAD_BUILDING_OFFSET, 1),
        ("PLAY_YEAR_OF_PLENTY", PLAY_YEAR_OF_PLENTY_OFFSET, 5),
        ("PLAY_MONOPOLY", PLAY_MONOPOLY_OFFSET, 5),
        ("BANK_TRADE", BANK_TRADE_OFFSET, 20),
        ("PLACE_ROBBER", PLACE_ROBBER_OFFSET, 19),
        ("STEAL_FROM_PLAYER", STEAL_FROM_PLAYER_OFFSET, 4),
        ("DISCARD_RESOURCE", DISCARD_RESOURCE_OFFSET, 5),
    ]

    # Resource indices
    WOOD = 0
    BRICK = 1
    SHEEP = 2
    WHEAT = 3
    ORE = 4

    # Building costs: [wood, brick, sheep, wheat, ore]
    ROAD_COST = np.array([1, 1, 0, 0, 0], dtype=np.int32)
    SETTLEMENT_COST = np.array([1, 1, 1, 1, 0], dtype=np.int32)
    CITY_COST = np.array([0, 0, 0, 2, 3], dtype=np.int32)
    DEV_CARD_COST = np.array([0, 0, 1, 1, 1], dtype=np.int32)

    @staticmethod
    def decode_action(action_id: int) -> tuple[str, int]:
        """Decode a flat action id into (action_type, parameter).

        Parameters
        ----------
        action_id : int
            Flat action index in [0, TOTAL_ACTIONS).

        Returns
        -------
        tuple[str, int]
            (action_type_name, parameter_within_category)
        """
        for name, offset, size in reversed(ActionSpace._CATEGORIES):
            if action_id >= offset:
                param = action_id - offset
                if param < size:
                    return name, param
                break
        raise ValueError(f"Invalid action_id: {action_id}")

    @staticmethod
    def encode_action(action_type: str, parameter: int = 0) -> int:
        """Encode an action type and parameter into a flat action id.

        Parameters
        ----------
        action_type : str
            One of the category names (e.g. "BUILD_ROAD").
        parameter : int
            Index within that category.

        Returns
        -------
        int
            Flat action index.
        """
        offset_attr = f"{action_type}_OFFSET"
        offset = getattr(ActionSpace, offset_attr, None)
        if offset is None:
            raise ValueError(f"Unknown action type: {action_type}")
        size = ActionSpace._SIZES[action_type]
        if not 0 <= parameter < size:
            raise ValueError(
                f"Parameter {parameter} out of range [0, {size}) for {action_type}"
            )
        return offset + parameter

    @staticmethod
    def get_action_mask(env) -> np.ndarray:
        """Return a boolean mask of legal actions for the current game state.

        Parameters
        ----------
        env : CatanEnv
            The environment instance (access its state attributes directly).

        Returns
        -------
        np.ndarray
            Boolean array of shape (TOTAL_ACTIONS,).
        """
        mask = np.zeros(ActionSpace.TOTAL_ACTIONS, dtype=np.bool_)
        phase = env.game_phase
        cp = env.current_player
        res = env.player_resources[cp]

        # ----------------------------------------------------------------
        # SETUP phases: place settlement then road
        # ----------------------------------------------------------------
        if phase in (env.PHASE_SETUP_FIRST, env.PHASE_SETUP_SECOND):
            if env.setup_sub_phase == 0:
                # Place settlement
                for v in range(54):
                    if env.vertex_building[v] == 0 and env._check_distance_rule(v):
                        # In setup, no connectivity requirement
                        mask[ActionSpace.BUILD_SETTLEMENT_OFFSET + v] = True
            else:
                # Place road adjacent to last placed settlement
                last_settle = env.last_setup_settlement
                for e in range(72):
                    v0, v1 = env.edge_vertices[e]
                    if env.edge_road[e] == 0 and (v0 == last_settle or v1 == last_settle):
                        mask[ActionSpace.BUILD_ROAD_OFFSET + e] = True
            return mask

        # ----------------------------------------------------------------
        # DISCARD phase
        # ----------------------------------------------------------------
        if phase == env.PHASE_DISCARD:
            if env.discard_remaining[cp] > 0:
                for r in range(5):
                    if res[r] > 0:
                        mask[ActionSpace.DISCARD_RESOURCE_OFFSET + r] = True
            return mask

        # ----------------------------------------------------------------
        # ROLL phase
        # ----------------------------------------------------------------
        if phase == env.PHASE_ROLL:
            mask[ActionSpace.ROLL_DICE_OFFSET] = True
            # Can play a knight before rolling
            if (
                env.player_dev_cards[cp, 0] > 0  # knight
                and not env.has_played_dev_card_this_turn
                and env.dev_card_bought_this_turn_idx != 0
            ):
                for h in range(19):
                    if h != env.robber_hex:
                        mask[ActionSpace.PLAY_KNIGHT_OFFSET + h] = True
            return mask

        # ----------------------------------------------------------------
        # ROBBER_PLACE phase (after rolling 7 or playing knight)
        # ----------------------------------------------------------------
        if phase == env.PHASE_ROBBER_PLACE:
            for h in range(19):
                if h != env.robber_hex:
                    mask[ActionSpace.PLACE_ROBBER_OFFSET + h] = True
            return mask

        # ----------------------------------------------------------------
        # ROBBER_STEAL phase
        # ----------------------------------------------------------------
        if phase == env.PHASE_ROBBER_STEAL:
            hex_verts = env.hex_vertices[env.robber_hex]
            for p in range(env.num_players):
                if p == cp:
                    continue
                if env.player_resources[p].sum() == 0:
                    continue
                # Check if player has a building adjacent to robber hex
                for v in hex_verts:
                    if env.vertex_owner[v] == p + 1:
                        mask[ActionSpace.STEAL_FROM_PLAYER_OFFSET + p] = True
                        break
            # If nobody to steal from, allow stealing from self (no-op)
            if not mask[ActionSpace.STEAL_FROM_PLAYER_OFFSET:
                        ActionSpace.STEAL_FROM_PLAYER_OFFSET + 4].any():
                mask[ActionSpace.STEAL_FROM_PLAYER_OFFSET + cp] = True
            return mask

        # ----------------------------------------------------------------
        # MAIN phase
        # ----------------------------------------------------------------
        if phase == env.PHASE_MAIN:
            # End turn is always legal in main phase
            mask[ActionSpace.END_TURN_OFFSET] = True

            # -- BUILD ROAD --
            if np.all(res >= ActionSpace.ROAD_COST) and env.player_roads_left[cp] > 0:
                for e in range(72):
                    if env.edge_road[e] == 0 and env._can_place_road(cp, e):
                        mask[ActionSpace.BUILD_ROAD_OFFSET + e] = True

            # -- BUILD SETTLEMENT --
            if np.all(res >= ActionSpace.SETTLEMENT_COST) and env.player_settlements_left[cp] > 0:
                for v in range(54):
                    if (
                        env.vertex_building[v] == 0
                        and env._check_distance_rule(v)
                        and env._vertex_connected_to_road(cp, v)
                    ):
                        mask[ActionSpace.BUILD_SETTLEMENT_OFFSET + v] = True

            # -- BUILD CITY --
            if np.all(res >= ActionSpace.CITY_COST) and env.player_cities_left[cp] > 0:
                for v in range(54):
                    if env.vertex_building[v] == 1 and env.vertex_owner[v] == cp + 1:
                        mask[ActionSpace.BUILD_CITY_OFFSET + v] = True

            # -- BUY DEV CARD --
            has_cost = np.all(res >= ActionSpace.DEV_CARD_COST)
            has_cards = env.dev_card_deck_index < len(env.dev_card_deck)
            if has_cost and has_cards:
                mask[ActionSpace.BUY_DEV_CARD_OFFSET] = True

            # -- PLAY DEV CARDS (max 1 per turn, except VP) --
            if not env.has_played_dev_card_this_turn:
                # Knight
                if env.player_dev_cards[cp, 0] > 0 and env.dev_card_bought_this_turn_idx != 0:
                    for h in range(19):
                        if h != env.robber_hex:
                            mask[ActionSpace.PLAY_KNIGHT_OFFSET + h] = True

                # Road Building
                if env.player_dev_cards[cp, 2] > 0 and env.dev_card_bought_this_turn_idx != 2:
                    if env.player_roads_left[cp] > 0:
                        mask[ActionSpace.PLAY_ROAD_BUILDING_OFFSET] = True

                # Year of Plenty
                if env.player_dev_cards[cp, 3] > 0 and env.dev_card_bought_this_turn_idx != 3:
                    for r in range(5):
                        mask[ActionSpace.PLAY_YEAR_OF_PLENTY_OFFSET + r] = True

                # Monopoly
                if env.player_dev_cards[cp, 4] > 0 and env.dev_card_bought_this_turn_idx != 4:
                    for r in range(5):
                        mask[ActionSpace.PLAY_MONOPOLY_OFFSET + r] = True

            # -- BANK TRADE --
            for give_r in range(5):
                trade_ratio = env._get_trade_ratio(cp, give_r)
                if res[give_r] >= trade_ratio:
                    for get_r in range(5):
                        if get_r != give_r:
                            # Encode: give_r * 4 + (get_r if get_r < give_r else get_r - 1)
                            idx = _bank_trade_index(give_r, get_r)
                            mask[ActionSpace.BANK_TRADE_OFFSET + idx] = True

            return mask

        # ----------------------------------------------------------------
        # ROAD_BUILDING special phase
        # ----------------------------------------------------------------
        if phase == env.PHASE_ROAD_BUILDING:
            if env.player_roads_left[cp] > 0:
                for e in range(72):
                    if env.edge_road[e] == 0 and env._can_place_road(cp, e):
                        mask[ActionSpace.BUILD_ROAD_OFFSET + e] = True
            # If no valid road placement, allow end turn
            if not mask.any():
                mask[ActionSpace.END_TURN_OFFSET] = True
            return mask

        # ----------------------------------------------------------------
        # YEAR_OF_PLENTY special phase
        # ----------------------------------------------------------------
        if phase == env.PHASE_YEAR_OF_PLENTY:
            for r in range(5):
                mask[ActionSpace.PLAY_YEAR_OF_PLENTY_OFFSET + r] = True
            return mask

        return mask


def _bank_trade_index(give_res: int, get_res: int) -> int:
    """Encode a bank trade (give_res, get_res) pair into [0..19].

    For each give resource (0-4), there are 4 possible get resources.
    We skip the case give==get.
    Index = give_res * 4 + (get_res if get_res < give_res else get_res - 1)
    """
    adjusted = get_res if get_res < give_res else get_res - 1
    return give_res * 4 + adjusted


def decode_bank_trade(index: int) -> tuple[int, int]:
    """Decode a bank trade index [0..19] back to (give_res, get_res)."""
    give_res = index // 4
    adjusted = index % 4
    get_res = adjusted if adjusted < give_res else adjusted + 1
    return give_res, get_res
