"""Tests for the CatanRL action space encoding/decoding."""

from __future__ import annotations

import pytest
from src.rl.env.action_space import ActionSpace, _bank_trade_index, decode_bank_trade


class TestEncodeDecodeRoundtrip:
    """Encode then decode should return the original (action_type, param)."""

    @pytest.mark.parametrize(
        "action_type,param",
        [
            ("ROLL_DICE", 0),
            ("END_TURN", 0),
            ("BUILD_ROAD", 0),
            ("BUILD_ROAD", 71),
            ("BUILD_SETTLEMENT", 0),
            ("BUILD_SETTLEMENT", 53),
            ("BUILD_CITY", 0),
            ("BUILD_CITY", 53),
            ("BUY_DEV_CARD", 0),
            ("PLAY_KNIGHT", 0),
            ("PLAY_KNIGHT", 18),
            ("PLAY_ROAD_BUILDING", 0),
            ("PLAY_YEAR_OF_PLENTY", 0),
            ("PLAY_YEAR_OF_PLENTY", 4),
            ("PLAY_MONOPOLY", 0),
            ("PLAY_MONOPOLY", 4),
            ("BANK_TRADE", 0),
            ("BANK_TRADE", 19),
            ("PLACE_ROBBER", 0),
            ("PLACE_ROBBER", 18),
            ("STEAL_FROM_PLAYER", 0),
            ("STEAL_FROM_PLAYER", 3),
            ("DISCARD_RESOURCE", 0),
            ("DISCARD_RESOURCE", 4),
        ],
    )
    def test_roundtrip(self, action_type: str, param: int):
        encoded = ActionSpace.encode_action(action_type, param)
        decoded_type, decoded_param = ActionSpace.decode_action(encoded)
        assert decoded_type == action_type
        assert decoded_param == param

    def test_all_actions_roundtrip(self):
        """Every action_id in [0, TOTAL_ACTIONS) should decode and re-encode cleanly."""
        for action_id in range(ActionSpace.TOTAL_ACTIONS):
            action_type, param = ActionSpace.decode_action(action_id)
            re_encoded = ActionSpace.encode_action(action_type, param)
            assert re_encoded == action_id, (
                f"Roundtrip failed for action_id={action_id}: "
                f"decoded=({action_type}, {param}), re-encoded={re_encoded}"
            )


class TestActionOffsets:
    """Offsets should be contiguous with no gaps or overlaps."""

    def test_offsets_contiguous(self):
        categories = ActionSpace._CATEGORIES
        for i in range(len(categories) - 1):
            _, offset_i, size_i = categories[i]
            _, offset_next, _ = categories[i + 1]
            assert offset_i + size_i == offset_next, (
                f"Gap between {categories[i][0]} (offset={offset_i}, size={size_i}) "
                f"and {categories[i + 1][0]} (offset={offset_next})"
            )

    def test_total_actions_matches(self):
        """TOTAL_ACTIONS should equal the sum of all category sizes."""
        total = sum(size for _, _, size in ActionSpace._CATEGORIES)
        assert total == ActionSpace.TOTAL_ACTIONS

    def test_total_actions_matches_last_offset_plus_size(self):
        last_name, last_offset, last_size = ActionSpace._CATEGORIES[-1]
        assert last_offset + last_size == ActionSpace.TOTAL_ACTIONS

    def test_first_offset_is_zero(self):
        _, first_offset, _ = ActionSpace._CATEGORIES[0]
        assert first_offset == 0


class TestDecodeKnownActions:
    """Decode known action IDs to verify correct (type, param)."""

    def test_roll_dice(self):
        assert ActionSpace.decode_action(0) == ("ROLL_DICE", 0)

    def test_end_turn(self):
        assert ActionSpace.decode_action(1) == ("END_TURN", 0)

    def test_first_road(self):
        assert ActionSpace.decode_action(2) == ("BUILD_ROAD", 0)

    def test_last_road(self):
        assert ActionSpace.decode_action(73) == ("BUILD_ROAD", 71)

    def test_first_settlement(self):
        assert ActionSpace.decode_action(74) == ("BUILD_SETTLEMENT", 0)

    def test_first_city(self):
        assert ActionSpace.decode_action(128) == ("BUILD_CITY", 0)

    def test_buy_dev_card(self):
        assert ActionSpace.decode_action(182) == ("BUY_DEV_CARD", 0)

    def test_last_action(self):
        assert ActionSpace.decode_action(260) == ("DISCARD_RESOURCE", 4)


class TestEncodeErrors:
    """Encoding invalid inputs should raise ValueError."""

    def test_unknown_action_type(self):
        with pytest.raises(ValueError, match="Unknown action type"):
            ActionSpace.encode_action("INVALID_TYPE", 0)

    def test_parameter_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            ActionSpace.encode_action("ROLL_DICE", 1)

    def test_negative_parameter(self):
        with pytest.raises(ValueError, match="out of range"):
            ActionSpace.encode_action("BUILD_ROAD", -1)


class TestDecodeErrors:

    def test_invalid_action_id(self):
        with pytest.raises(ValueError, match="Invalid action_id"):
            ActionSpace.decode_action(261)

    def test_negative_action_id(self):
        with pytest.raises(ValueError, match="Invalid action_id"):
            ActionSpace.decode_action(-1)


class TestBankTradeEncoding:
    """Bank trade index encode/decode roundtrip."""

    def test_all_bank_trades_roundtrip(self):
        seen_indices = set()
        for give_res in range(5):
            for get_res in range(5):
                if give_res == get_res:
                    continue
                idx = _bank_trade_index(give_res, get_res)
                assert 0 <= idx < 20
                seen_indices.add(idx)
                decoded_give, decoded_get = decode_bank_trade(idx)
                assert decoded_give == give_res
                assert decoded_get == get_res
        # Should cover all 20 indices
        assert len(seen_indices) == 20
