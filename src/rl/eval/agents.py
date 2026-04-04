"""Baseline agents for evaluation.

RandomAgent  — uniform random over legal actions.
HeuristicAgent — priority-based rule agent.
"""

from __future__ import annotations

import numpy as np


class RandomAgent:
    """Picks a uniformly random legal action."""

    def act(self, obs: dict) -> int:
        legal = np.where(obs["action_mask"])[0]
        return int(np.random.choice(legal))


class HeuristicAgent:
    """Simple priority-based heuristic.

    Priority order (highest to lowest):
        1. Roll dice            — always do immediately
        2. Discard resource     — forced
        3. Place robber         — forced after 7 / knight
        4. Steal from player    — forced
        5. Build city           — best VP gain
        6. Build settlement     — VP gain
        7. Buy dev card
        8. Play knight
        9. Build road
        10. Bank trade
        11. Other dev cards (road-building, YoP, monopoly)
        12. End turn
        13. Any remaining legal action
    """

    _ROLL = 0
    _END_TURN = 1
    _ROAD_S, _ROAD_E = 2, 73
    _SETTLE_S, _SETTLE_E = 74, 127
    _CITY_S, _CITY_E = 128, 181
    _BUY_DEV = 182
    _KNIGHT_S, _KNIGHT_E = 183, 201
    _ROAD_BUILD = 202
    _YOP_S, _YOP_E = 203, 207
    _MONO_S, _MONO_E = 208, 212
    _TRADE_S, _TRADE_E = 213, 232
    _ROBBER_S, _ROBBER_E = 233, 251
    _STEAL_S, _STEAL_E = 252, 255
    _DISCARD_S, _DISCARD_E = 256, 260

    def act(self, obs: dict) -> int:
        mask: np.ndarray = obs["action_mask"].astype(bool)

        def first_legal(start: int, end: int) -> int | None:
            for a in range(start, end + 1):
                if mask[a]:
                    return a
            return None

        checks = [
            (self._ROLL, self._ROLL),
            (self._DISCARD_S, self._DISCARD_E),
            (self._ROBBER_S, self._ROBBER_E),
            (self._STEAL_S, self._STEAL_E),
            (self._CITY_S, self._CITY_E),
            (self._SETTLE_S, self._SETTLE_E),
            (self._BUY_DEV, self._BUY_DEV),
            (self._KNIGHT_S, self._KNIGHT_E),
            (self._ROAD_S, self._ROAD_E),
            (self._TRADE_S, self._TRADE_E),
            (self._ROAD_BUILD, self._ROAD_BUILD),
            (self._YOP_S, self._YOP_E),
            (self._MONO_S, self._MONO_E),
            (self._END_TURN, self._END_TURN),
        ]
        for start, end in checks:
            a = first_legal(start, end)
            if a is not None:
                return a

        legal = np.where(mask)[0]
        return int(legal[0])
