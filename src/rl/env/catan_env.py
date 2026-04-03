"""Gymnasium-compatible 4-player Catan environment.

Full implementation of the Settlers of Catan board game with:
- Standard 19-hex board with randomized layout
- 4 players with resources, dev cards, buildings
- Complete game phases (setup, roll, robber, main, trade, discard, game over)
- Resource generation, building, robber, dev cards, trading, longest road, largest army
- Observation dict with hex/vertex/edge/player features + action mask
- Reward shaping for RL training
"""

from __future__ import annotations

from typing import Any

import gymnasium
import gymnasium.spaces as spaces
import numpy as np

from .action_space import ActionSpace, decode_bank_trade

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_HEX = 19
NUM_VERTICES = 54
NUM_EDGES = 72
NUM_RESOURCE_TYPES = 5
MAX_TURNS = 500
WIN_VP = 10

# Hex types (resource index or desert)
WOOD, BRICK, SHEEP, WHEAT, ORE, DESERT = 0, 1, 2, 3, 4, 5

# Standard Catan resource tile distribution (18 resource tiles + 1 desert)
_STANDARD_HEX_TYPES = (
    [WOOD] * 4
    + [BRICK] * 3
    + [SHEEP] * 4
    + [WHEAT] * 4
    + [ORE] * 3
    + [DESERT]
)

# Standard number tokens (placed on non-desert hexes in random order)
_STANDARD_NUMBERS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]

# Dev card types: 0=Knight, 1=VP, 2=RoadBuilding, 3=YearOfPlenty, 4=Monopoly
_STANDARD_DEV_DECK = (
    [0] * 14  # 14 knights
    + [1] * 5   # 5 VP cards
    + [2] * 2   # 2 road building
    + [3] * 2   # 2 year of plenty
    + [4] * 2   # 2 monopoly
)

# Building piece limits per player
MAX_ROADS_PER_PLAYER = 15
MAX_SETTLEMENTS_PER_PLAYER = 5
MAX_CITIES_PER_PLAYER = 4

# ---------------------------------------------------------------------------
# Board Topology
#
# Generated procedurally from cube coordinates at module load time.
# The 19 hexes form a standard Catan board (radius-2 hex grid).
# Vertex and edge indices are assigned by converting hex corners to
# pixel positions and merging duplicates.
#
# Result: 19 hexes, 54 vertices, 72 edges with correct adjacency.
# ---------------------------------------------------------------------------

def _generate_board_topology():
    """Generate hex-vertex and edge-vertex topology for a standard Catan board.

    Uses cube coordinates for the 19-hex grid (radius 2), converts hex corners
    to pixel positions, and merges duplicate vertices at shared corners.

    Returns (hex_vertices, edge_vertices, vertex_adjacent_vertices,
             vertex_adjacent_hexes, edge_index_map, port_info,
             vertex_positions, hex_centers).
    """
    import math

    # Build the 19 hexes using cube coordinates (q + r + s = 0, max coord <= 2)
    hex_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if abs(q) <= 2 and abs(r) <= 2 and abs(s) <= 2:
                hex_centers.append((q, r, s))
    assert len(hex_centers) == 19

    # Convert cube coords to pixel position (pointy-top hex layout)
    sqrt3 = math.sqrt(3.0)

    def cube_to_pixel(q, r):
        return (sqrt3 * (q + r / 2.0), 1.5 * r)

    def hex_corners(cx, cy):
        """Return 6 corners of a pointy-top hex, CW from top."""
        corners = []
        for i in range(6):
            angle_rad = math.radians(90 - 60 * i)
            corners.append((
                round(cx + math.cos(angle_rad), 6),
                round(cy + math.sin(angle_rad), 6),
            ))
        return corners

    # Collect all corner positions and merge duplicates
    all_corners = []  # (x, y, hex_idx, corner_idx)
    for hi, (q, r, s) in enumerate(hex_centers):
        cx, cy = cube_to_pixel(q, r)
        corners = hex_corners(cx, cy)
        for ci, (x, y) in enumerate(corners):
            all_corners.append((x, y, hi, ci))

    # Assign unique vertex IDs by merging close positions
    vertex_positions = []  # list of (x, y)
    vertex_id_map = {}  # (rounded_x, rounded_y) -> vertex_id
    hex_vertex_map = np.full((19, 6), -1, dtype=np.int32)

    def _round_key(x, y):
        return (round(x, 2), round(y, 2))

    for x, y, hi, ci in all_corners:
        key = _round_key(x, y)
        if key not in vertex_id_map:
            vertex_id_map[key] = len(vertex_positions)
            vertex_positions.append((x, y))
        hex_vertex_map[hi, ci] = vertex_id_map[key]

    num_verts = len(vertex_positions)
    assert num_verts == 54, f"Expected 54 vertices, got {num_verts}"

    # Build edge list: edges connect adjacent vertices on the same hex
    edge_set = set()
    for hi in range(19):
        verts = hex_vertex_map[hi]
        for ci in range(6):
            v0 = int(verts[ci])
            v1 = int(verts[(ci + 1) % 6])
            edge = (min(v0, v1), max(v0, v1))
            edge_set.add(edge)

    edges = sorted(edge_set)
    assert len(edges) == 72, f"Expected 72 edges, got {len(edges)}"
    edge_vertices_arr = np.array(edges, dtype=np.int32)

    # Build vertex adjacency
    vert_adj = [set() for _ in range(54)]
    for v0, v1 in edges:
        vert_adj[v0].add(v1)
        vert_adj[v1].add(v0)

    # Build vertex -> hex adjacency
    vert_hexes = [[] for _ in range(54)]
    for hi in range(19):
        for ci in range(6):
            v = int(hex_vertex_map[hi, ci])
            vert_hexes[v].append(hi)

    # Edge index map
    edge_index_map = {}
    for ei, (v0, v1) in enumerate(edges):
        edge_index_map[(v0, v1)] = ei
        edge_index_map[(v1, v0)] = ei

    # Port positions: 9 ports on the coast.
    # Each port is on 2 adjacent coastal vertices (an edge on the board boundary).
    # We identify coastal edges: edges that belong to only 1 hex.
    edge_hex_count = {}
    for hi in range(19):
        verts = hex_vertex_map[hi]
        for ci in range(6):
            v0 = int(verts[ci])
            v1 = int(verts[(ci + 1) % 6])
            edge_key = (min(v0, v1), max(v0, v1))
            edge_hex_count[edge_key] = edge_hex_count.get(edge_key, 0) + 1

    coastal_edges = [e for e, cnt in edge_hex_count.items() if cnt == 1]
    # Sort coastal edges to get a deterministic order going around the board
    # We'll pick 9 evenly spaced ones for ports
    # Sort by angle from center
    center_x = sum(p[0] for p in vertex_positions) / len(vertex_positions)
    center_y = sum(p[1] for p in vertex_positions) / len(vertex_positions)

    def edge_angle(e):
        mx = (vertex_positions[e[0]][0] + vertex_positions[e[1]][0]) / 2
        my = (vertex_positions[e[0]][1] + vertex_positions[e[1]][1]) / 2
        return math.atan2(my - center_y, mx - center_x)

    coastal_edges.sort(key=edge_angle)
    # Pick 9 ports roughly evenly spaced
    n_coastal = len(coastal_edges)
    port_edges_indices = [coastal_edges[i * n_coastal // 9] for i in range(9)]
    # Each port is (v0, v1, port_type) where port_type:
    #   0 = 3:1 generic, 1-5 = 2:1 for WOOD/BRICK/SHEEP/WHEAT/ORE
    # Standard: 4 generic 3:1 + one 2:1 for each resource
    port_types = [0, 0, 0, 0, 1, 2, 3, 4, 5]  # 4 generic + 5 specialized

    port_info = []
    for i, (v0, v1) in enumerate(port_edges_indices):
        port_info.append((v0, v1, port_types[i]))

    return (hex_vertex_map, edge_vertices_arr, vert_adj, vert_hexes,
            edge_index_map, port_info, vertex_positions, hex_centers)


# Generate topology once at module load
(_HEX_VERTICES, _EDGE_VERTICES, _VERT_ADJ, _VERT_HEXES,
 _EDGE_INDEX_MAP, _PORT_INFO, _VERTEX_POSITIONS, _HEX_CENTERS) = _generate_board_topology()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CatanEnv(gymnasium.Env):
    """4-player Settlers of Catan as a Gymnasium environment."""

    metadata = {"render_modes": ["ansi"], "name": "Catan-v0"}

    # Game phases
    PHASE_SETUP_FIRST = 0
    PHASE_SETUP_SECOND = 1
    PHASE_ROLL = 2
    PHASE_ROBBER_PLACE = 3
    PHASE_ROBBER_STEAL = 4
    PHASE_MAIN = 5
    PHASE_TRADE = 6
    PHASE_DISCARD = 7
    PHASE_GAME_OVER = 8
    PHASE_ROAD_BUILDING = 9
    PHASE_YEAR_OF_PLENTY = 10

    _PHASE_NAMES = [
        "SETUP_FIRST", "SETUP_SECOND", "ROLL", "ROBBER_PLACE",
        "ROBBER_STEAL", "MAIN", "TRADE", "DISCARD", "GAME_OVER",
        "ROAD_BUILDING", "YEAR_OF_PLENTY",
    ]

    def __init__(self, num_players: int = 4, render_mode: str | None = None):
        super().__init__()
        self.num_players = num_players
        self.render_mode = render_mode

        # Topology (shared, read-only)
        self.hex_vertices = _HEX_VERTICES  # (19, 6)
        self.edge_vertices = _EDGE_VERTICES  # (72, 2)
        self.vert_adj = _VERT_ADJ  # list of sets
        self.vert_hexes = _VERT_HEXES  # list of lists
        self.edge_index_map = _EDGE_INDEX_MAP
        self.port_info = _PORT_INFO  # list of (v0, v1, type)

        # Action / observation spaces
        self.action_space = spaces.Discrete(ActionSpace.TOTAL_ACTIONS)

        # Observation space (dict of Box spaces)
        hex_feat_dim = 7 + 1 + 1  # 6 type one-hot + desert + number(scaled) + robber
        # building_type one-hot (none/settlement/city) + owner one-hot
        vert_feat_dim = 3 + num_players
        edge_feat_dim = 1 + num_players  # has_road + owner one-hot
        player_feat_dim = 5 + 5 + 1 + 3  # resources(5) + dev_cards(5types) + vp + pieces_left(3)

        self.observation_space = spaces.Dict({
            "hex_features": spaces.Box(0, 1, shape=(19, hex_feat_dim), dtype=np.float32),
            "vertex_features": spaces.Box(0, 1, shape=(54, vert_feat_dim), dtype=np.float32),
            "edge_features": spaces.Box(0, 1, shape=(72, edge_feat_dim), dtype=np.float32),
            "player_features": spaces.Box(
                0, 1, shape=(num_players, player_feat_dim), dtype=np.float32,
            ),
            "current_player": spaces.Discrete(num_players),
            "game_phase": spaces.Discrete(11),
            "action_mask": spaces.MultiBinary(ActionSpace.TOTAL_ACTIONS),
        })

        self._rng = np.random.default_rng()

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # --- Board ---
        hex_types = list(_STANDARD_HEX_TYPES)
        self._rng.shuffle(hex_types)
        self.hex_types = np.array(hex_types, dtype=np.int32)  # (19,)

        # Assign number tokens to non-desert hexes
        self.hex_numbers = np.zeros(19, dtype=np.int32)
        numbers = list(_STANDARD_NUMBERS)
        self._rng.shuffle(numbers)
        ni = 0
        for hi in range(19):
            if self.hex_types[hi] != DESERT:
                self.hex_numbers[hi] = numbers[ni]
                ni += 1
            else:
                self.hex_numbers[hi] = 0

        # Robber starts on desert
        self.robber_hex = int(np.argmax(self.hex_types == DESERT))

        # Shuffle port types
        port_types = [0, 0, 0, 0, 1, 2, 3, 4, 5]
        self._rng.shuffle(port_types)
        self.port_info_current = [
            (pi[0], pi[1], port_types[i]) for i, pi in enumerate(_PORT_INFO)
        ]

        # --- Players ---
        self.player_resources = np.zeros((self.num_players, 5), dtype=np.int32)
        # Dev cards held: [knight, vp, road_building, year_of_plenty, monopoly]
        self.player_dev_cards = np.zeros((self.num_players, 5), dtype=np.int32)
        self.player_knights_played = np.zeros(self.num_players, dtype=np.int32)
        self.player_vp = np.zeros(self.num_players, dtype=np.int32)
        self.player_roads_left = np.full(self.num_players, MAX_ROADS_PER_PLAYER, dtype=np.int32)
        self.player_settlements_left = np.full(
            self.num_players, MAX_SETTLEMENTS_PER_PLAYER, dtype=np.int32,
        )
        self.player_cities_left = np.full(self.num_players, MAX_CITIES_PER_PLAYER, dtype=np.int32)

        # --- Board state ---
        # vertex_building: 0=empty, 1=settlement, 2=city
        self.vertex_building = np.zeros(54, dtype=np.int32)
        # vertex_owner: 0=nobody, 1-4=player (1-indexed)
        self.vertex_owner = np.zeros(54, dtype=np.int32)
        # edge_road: 0=no road, 1=has road
        self.edge_road = np.zeros(72, dtype=np.int32)
        # edge_owner: 0=nobody, 1-4=player
        self.edge_owner = np.zeros(72, dtype=np.int32)

        # --- Dev card deck ---
        deck = list(_STANDARD_DEV_DECK)
        self._rng.shuffle(deck)
        self.dev_card_deck = np.array(deck, dtype=np.int32)
        self.dev_card_deck_index = 0

        # --- Game state ---
        self.current_player = 0
        self.game_phase = self.PHASE_SETUP_FIRST
        self.dice_roll = 0
        self.turn_counter = 0

        # Setup tracking
        self.setup_sub_phase = 0  # 0 = place settlement, 1 = place road
        self.setup_settlements_placed = 0  # total across all players for current phase
        self.last_setup_settlement = -1

        # Setup order: player 0,1,2,3 then 3,2,1,0
        self.setup_order = list(range(self.num_players)) + list(range(self.num_players - 1, -1, -1))
        self.setup_index = 0

        # Longest road / largest army
        self.longest_road_player = -1  # -1 = nobody
        self.longest_road_length = 0
        self.largest_army_player = -1
        self.largest_army_size = 0

        # Dev card tracking per turn
        self.has_played_dev_card_this_turn = False
        # Track which dev card type was bought this turn (can't play same turn)
        # -1 means nothing bought this turn
        self.dev_card_bought_this_turn_idx = -1

        # Road building tracking
        self.road_building_roads_left = 0

        # Year of plenty tracking
        self.year_of_plenty_resources_left = 0

        # Discard tracking (how many more cards each player must discard)
        self.discard_remaining = np.zeros(self.num_players, dtype=np.int32)
        self.discard_player_index = 0

        # Winner
        self.winner = -1

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self.game_phase == self.PHASE_GAME_OVER:
            obs = self._get_observation()
            return obs, 0.0, True, False, self._get_info()

        action_type, param = ActionSpace.decode_action(action)
        cp = self.current_player
        reward = -0.01  # time penalty per step
        prev_vp = self.player_vp[cp]

        # Validate action is legal
        mask = self.get_action_mask()
        if not mask[action]:
            # Illegal action -- heavy penalty, no state change, don't end
            obs = self._get_observation()
            return obs, -1.0, False, False, self._get_info()

        terminated = False
        truncated = False

        # ---- SETUP PHASES ----
        if self.game_phase in (self.PHASE_SETUP_FIRST, self.PHASE_SETUP_SECOND):
            reward = self._handle_setup(action_type, param)

        # ---- ROLL ----
        elif self.game_phase == self.PHASE_ROLL:
            if action_type == "ROLL_DICE":
                reward += self._handle_roll()
            elif action_type == "PLAY_KNIGHT":
                reward += self._handle_play_knight(param)

        # ---- ROBBER PLACE ----
        elif self.game_phase == self.PHASE_ROBBER_PLACE:
            reward += self._handle_robber_place(param)

        # ---- ROBBER STEAL ----
        elif self.game_phase == self.PHASE_ROBBER_STEAL:
            reward += self._handle_robber_steal(param)

        # ---- DISCARD ----
        elif self.game_phase == self.PHASE_DISCARD:
            reward += self._handle_discard(param)

        # ---- MAIN PHASE ----
        elif self.game_phase == self.PHASE_MAIN:
            if action_type == "END_TURN":
                reward += self._handle_end_turn()
            elif action_type == "BUILD_ROAD":
                reward += self._handle_build_road(param, free=False)
            elif action_type == "BUILD_SETTLEMENT":
                reward += self._handle_build_settlement(param, free=False)
            elif action_type == "BUILD_CITY":
                reward += self._handle_build_city(param)
            elif action_type == "BUY_DEV_CARD":
                reward += self._handle_buy_dev_card()
            elif action_type == "PLAY_KNIGHT":
                reward += self._handle_play_knight(param)
            elif action_type == "PLAY_ROAD_BUILDING":
                reward += self._handle_play_road_building()
            elif action_type == "PLAY_YEAR_OF_PLENTY":
                reward += self._handle_play_year_of_plenty(param)
            elif action_type == "PLAY_MONOPOLY":
                reward += self._handle_play_monopoly(param)
            elif action_type == "BANK_TRADE":
                reward += self._handle_bank_trade(param)

        # ---- ROAD BUILDING PHASE ----
        elif self.game_phase == self.PHASE_ROAD_BUILDING:
            if action_type == "BUILD_ROAD":
                reward += self._handle_build_road(param, free=True)
                self.road_building_roads_left -= 1
                if self.road_building_roads_left <= 0:
                    self.game_phase = self.PHASE_MAIN
            elif action_type == "END_TURN":
                self.game_phase = self.PHASE_MAIN

        # ---- YEAR OF PLENTY PHASE ----
        elif self.game_phase == self.PHASE_YEAR_OF_PLENTY:
            if action_type == "PLAY_YEAR_OF_PLENTY":
                self.player_resources[cp, param] += 1
                self.year_of_plenty_resources_left -= 1
                if self.year_of_plenty_resources_left <= 0:
                    self.game_phase = self.PHASE_MAIN

        # VP reward shaping
        vp_gained = self.player_vp[cp] - prev_vp
        reward += 0.1 * vp_gained

        # Check win
        if self.player_vp[cp] >= WIN_VP and self.game_phase != self.PHASE_GAME_OVER:
            self.game_phase = self.PHASE_GAME_OVER
            self.winner = cp
            reward += 1.0
            terminated = True

        # Check truncation
        if self.turn_counter >= MAX_TURNS and not terminated:
            truncated = True
            self.game_phase = self.PHASE_GAME_OVER

        obs = self._get_observation()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    # -----------------------------------------------------------------------
    # Action handlers
    # -----------------------------------------------------------------------

    def _handle_setup(self, action_type: str, param: int) -> float:
        cp = self.current_player
        reward = 0.0

        if action_type == "BUILD_SETTLEMENT" and self.setup_sub_phase == 0:
            # Place free settlement
            self.vertex_building[param] = 1
            self.vertex_owner[param] = cp + 1
            self.player_settlements_left[cp] -= 1
            self.player_vp[cp] += 1
            self.last_setup_settlement = param
            self.setup_sub_phase = 1
            reward += 0.05

            # In second setup phase, give initial resources from adjacent hexes
            if self.game_phase == self.PHASE_SETUP_SECOND:
                for hi in self.vert_hexes[param]:
                    ht = self.hex_types[hi]
                    if ht != DESERT:
                        self.player_resources[cp, ht] += 1

        elif action_type == "BUILD_ROAD" and self.setup_sub_phase == 1:
            # Place free road
            self.edge_road[param] = 1
            self.edge_owner[param] = cp + 1
            self.player_roads_left[cp] -= 1
            self.setup_sub_phase = 0
            self.setup_index += 1
            reward += 0.03

            # Advance to next player or next phase
            if self.game_phase == self.PHASE_SETUP_FIRST:
                if self.setup_index >= self.num_players:
                    self.game_phase = self.PHASE_SETUP_SECOND
                    self.setup_index = self.num_players  # continues with reverse order
                    self.current_player = self.setup_order[self.setup_index]
                else:
                    self.current_player = self.setup_order[self.setup_index]
            elif self.game_phase == self.PHASE_SETUP_SECOND:
                if self.setup_index >= len(self.setup_order):
                    # Setup complete, start normal play
                    self.game_phase = self.PHASE_ROLL
                    self.current_player = 0
                    self.turn_counter = 1
                else:
                    self.current_player = self.setup_order[self.setup_index]

        return reward

    def _handle_roll(self) -> float:
        d1 = self._rng.integers(1, 7)
        d2 = self._rng.integers(1, 7)
        self.dice_roll = d1 + d2

        if self.dice_roll == 7:
            # Check for discards first
            must_discard = False
            for p in range(self.num_players):
                total = int(self.player_resources[p].sum())
                if total > 7:
                    self.discard_remaining[p] = total // 2
                    must_discard = True

            if must_discard:
                self.game_phase = self.PHASE_DISCARD
                # Find first player that must discard
                self.discard_player_index = self._find_next_discard_player(-1)
                self.current_player = self.discard_player_index
            else:
                self.game_phase = self.PHASE_ROBBER_PLACE
        else:
            # Distribute resources
            self._distribute_resources(self.dice_roll)
            self.game_phase = self.PHASE_MAIN

        return 0.0

    def _handle_robber_place(self, hex_id: int) -> float:
        self.robber_hex = hex_id
        # Check if any opponent has buildings adjacent to this hex
        hex_verts = self.hex_vertices[hex_id]
        can_steal = False
        cp = self.current_player
        for p in range(self.num_players):
            if p == cp:
                continue
            for v in hex_verts:
                if self.vertex_owner[v] == p + 1 and self.player_resources[p].sum() > 0:
                    can_steal = True
                    break
            if can_steal:
                break

        if can_steal:
            self.game_phase = self.PHASE_ROBBER_STEAL
        else:
            self.game_phase = self.PHASE_MAIN

        return 0.0

    def _handle_robber_steal(self, target_player: int) -> float:
        cp = self.current_player
        if target_player != cp and self.player_resources[target_player].sum() > 0:
            # Steal random resource
            res = self.player_resources[target_player]
            nonzero = np.where(res > 0)[0]
            if len(nonzero) > 0:
                stolen = self._rng.choice(nonzero)
                res[stolen] -= 1
                self.player_resources[cp, stolen] += 1

        self.game_phase = self.PHASE_MAIN
        return 0.0

    def _handle_discard(self, resource_type: int) -> float:
        cp = self.current_player
        if self.player_resources[cp, resource_type] > 0 and self.discard_remaining[cp] > 0:
            self.player_resources[cp, resource_type] -= 1
            self.discard_remaining[cp] -= 1

        # Check if current player is done discarding
        if self.discard_remaining[cp] <= 0:
            nxt = self._find_next_discard_player(cp)
            if nxt == -1:
                # All done discarding, move to robber placement
                self.game_phase = self.PHASE_ROBBER_PLACE
                # Restore current player to the one who rolled
                self.current_player = self._roller_player
            else:
                self.discard_player_index = nxt
                self.current_player = nxt

        return 0.0

    def _handle_end_turn(self) -> float:
        self._advance_turn()
        return 0.0

    def _handle_build_road(self, edge_id: int, free: bool = False) -> float:
        cp = self.current_player
        if not free:
            self.player_resources[cp] -= ActionSpace.ROAD_COST
        self.edge_road[edge_id] = 1
        self.edge_owner[edge_id] = cp + 1
        self.player_roads_left[cp] -= 1
        self._update_longest_road()
        return 0.03

    def _handle_build_settlement(self, vertex_id: int, free: bool = False) -> float:
        cp = self.current_player
        if not free:
            self.player_resources[cp] -= ActionSpace.SETTLEMENT_COST
        self.vertex_building[vertex_id] = 1
        self.vertex_owner[vertex_id] = cp + 1
        self.player_settlements_left[cp] -= 1
        self.player_vp[cp] += 1
        self._update_longest_road()  # settlements can break roads
        return 0.05

    def _handle_build_city(self, vertex_id: int) -> float:
        cp = self.current_player
        self.player_resources[cp] -= ActionSpace.CITY_COST
        self.vertex_building[vertex_id] = 2
        self.player_settlements_left[cp] += 1  # settlement returned
        self.player_cities_left[cp] -= 1
        self.player_vp[cp] += 1  # settlement was 1 VP, city is 2 VP (net +1)
        return 0.08

    def _handle_buy_dev_card(self) -> float:
        cp = self.current_player
        self.player_resources[cp] -= ActionSpace.DEV_CARD_COST
        card_type = int(self.dev_card_deck[self.dev_card_deck_index])
        self.dev_card_deck_index += 1
        self.player_dev_cards[cp, card_type] += 1
        self.dev_card_bought_this_turn_idx = card_type

        # VP cards take effect immediately
        if card_type == 1:
            self.player_vp[cp] += 1

        return 0.0

    def _handle_play_knight(self, hex_id: int) -> float:
        cp = self.current_player
        self.player_dev_cards[cp, 0] -= 1
        self.player_knights_played[cp] += 1
        self.has_played_dev_card_this_turn = True

        # Update largest army
        self._update_largest_army()

        # Move robber
        self.robber_hex = hex_id

        # Check if can steal
        hex_verts = self.hex_vertices[hex_id]
        can_steal = False
        for p in range(self.num_players):
            if p == cp:
                continue
            for v in hex_verts:
                if self.vertex_owner[v] == p + 1 and self.player_resources[p].sum() > 0:
                    can_steal = True
                    break
            if can_steal:
                break

        if can_steal:
            self.game_phase = self.PHASE_ROBBER_STEAL
        elif self.game_phase == self.PHASE_ROLL:
            # Played knight before rolling, go back to roll
            self.game_phase = self.PHASE_ROLL
        # else stay in MAIN

        return 0.0

    def _handle_play_road_building(self) -> float:
        cp = self.current_player
        self.player_dev_cards[cp, 2] -= 1
        self.has_played_dev_card_this_turn = True
        self.road_building_roads_left = min(2, int(self.player_roads_left[cp]))
        self.game_phase = self.PHASE_ROAD_BUILDING
        return 0.0

    def _handle_play_year_of_plenty(self, resource_type: int) -> float:
        cp = self.current_player
        if self.game_phase == self.PHASE_MAIN:
            # First use from main phase
            self.player_dev_cards[cp, 3] -= 1
            self.has_played_dev_card_this_turn = True
            self.player_resources[cp, resource_type] += 1
            self.year_of_plenty_resources_left = 1
            self.game_phase = self.PHASE_YEAR_OF_PLENTY
        else:
            # Second resource in YEAR_OF_PLENTY phase (handled in step)
            pass  # handled in step's YEAR_OF_PLENTY branch
        return 0.0

    def _handle_play_monopoly(self, resource_type: int) -> float:
        cp = self.current_player
        self.player_dev_cards[cp, 4] -= 1
        self.has_played_dev_card_this_turn = True
        total_stolen = 0
        for p in range(self.num_players):
            if p != cp:
                amount = int(self.player_resources[p, resource_type])
                self.player_resources[p, resource_type] = 0
                total_stolen += amount
        self.player_resources[cp, resource_type] += total_stolen
        return 0.0

    def _handle_bank_trade(self, trade_idx: int) -> float:
        cp = self.current_player
        give_res, get_res = decode_bank_trade(trade_idx)
        ratio = self._get_trade_ratio(cp, give_res)
        self.player_resources[cp, give_res] -= ratio
        self.player_resources[cp, get_res] += 1
        return 0.0

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    @property
    def _roller_player(self) -> int:
        """The player whose turn it is (who rolled the dice)."""
        # During discard phase, we need to remember who actually rolled.
        # We track this via turn order: current_player at ROLL phase.
        # Simple approach: the player whose turn it is based on turn_counter.
        return (self.turn_counter - 1) % self.num_players

    def _find_next_discard_player(self, after: int) -> int:
        """Find the next player who still needs to discard, after the given index."""
        for i in range(1, self.num_players + 1):
            p = (after + i) % self.num_players
            if self.discard_remaining[p] > 0:
                return p
        return -1

    def _distribute_resources(self, roll: int) -> None:
        """Give resources to players based on dice roll."""
        for hi in range(19):
            if self.hex_numbers[hi] != roll:
                continue
            if hi == self.robber_hex:
                continue
            resource = int(self.hex_types[hi])
            if resource == DESERT:
                continue
            for v in self.hex_vertices[hi]:
                owner = self.vertex_owner[v]
                if owner > 0:
                    p = owner - 1
                    amount = int(self.vertex_building[v])  # 1 for settlement, 2 for city
                    self.player_resources[p, resource] += amount

    def _check_distance_rule(self, vertex: int) -> bool:
        """Check that no adjacent vertex has a building (distance rule)."""
        for adj_v in self.vert_adj[vertex]:
            if self.vertex_building[adj_v] > 0:
                return False
        return True

    def _can_place_road(self, player: int, edge_id: int) -> bool:
        """Check if player can place a road on this edge (connected to their network)."""
        v0, v1 = self.edge_vertices[edge_id]
        p_code = player + 1

        # Check if either endpoint has player's building
        if self.vertex_owner[v0] == p_code or self.vertex_owner[v1] == p_code:
            return True

        # Check if either endpoint connects to player's existing road
        for v in (v0, v1):
            for adj_v in self.vert_adj[v]:
                edge_key = (min(v, adj_v), max(v, adj_v))
                ei = self.edge_index_map.get(edge_key)
                if ei is not None and self.edge_owner[ei] == p_code:
                    # Make sure there's no opponent's building blocking at vertex v
                    if self.vertex_owner[v] == 0 or self.vertex_owner[v] == p_code:
                        return True
        return False

    def _vertex_connected_to_road(self, player: int, vertex: int) -> bool:
        """Check if vertex is connected to player's road network."""
        p_code = player + 1
        for adj_v in self.vert_adj[vertex]:
            edge_key = (min(vertex, adj_v), max(vertex, adj_v))
            ei = self.edge_index_map.get(edge_key)
            if ei is not None and self.edge_owner[ei] == p_code:
                return True
        return False

    def _get_trade_ratio(self, player: int, resource: int) -> int:
        """Get the best trade ratio for a player trading a given resource."""
        p_code = player + 1
        ratio = 4  # default bank trade

        for v0, v1, port_type in self.port_info_current:
            # Check if player has a building on either port vertex
            if self.vertex_owner[v0] == p_code or self.vertex_owner[v1] == p_code:
                if port_type == 0:
                    # 3:1 generic
                    ratio = min(ratio, 3)
                elif port_type == resource + 1:
                    # 2:1 specific
                    ratio = min(ratio, 2)

        return ratio

    def _update_longest_road(self) -> None:
        """Recalculate longest road for all players."""
        for p in range(self.num_players):
            length = self._calc_longest_road(p)
            if length >= 5:
                if length > self.longest_road_length:
                    # Remove VP from previous holder
                    if self.longest_road_player >= 0:
                        self.player_vp[self.longest_road_player] -= 2
                    self.longest_road_player = p
                    self.longest_road_length = length
                    self.player_vp[p] += 2

    def _calc_longest_road(self, player: int) -> int:
        """Calculate the longest road length for a player using DFS."""
        p_code = player + 1
        # Get all edges belonging to this player
        player_edges = set()
        for ei in range(72):
            if self.edge_owner[ei] == p_code:
                player_edges.add(ei)

        if not player_edges:
            return 0

        # Build adjacency graph of player's edges
        # Two edges are connected if they share a vertex
        # AND no opponent building blocks the junction
        best = 0

        # For each edge, try DFS
        for start_edge in player_edges:
            visited: set[int] = set()
            length = self._dfs_road(start_edge, visited, player_edges, p_code)
            best = max(best, length)

        return best

    def _dfs_road(self, edge: int, visited: set[int], player_edges: set[int], p_code: int) -> int:
        """DFS to find longest path from this edge."""
        visited.add(edge)
        v0, v1 = self.edge_vertices[edge]
        best = 1

        for endpoint in (v0, v1):
            # Can only continue through this vertex if no opponent building
            if self.vertex_owner[endpoint] != 0 and self.vertex_owner[endpoint] != p_code:
                continue

            # Find connected edges through this vertex
            for adj_v in self.vert_adj[endpoint]:
                edge_key = (min(endpoint, adj_v), max(endpoint, adj_v))
                nei = self.edge_index_map.get(edge_key)
                if nei is not None and nei in player_edges and nei not in visited:
                    length = 1 + self._dfs_road(nei, visited, player_edges, p_code)
                    best = max(best, length)

        visited.remove(edge)
        return best

    def _update_largest_army(self) -> None:
        """Check and update largest army."""
        for p in range(self.num_players):
            knights = int(self.player_knights_played[p])
            if knights >= 3 and knights > self.largest_army_size:
                if self.largest_army_player >= 0:
                    self.player_vp[self.largest_army_player] -= 2
                self.largest_army_player = p
                self.largest_army_size = knights
                self.player_vp[p] += 2

    def _advance_turn(self) -> None:
        """End current player's turn and advance to next player."""
        self.current_player = (self.current_player + 1) % self.num_players
        self.turn_counter += 1
        self.game_phase = self.PHASE_ROLL
        self.has_played_dev_card_this_turn = False
        self.dev_card_bought_this_turn_idx = -1
        self.dice_roll = 0

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------

    def _get_observation(self) -> dict:
        # Hex features: 6 type one-hot + desert flag + number (scaled) + robber
        hex_feat = np.zeros((19, 9), dtype=np.float32)
        for hi in range(19):
            ht = self.hex_types[hi]
            if ht < 5:
                hex_feat[hi, ht] = 1.0
            if ht == DESERT:
                hex_feat[hi, 5] = 1.0  # explicit desert flag
            # Resource type already encoded; add column 6 for desert
            hex_feat[hi, 6] = 1.0 if ht == DESERT else 0.0
            hex_feat[hi, 7] = self.hex_numbers[hi] / 12.0  # scaled number
            hex_feat[hi, 8] = 1.0 if hi == self.robber_hex else 0.0

        # Vertex features: building type one-hot (none/settlement/city) + owner one-hot (4 players)
        vert_feat = np.zeros((54, 3 + self.num_players), dtype=np.float32)
        for v in range(54):
            b = self.vertex_building[v]
            if b == 0:
                vert_feat[v, 0] = 1.0  # empty
            elif b == 1:
                vert_feat[v, 1] = 1.0  # settlement
            elif b == 2:
                vert_feat[v, 2] = 1.0  # city
            o = self.vertex_owner[v]
            if o > 0:
                vert_feat[v, 3 + o - 1] = 1.0

        # Edge features: has_road + owner one-hot
        edge_feat = np.zeros((72, 1 + self.num_players), dtype=np.float32)
        for e in range(72):
            if self.edge_road[e]:
                edge_feat[e, 0] = 1.0
                o = self.edge_owner[e]
                if o > 0:
                    edge_feat[e, 1 + o - 1] = 1.0

        # Player features: resources(5) + dev_cards_count(5) + vp(1) + pieces_left(3)
        player_feat = np.zeros((self.num_players, 14), dtype=np.float32)
        for p in range(self.num_players):
            player_feat[p, 0:5] = self.player_resources[p] / 19.0  # scaled
            player_feat[p, 5:10] = self.player_dev_cards[p] / 14.0  # scaled
            player_feat[p, 10] = self.player_vp[p] / 10.0
            player_feat[p, 11] = self.player_roads_left[p] / MAX_ROADS_PER_PLAYER
            player_feat[p, 12] = self.player_settlements_left[p] / MAX_SETTLEMENTS_PER_PLAYER
            player_feat[p, 13] = self.player_cities_left[p] / MAX_CITIES_PER_PLAYER

        action_mask = self.get_action_mask()

        return {
            "hex_features": hex_feat,
            "vertex_features": vert_feat,
            "edge_features": edge_feat,
            "player_features": player_feat,
            "current_player": self.current_player,
            "game_phase": self.game_phase,
            "action_mask": action_mask.astype(np.int8),
        }

    def get_action_mask(self) -> np.ndarray:
        """Return boolean mask of legal actions."""
        return ActionSpace.get_action_mask(self)

    def _get_info(self) -> dict:
        return {
            "turn": self.turn_counter,
            "current_player": self.current_player,
            "phase": (
                self._PHASE_NAMES[self.game_phase]
                if self.game_phase < len(self._PHASE_NAMES)
                else str(self.game_phase)
            ),
            "dice_roll": self.dice_roll,
            "vp": self.player_vp.copy(),
            "winner": self.winner,
            "longest_road_player": self.longest_road_player,
            "largest_army_player": self.largest_army_player,
        }

    # -----------------------------------------------------------------------
    # Render
    # -----------------------------------------------------------------------

    def render(self) -> str | None:  # type: ignore[override]
        if self.render_mode != "ansi":
            return None

        lines = []
        phase_name = (
            self._PHASE_NAMES[self.game_phase]
            if self.game_phase < len(self._PHASE_NAMES)
            else self.game_phase
        )
        lines.append(
            f"=== CATAN === Turn {self.turn_counter} | "
            f"Player {self.current_player} | "
            f"Phase: {phase_name}"
        )
        lines.append(f"Dice: {self.dice_roll} | Robber on hex {self.robber_hex}")

        res_names = ["Wood", "Brick", "Sheep", "Wheat", "Ore"]
        for p in range(self.num_players):
            r = self.player_resources[p]
            res_str = ", ".join(f"{res_names[i]}:{r[i]}" for i in range(5))
            dc = self.player_dev_cards[p]
            dc_str = f"K:{dc[0]} VP:{dc[1]} RB:{dc[2]} YoP:{dc[3]} Mon:{dc[4]}"
            roads_used = MAX_ROADS_PER_PLAYER - self.player_roads_left[p]
            sett_used = MAX_SETTLEMENTS_PER_PLAYER - self.player_settlements_left[p]
            city_used = MAX_CITIES_PER_PLAYER - self.player_cities_left[p]
            lines.append(
                f"  P{p}: VP={self.player_vp[p]} | {res_str} | "
                f"DevCards=[{dc_str}] | "
                f"Roads:{roads_used}/{MAX_ROADS_PER_PLAYER} "
                f"Sett:{sett_used}/{MAX_SETTLEMENTS_PER_PLAYER} "
                f"City:{city_used}/{MAX_CITIES_PER_PLAYER}"
            )

        hex_names = ["W", "B", "S", "H", "O", "D"]
        hex_row_sizes = [3, 4, 5, 4, 3]
        hi = 0
        for row_idx, row_size in enumerate(hex_row_sizes):
            indent = "  " * (2 - abs(2 - row_idx))
            row_strs = []
            for _ in range(row_size):
                hn = hex_names[self.hex_types[hi]]
                num = self.hex_numbers[hi]
                rob = "*" if hi == self.robber_hex else " "
                row_strs.append(f"{hn}{num:2d}{rob}")
                hi += 1
            lines.append(indent + "  ".join(row_strs))

        if self.longest_road_player >= 0:
            lines.append(f"Longest Road: P{self.longest_road_player} ({self.longest_road_length})")
        if self.largest_army_player >= 0:
            lines.append(f"Largest Army: P{self.largest_army_player} ({self.largest_army_size})")

        text = "\n".join(lines)
        return text
