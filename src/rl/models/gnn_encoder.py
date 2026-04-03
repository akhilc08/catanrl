"""Heterogeneous Graph Attention Network encoder for Catan board state.

Encodes the Catan board (hex/vertex/edge nodes with player features) into a
fixed-size embedding using a 3-layer heterogeneous GAT with residual connections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, HeteroConv

# ---------------------------------------------------------------------------
# Board Topology (static, computed once)
# ---------------------------------------------------------------------------

def _build_catan_topology() -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Build the static Catan board edge indices for the heterogeneous graph.

    Imports the module-level topology arrays from the Catan env and converts
    them to edge_index tensors (COO format, shape [2, num_edges]) for each
    relation type.

    Returns
    -------
    tuple of 5 tensors
        (hex_to_vertex, vertex_to_hex,
         vertex_to_edge, edge_to_vertex,
         vertex_to_vertex)
        Each is a [2, E] int64 tensor.
    """
    from ..env.catan_env import _EDGE_VERTICES, _HEX_VERTICES, _VERT_ADJ

    # hex -> vertex: each hex connects to its 6 vertices
    hex_src: list[int] = []
    vert_dst: list[int] = []
    for hi in range(19):
        for ci in range(6):
            vid = int(_HEX_VERTICES[hi, ci])
            hex_src.append(hi)
            vert_dst.append(vid)
    hex_to_vertex = torch.tensor([hex_src, vert_dst], dtype=torch.long)
    vertex_to_hex = torch.tensor([vert_dst, hex_src], dtype=torch.long)

    # edge -> vertex: each edge connects to its 2 endpoint vertices
    edge_src: list[int] = []
    vert_dst2: list[int] = []
    for ei in range(72):
        v0, v1 = int(_EDGE_VERTICES[ei, 0]), int(_EDGE_VERTICES[ei, 1])
        edge_src.extend([ei, ei])
        vert_dst2.extend([v0, v1])
    edge_to_vertex = torch.tensor([edge_src, vert_dst2], dtype=torch.long)
    vertex_to_edge_src: list[int] = []
    vertex_to_edge_dst: list[int] = []
    for ei in range(72):
        v0, v1 = int(_EDGE_VERTICES[ei, 0]), int(_EDGE_VERTICES[ei, 1])
        vertex_to_edge_src.extend([v0, v1])
        vertex_to_edge_dst.extend([ei, ei])
    vertex_to_edge = torch.tensor(
        [vertex_to_edge_src, vertex_to_edge_dst], dtype=torch.long
    )

    # vertex -> vertex: bidirectional adjacency
    v2v_src: list[int] = []
    v2v_dst: list[int] = []
    for v, neighbors in enumerate(_VERT_ADJ):
        for n in neighbors:
            v2v_src.append(v)
            v2v_dst.append(n)
    vertex_to_vertex = torch.tensor([v2v_src, v2v_dst], dtype=torch.long)

    return (
        hex_to_vertex,
        vertex_to_hex,
        vertex_to_edge,
        edge_to_vertex,
        vertex_to_vertex,
    )


# Cache the topology so it's built at most once per process.
_TOPOLOGY_CACHE: (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
) = None


def _get_topology() -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    global _TOPOLOGY_CACHE
    if _TOPOLOGY_CACHE is None:
        _TOPOLOGY_CACHE = _build_catan_topology()
    return _TOPOLOGY_CACHE


# ---------------------------------------------------------------------------
# Heterogeneous GAT Encoder
# ---------------------------------------------------------------------------


class CatanGNNEncoder(nn.Module):
    """3-layer heterogeneous GAT that encodes the Catan board into a fixed-size embedding.

    Node types:
        - hex   (19 nodes)
        - vertex (54 nodes)
        - edge  (72 nodes)

    Edge (relation) types:
        - hex    -> vertex  (hex shares 6 adjacent vertices)
        - vertex -> hex     (reverse)
        - vertex -> edge    (vertex is endpoint of edge)
        - edge   -> vertex  (reverse)
        - vertex -> vertex  (vertices connected by board edges)

    After message passing, mean-pool each node type, concatenate pools + current
    player features, and project through an MLP to ``output_dim``.

    Parameters
    ----------
    hex_in_features : int
        Dimensionality of input hex node features (default env: 9).
    vertex_in_features : int
        Dimensionality of input vertex node features (default env: 7).
    edge_in_features : int
        Dimensionality of input edge node features (default env: 5).
    player_in_features : int
        Dimensionality of per-player feature vector (default env: 14).
    hidden_dim : int
        Hidden dimension for all GAT layers.
    num_heads : int
        Number of attention heads per GAT layer.
    num_layers : int
        Number of heterogeneous GAT layers.
    output_dim : int
        Dimension of the final board embedding.
    """

    # Node / relation type keys
    NODE_TYPES = ("hex", "vertex", "edge")

    RELATION_TYPES: list[tuple[str, str, str]] = [
        ("hex", "hex_to_vertex", "vertex"),
        ("vertex", "vertex_to_hex", "hex"),
        ("vertex", "vertex_to_edge", "edge"),
        ("edge", "edge_to_vertex", "vertex"),
        ("vertex", "vertex_to_vertex", "vertex"),
    ]

    def __init__(
        self,
        hex_in_features: int,
        vertex_in_features: int,
        edge_in_features: int,
        player_in_features: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        output_dim: int = 256,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.player_in_features = player_in_features

        # --- Input projections (node type -> hidden_dim) ---
        self.input_projs = nn.ModuleDict(
            {
                "hex": nn.Linear(hex_in_features, hidden_dim),
                "vertex": nn.Linear(vertex_in_features, hidden_dim),
                "edge": nn.Linear(edge_in_features, hidden_dim),
            }
        )

        # --- GAT layers ---
        self.conv_layers = nn.ModuleList()
        self.norms: nn.ModuleDict  # per-layer, per-node-type LayerNorm
        self.layer_norms = nn.ModuleList()  # list of ModuleDict

        for _ in range(num_layers):
            conv_dict: dict[tuple[str, str, str], GATConv] = {}
            for src, rel, dst in self.RELATION_TYPES:
                # GATConv expects (in_channels, out_channels, heads).
                # We use concat=False so output is hidden_dim (averaged heads).
                conv_dict[(src, rel, dst)] = GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=False,
                    add_self_loops=False,
                )
            self.conv_layers.append(HeteroConv(conv_dict, aggr="sum"))

            norms = nn.ModuleDict(
                {nt: nn.LayerNorm(hidden_dim) for nt in self.NODE_TYPES}
            )
            self.layer_norms.append(norms)

        # --- Output MLP ---
        # After pooling we concatenate the mean-pooled embeddings for each
        # node type (3 * hidden_dim) plus the current player feature vector.
        pool_dim = 3 * hidden_dim + player_in_features
        self.output_mlp = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Placeholder for attention weights (set during forward for explainability)
        self._attention_weights: dict[str, list[torch.Tensor]] = {}

        # Pre-build topology tensors (will be moved to correct device lazily)
        self._edge_indices: dict[tuple[str, str, str], torch.Tensor] | None = None
        self._edge_device: torch.device | None = None

    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------

    def _get_edge_indices(
        self, device: torch.device
    ) -> dict[tuple[str, str, str], torch.Tensor]:
        """Return cached edge indices on the requested device."""
        if self._edge_indices is not None and self._edge_device == device:
            return self._edge_indices

        (
            hex_to_vertex,
            vertex_to_hex,
            vertex_to_edge,
            edge_to_vertex,
            vertex_to_vertex,
        ) = _get_topology()

        raw = {
            ("hex", "hex_to_vertex", "vertex"): hex_to_vertex,
            ("vertex", "vertex_to_hex", "hex"): vertex_to_hex,
            ("vertex", "vertex_to_edge", "edge"): vertex_to_edge,
            ("edge", "edge_to_vertex", "vertex"): edge_to_vertex,
            ("vertex", "vertex_to_vertex", "vertex"): vertex_to_vertex,
        }
        self._edge_indices = {k: v.to(device) for k, v in raw.items()}
        self._edge_device = device
        return self._edge_indices

    # ------------------------------------------------------------------
    # Batching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_edge_index(
        edge_index: torch.Tensor,
        batch_size: int,
        src_count: int,
        dst_count: int,
    ) -> torch.Tensor:
        """Replicate a single-graph edge_index for a batch of graphs.

        Each graph in the batch gets its own offset so node indices don't
        overlap.

        Parameters
        ----------
        edge_index : Tensor [2, E]
            Single-graph edge index.
        batch_size : int
        src_count : int
            Number of source nodes per graph.
        dst_count : int
            Number of destination nodes per graph.

        Returns
        -------
        Tensor [2, E * batch_size]
        """
        if batch_size == 1:
            return edge_index

        device = edge_index.device
        src_offsets = torch.arange(batch_size, device=device) * src_count  # (B,)
        dst_offsets = torch.arange(batch_size, device=device) * dst_count  # (B,)

        # Expand: (B, E) for each row
        src_row = edge_index[0].unsqueeze(0).expand(batch_size, -1) + src_offsets.unsqueeze(1)
        dst_row = edge_index[1].unsqueeze(0).expand(batch_size, -1) + dst_offsets.unsqueeze(1)

        return torch.stack([src_row.reshape(-1), dst_row.reshape(-1)], dim=0)

    _NODE_COUNTS = {"hex": 19, "vertex": 54, "edge": 72}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hex_features: torch.Tensor,
        vertex_features: torch.Tensor,
        edge_features: torch.Tensor,
        player_features: torch.Tensor,
        current_player: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the Catan board state into a fixed-size embedding.

        Parameters
        ----------
        hex_features : Tensor (batch, 19, F_hex)
        vertex_features : Tensor (batch, 54, F_vert)
        edge_features : Tensor (batch, 72, F_edge)
        player_features : Tensor (batch, 4, F_player)
        current_player : Tensor (batch,) int — index of the acting player

        Returns
        -------
        Tensor (batch, output_dim)
            Board-level embedding.
        """
        device = hex_features.device
        batch_sz = hex_features.size(0)

        # Flatten batch dimension: (B, N, F) -> (B*N, F)
        x_dict: dict[str, torch.Tensor] = {
            "hex": hex_features.reshape(-1, hex_features.size(-1)),
            "vertex": vertex_features.reshape(-1, vertex_features.size(-1)),
            "edge": edge_features.reshape(-1, edge_features.size(-1)),
        }

        # Input projection
        x_dict = {nt: self.input_projs[nt](x_dict[nt]) for nt in self.NODE_TYPES}

        # Build batched edge indices
        single_edge_indices = self._get_edge_indices(device)
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}
        for rel_key in self.RELATION_TYPES:
            src_type, _, dst_type = rel_key
            edge_index_dict[rel_key] = self._batch_edge_index(
                single_edge_indices[rel_key],
                batch_sz,
                self._NODE_COUNTS[src_type],
                self._NODE_COUNTS[dst_type],
            )

        # Clear attention weight storage
        self._attention_weights = {rel[1]: [] for rel in self.RELATION_TYPES}

        # Message passing layers
        for layer_idx, (conv, norms) in enumerate(
            zip(self.conv_layers, self.layer_norms)
        ):
            residual = {nt: x_dict[nt] for nt in self.NODE_TYPES}

            # HeteroConv forward
            out_dict: dict[str, torch.Tensor] = conv(
                x_dict, edge_index_dict
            )

            # Residual + LayerNorm + ReLU
            x_dict = {}
            for nt in self.NODE_TYPES:
                if nt in out_dict:
                    h = norms[nt](out_dict[nt] + residual[nt])  # type: ignore[index]
                    x_dict[nt] = torch.relu(h)
                else:
                    # Shouldn't happen with our relation types, but be safe
                    x_dict[nt] = residual[nt]

            # Store attention weights for explainability
            for rel_key in self.RELATION_TYPES:
                # HeteroConv stores convs keyed by tuple in newer torch_geometric
                if tuple(rel_key) in conv.convs:  # type: ignore[operator]
                    gat_layer: GATConv = conv.convs[tuple(rel_key)]  # type: ignore[index]
                elif "__".join(rel_key) in conv.convs:  # type: ignore[operator]
                    gat_layer = conv.convs["__".join(rel_key)]  # type: ignore[index]
                else:
                    continue
                alpha = getattr(gat_layer, "_alpha", None)
                if alpha is not None:
                    self._attention_weights[rel_key[1]].append(alpha.detach())

        # ---- Pooling ----
        # Mean pool each node type across nodes within each graph.
        pooled = []
        for nt in self.NODE_TYPES:
            count = self._NODE_COUNTS[nt]
            # x_dict[nt] shape: (batch_sz * count, hidden_dim)
            h = x_dict[nt].view(batch_sz, count, self.hidden_dim)
            pooled.append(h.mean(dim=1))  # (batch_sz, hidden_dim)

        # Extract the current player's feature vector
        # player_features: (batch_sz, num_players, F_player)
        # current_player: (batch_sz,) int
        batch_idx = torch.arange(batch_sz, device=device)
        current_player_feat = player_features[batch_idx, current_player.long()]

        # Concatenate pools + player features -> MLP
        combined = torch.cat(pooled + [current_player_feat], dim=-1)
        return self.output_mlp(combined)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_attention_weights(self) -> dict[str, list[torch.Tensor]]:
        """Return stored attention weights from the last forward pass.

        Returns
        -------
        dict
            Mapping from relation name to a list of attention weight tensors
            (one per layer). Each tensor has shape (num_edges_in_batch, num_heads).
        """
        return self._attention_weights

    @classmethod
    def from_env_defaults(
        cls,
        num_players: int = 4,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        output_dim: int = 256,
    ) -> CatanGNNEncoder:
        """Construct an encoder with feature dimensions matching :class:`CatanEnv`.

        Parameters
        ----------
        num_players : int
            Number of players (determines owner one-hot sizes).
        hidden_dim, num_heads, num_layers, output_dim
            Architecture hyper-parameters.
        """
        hex_feat_dim = 9  # 6 type one-hot + desert + number(scaled) + robber
        vert_feat_dim = 3 + num_players  # building type one-hot + owner one-hot
        edge_feat_dim = 1 + num_players  # has_road + owner one-hot
        player_feat_dim = 14  # resources(5) + dev_cards(5) + vp + pieces_left(3)

        return cls(
            hex_in_features=hex_feat_dim,
            vertex_in_features=vert_feat_dim,
            edge_in_features=edge_feat_dim,
            player_in_features=player_feat_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=output_dim,
        )
