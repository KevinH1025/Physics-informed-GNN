"""Subcircuit attention pooling and DAG helpers for the tower architecture."""

import torch
import torch.nn as nn


class SubcircuitAttentionPool(nn.Module):
    """Multi-head attention pooling for subcircuit terminal embeddings.

    Learnable query tokens attend over variable-length terminal embeddings
    to produce a fixed-size subcircuit representation.
    """

    def __init__(self, input_dim: int = 128, output_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.output_dim = output_dim
        self.queries = nn.Parameter(torch.randn(1, num_heads, self.head_dim) * 0.02)
        self.key_proj = nn.Linear(input_dim, self.head_dim)
        self.value_proj = nn.Linear(input_dim, self.head_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self._scale = self.head_dim ** -0.5

    def forward(self, terminal_embs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Pool terminal embeddings to subcircuit embedding.

        Args:
            terminal_embs: [B, T, input_dim] padded terminal embeddings
            mask: [B, T] bool, True for valid terminals

        Returns:
            [B, output_dim] subcircuit embeddings
        """
        B, T, _ = terminal_embs.shape
        K = self.key_proj(terminal_embs)   # [B, T, head_dim]
        V = self.value_proj(terminal_embs)  # [B, T, head_dim]
        Q = self.queries.expand(B, -1, -1)  # [B, num_heads, head_dim]

        scores = torch.bmm(Q, K.transpose(1, 2)) * self._scale  # [B, num_heads, T]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        weights = torch.softmax(scores, dim=-1)

        out = torch.bmm(weights, V)  # [B, num_heads, head_dim]
        out = out.reshape(B, self.output_dim)
        out = self.norm(self.output_proj(out))
        return out


def build_subcircuit_assignment(sc_groups) -> torch.Tensor:
    """Build assignment: mosfet index → subcircuit index."""
    assignment = torch.full((24,), -1, dtype=torch.long)
    for sc_idx, (sc_name, device_indices) in enumerate(sc_groups.items()):
        for dev_idx in device_indices:
            assignment[dev_idx] = sc_idx
    return assignment


def build_dag_topo_order(sc_edges, num_groups: int) -> list:
    """Build topological order from DAG edges."""
    parent_map = {i: [] for i in range(num_groups)}
    for src, dst in sc_edges:
        parent_map[dst].append(src)
    # Process non-root nodes in dependency order
    topo_order = []
    processed = set(i for i in range(num_groups) if not parent_map[i])
    remaining = set(range(num_groups)) - processed
    while remaining:
        for child in sorted(remaining):
            if all(p in processed for p in parent_map[child]):
                topo_order.append((child, parent_map[child]))
                processed.add(child)
        remaining -= processed
    return topo_order
