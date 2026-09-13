"""Multi-head self-attention restricted to net nodes (the wires/voltage nodes).

Allows direct net-to-net information flow without going through 2-6 hops of
local message passing. Useful for V prediction because:
  - Voltage dividers couple 2-3 nets that mutually constrain each other.
  - Current mirrors enforce equal gate voltages on potentially distant nets.
  - Multi-stage opamps have long-range V dependencies (input → output stage).

Terminal/net split is positional: nodes 0..num_terminals-1 are device terminals,
nodes num_terminals..N-1 are nets. Operates per graph (no cross-graph attention).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


class NetSelfAttention(nn.Module):
    """Net-to-net multi-head self-attention with residual + LayerNorm."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor,
                num_terminals, ptr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N_total, hidden_dim]
            batch_idx: [N_total] graph index per node
            num_terminals: int or [B] tensor — number of terminal nodes per graph
            ptr: [B+1] node boundaries
        Returns:
            x with net-node positions updated (terminal positions unchanged).
        """
        device = x.device
        N = x.size(0)
        local_idx = torch.arange(N, device=device) - ptr[batch_idx]
        if isinstance(num_terminals, int):
            net_mask = local_idx >= num_terminals
        else:
            net_mask = local_idx >= num_terminals[batch_idx]

        if not net_mask.any():
            return x

        x_nets = x[net_mask]
        batch_nets = batch_idx[net_mask]

        # Pad to [B, max_nets, d] with mask
        x_dense, valid_mask = to_dense_batch(x_nets, batch_nets)
        key_padding_mask = ~valid_mask  # MHA convention: True = pad/skip

        attn_out, _ = self.attn(
            x_dense, x_dense, x_dense,
            key_padding_mask=key_padding_mask, need_weights=False)
        # Residual + post-norm
        out = self.norm(attn_out + x_dense)

        # Gather back to flat [N_nets, d]
        out_flat = out[valid_mask]

        # Write back into a copy at net positions
        x_new = x.clone()
        x_new[net_mask] = out_flat
        return x_new
