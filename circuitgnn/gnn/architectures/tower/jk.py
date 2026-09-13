"""Jumping Knowledge aggregation for the tower architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...components.layers import get_activation


class JKAggregation(nn.Module):
    """Reusable Jumping Knowledge aggregation module."""

    def __init__(
        self,
        hidden_dim: int,
        num_outputs: int,
        mode: str = 'last',
        attention: bool = False,
        learn_temperature: bool = False,
        gated_residual: bool = False,
        act_type: str = 'relu',
    ):
        super().__init__()
        self.mode = mode
        self.attention = attention
        self.per_feature = (mode == 'attention')  # per-feature softmax across layers
        self.hidden_dim = hidden_dim
        self.num_outputs = num_outputs
        self.gated_residual = gated_residual
        self.act_type = act_type

        if self.per_feature:
            # Per-feature softmax: each feature independently picks its layer mix
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                get_activation(self.act_type),
                nn.Linear(hidden_dim // 2, hidden_dim),
            )
            self.linear = None
        elif gated_residual:
            # Attention over layers 0..N-2, gated addition to last layer
            n_early = max(num_outputs - 1, 1)
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                get_activation(self.act_type),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.register_buffer('temperature', torch.tensor(1.0))
            self.gate_proj = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)
            self.linear = None
        elif attention:
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                get_activation(self.act_type),
                nn.Linear(hidden_dim // 2, 1),
            )
            if learn_temperature:
                self.temperature = nn.Parameter(torch.tensor(1.0))
            else:
                self.register_buffer('temperature', torch.tensor(1.0))

            if mode == 'cat':
                self.linear = nn.Linear(hidden_dim * num_outputs, hidden_dim)
            else:
                self.linear = None
        elif mode == 'cat':
            self.linear = nn.Linear(hidden_dim * num_outputs, hidden_dim)
            self.attn = None
        else:
            self.linear = None
            self.attn = None

    def forward(self, layer_outputs: list) -> torch.Tensor:
        if self.gated_residual:
            x_last = layer_outputs[-1]  # [N, hidden_dim] — base
            if len(layer_outputs) > 1:
                early = torch.stack(layer_outputs[:-1], dim=0)  # [L-1, N, hidden_dim]
                attn_scores = self.attn(early)
                temperature = self.temperature.clamp(min=0.1)
                attn_weights = F.softmax(attn_scores / temperature, dim=0)
                x_jk = (early * attn_weights).sum(dim=0)  # [N, hidden_dim]
                gate = torch.sigmoid(self.gate_proj(x_last))  # [N, 1]
                x = x_last + gate * x_jk
            else:
                x = x_last
            return x

        if self.per_feature:
            layer_stack = torch.stack(layer_outputs, dim=0)  # [L, N, H]
            attn_scores = self.attn(layer_stack)              # [L, N, H]
            attn_weights = F.softmax(attn_scores, dim=0)      # [L, N, H] sums to 1 across layers
            return (attn_weights * layer_stack).sum(dim=0)     # [N, H]

        if self.attention:
            layer_stack = torch.stack(layer_outputs, dim=0)
            attn_scores = self.attn(layer_stack)
            temperature = self.temperature.clamp(min=0.1)
            attn_weights = F.softmax(attn_scores / temperature, dim=0)

            if self.linear is not None:
                weighted_stack = layer_stack * attn_weights
                x = weighted_stack.permute(1, 0, 2).reshape(layer_stack.size(1), -1)
                x = self.linear(x)
            else:
                x = (layer_stack * attn_weights).sum(dim=0)

        elif self.mode == 'cat':
            x = torch.cat(layer_outputs, dim=-1)
            x = self.linear(x)

        elif self.mode == 'max':
            x = torch.stack(layer_outputs, dim=0).max(dim=0)[0]

        elif self.mode == 'sum':
            x = torch.stack(layer_outputs, dim=0).sum(dim=0)

        else:  # 'last'
            x = layer_outputs[-1]

        return x
