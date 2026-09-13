"""Specialized gm/gds readout heads for the tower architecture."""

import torch
import torch.nn as nn

from ...components.layers import get_activation


class AttentionSSHead(nn.Module):
    """Terminal attention SS head: self-attention over terminal embeddings,
    then separate learned queries for gm and gds attend over the tokens."""

    def __init__(self, terminal_dim: int, ctx_dim: int = 0, hidden_dim: int = 512,
                 num_attn_layers: int = 1, num_heads: int = 4, mlp_layers: int = 2,
                 dropout: float = 0.0, act_type: str = 'relu'):
        super().__init__()
        self.act_type = act_type
        self.token_dim = terminal_dim + ctx_dim  # each terminal gets its context appended
        # Project all tokens to a common dim divisible by num_heads
        d_model = hidden_dim
        self.token_proj = nn.Linear(self.token_dim, d_model)

        # Self-attention layers over 4 terminal tokens
        attn_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(attn_layer, num_layers=num_attn_layers)

        # Learned query vectors for gm and gds
        self.gm_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.gds_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Final MLPs
        self.gm_mlp = self._build_mlp(d_model, hidden_dim, mlp_layers, dropout)
        self.gds_mlp = self._build_mlp(d_model, hidden_dim, mlp_layers, dropout)

    def _build_mlp(self, in_dim, hidden_dim, num_layers, dropout):
        layers = []
        curr = in_dim
        for i in range(num_layers):
            out = hidden_dim // (2 ** i) if num_layers > 1 else hidden_dim
            out = max(out, 1)
            layers.extend([nn.Linear(curr, out), nn.LayerNorm(out), get_activation(self.act_type)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr = out
        layers.append(nn.Linear(curr, 1))
        return nn.Sequential(*layers)

    def forward(self, gate, drain, source, bulk, ctx_scalars=None):
        """
        Args: gate/drain/source/bulk: [N, terminal_dim], ctx_scalars: [N, ctx_dim] or None
        Returns: gm_pred [N], gds_pred [N]
        """
        tokens = [gate, drain, source, bulk]
        if ctx_scalars is not None:
            # Append per-terminal voltage context: Vgs→gate, Vds→drain, Vbs→bulk, 0→source
            n = gate.shape[0]
            # Split ctx: [Vgs, Vds, Vbs] + optional [ID] + optional [Vov]
            ctx_dim = ctx_scalars.shape[-1]
            # Each terminal gets the full context (terminal-agnostic)
            tokens = [torch.cat([t, ctx_scalars], dim=-1) for t in tokens]

        # Stack: [N, 4, token_dim] → project → [N, 4, d_model]
        x = torch.stack(tokens, dim=1)
        x = self.token_proj(x)

        # Self-attention over terminals
        x = self.self_attn(x)  # [N, 4, d_model]

        # Cross-attention: learned query attends over terminal tokens
        n = x.shape[0]
        gm_q = self.gm_query.expand(n, -1, -1)   # [N, 1, d_model]
        gds_q = self.gds_query.expand(n, -1, -1)  # [N, 1, d_model]

        gm_repr, _ = self.cross_attn(gm_q, x, x)   # [N, 1, d_model]
        gds_repr, _ = self.cross_attn(gds_q, x, x)  # [N, 1, d_model]

        gm_pred = self.gm_mlp(gm_repr.squeeze(1)).squeeze(-1)    # [N]
        gds_pred = self.gds_mlp(gds_repr.squeeze(1)).squeeze(-1)  # [N]
        return gm_pred, gds_pred


class GatedSSHead(nn.Module):
    """Gated pairwise SS head: explicit pairwise terminal interactions
    with learned gates selecting relevant interactions for gm vs gds."""

    def __init__(self, terminal_dim: int, ctx_dim: int = 0, hidden_dim: int = 512,
                 mlp_layers: int = 2, dropout: float = 0.0, act_type: str = 'relu'):
        super().__init__()
        self.act_type = act_type
        self.terminal_dim = terminal_dim

        # Pairwise interaction projections
        pair_dim = terminal_dim
        self.gate_source_proj = nn.Linear(terminal_dim, pair_dim)
        self.drain_source_proj = nn.Linear(terminal_dim, pair_dim)
        self.gate_drain_proj = nn.Linear(terminal_dim, pair_dim)

        # Gating: separate gates for gm and gds over the 3 pairwise + 4 terminal features
        # Input: 3 pairwise interactions + 4 terminal embeddings = 7 * pair_dim
        gate_input_dim = 7 * pair_dim + ctx_dim
        self.gm_gate = nn.Sequential(
            nn.Linear(gate_input_dim, 7),
            nn.Sigmoid(),
        )
        self.gds_gate = nn.Sequential(
            nn.Linear(gate_input_dim, 7),
            nn.Sigmoid(),
        )

        # Final MLPs take gated features (7 * pair_dim + ctx_dim)
        mlp_in = 7 * pair_dim + ctx_dim
        self.gm_mlp = self._build_mlp(mlp_in, hidden_dim, mlp_layers, dropout)
        self.gds_mlp = self._build_mlp(mlp_in, hidden_dim, mlp_layers, dropout)

    def _build_mlp(self, in_dim, hidden_dim, num_layers, dropout):
        layers = []
        curr = in_dim
        for i in range(num_layers):
            out = hidden_dim // (2 ** i) if num_layers > 1 else hidden_dim
            out = max(out, 1)
            layers.extend([nn.Linear(curr, out), nn.LayerNorm(out), get_activation(self.act_type)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr = out
        layers.append(nn.Linear(curr, 1))
        return nn.Sequential(*layers)

    def forward(self, gate, drain, source, bulk, ctx_scalars=None):
        """
        Args: gate/drain/source/bulk: [N, terminal_dim], ctx_scalars: [N, ctx_dim] or None
        Returns: gm_pred [N], gds_pred [N]
        """
        # Pairwise interactions (element-wise product → project)
        gs = self.gate_source_proj(gate * source)  # gm-relevant
        ds = self.drain_source_proj(drain * source)  # gds-relevant
        gd = self.gate_drain_proj(gate * drain)

        # Stack all 7 features: 4 terminals + 3 pairwise
        all_features = [gate, drain, source, bulk, gs, ds, gd]  # each [N, dim]
        stacked = torch.stack(all_features, dim=1)  # [N, 7, dim]

        # Flat concat for gate computation
        flat = torch.cat(all_features, dim=-1)  # [N, 7*dim]
        if ctx_scalars is not None:
            flat_with_ctx = torch.cat([flat, ctx_scalars], dim=-1)  # [N, 7*dim + ctx]
        else:
            flat_with_ctx = flat

        # Compute gates: [N, 7] — one weight per feature group
        gm_g = self.gm_gate(flat_with_ctx).unsqueeze(-1)    # [N, 7, 1]
        gds_g = self.gds_gate(flat_with_ctx).unsqueeze(-1)   # [N, 7, 1]

        # Apply gates and flatten
        gm_gated = (stacked * gm_g).reshape(stacked.shape[0], -1)    # [N, 7*dim]
        gds_gated = (stacked * gds_g).reshape(stacked.shape[0], -1)  # [N, 7*dim]

        # Append context scalars to MLP input
        if ctx_scalars is not None:
            gm_gated = torch.cat([gm_gated, ctx_scalars], dim=-1)
            gds_gated = torch.cat([gds_gated, ctx_scalars], dim=-1)

        gm_pred = self.gm_mlp(gm_gated).squeeze(-1)
        gds_pred = self.gds_mlp(gds_gated).squeeze(-1)
        return gm_pred, gds_pred
