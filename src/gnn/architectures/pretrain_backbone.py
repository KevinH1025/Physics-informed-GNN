"""Pretraining backbone — minimal architecture for masked feature prediction.

Reuses the same building blocks (input_linear, backbone GENConv layers, VN,
loop attention) as TowerGENConv with matching state-dict prefixes so the
pretrained weights can be loaded into a TowerGENConv via `strict=False`.

Pretraining task: mask MOSFET design features (W, L, wl_ratio, M = x[:,0:4])
at a random subset of MOSFETs, then predict them back from the backbone
embedding at the masked terminal nodes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gnn.components.layers import create_deepgcn_layer, get_activation
from src.gnn.components.virtual_node import VirtualNode
from src.gnn.components.loop_attention import LoopAttention


class PretrainBackbone(nn.Module):
    """Backbone-only model with a small mask-prediction head.

    State-dict prefixes mirror TowerGENConv:
      input_linear.*, backbone.*, virtual_node.*, loop_attn_layers.*

    The mask head (mask_head.*) is not part of TowerGENConv and is dropped
    when fine-tuning (strict=False).
    """

    def __init__(
        self,
        node_feature_dim: int,
        type_feature_dim: int,
        hidden_dim: int,
        backbone_layers: int = 8,
        genconv_num_layers: int = 2,
        norm_type: str = 'layer',
        act_type: str = 'gelu',
        dropout: float = 0.0,
        skip_connection: bool = False,
        conv_type: str = 'gin',
        mlp_depth: int = 3,
        # Virtual node
        use_virtual_node: bool = True,
        vn_mode: str = 'mha',
        vn_num_heads: int = 4,
        vn_head_dim: int = 32,
        vn_gate_broadcast: bool = True,
        # Loop attention
        use_loop_attention: bool = True,
        loop_num_heads: int = 4,
        loop_head_dim: int = 32,
        loop_fusion: str = 'gate',
        loop_warmup_epochs: int = 0,
        loop_warmup_duration: int = 0,
        # Mask head
        mask_target_dim: int = 4,
        mask_head_hidden: int = 128,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.type_feature_dim = type_feature_dim
        self.hidden_dim = hidden_dim
        self.skip_connection = skip_connection
        self.conv_type = conv_type
        self.act_type = act_type
        self.use_loop_attention = use_loop_attention
        self.loop_attn_warmup_epochs = loop_warmup_epochs
        self.loop_attn_warmup_duration = loop_warmup_duration
        self.current_epoch = 0  # set by training loop

        input_dim = node_feature_dim + type_feature_dim
        self.input_linear = nn.Linear(input_dim, hidden_dim)

        # Backbone GENConv stack (no edge features)
        self.backbone = nn.ModuleList([
            create_deepgcn_layer(
                hidden_dim, genconv_num_layers, norm_type, dropout,
                edge_dim=None, act_type=act_type, conv_type=conv_type,
                mlp_depth=mlp_depth,
            )
            for _ in range(backbone_layers)
        ])

        # Virtual node
        if use_virtual_node:
            self.virtual_node = VirtualNode(
                hidden_dim=hidden_dim,
                use_attention_pooling=True,
                gate_broadcast=vn_gate_broadcast,
                mode=vn_mode,
                num_heads=vn_num_heads,
                head_dim=vn_head_dim,
                num_layers=backbone_layers,
                act_type=act_type,
            )
        else:
            self.virtual_node = None

        # Loop attention (one per backbone layer)
        if use_loop_attention:
            self.loop_attn_layers = nn.ModuleList([
                LoopAttention(
                    hidden_dim=hidden_dim,
                    num_heads=loop_num_heads,
                    head_dim=loop_head_dim,
                    fusion=loop_fusion,
                    dropout=dropout,
                    level='device',
                )
                for _ in range(backbone_layers)
            ])
        else:
            self.loop_attn_layers = None

        # Loop topology cache (computed once per topology and reused)
        self._cached_loop_ei = None
        self._cached_dtm = None
        self._cached_node_loop_ei = None
        self._loop_edges_printed = False

        # Mask prediction head — predicts (W, L, wl_ratio, M) from backbone
        # embedding at MOSFET terminal nodes.
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, mask_head_hidden),
            get_activation(act_type),
            nn.Linear(mask_head_hidden, mask_target_dim),
        )

    # ── Loop topology helper for mixed-topology batches.
    @staticmethod
    def _compute_single_topo_loop(
        mi_single: torch.Tensor,
        ci_single: Optional[torch.Tensor],
        num_terminals: int,
    ):
        """Compute device-level loop info for a single-graph topology.

        Returns (loop_ei_single, dtm_single, num_devices, num_cycles, num_loop_edges).
        """
        from collections import defaultdict
        import networkx as nx

        device_terms = []
        device_nets = []
        for row in mi_single:
            gate_idx = row[0].item()
            drain_idx = row[1].item()
            source_idx = row[2].item()
            gate_net = row[3].item()
            drain_net = row[4].item()
            source_net = row[5].item()
            bulk_idx = drain_idx + 3
            terms = [gate_idx, drain_idx, source_idx]
            if bulk_idx < num_terminals:
                terms.append(bulk_idx)
            device_terms.append(terms)
            device_nets.append({gate_net, drain_net, source_net})

        if ci_single is not None and ci_single.shape[0] > 0:
            for row in ci_single:
                t0, t1 = int(row[0].item()), int(row[1].item())
                n0, n1 = int(row[2].item()), int(row[3].item())
                device_terms.append([t0, t1])
                device_nets.append({n0, n1})

        num_devices = len(device_terms)
        if num_devices == 0:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 4), dtype=torch.long),
                0, 0, 0,
            )

        net_to_devs = defaultdict(set)
        for d, nets in enumerate(device_nets):
            for net in nets:
                net_to_devs[net].add(d)

        G = nx.Graph()
        G.add_nodes_from(range(num_devices))
        for _net, devs in net_to_devs.items():
            devs = list(devs)
            for i in range(len(devs)):
                for j in range(i + 1, len(devs)):
                    G.add_edge(devs[i], devs[j])

        cycles = nx.cycle_basis(G)
        loop_edges = set()
        for cycle in cycles:
            for i in range(len(cycle)):
                for j in range(i + 1, len(cycle)):
                    a, b = min(cycle[i], cycle[j]), max(cycle[i], cycle[j])
                    loop_edges.add((a, b))
        if loop_edges:
            src, dst = [], []
            for a, b in loop_edges:
                src.extend([a, b])
                dst.extend([b, a])
            loop_ei_single = torch.tensor([src, dst], dtype=torch.long)
        else:
            loop_ei_single = torch.zeros((2, 0), dtype=torch.long)

        max_terms = 4
        dtm_single = torch.full((num_devices, max_terms), -1, dtype=torch.long)
        for d, terms in enumerate(device_terms):
            for t, idx in enumerate(terms[:max_terms]):
                dtm_single[d, t] = idx

        return loop_ei_single, dtm_single, num_devices, len(cycles), loop_ei_single.shape[1]

    def _get_loop_data(self, data):
        """Compute batched loop_edge_index + device_terminal_map for the full
        mixed-topology batch (cached after first call).

        Iterates over `data.topo_node_slices`, computes per-topo loop info
        from the first graph's slice, replicates to per_topo graphs within
        that topology block, and concatenates with proper offsets.
        """
        if self._cached_loop_ei is not None:
            return self._cached_loop_ei, self._cached_dtm, None

        device = data.ptr.device
        topo_slices = getattr(data, 'topo_node_slices', None)
        if topo_slices is None:
            raise RuntimeError(
                'PretrainBackbone requires data.topo_node_slices to compute '
                'mixed-topology loop info.'
            )

        # Walk graphs in batch order. For each topo block we need the slice
        # of mosfet_info / capacitor_info that belongs to its first graph.
        mosfet_ptr = data.mosfet_ptr
        capacitor_ptr = data.capacitor_ptr
        ptr = data.ptr
        num_terminals = data.num_terminals
        loop_ei_parts = []
        dtm_parts = []
        device_offset = 0
        report_lines = []

        graph_idx = 0
        for t_name, (n_start, n_end, n_nodes, n_graphs_t) in topo_slices.items():
            # First graph in this topo block
            mi_start = mosfet_ptr[graph_idx].item()
            mi_end = mosfet_ptr[graph_idx + 1].item()
            ci_start = capacitor_ptr[graph_idx].item()
            ci_end = capacitor_ptr[graph_idx + 1].item()
            mi_single = data.mosfet_info[mi_start:mi_end].clone()
            # Strip the node-offset on mi terminal cols so we work in
            # per-graph local space (single-graph computation).
            node_offset_local = ptr[graph_idx].item()
            mi_single[:, 0] -= node_offset_local
            mi_single[:, 1] -= node_offset_local
            mi_single[:, 2] -= node_offset_local
            ci_single = data.capacitor_info[ci_start:ci_end].clone()
            if ci_single.shape[0] > 0:
                ci_single[:, 0] -= node_offset_local
                ci_single[:, 1] -= node_offset_local
            num_term_t = num_terminals[graph_idx].item() \
                if isinstance(num_terminals, torch.Tensor) else int(num_terminals)

            (loop_ei_single, dtm_single,
             num_devs, num_cycles, num_loop_e) = self._compute_single_topo_loop(
                mi_single, ci_single, num_term_t,
            )
            loop_ei_single = loop_ei_single.to(device)
            dtm_single = dtm_single.to(device)
            report_lines.append(
                f'  {t_name}: {num_devs} devices, {num_cycles} cycles, {num_loop_e} loop edges/graph'
            )

            # Replicate to all `n_graphs_t` graphs in this topo block
            E_loop = loop_ei_single.shape[1]
            if E_loop > 0:
                dev_offsets = (
                    torch.arange(n_graphs_t, device=device) * num_devs + device_offset
                )
                loop_ei_block = (
                    loop_ei_single.repeat(1, n_graphs_t)
                    + dev_offsets.repeat_interleave(E_loop).unsqueeze(0)
                )
                loop_ei_parts.append(loop_ei_block)

            dtm_block = dtm_single.repeat(n_graphs_t, 1)
            graph_node_offsets = ptr[graph_idx:graph_idx + n_graphs_t].repeat_interleave(num_devs)
            pad_mask = dtm_block >= 0
            dtm_block[pad_mask] += graph_node_offsets.unsqueeze(1).expand_as(dtm_block)[pad_mask]
            dtm_parts.append(dtm_block)

            device_offset += num_devs * n_graphs_t
            graph_idx += n_graphs_t

        loop_ei = (
            torch.cat(loop_ei_parts, dim=1)
            if loop_ei_parts
            else torch.zeros((2, 0), dtype=torch.long, device=device)
        )
        dtm = torch.cat(dtm_parts, dim=0)

        if not self._loop_edges_printed:
            print('[PretrainBackbone] loop topology (per-topo):')
            for line in report_lines:
                print(line)
            print(f'  total: {dtm.shape[0]} devices in batch, {loop_ei.shape[1]} loop edges')
            self._loop_edges_printed = True

        self._cached_loop_ei = loop_ei
        self._cached_dtm = dtm
        self._cached_node_loop_ei = None
        return loop_ei, dtm, None

    def _get_input_features(self, data) -> torch.Tensor:
        x = data.x
        if hasattr(data, 'type_tens') and data.type_tens is not None:
            x = torch.cat([x, data.type_tens], dim=-1)
        return x

    def forward(
        self,
        data,
        mask_indices: Optional[torch.Tensor] = None,
    ):
        """Run backbone and predict masked features.

        Args:
            data: PretrainBatch with x (already masked) + structural tensors.
            mask_indices: [K] node indices (in batched node space) where the
                mask head should produce predictions. If None, predict at all
                MOSFET terminal nodes (`data.batched_mosfet_term`).

        Returns:
            dict with:
              'mask_pred': [K, mask_target_dim] predicted features
              'backbone_hidden': [N, hidden_dim] full backbone embedding
        """
        x_in = self._get_input_features(data)
        x = self.input_linear(x_in)

        batch_vec = data.batch
        num_graphs = data.num_graphs

        vn_emb = None
        if self.virtual_node is not None:
            vn_emb = self.virtual_node.init_embedding(num_graphs)

        # Loop attention topology
        _loop_ei = _loop_dtm = _node_loop_ei = None
        if self.use_loop_attention and self.loop_attn_layers is not None:
            _loop_ei, _loop_dtm, _node_loop_ei = self._get_loop_data(data)

        vn_is_mha = (
            self.virtual_node is not None
            and self.virtual_node.mode == 'mha'
        )
        vn_is_default = (
            self.virtual_node is not None
            and self.virtual_node.mode == 'default'
            and vn_emb is not None
        )

        # For VN MHA with mixed topologies, apply MHA per-topo slice
        # (each slice contains per_topo graphs of uniform node count).
        topo_slices = getattr(data, 'topo_node_slices', None)

        for layer_idx, layer in enumerate(self.backbone):
            # VN injection (MHA-style: stateless global attention)
            if vn_is_mha:
                if topo_slices is not None:
                    mha_out = torch.zeros_like(x)
                    for (start, end, _, _) in topo_slices.values():
                        x_slice = x[start:end]
                        batch_slice = batch_vec[start:end] - batch_vec[start]
                        mha_out[start:end] = self.virtual_node.global_mha(
                            x_slice, batch_slice, layer_idx=layer_idx,
                        )
                    x = x + mha_out
                else:
                    x = x + self.virtual_node.global_mha(x, batch_vec, layer_idx=layer_idx)
            elif vn_is_default:
                x = x + self.virtual_node.broadcast(vn_emb, batch_vec, layer_idx=layer_idx)

            # Message passing
            x = layer(x, data.edge_index)

            # Loop attention (after warmup)
            if (self.loop_attn_layers is not None
                    and self.current_epoch >= self.loop_attn_warmup_epochs):
                la_layer = self.loop_attn_layers[layer_idx]
                if _loop_dtm is not None and _loop_ei is not None:
                    x_loop = la_layer(x, _loop_dtm, _loop_ei)
                    if self.loop_attn_warmup_duration > 0:
                        progress = self.current_epoch - self.loop_attn_warmup_epochs
                        alpha = min(1.0, progress / self.loop_attn_warmup_duration)
                        x = x + alpha * x_loop
                    else:
                        x = x + x_loop

            # Update default-mode VN
            if vn_is_default:
                vn_emb, _ = self.virtual_node(x, vn_emb, batch_vec, num_graphs)

        # Mask prediction at requested nodes
        if mask_indices is None:
            mask_indices = data.batched_mosfet_term
        h_masked = x[mask_indices]
        mask_pred = self.mask_head(h_masked)

        return {
            'mask_pred': mask_pred,
            'backbone_hidden': x,
        }
