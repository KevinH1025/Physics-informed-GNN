"""Stratified mixed-topology loader for pretraining.

The combined dataset has 4 fixed topologies. Within each topology every
sample shares the same structural tensors (edge_index, mosfet_info,
loop_edge_index, device_terminal_map, masks, type_tens, edge_attr,
capacitor_info, vsource_info). Only `x` and a few per-sample tensors vary.

This loader:
  1. Loads the pkl once, groups samples by topology, stacks the variable
     tensors per topology onto GPU.
  2. Builds a single fixed batched-topology template containing B/4 graphs
     from each of the 4 topologies (offsets, batch vector, ptr already
     baked in).
  3. Per step: samples B/4 indices from each topology and gathers `x` (and
     the small set of per-sample tensors actually needed by the backbone)
     by reshape+slice — no PyG `Batch.from_data_list`, no Python loop.

The result is a Data-like object the existing TowerGENConv backbone can
consume unchanged.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from typing import Dict, List, Optional

import torch

from circuitgnn.data import net_roles


# Per-sample varying tensors actually needed by the backbone forward path.
# Everything else is per-topology fixed and pre-batched into the template.
_VARYING_NEEDED = ('x',)


class PretrainBatch:
    """Lightweight Data-like wrapper exposing tensors as attributes."""

    def __init__(self, attrs: Dict[str, object]):
        self.__dict__.update(attrs)

    def keys(self):
        return self.__dict__.keys()


class PretrainCombinedLoader:
    """Stratified mixed-topology pretraining loader.

    Args:
        path: pickle path (list of dicts with 'topology' + 'graph').
        batch_size: total batch size (must be divisible by num topologies).
        device: torch device for tensors.
        shuffle: shuffle within each topology each epoch.
        drop_last: drop the trailing partial batch.
    """

    def __init__(
        self,
        path: str,
        batch_size: int,
        device: torch.device,
        shuffle: bool = True,
        drop_last: bool = True,
        topology_filter: Optional[str] = None,
        max_per_topo: Optional[int] = None,
        exclude_topology: Optional[str] = None,
    ):
        with open(path, 'rb') as f:
            samples = pickle.load(f)

        by_topo = defaultdict(list)
        for s in samples:
            by_topo[s['topology']].append(s)

        # Optional per-topology restriction: useful for per-topology baselines.
        if topology_filter is not None:
            if topology_filter not in by_topo:
                raise ValueError(
                    f'topology_filter={topology_filter!r} not in dataset; '
                    f'available: {sorted(by_topo.keys())}'
                )
            by_topo = defaultdict(list, {topology_filter: by_topo[topology_filter]})

        # Optional held-out topology: drops one topology entirely. Used for
        # zero-shot transfer training (train on N-1 topos, evaluate on the
        # held-out one separately).
        if exclude_topology is not None:
            if exclude_topology not in by_topo:
                raise ValueError(
                    f'exclude_topology={exclude_topology!r} not in dataset; '
                    f'available: {sorted(by_topo.keys())}'
                )
            by_topo = defaultdict(list, {
                t: lst for t, lst in by_topo.items() if t != exclude_topology
            })

        # Optional few-shot mode: cap each topology to the first N samples.
        # Joint training with max_per_topo=N → N samples per topo (4N total).
        # Per-topo training with max_per_topo=N → N total (since only 1 topo).
        if max_per_topo is not None:
            by_topo = defaultdict(list, {
                t: lst[:max_per_topo] for t, lst in by_topo.items()
            })

        self.topo_names = sorted(by_topo.keys())
        self.num_topos = len(self.topo_names)
        if batch_size % self.num_topos != 0:
            raise ValueError(
                f'batch_size ({batch_size}) must be divisible by num topologies '
                f'({self.num_topos})'
            )
        self.batch_size = batch_size
        self.per_topo = batch_size // self.num_topos
        self.device = device
        self.shuffle = shuffle
        self.drop_last = drop_last

        # ── Per-topology variable x stack (only x varies in cols 0-3, but we
        # stack the full row to keep simple). All other per-sample tensors are
        # not needed by the backbone for pretraining (no targets needed).
        self._topo_x: Dict[str, torch.Tensor] = {}
        self._topo_size: Dict[str, int] = {}

        # Per-topology mosfet terminal node indices (within the single graph).
        # Used for masking (cols 0-3 at these positions only).
        self._topo_mosfet_term: Dict[str, torch.Tensor] = {}

        # Per-topology stacks for physics-pretrain extras (varying per sample).
        # These are present after running scripts/patch_pretrain_combined.py.
        self._topo_wl_um: Dict[str, torch.Tensor] = {}     # [n, M, 2] µm
        self._topo_m: Dict[str, torch.Tensor] = {}          # [n, M] effective multiplier
        self._topo_v_targets: Dict[str, torch.Tensor] = {}  # [n, N] node V (SPICE truth)
        # Supervised-pretrain targets (per-sample): currents, gm/gds.
        self._topo_i_targets: Dict[str, torch.Tensor] = {}        # [n, N] terminal currents (signed amps)
        self._topo_has_curr_mask: Dict[str, torch.Tensor] = {}    # [n, N] bool mask
        self._topo_term_curr_sign: Dict[str, torch.Tensor] = {}   # [n, N] sign for KCL/I targets
        self._topo_log_gm: Dict[str, torch.Tensor] = {}           # [n, N] log10 gm broadcast to drain nodes
        self._topo_log_gds: Dict[str, torch.Tensor] = {}          # [n, N] log10 gds broadcast to drain nodes
        # Note: node_voltage_targets are loaded so we can inject the KNOWN
        # boundary V values (where known_voltage_mask=True) into the physics
        # chain. We do NOT compute loss on these — they're inputs.

        for t in self.topo_names:
            topo_samples = by_topo[t]
            self._topo_size[t] = len(topo_samples)
            xs = torch.stack([s['graph']['x'] for s in topo_samples], dim=0)
            self._topo_x[t] = xs.to(device)
            mi = topo_samples[0]['graph']['mosfet_info']
            term_idx = self._compute_mosfet_terminals(
                mi, topo_samples[0]['graph']['num_terminals']
            )
            self._topo_mosfet_term[t] = term_idx.to(device)

            # Optional physics-extras (only present if dataset was patched).
            if 'mosfet_wl_um' in topo_samples[0]['graph']:
                wls = torch.stack(
                    [s['graph']['mosfet_wl_um'] for s in topo_samples], dim=0,
                )
                self._topo_wl_um[t] = wls.to(device)
            if 'mosfet_m' in topo_samples[0]['graph']:
                ms = torch.stack(
                    [s['graph']['mosfet_m'] for s in topo_samples], dim=0,
                )
                self._topo_m[t] = ms.to(device)
            if 'node_voltage_targets' in topo_samples[0]['graph']:
                vts = torch.stack(
                    [s['graph']['node_voltage_targets'] for s in topo_samples], dim=0,
                )
                self._topo_v_targets[t] = vts.to(device)
            if 'node_current_targets' in topo_samples[0]['graph']:
                its = torch.stack(
                    [s['graph']['node_current_targets'] for s in topo_samples], dim=0,
                )
                self._topo_i_targets[t] = its.to(device)
            if 'has_current_mask' in topo_samples[0]['graph']:
                hcm = torch.stack(
                    [s['graph']['has_current_mask'] for s in topo_samples], dim=0,
                )
                self._topo_has_curr_mask[t] = hcm.to(device)
            if 'terminal_current_sign' in topo_samples[0]['graph']:
                tcs = torch.stack(
                    [s['graph']['terminal_current_sign'] for s in topo_samples], dim=0,
                )
                self._topo_term_curr_sign[t] = tcs.to(device)
            if 'node_log_gm' in topo_samples[0]['graph']:
                lgm = torch.stack(
                    [s['graph']['node_log_gm'] for s in topo_samples], dim=0,
                )
                self._topo_log_gm[t] = lgm.to(device)
            if 'node_log_gds' in topo_samples[0]['graph']:
                lgds = torch.stack(
                    [s['graph']['node_log_gds'] for s in topo_samples], dim=0,
                )
                self._topo_log_gds[t] = lgds.to(device)

        # ── Build the fixed batched template (per_topo graphs of every topo)
        self._template = self._build_template(by_topo)

        # ── Mirror groups: transistors sharing GATE NET (current mirrors).
        # ── Stage groups: transistors sharing (W, L, M_base) → same param group.
        # Both are computed in BATCHED MOSFET-index space (0..M_total-1) where
        # M_total spans all 4 topo blocks × per_topo graphs.
        self.batched_mirror_groups = self._compute_batched_mirror_groups(by_topo)
        self.batched_stage_groups = self._compute_batched_stage_groups(by_topo)

        # Indices for shuffling per topology
        self._epoch_indices: Dict[str, torch.Tensor] = {}
        self._cursor: Dict[str, int] = {t: 0 for t in self.topo_names}

        # Length: one pass through the smallest topology
        min_topo = min(self._topo_size[t] for t in self.topo_names)
        self._batches_per_epoch = min_topo // self.per_topo
        if not self.drop_last and (min_topo % self.per_topo) != 0:
            self._batches_per_epoch += 1

    # ── Public API ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._batches_per_epoch

    def __iter__(self):
        # Reshuffle per epoch
        for t in self.topo_names:
            n = self._topo_size[t]
            if self.shuffle:
                self._epoch_indices[t] = torch.randperm(n, device=self.device)
            else:
                self._epoch_indices[t] = torch.arange(n, device=self.device)
            self._cursor[t] = 0
        for _ in range(self._batches_per_epoch):
            yield self._next_batch()

    # ── Internals ─────────────────────────────────────────────────────────

    def _compute_batched_mirror_groups(self, by_topo) -> list:
        """For each topo, group MOSFET indices by GATE NET (current mirror partners
        share gate net). Then offset by topo's batched MOSFET-index start so the
        groups are valid in the batched M_total space.

        Returns: flat list of lists. Each inner list contains MOSFET indices
        (in batched space) that share a gate net within a single graph.
        """
        groups_all = []
        topo_mos_slices = self._template['topo_mosfet_slices']  # topo -> (start, end, M_per_t)
        for t in self.topo_names:
            ms_start, ms_end, M_per_t = topo_mos_slices[t]
            mi = by_topo[t][0]['graph']['mosfet_info']  # [M_per_t, 7]
            # Group by gate net (col 3)
            net_to_mos = {}
            for i in range(M_per_t):
                net = int(mi[i, 3].item())
                net_to_mos.setdefault(net, []).append(i)
            # For each per_topo graph in this topo's block, replicate the groups
            # with proper MOSFET-index offsets.
            for gi in range(self.per_topo):
                graph_mos_offset = ms_start + gi * M_per_t
                for net, members in net_to_mos.items():
                    if len(members) >= 1:
                        groups_all.append([m + graph_mos_offset for m in members])
        return groups_all

    def _compute_batched_stage_groups(self, by_topo) -> list:
        """For each topo, group MOSFET indices by (W, L, M_base) — within a
        topology, transistors sharing param group will have identical (W, L, M)
        in the FIRST sample. We use that to detect groupings (deterministic
        per topo since topology is fixed).

        Returns: list of lists in batched MOSFET-index space.
        """
        groups_all = []
        topo_mos_slices = self._template['topo_mosfet_slices']
        for t in self.topo_names:
            ms_start, ms_end, M_per_t = topo_mos_slices[t]
            sample0 = by_topo[t][0]['graph']
            wl = sample0.get('mosfet_wl_um')
            mfac = sample0.get('mosfet_m')
            if wl is None or mfac is None:
                # Dataset not patched — fall back to single big group per topo
                continue
            keys = []
            for i in range(M_per_t):
                k = (
                    round(float(wl[i, 0].item()), 4),
                    round(float(wl[i, 1].item()), 4),
                    round(float(mfac[i].item()), 1),
                )
                keys.append(k)
            key_to_mos = {}
            for i, k in enumerate(keys):
                key_to_mos.setdefault(k, []).append(i)
            for gi in range(self.per_topo):
                graph_mos_offset = ms_start + gi * M_per_t
                for _, members in key_to_mos.items():
                    if len(members) >= 1:
                        groups_all.append([m + graph_mos_offset for m in members])
        return groups_all

    @staticmethod
    def _compute_mosfet_terminals(
        mosfet_info: torch.Tensor, num_terminals: int
    ) -> torch.Tensor:
        """Return [4 * M] node indices: drain, gate, source, bulk for each MOSFET."""
        if mosfet_info.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        gate_idx = mosfet_info[:, 0]
        drain_idx = mosfet_info[:, 1]
        source_idx = mosfet_info[:, 2]
        bulk_idx = drain_idx + 3
        bulk_valid = bulk_idx < num_terminals
        # Build a flat list — keep bulk only where valid, fill with -1 otherwise
        bulk = torch.where(bulk_valid, bulk_idx, torch.full_like(bulk_idx, -1))
        out = torch.stack([drain_idx, gate_idx, source_idx, bulk], dim=1).flatten()
        return out[out >= 0]

    # Net role classification: 5 classes (VDD/GND/SIG_IN/SIG_OUT/INTERNAL).
    # Same logic as scripts/patch_dataset_net_role.py — shared core lives in
    # circuitgnn.data.net_roles so the combined corpus doesn't need a
    # separate patch step.
    _NET_ROLE_DIM = net_roles.NET_ROLE_DIM
    _ROLE_VDD = net_roles.ROLE_VDD
    _ROLE_GND = net_roles.ROLE_GND
    _ROLE_SIG_IN = net_roles.ROLE_SIG_IN
    _ROLE_SIG_OUT = net_roles.ROLE_SIG_OUT
    _ROLE_INTERNAL = net_roles.ROLE_INTERNAL

    @classmethod
    def _classify_net(cls, name: str) -> int:
        return net_roles.classify_net_role(name)

    def _compute_net_role_for_graph(self, g0) -> torch.Tensor:
        """Build [n_nodes, 5] net_role one-hot for one graph. Net nodes get a
        single 1.0; terminal nodes get all zeros."""
        n_nodes = g0['x'].shape[0]
        node_names = g0.get('node_names')
        node_types = g0.get('node_types')
        out = torch.zeros((n_nodes, self._NET_ROLE_DIM), dtype=torch.float32)
        if node_names is None or node_types is None:
            return out
        for i, (name, ntype) in enumerate(zip(node_names, node_types)):
            if ntype and ntype[0] == 'VNode':
                r = self._classify_net(name)
                out[i, r] = 1.0
        return out

    def _build_template(self, by_topo) -> Dict[str, torch.Tensor]:
        """Pre-batch the structural tensors for (per_topo × num_topos) graphs.

        Result attrs (all on self.device):
          edge_index, edge_attr, batch, ptr, num_graphs,
          mosfet_info, mosfet_ptr, resistor_info (zero rows), capacitor_info,
          capacitor_ptr, loop_edge_index, device_terminal_map, num_terminals,
          type_tens, terminal_train_mask, known_voltage_mask, train_mask,
          mosfet_drain_mask, kcl_include_mask,
          # MOSFET terminal indices in batched node space (for masking head):
          batched_mosfet_term, batched_mosfet_term_topo (which topo each term came from),
          # Per-graph offsets (so we can gather batched x quickly):
          topo_node_slices: Dict[topo, (batch_node_start, batch_node_end, n_nodes)]
        """
        device = self.device
        edge_index_parts = []
        edge_attr_parts = []
        type_tens_parts = []
        net_role_parts = []   # per-node 5-dim one-hot, computed on the fly
        batch_parts = []
        ptr = [0]
        mosfet_info_parts = []
        mosfet_ptr = [0]
        capacitor_info_parts = []
        capacitor_ptr = [0]
        loop_ei_parts = []
        dtm_parts = []
        num_terminals_parts = []
        terminal_train_mask_parts = []
        known_voltage_mask_parts = []
        train_mask_parts = []
        mosfet_drain_mask_parts = []
        kcl_include_mask_parts = []
        # For backbone-only forward we don't need most of these masks, but pass
        # them for API compatibility with TowerGENConv.

        # Track per-topo node ranges (start, end) within the batched node space
        topo_node_slices = {}  # topo -> (start_node, end_node, n_nodes_per, n_graphs)
        topo_mosfet_slices = {}  # topo -> (start_mos, end_mos, n_mosfets_per)
        graph_idx_global = 0

        # We'll also build a flat list of MOSFET terminal indices (in batched
        # node space) and which topology each came from — used for masking.
        batched_mosfet_term_parts = []

        for t in self.topo_names:
            sample0 = by_topo[t][0]
            g0 = sample0['graph']
            n_nodes = g0['x'].shape[0]
            n_edges = g0['edge_index'].shape[1]
            n_mosfets = g0['mosfet_info'].shape[0]
            n_caps = g0['capacitor_info'].shape[0]
            n_loop = g0['loop_edge_index'].shape[1]
            n_devs = g0['device_terminal_map'].shape[0]
            num_term_t = int(g0['num_terminals'])

            topo_node_start = ptr[-1]
            topo_mosfet_start = mosfet_ptr[-1]

            for gi in range(self.per_topo):
                node_offset = ptr[-1]
                edge_index_parts.append(g0['edge_index'] + node_offset)
                if 'edge_attr' in g0:
                    edge_attr_parts.append(g0['edge_attr'])
                type_tens_parts.append(g0['type_tens'])
                # Per-node net_role 5-dim one-hot. Net nodes get a single 1.0
                # in their class column based on net name; terminals get zeros.
                net_role_parts.append(self._compute_net_role_for_graph(g0))
                batch_parts.append(
                    torch.full((n_nodes,), graph_idx_global, dtype=torch.long)
                )
                ptr.append(ptr[-1] + n_nodes)

                # mosfet_info: KEEP cols 0,1,2 as per-graph terminal indices.
                # The downstream device_current_head computes offsets via
                # mosfet_ptr+ptr and adds them itself — pre-offsetting here
                # double-offsets and indexes OOB. Cols 3-6 stay per-graph too.
                mi = g0['mosfet_info'].clone()
                mosfet_info_parts.append(mi)
                mosfet_ptr.append(mosfet_ptr[-1] + n_mosfets)

                # capacitor_info: KEEP per-graph indices (head computes offset).
                ci = g0['capacitor_info'].clone()
                capacitor_info_parts.append(ci)
                capacitor_ptr.append(capacitor_ptr[-1] + n_caps)

                # loop_edge_index — pre-stored is node-level (terminal indices)
                # in the original openloop dataset; in this combined dataset
                # each graph has its own correct loop_edge_index. We DON'T pass
                # it pre-batched — instead let the model recompute via
                # _get_loop_data so the validation check in tower_genconv.py
                # falls back correctly. We omit loop_edge_index/device_terminal_map
                # from the template entirely.

                num_terminals_parts.append(num_term_t)
                terminal_train_mask_parts.append(g0['terminal_train_mask'])
                known_voltage_mask_parts.append(g0['known_voltage_mask'])
                train_mask_parts.append(g0['train_mask'])
                mosfet_drain_mask_parts.append(g0['mosfet_drain_mask'])
                kcl_include_mask_parts.append(g0['kcl_include_mask'])

                # MOSFET terminal indices in batched node space
                term_local = self._topo_mosfet_term[t].cpu()
                batched_mosfet_term_parts.append(term_local + node_offset)

                graph_idx_global += 1

            topo_node_end = ptr[-1]
            topo_node_slices[t] = (
                topo_node_start, topo_node_end,
                n_nodes, self.per_topo,
            )
            topo_mosfet_slices[t] = (
                topo_mosfet_start, mosfet_ptr[-1], n_mosfets,
            )

        edge_index = torch.cat(edge_index_parts, dim=1).to(device)
        edge_attr = (
            torch.cat(edge_attr_parts, dim=0).to(device)
            if edge_attr_parts else None
        )
        type_tens = torch.cat(type_tens_parts, dim=0).to(device)
        net_role = torch.cat(net_role_parts, dim=0).to(device)
        batch_vec = torch.cat(batch_parts, dim=0).to(device)
        ptr_t = torch.tensor(ptr, dtype=torch.long, device=device)
        mosfet_info = torch.cat(mosfet_info_parts, dim=0).to(device)
        mosfet_ptr_t = torch.tensor(mosfet_ptr, dtype=torch.long, device=device)
        capacitor_info = torch.cat(capacitor_info_parts, dim=0).to(device)
        capacitor_ptr_t = torch.tensor(capacitor_ptr, dtype=torch.long, device=device)
        num_terminals_t = torch.tensor(num_terminals_parts, dtype=torch.long, device=device)
        terminal_train_mask = torch.cat(terminal_train_mask_parts, dim=0).to(device)
        known_voltage_mask = torch.cat(known_voltage_mask_parts, dim=0).to(device)
        train_mask = torch.cat(train_mask_parts, dim=0).to(device)
        mosfet_drain_mask = torch.cat(mosfet_drain_mask_parts, dim=0).to(device)
        kcl_include_mask = torch.cat(kcl_include_mask_parts, dim=0).to(device)
        batched_mosfet_term = torch.cat(batched_mosfet_term_parts, dim=0).to(device)

        # Empty resistor_info — we don't pre-batch resistor info per-sample; the
        # backbone path doesn't read it for the architecture used here.
        resistor_info = torch.zeros((0, 5), dtype=torch.float32, device=device)
        resistor_ptr_t = torch.zeros((graph_idx_global + 1,), dtype=torch.long, device=device)

        # Pre-allocated batched x buffer, written into per-step.
        # Shape: (total_nodes, 9). x dtype is float32.
        x_dim = self._topo_x[self.topo_names[0]].shape[-1]
        batched_x = torch.zeros(
            (ptr[-1], x_dim), dtype=torch.float32, device=device
        )

        # Optional physics-extra buffers — only allocated if all topos have them.
        has_wl_um = all(t in self._topo_wl_um for t in self.topo_names)
        has_m = all(t in self._topo_m for t in self.topo_names)
        has_v_targets = all(t in self._topo_v_targets for t in self.topo_names)
        batched_wl_um = (
            torch.zeros((mosfet_ptr[-1], 2), dtype=torch.float32, device=device)
            if has_wl_um else None
        )
        batched_m = (
            torch.zeros((mosfet_ptr[-1],), dtype=torch.float32, device=device)
            if has_m else None
        )
        batched_v_targets = (
            torch.zeros((ptr[-1],), dtype=torch.float32, device=device)
            if has_v_targets else None
        )

        # Supervised-pretrain extra buffers (per-node current & gm/gds).
        has_i_targets = all(t in self._topo_i_targets for t in self.topo_names)
        has_curr_mask = all(t in self._topo_has_curr_mask for t in self.topo_names)
        has_term_sign = all(t in self._topo_term_curr_sign for t in self.topo_names)
        has_log_gm = all(t in self._topo_log_gm for t in self.topo_names)
        has_log_gds = all(t in self._topo_log_gds for t in self.topo_names)
        batched_i_targets = (
            torch.zeros((ptr[-1],), dtype=torch.float32, device=device)
            if has_i_targets else None
        )
        batched_has_curr_mask = (
            torch.zeros((ptr[-1],), dtype=torch.bool, device=device)
            if has_curr_mask else None
        )
        batched_term_curr_sign = (
            torch.zeros((ptr[-1],), dtype=torch.float32, device=device)
            if has_term_sign else None
        )
        batched_log_gm = (
            torch.zeros((ptr[-1],), dtype=torch.float32, device=device)
            if has_log_gm else None
        )
        batched_log_gds = (
            torch.zeros((ptr[-1],), dtype=torch.float32, device=device)
            if has_log_gds else None
        )

        return {
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'type_tens': type_tens,
            'net_role': net_role,
            'batch': batch_vec,
            'ptr': ptr_t,
            'num_graphs': graph_idx_global,
            'mosfet_info': mosfet_info,
            'mosfet_ptr': mosfet_ptr_t,
            'resistor_info': resistor_info,
            'resistor_ptr': resistor_ptr_t,
            'capacitor_info': capacitor_info,
            'capacitor_ptr': capacitor_ptr_t,
            'num_terminals': num_terminals_t,
            'terminal_train_mask': terminal_train_mask,
            'known_voltage_mask': known_voltage_mask,
            'train_mask': train_mask,
            'mosfet_drain_mask': mosfet_drain_mask,
            'kcl_include_mask': kcl_include_mask,
            'batched_mosfet_term': batched_mosfet_term,
            'topo_node_slices': topo_node_slices,
            'topo_mosfet_slices': topo_mosfet_slices,
            'batched_x': batched_x,
            'batched_wl_um': batched_wl_um,
            'batched_m': batched_m,
            'batched_v_targets': batched_v_targets,
            'batched_i_targets': batched_i_targets,
            'batched_has_curr_mask': batched_has_curr_mask,
            'batched_term_curr_sign': batched_term_curr_sign,
            'batched_log_gm': batched_log_gm,
            'batched_log_gds': batched_log_gds,
        }

    def _next_batch(self) -> PretrainBatch:
        tpl = self._template
        x_buf = tpl['batched_x']
        wl_buf = tpl['batched_wl_um']
        m_buf = tpl['batched_m']
        v_buf = tpl['batched_v_targets']
        i_buf = tpl['batched_i_targets']
        hcm_buf = tpl['batched_has_curr_mask']
        tcs_buf = tpl['batched_term_curr_sign']
        lgm_buf = tpl['batched_log_gm']
        lgds_buf = tpl['batched_log_gds']
        # Gather per-topo and write into the pre-allocated batched_x buffer
        for t in self.topo_names:
            ep_idx = self._epoch_indices[t]
            cur = self._cursor[t]
            sel = ep_idx[cur:cur + self.per_topo]
            self._cursor[t] += self.per_topo
            # If short on samples (last batch when not drop_last), wrap around
            if sel.shape[0] < self.per_topo:
                wrap = self.per_topo - sel.shape[0]
                extra = ep_idx[:wrap]
                sel = torch.cat([sel, extra], dim=0)
            xs = self._topo_x[t][sel]  # [per_topo, n_nodes_t, 9]
            start, end, n_nodes, n_graphs = tpl['topo_node_slices'][t]
            x_buf[start:end] = xs.reshape(n_graphs * n_nodes, -1)
            # Physics extras
            if wl_buf is not None and t in self._topo_wl_um:
                wls = self._topo_wl_um[t][sel]  # [per_topo, M_t, 2]
                ms_start, ms_end, m_per_t = tpl['topo_mosfet_slices'][t]
                wl_buf[ms_start:ms_end] = wls.reshape(n_graphs * m_per_t, 2)
            if m_buf is not None and t in self._topo_m:
                ms_vals = self._topo_m[t][sel]  # [per_topo, M_t]
                ms_start, ms_end, m_per_t = tpl['topo_mosfet_slices'][t]
                m_buf[ms_start:ms_end] = ms_vals.reshape(n_graphs * m_per_t)
            if v_buf is not None and t in self._topo_v_targets:
                vts = self._topo_v_targets[t][sel]  # [per_topo, n_nodes_t]
                v_buf[start:end] = vts.reshape(n_graphs * n_nodes)
            if i_buf is not None and t in self._topo_i_targets:
                its = self._topo_i_targets[t][sel]
                i_buf[start:end] = its.reshape(n_graphs * n_nodes)
            if hcm_buf is not None and t in self._topo_has_curr_mask:
                hcm = self._topo_has_curr_mask[t][sel]
                hcm_buf[start:end] = hcm.reshape(n_graphs * n_nodes)
            if tcs_buf is not None and t in self._topo_term_curr_sign:
                tcs = self._topo_term_curr_sign[t][sel]
                tcs_buf[start:end] = tcs.reshape(n_graphs * n_nodes)
            if lgm_buf is not None and t in self._topo_log_gm:
                lgm = self._topo_log_gm[t][sel]
                lgm_buf[start:end] = lgm.reshape(n_graphs * n_nodes)
            if lgds_buf is not None and t in self._topo_log_gds:
                lgds = self._topo_log_gds[t][sel]
                lgds_buf[start:end] = lgds.reshape(n_graphs * n_nodes)

        # Build attrs dict for the batch (all reused from template + the
        # in-place written batched_x). num_graphs is an int.
        attrs = {
            'x': x_buf,
            'edge_index': tpl['edge_index'],
            'edge_attr': tpl['edge_attr'],
            'type_tens': tpl['type_tens'],
            'net_role': tpl['net_role'],
            'batch': tpl['batch'],
            'ptr': tpl['ptr'],
            'num_graphs': tpl['num_graphs'],
            'mosfet_info': tpl['mosfet_info'],
            'mosfet_ptr': tpl['mosfet_ptr'],
            'resistor_info': tpl['resistor_info'],
            'resistor_ptr': tpl['resistor_ptr'],
            'capacitor_info': tpl['capacitor_info'],
            'capacitor_ptr': tpl['capacitor_ptr'],
            'num_terminals': tpl['num_terminals'],
            'terminal_train_mask': tpl['terminal_train_mask'],
            'known_voltage_mask': tpl['known_voltage_mask'],
            'train_mask': tpl['train_mask'],
            'mosfet_drain_mask': tpl['mosfet_drain_mask'],
            'kcl_include_mask': tpl['kcl_include_mask'],
            'batched_mosfet_term': tpl['batched_mosfet_term'],
            # Per-topo node ranges so the model can apply per-topo MHA
            # (each topo slice contains per_topo graphs of identical size).
            'topo_node_slices': tpl['topo_node_slices'],
            # Physics-pretrain extras (None if dataset wasn't patched)
            'mosfet_wl_um': wl_buf,
            'mosfet_m': m_buf,
            'node_voltage_targets': v_buf,
            # Supervised-pretrain extras (None if dataset lacks them)
            'node_current_targets': i_buf,
            'has_current_mask': hcm_buf,
            'terminal_current_sign': tcs_buf,
            'node_log_gm': lgm_buf,
            'node_log_gds': lgds_buf,
        }
        return PretrainBatch(attrs)
