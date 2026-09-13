"""On-disk graph schema for circuit datasets.

Declares the attribute layout of the pickled PyG ``Data``/``Batch`` objects
produced by ``CircuitGraphBuilder.build_from_netlist`` and the
``scripts/patch_dataset_*.py`` migrations. Every attribute name, shape and
column meaning below is a load-bearing on-disk contract: pickled datasets,
the prebatched variants and all consumers (loaders, losses, models, eval
scripts) rely on it byte for byte. This module is documentation plus
constants — existing call sites are not required to import it.

Node ordering contract
----------------------

Nodes are ordered TERMINALS FIRST, THEN NETS:

  - Terminal nodes appear in netlist component order. Within one MOSFET the
    terminals are created in ``TERMINAL_ORDER`` = (drain, gate, source, bulk),
    so a MOSFET's four terminal nodes are consecutive:
    drain = k, gate = k + 1, source = k + 2, bulk = k + 3.
    Equivalently bulk = source + 1 (see ``bulk_index``); loaders also use
    bulk = drain + 3. Two-terminal devices (R, C, V, I) create (p, n).
  - Net nodes follow, one per net name, sorted alphabetically
    (``sorted(all_nets)``).

``num_terminals`` and ``num_nets`` record the split point;
``node_names`` / ``node_types`` list all nodes in this order.

Per-node tensors ([N] or [N, F], N = num_terminals + num_nets)
--------------------------------------------------------------

x : float [N, 9]
    Columns 0-3: device properties, zero-padded per node type
      MOSFET terminal: [W_norm, L_norm, W/L_norm, M_norm]
        (min-max normalized per param spec; M log-normalized, see
        ``CircuitGraphBuilder._normalize_mosfet_props``)
      V source terminal: [dc_norm, 0, 0, 0]  (2 V/VDD - 1)
      I source terminal: [dc, 0, 0, 0]       (raw amps)
      R / C terminal:    [value_norm, 0, 0, 0]  (log min-max)
      Net node:          [0, 0, 0, 0]
    Columns 4-8: graph-level globals, identical on every node
      [vdd_norm, vcm_norm, vin_p_norm, vin_n_norm, log10(i_ref) + 5]
      (voltages normalized as 2 V/VDD - 1)
    Datasets before patch_dataset_add_m.py had x [N, 8] (no M column).

type_tens : float [N, TYPE_DIM]
    Hierarchical trie one-hot of the node type tuple, zero-padded to the
    deepest encoding (see ``circuitgnn.data.encoding.build_circuit_trie``).
    Node type tuples: ('M', 'N'|'P', 'D'|'G'|'S'|'B') for MOSFET terminals,
    (dev_type, 'P'|'M') for 2-terminal devices, ('VNode', 'GND'|'NGND')
    for nets.

node_voltage_targets : float [N]   raw SPICE DC voltage per node (volts).
node_current_targets : float [N]   |I| magnitude per terminal (amps, raw in
    dataset.pkl; log10 z-scored in place by the training pipeline).
terminal_current_sign : float [N]  KCL sign convention per terminal:
    MOSFET drain/source fixed by polarity (NMOS drain -1, source +1;
    PMOS mirrored), others follow the actual SPICE current direction,
    0 for gate/bulk and non-terminal nodes.
node_log_gm, node_log_gds : float [N]  log10(gm), log10(gds) scattered to
    each MOSFET's DRAIN terminal node (0 elsewhere; see mosfet_drain_mask).
node_mosfet_vth : float [N]  SPICE Vth scattered to drain terminal nodes.

Per-node boolean masks ([N])
----------------------------

output_node_mask     non-GND net nodes (voltage prediction sites).
known_voltage_mask   nodes tied to gnd/vdd/input nets (boundary values).
train_mask           output_node_mask & ~known_voltage_mask (V supervision).
terminal_train_mask  terminal nodes on unknown internal nets.
has_current_mask     terminals with a supervised current target.
kcl_include_mask     terminals included in the KCL residual.
mosfet_drain_mask    drain terminal nodes with valid gm targets.

Voltage target slices
---------------------

vdc : float [n_train, 1]           node_voltage_targets[train_mask].
terminal_vdc : float [n_term_train, 1]
                                   node_voltage_targets[terminal_train_mask].

edge_index / edge_attr
----------------------

edge_index : long [2, E]  bidirectional terminal <-> net edges.
edge_attr generations (per-column layout in ``EDGE_ATTR_COLUMNS``):
  6-dim  (builder)  one-hot of the terminal type of the edge's terminal
         endpoint, column order ``EDGE_TYPE_MAP``:
         gate=0, drain=1, source=2, bulk=3, p=4, n=5.
  10-dim (patch_dataset_edge_features.py) appends 4 structural flags:
         6 loop_flag  — edge is in loop_edge_index
         7 bound_flag — destination node has known voltage (VDD/GND/V-src)
         8 out_flag   — destination node is an output node
         9 cap_flag   — destination node is a capacitor terminal
  11-dim (patch_dataset_edge_current_sign.py) appends:
         10 current_sign — terminal_current_sign at the source endpoint
            (negated on the reverse net→terminal edge; 0 if neither
            endpoint is a current-carrying terminal).

Per-device info tensors
-----------------------

mosfet_info : long [M, 7]  columns ``MOSFET_INFO_COLUMNS``:
    [gate_term, drain_term, source_term, gate_net, drain_net, source_net,
     is_nmos]  (terminal/net node indices local to the graph; is_nmos 1/0).
mosfet_terminal_idx : long [M, 4]  [gate, drain, source, bulk] terminal
    node indices, -1 where missing (patch_dataset_mosfet_wl.py backfills).
mosfet_wl_um : float [M, 2]  raw (W, L) in micrometers, for LUT queries.
mosfet_m : float [M]  raw multiplier (fingers); scales LUT currents linearly.
mosfet_device_names : list[str] [M]  device names aligned with mosfet_info.
mosfet_region_labels : long [M]  SPICE operating region,
    ``REGION_LABELS``: cutoff=0, triode=1, saturation=2, unknown=-1.
mosfet_vth : float [M]  SPICE threshold voltage (volts).
mosfet_gm, mosfet_gds : float [M]  SPICE small-signal params (siemens).

resistor_info : float [R, 5]   [term_p, term_n, net_p, net_n, r_value_norm]
capacitor_info : long [C, 4]   [term_p, term_n, net_p, net_n]
vsource_info : long [V, 4]     [term_p, term_n, net_p, net_n]
isource_info : float [I, 5]    [term_p, term_n, net_p, net_n, log10(i_ref)]

loop_edge_index : long [2, E_loop]  device-level edges connecting devices
    that share a fundamental cycle (see ``circuitgnn.data.loop_info``).
device_terminal_map : long [D, 4]  terminal node indices per device,
    -1 padded; MOSFET rows follow terminal creation order
    (drain, gate, source, bulk).
num_devices : int  D (MOSFETs + R + C + V + I sources).

Batched ptr tensors (added by ``batching._add_device_ptr_tensors``)
-------------------------------------------------------------------

mosfet_ptr, resistor_ptr, isource_ptr, capacitor_ptr : long [num_graphs + 1]
    Cumulative device counts per graph (PyG ptr convention). The info
    tensors keep PER-GRAPH LOCAL node indices after batching; consumers
    combine ptr (node offsets) with these device ptrs to index the batched
    node dimension. FixedTopologyDataset additionally emits device_ptr for
    loop attention.

Graph-level scalars
-------------------

ac_ugbw, ac_pm, ac_am, ac_dc_gain : float [1]  AC metrics from .ac SPICE
    analysis (Hz, degrees, dB); ac_valid : bool [1]  metrics usable
    (requires positive DC gain).
num_terminals, num_nets : int  node ordering split (see above).

Patch-added per-graph attributes (present only on patched datasets)
-------------------------------------------------------------------

net_role : float [N, 5]  one-hot net role for net nodes, zeros for
    terminals (patch_dataset_net_role.py; roles in
    ``circuitgnn.data.net_roles``: VDD=0, GND=1, SIG_IN=2, SIG_OUT=3,
    INTERNAL=4).
mosfet_role : float [M, 7]  functional role one-hot for the fan_smc
    topology (patch_dataset_mosfet_role.py).
mosfet_descriptor : float [M, 5]  LUT-derived physical descriptor at
    canonical bias points (patch_dataset_mosfet_descriptor.py).
node_lut_features : float [N, 3]  pass-1 LUT (id, gm, gds), z-scored,
    scattered to MOSFET terminal nodes (patch_dataset_lut_op_features.py).
node_stack_features : float [N, 4]  pass-1 predictions (V, I, gm, gds),
    z-scored (patch_dataset_stack_features.py).
"""

from typing import Dict, Tuple

# ── Node ordering ───────────────────────────────────────────────────────

# Terminal creation order per MOSFET. A MOSFET's terminal nodes are
# consecutive in this order: drain = k, gate = k+1, source = k+2, bulk = k+3.
TERMINAL_ORDER: Tuple[str, ...] = ('drain', 'gate', 'source', 'bulk')

# Terminal creation order per 2-terminal device (R, C, V, I).
TWO_TERMINAL_ORDER: Tuple[str, ...] = ('p', 'n')


def bulk_index(source_idx: int) -> int:
    """Bulk terminal node index from the source terminal node index.

    Terminal creation order per MOSFET is (drain, gate, source, bulk), so
    bulk = source + 1 (equivalently drain + 3 — the form used by
    FixedTopologyDataset and PretrainCombinedLoader). Callers must check the
    result against num_terminals: netlists without an explicit bulk terminal
    have no such node.
    """
    return source_idx + 1


# ── Node features ───────────────────────────────────────────────────────

NUM_DEVICE_PROP_FEATURES = 4   # x columns 0-3 (W, L, W/L, M for MOSFETs)
NUM_GLOBAL_FEATURES = 5        # x columns 4-8 (vdd, vcm, vin_p, vin_n, i_ref)
NUM_NODE_FEATURES = NUM_DEVICE_PROP_FEATURES + NUM_GLOBAL_FEATURES  # x [N, 9]

# ── Edge features ───────────────────────────────────────────────────────

# Column of the terminal-type one-hot (edge_attr columns 0-5). Matches
# EDGE_TYPE_MAP in CircuitGraphBuilder.build_from_netlist.
EDGE_TYPE_MAP: Dict[str, int] = {
    'gate': 0, 'drain': 1, 'source': 2, 'bulk': 3, 'p': 4, 'n': 5,
}

# edge_attr generations (see module docstring for per-column meaning).
EDGE_ATTR_DIM_BASE = 6        # builder output: terminal-type one-hot
EDGE_ATTR_DIM_STRUCTURAL = 10  # + loop/bound/out/cap flags
EDGE_ATTR_DIM_CURRENT_SIGN = 11  # + per-edge current direction sign

EDGE_ATTR_COLUMNS: Tuple[str, ...] = (
    'gate', 'drain', 'source', 'bulk', 'p', 'n',   # 0-5 one-hot
    'loop_flag',      # 6
    'bound_flag',     # 7
    'out_flag',       # 8
    'cap_flag',       # 9
    'current_sign',   # 10
)

# ── Per-device info tensor columns ──────────────────────────────────────

# mosfet_info [M, 7] column names.
MOSFET_INFO_COLUMNS: Tuple[str, ...] = (
    'gate_term', 'drain_term', 'source_term',
    'gate_net', 'drain_net', 'source_net',
    'is_nmos',
)

# mosfet_terminal_idx [M, 4] column order (-1 where missing).
MOSFET_TERMINAL_IDX_COLUMNS: Tuple[str, ...] = ('gate', 'drain', 'source', 'bulk')

RESISTOR_INFO_COLUMNS: Tuple[str, ...] = (
    'term_p', 'term_n', 'net_p', 'net_n', 'r_value_norm',
)
CAPACITOR_INFO_COLUMNS: Tuple[str, ...] = ('term_p', 'term_n', 'net_p', 'net_n')
VSOURCE_INFO_COLUMNS: Tuple[str, ...] = ('term_p', 'term_n', 'net_p', 'net_n')
ISOURCE_INFO_COLUMNS: Tuple[str, ...] = (
    'term_p', 'term_n', 'net_p', 'net_n', 'i_ref_log10',
)

# ── Operating region labels (mosfet_region_labels) ──────────────────────

REGION_LABELS: Dict[str, int] = {
    'cutoff': 0, 'triode': 1, 'saturation': 2, 'unknown': -1,
}
