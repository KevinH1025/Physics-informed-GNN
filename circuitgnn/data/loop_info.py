"""Device-level loop detection for loop attention.

Shared core used by both graph construction paths:
  - ``CircuitGraphBuilder._create_loop_info`` (netlist build time, all device
    types, nets keyed by name)
  - ``FixedTopologyDataset._compute_loop_info`` (derived from stored
    mosfet/resistor/capacitor info tensors, nets keyed by net index)

The callers differ only in how they enumerate devices and their terminals;
the device-graph construction, NetworkX cycle-basis detection, loop edge
index and device terminal map assembly are identical and live here.
"""

from collections import defaultdict
from typing import List, Sequence, Set, Tuple

import networkx as nx
import torch


def build_loop_info(
    device_terms: Sequence[Sequence[int]],
    device_nets: Sequence[Set],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build loop_edge_index and device_terminal_map from per-device info.

    Args:
        device_terms: per device, the list of its terminal node indices.
        device_nets: per device, the set of nets it touches. Nets may be
            keyed by any hashable (net name or net index) — only shared
            membership between devices matters.

    Returns:
        loop_edge_index: [2, E_loop] device-level edges connecting devices in same loop
        device_terminal_map: [D, 4] terminal node indices per device (-1 padded)
    """
    num_devices = len(device_terms)

    # net → set of device indices
    net_to_devices = defaultdict(set)
    for d, nets in enumerate(device_nets):
        for net in nets:
            net_to_devices[net].add(d)

    # Build device-level graph: edge if two devices share a net
    G = nx.Graph()
    G.add_nodes_from(range(num_devices))
    for net, devs in net_to_devices.items():
        devs = list(devs)
        for i in range(len(devs)):
            for j in range(i + 1, len(devs)):
                G.add_edge(devs[i], devs[j])

    # Find fundamental cycles
    cycles = nx.cycle_basis(G)

    # Build loop_edge_index: connect all device pairs within each cycle
    loop_edges = set()
    for cycle in cycles:
        for i in range(len(cycle)):
            for j in range(i + 1, len(cycle)):
                a, b = min(cycle[i], cycle[j]), max(cycle[i], cycle[j])
                loop_edges.add((a, b))

    # Make bidirectional
    if loop_edges:
        src, dst = [], []
        for a, b in loop_edges:
            src.extend([a, b])
            dst.extend([b, a])
        loop_edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        loop_edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Build device_terminal_map: [D, 4] padded with -1
    max_terms = 4
    device_terminal_map = torch.full((num_devices, max_terms), -1, dtype=torch.long)
    for d, terms in enumerate(device_terms):
        for t, idx in enumerate(terms[:max_terms]):
            device_terminal_map[d, t] = idx

    return loop_edge_index, device_terminal_map
