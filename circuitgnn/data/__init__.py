"""
Data processing utilities for circuit GNN training.

This module provides utilities for:
- Graph building from circuit netlists
- Parameter sampling (LHS)
- Pre-batching for efficient training
- Visualization and plotting

The plotting functions (and their matplotlib dependency) are loaded lazily
so headless consumers don't pay the matplotlib import cost.
"""

from circuitgnn.data.encoding import Trie, build_circuit_trie
from circuitgnn.data.graph_builder import CircuitGraphBuilder, TerminalNode, NetNode
from circuitgnn.data.sampling import (
    generate_lhs_samples,
    generate_netlist,
    worker_generate_sample,
)
from circuitgnn.data.batching import create_prebatched_dataset

__all__ = [
    # Encoding
    'Trie',
    'build_circuit_trie',
    # Graph builder
    'CircuitGraphBuilder',
    'TerminalNode',
    'NetNode',
    # Sampling
    'generate_lhs_samples',
    'generate_netlist',
    'worker_generate_sample',
    # Batching
    'create_prebatched_dataset',
    # Plotting
    'plot_parameter_distributions',
    'plot_node_voltage_distributions',
    'plot_device_current_distributions',
    'plot_target_normalization_distributions',
]

_PLOTTING_EXPORTS = (
    'plot_parameter_distributions',
    'plot_node_voltage_distributions',
    'plot_device_current_distributions',
    'plot_target_normalization_distributions',
)


def __getattr__(name):
    if name in _PLOTTING_EXPORTS:
        from circuitgnn.data import plotting
        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
