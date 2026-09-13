"""
Reusable GNN components.

This module provides building blocks for constructing GNN architectures:
- layers: MLP and DeepGCNLayer factories
- virtual_node: Virtual node module for global information flow
- loop_attention: Attention over feedback-loop membership
- net_self_attention: Self-attention among net nodes
- device_current_head: Device-pooled current prediction head
- differentiable_iv: Differentiable IV surface model
- lut: SKY130 LUT pipeline (IVEmbedder, LUTIdQuery, LUTOpQuery)

Legacy deepgen-only components (current_from_voltage, current_gnn,
device_aggregation) live in circuitgnn.gnn.legacy; their public names stay
importable from here and from their old module paths.
"""

from .layers import build_mlp, create_deepgcn_layer, get_activation
from .virtual_node import VirtualNode

# Names re-exported from circuitgnn.gnn.legacy, resolved lazily because the
# legacy modules themselves import circuitgnn.gnn.components.layers.
_LEGACY_EXPORTS = {
    'MOSFETCurrentMLP': 'circuitgnn.gnn.legacy.current_from_voltage',
    'compute_mosfet_currents': 'circuitgnn.gnn.legacy.current_from_voltage',
}

__all__ = [
    'build_mlp',
    'create_deepgcn_layer',
    'get_activation',
    'VirtualNode',
    'MOSFETCurrentMLP',
    'compute_mosfet_currents',
]


def __getattr__(name):
    if name in _LEGACY_EXPORTS:
        import importlib
        value = getattr(importlib.import_module(_LEGACY_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
