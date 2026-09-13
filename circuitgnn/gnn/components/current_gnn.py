"""Re-export shim: the code now lives in circuitgnn.gnn.legacy.current_gnn."""

from circuitgnn.gnn.legacy.current_gnn import (
    CurrentGNNBackbone,
    propagate_voltages_to_terminals,
)

__all__ = [
    'CurrentGNNBackbone',
    'propagate_voltages_to_terminals',
]
