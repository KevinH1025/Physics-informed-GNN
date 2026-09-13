"""
Compatibility shim: the tower architecture now lives in the
``circuitgnn.gnn.architectures.tower`` package. Import from there in new code.
"""

from .tower import (
    DEFAULT_SUBCIRCUIT_GROUPS,
    DEFAULT_DAG_EDGES,
    JKAggregation,
    AttentionSSHead,
    GatedSSHead,
    SubcircuitAttentionPool,
    TowerGENConv,
    log10_add,
)

__all__ = [
    'DEFAULT_SUBCIRCUIT_GROUPS',
    'DEFAULT_DAG_EDGES',
    'JKAggregation',
    'AttentionSSHead',
    'GatedSSHead',
    'SubcircuitAttentionPool',
    'TowerGENConv',
    'log10_add',
]
