"""
Tower GNN architecture package: shared backbone + task-specific towers.

Modules:
    - constants: fixed 3-stage opamp topology constants
    - physics: log10-space DC gain and UGBW formulas
    - jk: Jumping Knowledge aggregation
    - ss_heads: attention and gated gm/gds readout heads
    - subcircuit: subcircuit attention pooling and DAG helpers
    - model: the TowerGENConv model itself
"""

from .constants import (
    DEFAULT_SUBCIRCUIT_GROUPS,
    DEFAULT_DAG_EDGES,
    SIGNAL_PATH_MOSFET_INDICES,
)
from .physics import log10_add, dc_gain_log10_formula, ugbw_log10_estimate
from .jk import JKAggregation
from .ss_heads import AttentionSSHead, GatedSSHead
from .subcircuit import (
    SubcircuitAttentionPool,
    build_subcircuit_assignment,
    build_dag_topo_order,
)
from .model import TowerGENConv

__all__ = [
    'DEFAULT_SUBCIRCUIT_GROUPS',
    'DEFAULT_DAG_EDGES',
    'SIGNAL_PATH_MOSFET_INDICES',
    'log10_add',
    'dc_gain_log10_formula',
    'ugbw_log10_estimate',
    'JKAggregation',
    'AttentionSSHead',
    'GatedSSHead',
    'SubcircuitAttentionPool',
    'build_subcircuit_assignment',
    'build_dag_topo_order',
    'TowerGENConv',
]
