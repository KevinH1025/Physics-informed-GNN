"""
Legacy components used only by the DeepGENConv baseline.

These modules implement the original current-prediction pipeline
(voltage-derived currents, the current GNN backbone, device aggregation
and the standalone Device MLP). The tower architecture does not use them;
they are kept so existing DeepGENConv checkpoints and configs keep working.
"""

from .current_from_voltage import (
    MOSFETCurrentMLP,
    compute_mosfet_currents,
    compute_resistor_currents,
    compute_capacitor_currents,
    compute_isource_currents,
)
from .current_gnn import CurrentGNNBackbone, propagate_voltages_to_terminals
from .device_aggregation import DeviceAggregationLayer
from .device_mlp import (
    ResidualBlock,
    DeviceMLP,
    DeviceMLPWithNormalization,
    create_device_mlp,
)

__all__ = [
    'MOSFETCurrentMLP',
    'compute_mosfet_currents',
    'compute_resistor_currents',
    'compute_capacitor_currents',
    'compute_isource_currents',
    'CurrentGNNBackbone',
    'propagate_voltages_to_terminals',
    'DeviceAggregationLayer',
    'ResidualBlock',
    'DeviceMLP',
    'DeviceMLPWithNormalization',
    'create_device_mlp',
]
