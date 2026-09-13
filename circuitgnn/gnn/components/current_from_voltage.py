"""Re-export shim: the code now lives in circuitgnn.gnn.legacy.current_from_voltage."""

from circuitgnn.gnn.legacy.current_from_voltage import (
    MOSFETCurrentMLP,
    compute_mosfet_currents,
    compute_resistor_currents,
    compute_capacitor_currents,
    compute_isource_currents,
)

__all__ = [
    'MOSFETCurrentMLP',
    'compute_mosfet_currents',
    'compute_resistor_currents',
    'compute_capacitor_currents',
    'compute_isource_currents',
]
