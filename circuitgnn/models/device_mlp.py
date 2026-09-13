"""Re-export shim: the code now lives in circuitgnn.gnn.legacy.device_mlp."""

from circuitgnn.gnn.legacy.device_mlp import (
    ResidualBlock,
    DeviceMLP,
    DeviceMLPWithNormalization,
    create_device_mlp,
)

__all__ = [
    'ResidualBlock',
    'DeviceMLP',
    'DeviceMLPWithNormalization',
    'create_device_mlp',
]
