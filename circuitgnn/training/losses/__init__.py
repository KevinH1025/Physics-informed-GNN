"""
Loss computation utilities for GNN training.

This module provides loss functions for voltage and current prediction.
To add a new loss type:
  1. Create a function: def <name>_loss(pred, target) -> Tensor
  2. Add it to LOSS_FUNCTIONS dict

Implementations live in the submodules of this package: data_terms, kcl,
device_physics, supervised_heads, device_index and aggregate.
"""

from ..current_constraints import (
    compute_current_constraint_loss,
    compute_hardcoded_mirror_loss,
)
from .aggregate import UncertaintyWeights, compute_combined_loss
from .data_terms import (
    LOSS_FUNCTIONS,
    compute_current_loss,
    compute_device_consistency_loss,
    compute_loss,
    compute_voltage_loss,
    huber_loss,
    mae_loss,
    mse_loss,
)
from .device_index import _batch_offsets, get_device_graph_idx
from .device_physics import (
    compute_cutoff_physics_loss,
    compute_gm_physics_loss,
    compute_smaxt_gm_loss,
    compute_triode_physics_loss,
)
from .kcl import (
    apply_kcl_conservation_projection,
    compute_kcl_loss,
    compute_kcl_per_net_debug,
)
from .supervised_heads import (
    compute_ac_loss,
    compute_gm_id_aux_loss,
    compute_gm_id_consistency_loss,
    compute_iv_id_loss,
    compute_region_loss,
    compute_region_loss_coral,
    compute_ss_loss,
    compute_vov_loss,
    compute_vth_loss,
)
