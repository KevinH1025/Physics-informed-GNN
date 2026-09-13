"""Voltage, current and base data-fitting losses."""

import torch
import torch.nn.functional as F

from .device_index import _batch_offsets


# Individual Loss Functions
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss."""
    return F.mse_loss(pred, target)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss (smooth L1). Less sensitive to outliers than MSE."""
    return F.huber_loss(pred, target, delta=delta)


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error loss (L1)."""
    return F.l1_loss(pred, target)


# Loss Function Registry
LOSS_FUNCTIONS = {
    'mse': mse_loss,
    'huber': huber_loss,
    'mae': mae_loss,
}


# Base Loss Computation
def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = 'mse',
    huber_delta: float = 1.0
) -> torch.Tensor:
    """
    Compute loss using the specified loss function.

    Args:
        pred: Predicted values
        target: Target values
        loss_type: Loss function name ('mse', 'huber', 'mae')
        huber_delta: Delta parameter for Huber loss

    Returns:
        Computed loss tensor

    Raises:
        ValueError: If loss_type is not recognized
    """
    if loss_type not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss type '{loss_type}'. Available: {list(LOSS_FUNCTIONS.keys())}")

    loss_fn = LOSS_FUNCTIONS[loss_type]
    if loss_type == 'huber':
        return loss_fn(pred, target, delta=huber_delta)
    return loss_fn(pred, target)


# Component Losses
def compute_voltage_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = 'mse',
    huber_delta: float = 1.0,
    node_weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute voltage prediction loss with optional per-node weighting.

    Args:
        pred: Predicted voltage values (normalized)
        target: Target voltage values (normalized)
        loss_type: Loss function name
        huber_delta: Delta for Huber loss
        node_weights: Optional weights per node (e.g., 2.0 for stage2 nodes)

    Returns:
        Voltage loss tensor (weighted mean if node_weights provided)
    """
    if node_weights is None:
        return compute_loss(pred, target, loss_type, huber_delta)

    # Compute per-element loss
    if loss_type == 'mse':
        element_loss = (pred - target) ** 2
    elif loss_type == 'mae':
        element_loss = (pred - target).abs()
    elif loss_type == 'huber':
        element_loss = F.huber_loss(pred, target, delta=huber_delta, reduction='none')
    else:
        raise ValueError(f"Unknown loss type '{loss_type}'")

    # Weighted mean
    return (element_loss * node_weights).sum() / node_weights.sum()


def compute_current_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = 'mse',
    huber_delta: float = 1.0
) -> torch.Tensor:
    """
    Compute current prediction loss (masked).

    Args:
        pred: Predicted current values (normalized, all nodes)
        target: Target current values (normalized, all nodes)
        mask: Boolean mask for nodes with current targets
        loss_type: Loss function name
        huber_delta: Delta for Huber loss

    Returns:
        Current loss tensor (0 if no valid targets)
    """
    if pred is None or mask is None or not mask.any():
        return torch.tensor(0.0, device=target.device if target is not None else 'cpu')

    pred_masked = pred[mask]
    target_masked = target[mask]
    return compute_loss(pred_masked, target_masked, loss_type, huber_delta)


def compute_device_consistency_loss(
    node_currents: torch.Tensor,
    batch,
) -> torch.Tensor:
    """MSE between terminal predictions of the same device.

    Penalizes drain != source for MOSFETs, p != n for 2-terminal devices.
    Operates in normalized z-score space (same as model output).
    """
    total = torch.tensor(0.0, device=node_currents.device)
    count = 0
    ptr = getattr(batch, 'ptr', None)
    is_batched = ptr is not None and len(ptr) > 2

    # MOSFETs: drain vs source
    if hasattr(batch, 'mosfet_info') and batch.mosfet_info is not None and batch.mosfet_info.numel() > 0:
        d = batch.mosfet_info[:, 1].long()
        s = batch.mosfet_info[:, 2].long()
        if is_batched:
            offsets = _batch_offsets(len(d), getattr(batch, 'mosfet_ptr', None), ptr, d.device)
            d = d + offsets
            s = s + offsets
        total = total + (node_currents[d] - node_currents[s]).pow(2).sum()
        count += d.shape[0]

    # 2-terminal devices: p vs n
    for attr, ptr_attr in [('resistor_info', 'resistor_ptr'), ('capacitor_info', 'capacitor_ptr'),
                           ('vsource_info', None), ('isource_info', 'isource_ptr')]:
        info = getattr(batch, attr, None)
        if info is not None and info.numel() > 0:
            p_idx = info[:, 0].long()
            n_idx = info[:, 1].long()
            if is_batched:
                dev_ptr = getattr(batch, ptr_attr, None) if ptr_attr else None
                offsets = _batch_offsets(len(p_idx), dev_ptr, ptr, p_idx.device)
                p_idx = p_idx + offsets
                n_idx = n_idx + offsets
            total = total + (node_currents[p_idx] - node_currents[n_idx]).pow(2).sum()
            count += p_idx.shape[0]

    return total / max(count, 1)
