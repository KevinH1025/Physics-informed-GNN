"""Supervised auxiliary head losses (vov, vth, ss, gm/Id, IV id, AC, region)."""

import torch
import torch.nn.functional as F


def compute_vov_loss(vov_pred: torch.Tensor, mosfet_gt_vov: torch.Tensor,
                     mosfet_vov_valid: torch.Tensor = None) -> torch.Tensor:
    """MSE loss on predicted vs GT Vov, skipping cutoff devices."""
    if mosfet_vov_valid is not None:
        mask = mosfet_vov_valid
    else:
        mask = (mosfet_gt_vov > 0) & (mosfet_gt_vov < 1.8)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=vov_pred.device)
    return F.mse_loss(vov_pred[mask], mosfet_gt_vov[mask])


def compute_vth_loss(vth_pred: torch.Tensor, mosfet_vth: torch.Tensor) -> torch.Tensor:
    """MSE loss on predicted vs GT Vth, skipping missing values."""
    valid = mosfet_vth.abs() > 1e-6
    if valid.sum() == 0:
        return torch.tensor(0.0, device=vth_pred.device)
    return F.mse_loss(vth_pred[valid], mosfet_vth[valid])


# Supervised gm/gds Prediction Loss (mask-based, like voltage/current)
def compute_ss_loss(
    gm_pred: torch.Tensor,
    gds_pred: torch.Tensor,
    mosfet_gm: torch.Tensor,
    mosfet_gds: torch.Tensor,
    gm_mean: float,
    gm_std: float,
    gds_mean: float,
    gds_std: float,
    huber_delta: float = 0.0,
    region_stats: dict = None,
    mosfet_region_labels: torch.Tensor = None,
) -> tuple:
    """
    Supervised loss for per-MOSFET gm/gds predictions.

    Predictions are per-MOSFET (from gate+drain+source terminal concat).
    Targets are raw linear-scale mosfet_gm/mosfet_gds, normalized inline.

    If region_stats is provided, uses per-region (mean, std) for target normalization.

    Args:
        gm_pred: [num_mosfets] predicted z-scored log10(gm)
        gds_pred: [num_mosfets] predicted z-scored log10(gds)
        mosfet_gm: [num_mosfets] raw gm values (linear scale)
        mosfet_gds: [num_mosfets] raw gds values (linear scale)
        gm_mean/gm_std: global z-score normalization stats for log10(gm)
        gds_mean/gds_std: global z-score normalization stats for log10(gds)
        region_stats: Optional per-region stats dict {0: {'gm_mean', 'gm_std', ...}, ...}
        mosfet_region_labels: Optional [num_mosfets] region labels (0=cutoff, 1=triode, 2=sat)

    Returns:
        Tuple of (gm_loss, gds_loss) scalar tensors
    """
    device = gm_pred.device
    mosfet_gm = mosfet_gm.to(device)
    mosfet_gds = mosfet_gds.to(device)
    valid = mosfet_gm > 1e-12

    if valid.any():
        log_gm = torch.log10(mosfet_gm[valid])
        log_gds = torch.log10(mosfet_gds[valid].clamp(min=1e-20))

        if region_stats is not None and mosfet_region_labels is not None:
            labels = mosfet_region_labels.to(device)[valid]
            gm_means = torch.full_like(log_gm, gm_mean)
            gm_stds = torch.full_like(log_gm, gm_std)
            gds_means = torch.full_like(log_gds, gds_mean)
            gds_stds = torch.full_like(log_gds, gds_std)
            for r in range(3):
                rmask = (labels == r)
                if rmask.any():
                    gm_means[rmask] = region_stats[r]['gm_mean']
                    gm_stds[rmask] = region_stats[r]['gm_std']
                    gds_means[rmask] = region_stats[r]['gds_mean']
                    gds_stds[rmask] = region_stats[r]['gds_std']
            gm_target = (log_gm - gm_means) / gm_stds
            gds_target = (log_gds - gds_means) / gds_stds
        else:
            gm_target = (log_gm - gm_mean) / gm_std
            gds_target = (log_gds - gds_mean) / gds_std

        if huber_delta > 0:
            gm_loss = F.huber_loss(gm_pred[valid], gm_target, delta=huber_delta)
            gds_loss = F.huber_loss(gds_pred[valid], gds_target, delta=huber_delta)
        else:
            gm_loss = F.mse_loss(gm_pred[valid], gm_target)
            gds_loss = F.mse_loss(gds_pred[valid], gds_target)
    else:
        gm_loss = torch.tensor(0.0, device=device)
        gds_loss = torch.tensor(0.0, device=device)

    return gm_loss, gds_loss


def compute_gm_id_consistency_loss(
    gm_pred: torch.Tensor,
    current_pred: torch.Tensor,
    mosfet_info: torch.Tensor,
    node_current_targets: torch.Tensor,
    mosfet_gm: torch.Tensor,
    gm_mean: float,
    gm_std: float,
    current_mean: float,
    current_std: float,
    ptr: torch.Tensor,
    mosfet_ptr: torch.Tensor = None,
) -> torch.Tensor:
    """
    Consistency loss: log10(gm/Id) predicted vs ground truth.

    Ensures gm and I_d predictions are mutually consistent via their ratio.
    Operates in log10 space: log10(gm) - log10(|Id|).

    Args:
        gm_pred: [num_mosfets] z-scored log10(gm) from SS head
        current_pred: [num_nodes] z-scored log10(|I|) from current head
        mosfet_info: [num_mosfets, 7] MOSFET info (col 1 = drain terminal idx)
        node_current_targets: [num_nodes] z-scored log10(|I|) ground truth
        mosfet_gm: [num_mosfets] raw gm (linear scale)
        gm_mean/gm_std: z-score stats for log10(gm)
        current_mean/current_std: z-score stats for log10(|I|)
        ptr: node boundaries per graph
        mosfet_ptr: cumulative MOSFET counts per graph
    """
    device = gm_pred.device
    num_mosfets = mosfet_info.shape[0]
    drain_idx = mosfet_info[:, 1].long()

    # Compute per-MOSFET node offsets for batched graphs
    num_graphs = ptr.shape[0] - 1
    if mosfet_ptr is not None:
        mg_idx = torch.bucketize(
            torch.arange(num_mosfets, device=device),
            mosfet_ptr[1:].to(device), right=True,
        )
    else:
        devices_per_graph = num_mosfets // num_graphs
        mg_idx = torch.arange(num_mosfets, device=device) // devices_per_graph
    offsets = ptr[mg_idx]

    # Predicted log10(gm) and log10(|Id|) at drain terminals
    pred_log_gm = gm_pred * gm_std + gm_mean
    pred_log_id = current_pred[drain_idx + offsets] * current_std + current_mean
    pred_ratio = pred_log_gm - pred_log_id

    # Ground truth log10(gm/Id)
    mosfet_gm = mosfet_gm.to(device)
    valid = mosfet_gm > 1e-12
    gt_log_gm = torch.log10(mosfet_gm.clamp(min=1e-20))
    gt_log_id = node_current_targets.to(device)[drain_idx + offsets] * current_std + current_mean
    gt_ratio = gt_log_gm - gt_log_id

    valid = valid & torch.isfinite(gt_ratio) & torch.isfinite(pred_ratio)
    if valid.any():
        return F.mse_loss(pred_ratio[valid], gt_ratio[valid])
    return torch.tensor(0.0, device=device)


def compute_gm_id_aux_loss(
    gm_id_pred: torch.Tensor,
    mosfet_gm: torch.Tensor,
    node_current_targets: torch.Tensor,
    mosfet_info: torch.Tensor,
    current_mean: float,
    current_std: float,
    gm_id_mean: float,
    gm_id_std: float,
    ptr: torch.Tensor,
    mosfet_ptr: torch.Tensor = None,
) -> torch.Tensor:
    """
    Auxiliary loss for per-MOSFET log10(gm/Id) prediction.

    Target: z-scored log10(gm/Id) = (log10(gm) - log10(|Id|) - mean) / std.
    """
    device = gm_id_pred.device
    num_mosfets = mosfet_info.shape[0]
    drain_idx = mosfet_info[:, 1].long()

    num_graphs = ptr.shape[0] - 1
    if mosfet_ptr is not None:
        mg_idx = torch.bucketize(
            torch.arange(num_mosfets, device=device),
            mosfet_ptr[1:].to(device), right=True,
        )
    else:
        devices_per_graph = num_mosfets // num_graphs
        mg_idx = torch.arange(num_mosfets, device=device) // devices_per_graph
    offsets = ptr[mg_idx]

    mosfet_gm = mosfet_gm.to(device)
    valid = mosfet_gm > 1e-12
    gt_log_gm = torch.log10(mosfet_gm.clamp(min=1e-20))
    gt_log_id = node_current_targets.to(device)[drain_idx + offsets] * current_std + current_mean
    gt_gm_id = gt_log_gm - gt_log_id
    gt_z = (gt_gm_id - gm_id_mean) / max(gm_id_std, 1e-6)

    valid = valid & torch.isfinite(gt_z)
    if valid.any():
        return F.mse_loss(gm_id_pred[valid], gt_z[valid])
    return torch.tensor(0.0, device=device)


# IV Model ID Prediction Loss
def compute_iv_id_loss(
    log_abs_id_pred: torch.Tensor,
    mosfet_info: torch.Tensor,
    node_current_targets: torch.Tensor,
    current_mean: float,
    current_std: float,
    ptr: torch.Tensor,
    mosfet_ptr: torch.Tensor = None,
) -> torch.Tensor:
    """
    MSE loss on IV model's log10(|ID|) vs GT drain current.

    GT drain current comes from node_current_targets at drain terminal indices,
    which are stored in z-scored log10(|I|) space.

    Args:
        log_abs_id_pred: [num_mosfets] predicted log10(|ID|) from IV model
        mosfet_info: [num_mosfets, 7] MOSFET info tensor
        node_current_targets: [num_nodes] z-scored log10(|I|) targets
        current_mean: Mean for current denormalization (log10 scale)
        current_std: Std for current denormalization (log10 scale)
        ptr: Node boundaries per graph [num_graphs + 1]
        mosfet_ptr: Cumulative MOSFET counts per graph [num_graphs + 1], or None

    Returns:
        Scalar loss tensor
    """
    device = log_abs_id_pred.device
    num_mosfets = mosfet_info.shape[0]
    drain_idx = mosfet_info[:, 1].long()

    # Compute per-MOSFET node offsets for batched graphs
    num_graphs = ptr.shape[0] - 1
    if mosfet_ptr is not None:
        mg_idx = torch.bucketize(
            torch.arange(num_mosfets, device=device),
            mosfet_ptr[1:].to(device), right=True,
        )
    else:
        devices_per_graph = num_mosfets // num_graphs
        mg_idx = torch.arange(num_mosfets, device=device) // devices_per_graph
    offsets = ptr[mg_idx]

    # GT drain current: z-scored log10 -> denorm to log10(|I|)
    gt_zscore = node_current_targets.to(device)[drain_idx + offsets]
    gt_log_abs_id = gt_zscore * current_std + current_mean  # log10(|ID|)

    valid = torch.isfinite(gt_log_abs_id) & (gt_log_abs_id > -15)
    if valid.any():
        return F.mse_loss(log_abs_id_pred[valid], gt_log_abs_id[valid])
    return torch.tensor(0.0, device=device)


# AC Prediction Loss
def compute_ac_loss(
    ac_pred: torch.Tensor,
    ac_valid: torch.Tensor,
    ac_mean: torch.Tensor,
    ac_std: torch.Tensor,
    ac_components: list,
    ac_ugbw: torch.Tensor = None,
    ac_pm: torch.Tensor = None,
    ac_am: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute AC prediction loss for configurable components.

    Each AC component (UGBW, PM, AM) can be individually enabled.
    Only uses samples where ac_valid is True (SPICE AC analysis converged).
    Targets are z-score normalized: (value - mean) / std.

    Args:
        ac_pred: [B, N] predicted AC quantities (N = number of enabled components)
        ac_valid: [B] boolean mask for valid AC measurements
        ac_mean: [N] mean for each enabled component from training set
        ac_std: [N] std for each enabled component from training set
        ac_components: List of enabled component names, e.g. ['ugbw', 'pm', 'am']
        ac_ugbw: [B] ground truth UGBW in Hz (if 'ugbw' in ac_components)
        ac_pm: [B] ground truth phase margin in degrees (if 'pm' in ac_components)
        ac_am: [B] ground truth gain margin in dB (if 'am' in ac_components)

    Returns:
        Scalar loss tensor (MSE on normalized targets, only valid samples)
    """
    device = ac_pred.device
    zero = torch.tensor(0.0, device=device)

    if ac_valid is None or not ac_valid.any() or len(ac_components) == 0:
        per_component = {comp: zero for comp in ac_components} if ac_components else {}
        return zero, per_component

    # Filter to valid samples
    valid_pred = ac_pred[ac_valid]  # [N_valid, num_components]

    # Build target tensor dynamically based on enabled components
    target_parts = []
    for comp in ac_components:
        if comp == 'ugbw':
            target_parts.append(torch.log10(ac_ugbw[ac_valid].clamp(min=1.0).to(device)))
        elif comp == 'pm':
            target_parts.append(ac_pm[ac_valid].to(device))
        elif comp == 'am':
            target_parts.append(ac_am[ac_valid].to(device))
    target = torch.stack(target_parts, dim=-1)  # [N_valid, num_components]

    # Z-score normalize targets
    ac_mean = ac_mean.to(device)
    ac_std = ac_std.to(device)
    target_norm = (target - ac_mean) / ac_std.clamp(min=1e-6)

    # Per-component losses
    per_component = {}
    for i, comp in enumerate(ac_components):
        per_component[comp] = F.mse_loss(valid_pred[:, i], target_norm[:, i])

    return F.mse_loss(valid_pred, target_norm), per_component


# Region Classification Loss
def compute_region_loss(
    region_pred: torch.Tensor,
    mosfet_drain_mask: torch.Tensor,
    node_region_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Ordinal regression loss for MOSFET operating region prediction.

    Uses weighted MSE with continuous targets (cutoff=0, triode=1, saturation=2).
    Smooth gradients respect the physical ordering of regions.

    Args:
        region_pred: [num_nodes] scalar prediction per node
        mosfet_drain_mask: [num_nodes] bool - True at valid drain terminals
        node_region_labels: [num_nodes] long - region label at drain nodes, -1 elsewhere

    Returns:
        Scalar region loss
    """
    device = region_pred.device
    mask = mosfet_drain_mask.to(device) & (node_region_labels.to(device) >= 0)

    if not mask.any():
        return torch.tensor(0.0, device=device)

    pred = region_pred[mask]                        # [num_valid]
    target = node_region_labels[mask].float().to(device)  # 0.0, 1.0, 2.0

    return F.mse_loss(pred, target)


def compute_region_loss_coral(
    region_logits: torch.Tensor,
    mosfet_region_labels: torch.Tensor,
) -> torch.Tensor:
    """
    CORAL ordinal cross-entropy loss for MOSFET operating region prediction.

    Uses two binary classifiers for ordinal thresholds:
      - t1: P(region >= triode) = sigma(logit_1)
      - t2: P(region >= saturation) = sigma(logit_2)

    Args:
        region_logits: [num_mosfets, 2] raw logits for ordinal thresholds
        mosfet_region_labels: [num_mosfets] 0=cutoff, 1=triode, 2=saturation

    Returns:
        Scalar CORAL loss (mean BCE over both thresholds)
    """
    device = region_logits.device
    labels = mosfet_region_labels.to(device).long()

    # Binary targets for each threshold
    # t1: 1 if region >= 1 (triode or saturation), 0 if cutoff
    # t2: 1 if region >= 2 (saturation), 0 if cutoff or triode
    t1_target = (labels >= 1).float()
    t2_target = (labels >= 2).float()

    # BCE on each threshold
    loss_t1 = F.binary_cross_entropy_with_logits(region_logits[:, 0], t1_target)
    loss_t2 = F.binary_cross_entropy_with_logits(region_logits[:, 1], t2_target)

    return (loss_t1 + loss_t2) / 2
