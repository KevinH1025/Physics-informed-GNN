"""Combined loss aggregation and uncertainty weighting."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..current_constraints import compute_current_constraint_loss, compute_hardcoded_mirror_loss
from .data_terms import (
    compute_current_loss,
    compute_device_consistency_loss,
    compute_voltage_loss,
)
from .device_index import get_device_graph_idx
from .device_physics import (
    compute_cutoff_physics_loss,
    compute_gm_physics_loss,
    compute_triode_physics_loss,
)
from .kcl import apply_kcl_conservation_projection, compute_kcl_loss
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


class UncertaintyWeights(nn.Module):
    """Kendall et al. 2018 with normalized weights — prevents magnitude explosion.

    Weights are normalized to sum to N_tasks, preserving relative adaptation
    while keeping total gradient magnitude constant.
    """
    def __init__(self, task_names):
        super().__init__()
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in task_names
        })

    def get_weights(self):
        """Normalized weights that sum to N_tasks."""
        raw = {name: torch.exp(-lv) for name, lv in self.log_vars.items()}
        total = sum(raw.values())
        n = len(raw)
        return {name: n * w / total for name, w in raw.items()}

    def regularizer(self):
        """Sum of log_var / 2 — prevents all weights from collapsing to equal."""
        return sum(lv for lv in self.log_vars.values()) / 2


# Combined Loss
def compute_combined_loss(
    voltage_pred: torch.Tensor,
    voltage_target: torch.Tensor,
    current_pred: torch.Tensor = None,
    current_pred_raw: torch.Tensor = None,
    current_target: torch.Tensor = None,
    current_mask: torch.Tensor = None,
    current_weight: float = 1.0,
    voltage_weight: float = 1.0,
    loss_type: str = 'mse',
    huber_delta: float = 1.0,
    kcl_weight: float = 0.0,
    edge_index: torch.Tensor = None,
    num_terminals: torch.Tensor = None,
    train_mask: torch.Tensor = None,
    ptr: torch.Tensor = None,
    terminal_current_sign: torch.Tensor = None,
    current_mean: float = 0.0,
    current_std: float = 1.0,
    kcl_include_mask: torch.Tensor = None,
    kcl_min_current: float = 1e-9,
    kcl_mode: str = 'logsumexp',
    kcl_violation_threshold: float = 0.0,
    kcl_huber_delta: float = 0.0,
    kcl_gt_filter: float = 0.1,
    kcl_exclusive: bool = False,
    kcl_conservation: bool = False,
    kcl_detach_backbone: bool = False,
    kcl_skip_two_term: bool = False,
    kcl_only_two_term: bool = False,
    node_embeddings: torch.Tensor = None,
    current_head: torch.nn.Module = None,
    # Physics constraint parameters
    constraint_weight: float = 0.0,
    mosfet_info: torch.Tensor = None,
    terminal_features: torch.Tensor = None,
    diff_pair_constraints: torch.Tensor = None,
    mirror_constraints: torch.Tensor = None,
    output_stage_constraints: torch.Tensor = None,
    lambda_mirror_constraints: torch.Tensor = None,
    # Lambda mirror parameters
    node_names: list = None,
    vdc_mean: float = 0.0,
    vdc_std: float = 1.0,
    lambda_n: float = 0.05,
    ref_vds_node: str = 'nbias',
    mir_vds_node: str = 'vout',
    # Full voltage predictions (all nodes, not masked) for lambda mirror
    full_voltage_pred: torch.Tensor = None,
    # Voltage node weights for stage2 upweighting
    voltage_node_weights: torch.Tensor = None,
    # gm physics self-consistency loss parameters
    gm_physics_loss_weight: float = 0.0,
    gm_physics_min_vov: float = 0.0,
    gm_physics_use_clm: bool = False,
    gm_physics_use_smaxt: bool = False,
    node_mosfet_vth: torch.Tensor = None,
    mosfet_region_labels: torch.Tensor = None,
    ss_gm_mean: float = 0.0,
    ss_gm_std: float = 1.0,
    ss_gds_mean: float = 0.0,
    ss_gds_std: float = 1.0,
    # AC prediction loss parameters
    ac_loss_weight: float = 0.0,
    ac_pred: torch.Tensor = None,
    ac_ugbw: torch.Tensor = None,
    ac_pm: torch.Tensor = None,
    ac_am: torch.Tensor = None,
    ac_valid: torch.Tensor = None,
    ac_mean: torch.Tensor = None,
    ac_std: torch.Tensor = None,
    ac_components: list = None,
    # Supervised gm/gds prediction loss parameters (per-MOSFET)
    ss_gm_loss_weight: float = 0.0,
    ss_gds_loss_weight: float = 0.0,
    ss_gm_pred: torch.Tensor = None,
    ss_gds_pred: torch.Tensor = None,
    mosfet_gm: torch.Tensor = None,
    mosfet_gds: torch.Tensor = None,
    ss_huber_delta: float = 0.0,
    ss_region_stats: dict = None,
    # Triode physics regularizer loss parameters
    triode_physics_loss_weight: float = 0.0,
    triode_physics_config: dict = None,
    # Cutoff/subthreshold physics loss parameters
    cutoff_physics_loss_weight: float = 0.0,
    cutoff_physics_n_nmos: float = 1.5,
    cutoff_physics_n_pmos: float = 2.0,
    # Ground truth voltages for physics loss Vov computation
    node_voltage_targets: torch.Tensor = None,
    # Ground truth Vov for physics loss filtering
    mosfet_gt_vov: torch.Tensor = None,
    # Device ptr tensors for variable-topology batches
    mosfet_ptr: torch.Tensor = None,
    # Region loss parameters
    region_loss_weight: float = 0.0,
    region_pred: torch.Tensor = None,
    node_region_labels: torch.Tensor = None,
    mosfet_drain_mask: torch.Tensor = None,
    region_logits: torch.Tensor = None,
    # Device consistency loss
    device_consistency_weight: float = 0.0,
    batch=None,
    # Intermediate KCL (deep supervision)
    kcl_intermediate_weight: float = 0.0,
    aux_node_currents: torch.Tensor = None,
    # Vov prediction loss
    vov_loss_weight: float = 0.0,
    vov_pred: torch.Tensor = None,
    mosfet_vov_valid: torch.Tensor = None,
    # Vth prediction loss
    vth_loss_weight: float = 0.0,
    vth_pred: torch.Tensor = None,
    mosfet_vth: torch.Tensor = None,
    # IV model ID prediction loss
    iv_id_loss_weight: float = 0.0,
    iv_log_abs_id: torch.Tensor = None,
    # DC gain prediction loss
    dc_gain_loss_weight: float = 0.0,
    dc_gain_pred: torch.Tensor = None,
    dc_gain_target: torch.Tensor = None,
    dc_gain_mean: float = 0.0,
    dc_gain_std: float = 1.0,
    # gm/Id consistency loss
    gm_id_consistency_weight: float = 0.0,
    node_current_targets: torch.Tensor = None,
    # gm/Id auxiliary prediction loss
    gm_id_aux_weight: float = 0.0,
    gm_id_pred: torch.Tensor = None,
    gm_id_mean: float = 0.0,
    gm_id_std: float = 1.0,
    # Uncertainty weighting
    uncertainty_weights = None,
    # Hardcoded mirror pair loss
    mirror_pair_indices: torch.Tensor = None,
    mirror_pair_ratios: torch.Tensor = None,
    mirror_pair_names: list = None,
    # Vgs/Vds auxiliary loss (from net voltage consistency)
    vdiff_loss_weight: float = 0.0,
    # Vgs/Vds head loss (supervised per-device prediction)
    vgsvds_loss_weight: float = 0.0,
    vgsvds_pred: torch.Tensor = None,
    vgsvds_mean: torch.Tensor = None,  # [2] mean for Vgs, Vds
    vgsvds_std: torch.Tensor = None,   # [2] std for Vgs, Vds
) -> tuple:
    """
    Compute combined loss from all components.

    Total loss = voltage_loss + current_weight * current_loss + kcl_weight * kcl_loss
                 + constraint_weight * constraint_loss

    Args:
        voltage_pred: Predicted voltage values
        voltage_target: Target voltage values
        current_pred: Predicted current values (optional)
        current_target: Target current values (optional)
        current_mask: Mask for nodes with current targets (optional)
        current_weight: Weight for current loss term
        loss_type: Loss function name ('mse', 'huber', 'mae')
        huber_delta: Delta parameter for Huber loss
        kcl_weight: Weight for KCL physics loss (0 to disable)
        edge_index: Graph connectivity [2, num_edges] (for KCL)
        num_terminals: Number of terminals per graph [batch_size] (for KCL)
        train_mask: Mask for internal nodes (for KCL)
        ptr: Node boundaries between graphs [batch_size + 1] (for KCL)
        terminal_current_sign: Sign for each terminal (+1 drain, -1 source) (for KCL)
        current_mean: Mean for current denormalization (log10 scale, for KCL)
        current_std: Std for current denormalization (log10 scale, for KCL)
        kcl_include_mask: Mask for terminals to include in KCL (False for gate/bulk)
        kcl_min_current: Minimum total current for a net to be included in KCL
        constraint_weight: Weight for physics constraint losses (0 to disable)
        mosfet_info: MOSFET info tensor [num_mosfets, 7] (for constraints)
        terminal_features: Node features [num_nodes, num_features] (for constraints)
        diff_pair_constraints: Diff pair constraint indices [N, 3]
        mirror_constraints: Mirror constraint indices [M, 2]
        output_stage_constraints: Output stage constraint indices [K, 2]
        lambda_mirror_constraints: Lambda-corrected mirror indices [L, 2]
        node_names: Node names per graph (for lambda mirror)
        vdc_mean: Mean for voltage denormalization
        vdc_std: Std for voltage denormalization
        lambda_n: Channel length modulation parameter
        ref_vds_node: Node name for reference Vds (lambda mirror)
        mir_vds_node: Node name for mirror Vds (lambda mirror)
        full_voltage_pred: Full voltage predictions (all nodes, not masked) for lambda mirror

    Returns:
        Tuple of (total_loss, voltage_loss, current_loss, kcl_loss, diff_pair_loss,
                  mirror_loss, output_stage_loss, lambda_mirror_loss, gm_physics_loss,
                  ac_loss, ss_loss, triode_physics_loss, triode_eq1/2/3_loss)
    """
    voltage_loss = compute_voltage_loss(
        voltage_pred, voltage_target, loss_type, huber_delta,
        node_weights=voltage_node_weights
    )

    # Apply KCL conservation projection (structural enforcement, replaces soft KCL loss)
    if kcl_conservation and current_pred is not None and edge_index is not None and ptr is not None:
        current_pred = apply_kcl_conservation_projection(
            current_pred, edge_index, num_terminals, train_mask, ptr,
            terminal_current_sign, current_mean, current_std, kcl_include_mask,
        )

    # Compute KCL first to get kcl_supervised_mask (needed for kcl_exclusive current loss)
    kcl_loss = torch.tensor(0.0, device=voltage_loss.device)
    kcl_supervised_mask = None
    if not kcl_conservation and current_pred is not None and edge_index is not None and ptr is not None:
        # When kcl_detach_backbone=True, recompute currents from detached embeddings
        # so KCL gradients only update the current head, not the backbone
        # Use pre-projection (raw) currents if available, so KCL loss sees true violations
        kcl_current_pred = current_pred_raw if current_pred_raw is not None else current_pred
        if kcl_detach_backbone and node_embeddings is not None and current_head is not None:
            kcl_current_pred = current_head(node_embeddings.detach()).squeeze(-1)

        kcl_loss, kcl_supervised_mask = compute_kcl_loss(
            kcl_current_pred, edge_index, num_terminals, train_mask, ptr,
            terminal_current_sign=terminal_current_sign,
            current_mean=current_mean,
            current_std=current_std,
            gt_currents=current_target,
            kcl_include_mask=kcl_include_mask,
            kcl_min_current=kcl_min_current,
            kcl_mode=kcl_mode,
            kcl_violation_threshold=kcl_violation_threshold,
            kcl_skip_two_term=kcl_skip_two_term,
            kcl_only_two_term=kcl_only_two_term,
            kcl_huber_delta=kcl_huber_delta,
            kcl_gt_filter=kcl_gt_filter,
        )

    # Intermediate KCL (deep supervision on auxiliary current predictions)
    kcl_intermediate_loss = torch.tensor(0.0, device=voltage_loss.device)
    if kcl_intermediate_weight > 0 and aux_node_currents is not None and edge_index is not None and ptr is not None:
        kcl_intermediate_loss, _ = compute_kcl_loss(
            aux_node_currents, edge_index, num_terminals, train_mask, ptr,
            terminal_current_sign=terminal_current_sign,
            current_mean=current_mean,
            current_std=current_std,
            gt_currents=current_target,
            kcl_include_mask=kcl_include_mask,
            kcl_min_current=kcl_min_current,
            kcl_mode=kcl_mode,
            kcl_violation_threshold=kcl_violation_threshold,
            kcl_skip_two_term=kcl_skip_two_term,
            kcl_only_two_term=kcl_only_two_term,
            kcl_huber_delta=kcl_huber_delta,
            kcl_gt_filter=kcl_gt_filter,
        )

    # When kcl_exclusive=True, exclude KCL-supervised terminals from current MSE
    effective_current_mask = current_mask
    if kcl_exclusive and kcl_supervised_mask is not None and current_mask is not None:
        effective_current_mask = current_mask & ~kcl_supervised_mask

    current_loss = compute_current_loss(
        current_pred, current_target, effective_current_mask, loss_type, huber_delta
    )
    if current_loss.device != voltage_loss.device:
        current_loss = current_loss.to(voltage_loss.device)

    # Physics-based current constraints
    diff_pair_loss = torch.tensor(0.0, device=voltage_loss.device)
    mirror_loss = torch.tensor(0.0, device=voltage_loss.device)
    output_stage_loss = torch.tensor(0.0, device=voltage_loss.device)
    lambda_mirror_loss = torch.tensor(0.0, device=voltage_loss.device)

    if constraint_weight > 0 and current_pred is not None and mosfet_info is not None and ptr is not None:
        # Note: For lambda_mirror, we need full voltage predictions, not masked ones
        # voltage_pred here is masked (only internal nodes). We pass current_pred which is
        # full (all terminals), so lambda_mirror uses current_pred's indexing for voltages.
        # Actually, we need to pass full_voltage_pred separately - see loops.py
        _, diff_pair_loss, mirror_loss, output_stage_loss, lambda_mirror_loss = compute_current_constraint_loss(
            pred_currents=current_pred,
            mosfet_info=mosfet_info,
            terminal_features=terminal_features,
            diff_pair_constraints=diff_pair_constraints,
            mirror_constraints=mirror_constraints,
            output_stage_constraints=output_stage_constraints,
            lambda_mirror_constraints=lambda_mirror_constraints,
            ptr=ptr,
            current_mean=current_mean,
            current_std=current_std,
            pred_voltages=full_voltage_pred,
            node_names=node_names,
            vdc_mean=vdc_mean,
            vdc_std=vdc_std,
            lambda_n=lambda_n,
            ref_vds_node=ref_vds_node,
            mir_vds_node=mir_vds_node,
        )

    constraint_loss = diff_pair_loss + mirror_loss + output_stage_loss + lambda_mirror_loss

    # Hardcoded mirror pair loss (MSE in normalized space)
    hardcoded_mirror_loss = torch.tensor(0.0, device=voltage_loss.device)
    mirror_pair_losses = {}
    if constraint_weight > 0 and current_pred is not None and mosfet_info is not None and mirror_pair_indices is not None and len(mirror_pair_indices) > 0:
        hardcoded_mirror_loss, mirror_pair_losses = compute_hardcoded_mirror_loss(
            current_pred=current_pred,
            mosfet_info=mosfet_info,
            mirror_pair_indices=mirror_pair_indices,
            mirror_pair_ratios=mirror_pair_ratios,
            mirror_pair_names=mirror_pair_names,
            ptr=ptr,
            current_std=current_std,
            mosfet_ptr=mosfet_ptr,
        )

    # gm physics self-consistency loss
    gm_physics_loss = torch.tensor(0.0, device=voltage_loss.device)
    if gm_physics_loss_weight > 0 and ss_gm_pred is not None and current_pred is not None and full_voltage_pred is not None:
        gm_physics_loss = compute_gm_physics_loss(
            ss_gm_pred=ss_gm_pred,
            pred_currents=current_pred,
            full_voltage_pred=full_voltage_pred,
            mosfet_info=mosfet_info,
            node_mosfet_vth=node_mosfet_vth,
            mosfet_region_labels=mosfet_region_labels,
            ptr=ptr,
            vdc_mean=vdc_mean,
            vdc_std=vdc_std,
            current_mean=current_mean,
            current_std=current_std,
            ss_gm_mean=ss_gm_mean,
            ss_gm_std=ss_gm_std,
            min_vov=gm_physics_min_vov,
            mosfet_gt_vov=mosfet_gt_vov,
            mosfet_ptr=mosfet_ptr,
            ss_gds_pred=ss_gds_pred,
            ss_gds_mean=ss_gds_mean,
            ss_gds_std=ss_gds_std,
            use_clm=gm_physics_use_clm,
            node_voltage_targets=node_voltage_targets,
            use_smaxt=gm_physics_use_smaxt,
        )

    # AC prediction loss
    ac_loss = torch.tensor(0.0, device=voltage_loss.device)
    ac_per_component = {}
    if ac_loss_weight > 0 and ac_pred is not None and ac_mean is not None and ac_components:
        ac_loss, ac_per_component = compute_ac_loss(
            ac_pred=ac_pred,
            ac_valid=ac_valid,
            ac_mean=ac_mean,
            ac_std=ac_std,
            ac_components=ac_components,
            ac_ugbw=ac_ugbw,
            ac_pm=ac_pm,
            ac_am=ac_am,
        )

    # Supervised gm/gds prediction loss (per-MOSFET)
    ss_gm_loss = torch.tensor(0.0, device=voltage_loss.device)
    ss_gds_loss = torch.tensor(0.0, device=voltage_loss.device)
    if (ss_gm_loss_weight > 0 or ss_gds_loss_weight > 0) and ss_gm_pred is not None and ss_gds_pred is not None and mosfet_gm is not None:
        ss_gm_loss, ss_gds_loss = compute_ss_loss(
            gm_pred=ss_gm_pred,
            gds_pred=ss_gds_pred,
            mosfet_gm=mosfet_gm,
            mosfet_gds=mosfet_gds,
            gm_mean=ss_gm_mean,
            gm_std=ss_gm_std,
            gds_mean=ss_gds_mean,
            gds_std=ss_gds_std,
            huber_delta=ss_huber_delta,
            region_stats=ss_region_stats,
            mosfet_region_labels=mosfet_region_labels,
        )

    # Triode physics regularizer loss
    triode_physics_loss = torch.tensor(0.0, device=voltage_loss.device)
    triode_eq1_loss = torch.tensor(0.0, device=voltage_loss.device)
    triode_eq2_loss = torch.tensor(0.0, device=voltage_loss.device)
    triode_eq3_loss = torch.tensor(0.0, device=voltage_loss.device)
    if triode_physics_loss_weight > 0 and ss_gm_pred is not None and ss_gds_pred is not None and full_voltage_pred is not None:
        tri_cfg = triode_physics_config or {}
        triode_physics_loss, triode_eq1_loss, triode_eq2_loss, triode_eq3_loss = compute_triode_physics_loss(
            ss_gm_pred=ss_gm_pred,
            ss_gds_pred=ss_gds_pred,
            pred_currents=current_pred,
            full_voltage_pred=full_voltage_pred,
            mosfet_info=mosfet_info,
            node_mosfet_vth=node_mosfet_vth,
            mosfet_region_labels=mosfet_region_labels,
            ptr=ptr,
            vdc_mean=vdc_mean,
            vdc_std=vdc_std,
            current_mean=current_mean,
            current_std=current_std,
            ss_gm_mean=ss_gm_mean,
            ss_gm_std=ss_gm_std,
            ss_gds_mean=ss_gds_mean,
            ss_gds_std=ss_gds_std,
            min_vov=tri_cfg.get('min_vov', 0.0),
            eq1_enabled=tri_cfg.get('eq1_enabled', True),
            eq1_weight=tri_cfg.get('eq1_weight', 1.0),
            eq2_enabled=tri_cfg.get('eq2_enabled', True),
            eq2_weight=tri_cfg.get('eq2_weight', 1.0),
            eq2_max_vds_vov=tri_cfg.get('eq2_max_vds_vov', 1.0),
            eq3_enabled=tri_cfg.get('eq3_enabled', True),
            eq3_weight=tri_cfg.get('eq3_weight', 1.0),
            mosfet_gt_vov=mosfet_gt_vov,
            mosfet_ptr=mosfet_ptr,
            node_voltage_targets=node_voltage_targets,
        )

    # Cutoff/subthreshold physics loss
    cutoff_physics_loss = torch.tensor(0.0, device=voltage_loss.device)
    if cutoff_physics_loss_weight > 0 and ss_gm_pred is not None and current_pred is not None:
        cutoff_physics_loss = compute_cutoff_physics_loss(
            ss_gm_pred=ss_gm_pred,
            pred_currents=current_pred,
            mosfet_info=mosfet_info,
            mosfet_region_labels=mosfet_region_labels,
            ptr=ptr,
            current_mean=current_mean,
            current_std=current_std,
            ss_gm_mean=ss_gm_mean,
            ss_gm_std=ss_gm_std,
            n_nmos=cutoff_physics_n_nmos,
            n_pmos=cutoff_physics_n_pmos,
            mosfet_ptr=mosfet_ptr,
        )

    # Region loss (CORAL ordinal cross-entropy or MSE fallback)
    region_loss = torch.tensor(0.0, device=voltage_loss.device)
    if region_loss_weight > 0:
        if region_logits is not None and mosfet_region_labels is not None:
            # CORAL: per-MOSFET ordinal logits
            region_loss = compute_region_loss_coral(
                region_logits=region_logits,
                mosfet_region_labels=mosfet_region_labels,
            )
        elif region_pred is not None and node_region_labels is not None and mosfet_drain_mask is not None:
            # Legacy: per-node MSE
            region_loss = compute_region_loss(
                region_pred=region_pred,
                mosfet_drain_mask=mosfet_drain_mask,
                node_region_labels=node_region_labels,
            )

    # Device consistency loss
    if device_consistency_weight > 0 and current_pred is not None and batch is not None:
        dev_consistency_loss = compute_device_consistency_loss(current_pred, batch)
    else:
        dev_consistency_loss = torch.tensor(0.0, device=voltage_pred.device)

    # Vov prediction loss
    vov_loss = torch.tensor(0.0, device=voltage_pred.device)
    if vov_loss_weight > 0 and vov_pred is not None and mosfet_gt_vov is not None:
        vov_loss = compute_vov_loss(vov_pred, mosfet_gt_vov, mosfet_vov_valid)

    # Vth prediction loss
    vth_loss = torch.tensor(0.0, device=voltage_pred.device)
    if vth_loss_weight > 0 and vth_pred is not None and mosfet_vth is not None:
        vth_loss = compute_vth_loss(vth_pred, mosfet_vth.to(voltage_pred.device))

    # IV model ID prediction loss
    iv_id_loss = torch.tensor(0.0, device=voltage_pred.device)
    if iv_id_loss_weight > 0 and iv_log_abs_id is not None and current_target is not None:
        iv_id_loss = compute_iv_id_loss(
            log_abs_id_pred=iv_log_abs_id,
            mosfet_info=mosfet_info,
            node_current_targets=current_target,
            current_mean=current_mean,
            current_std=current_std,
            ptr=ptr,
            mosfet_ptr=mosfet_ptr,
        )

    # DC gain prediction loss (z-scored MSE, all samples)
    dc_gain_loss = torch.tensor(0.0, device=voltage_pred.device)
    if dc_gain_loss_weight > 0 and dc_gain_pred is not None and dc_gain_target is not None:
        dc_gain_target_z = (dc_gain_target - dc_gain_mean) / max(dc_gain_std, 1e-6)
        dc_gain_loss = F.mse_loss(dc_gain_pred, dc_gain_target_z)

    # gm/Id consistency loss
    gm_id_loss = torch.tensor(0.0, device=voltage_pred.device)
    if gm_id_consistency_weight > 0 and ss_gm_pred is not None and current_pred is not None and mosfet_info is not None and node_current_targets is not None and mosfet_gm is not None:
        gm_id_loss = compute_gm_id_consistency_loss(
            gm_pred=ss_gm_pred, current_pred=current_pred.float(),
            mosfet_info=mosfet_info, node_current_targets=node_current_targets,
            mosfet_gm=mosfet_gm, gm_mean=ss_gm_mean, gm_std=ss_gm_std,
            current_mean=current_mean, current_std=current_std,
            ptr=ptr, mosfet_ptr=mosfet_ptr,
        )

    # gm/Id auxiliary prediction loss
    gm_id_aux_loss = torch.tensor(0.0, device=voltage_pred.device)
    if gm_id_aux_weight > 0 and gm_id_pred is not None and mosfet_info is not None and node_current_targets is not None and mosfet_gm is not None:
        gm_id_aux_loss = compute_gm_id_aux_loss(
            gm_id_pred=gm_id_pred, mosfet_gm=mosfet_gm,
            node_current_targets=node_current_targets,
            mosfet_info=mosfet_info, current_mean=current_mean, current_std=current_std,
            gm_id_mean=gm_id_mean, gm_id_std=gm_id_std,
            ptr=ptr, mosfet_ptr=mosfet_ptr,
        )

    # Vgs/Vds auxiliary loss — supervise voltage differences across MOSFET terminals
    vdiff_loss = torch.tensor(0.0, device=voltage_loss.device)
    if vdiff_loss_weight > 0 and mosfet_info is not None and full_voltage_pred is not None and node_voltage_targets is not None:
        mi = mosfet_info.long()
        num_mosfets = mi.shape[0]
        num_graphs = ptr.shape[0] - 1
        offsets = get_device_graph_idx(num_mosfets, num_graphs, mosfet_ptr, mi.device)
        offsets = ptr[offsets]
        g_idx = mi[:, 0] + offsets
        d_idx = mi[:, 1] + offsets
        s_idx = mi[:, 2] + offsets
        # Predicted Vgs, Vds from node voltage predictions
        vgs_pred = full_voltage_pred[g_idx] - full_voltage_pred[s_idx]
        vds_pred = full_voltage_pred[d_idx] - full_voltage_pred[s_idx]
        # Ground truth Vgs, Vds from node voltage targets
        vgs_gt = node_voltage_targets[g_idx] - node_voltage_targets[s_idx]
        vds_gt = node_voltage_targets[d_idx] - node_voltage_targets[s_idx]
        vdiff_loss = F.mse_loss(vgs_pred, vgs_gt) + F.mse_loss(vds_pred, vds_gt)

    # Vgs/Vds head loss — supervised per-device terminal voltage prediction
    vgsvds_loss = torch.tensor(0.0, device=voltage_loss.device)
    if vgsvds_loss_weight > 0 and vgsvds_pred is not None and mosfet_info is not None and node_voltage_targets is not None:
        mi = mosfet_info.long()
        num_mosfets = mi.shape[0]
        num_graphs = ptr.shape[0] - 1
        offsets = get_device_graph_idx(num_mosfets, num_graphs, mosfet_ptr, mi.device)
        offsets = ptr[offsets]
        g_idx = mi[:, 0] + offsets
        d_idx = mi[:, 1] + offsets
        s_idx = mi[:, 2] + offsets
        vgs_gt_raw = node_voltage_targets[g_idx] - node_voltage_targets[s_idx]
        vds_gt_raw = node_voltage_targets[d_idx] - node_voltage_targets[s_idx]
        # Normalize to z-score space
        if vgsvds_mean is not None and vgsvds_std is not None:
            vgs_gt = (vgs_gt_raw - vgsvds_mean[0]) / vgsvds_std[0].clamp(min=1e-6)
            vds_gt = (vds_gt_raw - vgsvds_mean[1]) / vgsvds_std[1].clamp(min=1e-6)
        else:
            vgs_gt = vgs_gt_raw
            vds_gt = vds_gt_raw
        vgsvds_gt = torch.stack([vgs_gt, vds_gt], dim=-1)  # [M, 2]
        vgsvds_loss = F.mse_loss(vgsvds_pred, vgsvds_gt)

    # Terms that always use their static weight, in both the uncertainty-weighted
    # and the plain branch. Both branches sum this one list so they cannot drift
    # apart as terms are added.
    statically_weighted = (
        (kcl_weight, kcl_loss),
        (constraint_weight, constraint_loss),
        (gm_physics_loss_weight, gm_physics_loss),
        (ac_loss_weight, ac_loss),
        (triode_physics_loss_weight, triode_physics_loss),
        (cutoff_physics_loss_weight, cutoff_physics_loss),
        (region_loss_weight, region_loss),
        (device_consistency_weight, dev_consistency_loss),
        (kcl_intermediate_weight, kcl_intermediate_loss),
        (vth_loss_weight, vth_loss),
        (iv_id_loss_weight, iv_id_loss),
        (dc_gain_loss_weight, dc_gain_loss),
        (gm_id_consistency_weight, gm_id_loss),
        (gm_id_aux_weight, gm_id_aux_loss),
        (constraint_weight, hardcoded_mirror_loss),
        (vdiff_loss_weight, vdiff_loss),
        (vgsvds_loss_weight, vgsvds_loss),
    )

    if uncertainty_weights is not None:
        # Normalized uncertainty weighting (Kendall et al. 2018, weights sum to N_tasks)
        uw = uncertainty_weights.get_weights()
        total_loss = torch.tensor(0.0, device=voltage_pred.device)
        if voltage_weight > 0:
            total_loss = total_loss + uw['voltage'] * voltage_loss
        if current_weight > 0:
            total_loss = total_loss + uw['current'] * current_loss
        if ss_gm_loss_weight > 0:
            total_loss = total_loss + uw['ss_gm'] * ss_gm_loss
        if ss_gds_loss_weight > 0:
            total_loss = total_loss + uw['ss_gds'] * ss_gds_loss
        if vov_loss_weight > 0:
            total_loss = total_loss + uw['vov'] * vov_loss
        # Log_var regularizer (prevents all weights from becoming equal)
        total_loss = total_loss + uncertainty_weights.regularizer()
    else:
        total_loss = (voltage_weight * voltage_loss +
                      current_weight * current_loss +
                      ss_gm_loss_weight * ss_gm_loss +
                      ss_gds_loss_weight * ss_gds_loss +
                      vov_loss_weight * vov_loss)

    for term_weight, term_loss in statically_weighted:
        total_loss = total_loss + term_weight * term_loss

    return total_loss, voltage_loss, current_loss, kcl_loss, diff_pair_loss, mirror_loss, output_stage_loss, lambda_mirror_loss, gm_physics_loss, ac_loss, ss_gm_loss, ss_gds_loss, triode_physics_loss, triode_eq1_loss, triode_eq2_loss, triode_eq3_loss, region_loss, cutoff_physics_loss, kcl_intermediate_loss, vov_loss, vth_loss, iv_id_loss, ac_per_component, dc_gain_loss, gm_id_loss, gm_id_aux_loss, hardcoded_mirror_loss, mirror_pair_losses, vdiff_loss, vgsvds_loss
