"""MOSFET device physics self-consistency losses (gm, smaxt, cutoff, triode)."""

import torch
import torch.nn.functional as F

from .device_index import get_device_graph_idx


# gm Self-Consistency Physics Loss
def compute_gm_physics_loss(
    ss_gm_pred: torch.Tensor,
    pred_currents: torch.Tensor,
    full_voltage_pred: torch.Tensor,
    mosfet_info: torch.Tensor,
    node_mosfet_vth: torch.Tensor,
    mosfet_region_labels: torch.Tensor,
    ptr: torch.Tensor,
    vdc_mean: float = 0.0,
    vdc_std: float = 1.0,
    current_mean: float = 0.0,
    current_std: float = 1.0,
    ss_gm_mean: float = 0.0,
    ss_gm_std: float = 1.0,
    min_vov: float = 0.0,
    mosfet_gt_vov: torch.Tensor = None,
    mosfet_ptr: torch.Tensor = None,
    ss_gds_pred: torch.Tensor = None,
    ss_gds_mean: float = 0.0,
    ss_gds_std: float = 1.0,
    use_clm: bool = False,
    node_voltage_targets: torch.Tensor = None,
    use_smaxt: bool = False,
) -> torch.Tensor:
    """
    Saturation gm physics self-consistency loss.

    Without CLM correction (use_clm=False):
        gm_pred vs gm_physics = 2*I_D / Vov
        Compared in log10 space.

    With CLM correction (use_clm=True):
        gm·Vov + 2·gds·Vds = 2·Id
        Compared as: log10(gm·Vov + 2·gds·Vds) vs log10(2·Id)
        This ties together all four prediction heads (V, I, gm, gds)
        and accounts for channel length modulation.

    Only applied to MOSFETs in saturation with Vov >= min_vov.

    Args:
        ss_gm_pred: [num_nodes] SS head gm prediction (z-score normalized log10)
        pred_currents: [num_nodes] current head predictions (z-score normalized log10)
        full_voltage_pred: [num_nodes] voltage predictions (z-score normalized)
        mosfet_info: [num_mosfets, 7] cols: gate_t, drain_t, source_t, gate_n, drain_n, source_n, is_nmos
        node_mosfet_vth: [num_nodes] SPICE Vth at drain terminal positions (|Vth| in volts)
        mosfet_region_labels: [num_mosfets] 0=cutoff, 1=triode, 2=saturation
        ptr: [batch_size+1] node boundaries
        vdc_mean/std: voltage denormalization (V_real = pred * vdc_std + vdc_mean)
        current_mean/std: current denormalization (I_real = 10^(pred * std + mean))
        ss_gm_mean/std: SS gm denormalization (log10_gm = pred * std + mean)
        min_vov: minimum overdrive voltage (V) — transistors with Vov < min_vov are excluded.
        mosfet_gt_vov: [num_mosfets] pre-computed ground truth Vov (from SPICE gm/I_D).
                       If provided, used for min_vov filtering instead of predicted Vov.
        ss_gds_pred: [num_nodes] SS head gds prediction (z-score normalized log10). Required if use_clm=True.
        ss_gds_mean/std: SS gds denormalization (log10_gds = pred * std + mean)
        use_clm: if True, use CLM-corrected equation gm·Vov + 2·gds·Vds = 2·Id
    """
    device = ss_gm_pred.device

    if (mosfet_info is None or len(mosfet_info) == 0 or
        pred_currents is None or full_voltage_pred is None or
        node_mosfet_vth is None):
        return torch.tensor(0.0, device=device)

    if use_smaxt:
        return compute_smaxt_gm_loss(
            ss_gm_pred=ss_gm_pred,
            pred_currents=pred_currents,
            full_voltage_pred=full_voltage_pred,
            mosfet_info=mosfet_info,
            node_mosfet_vth=node_mosfet_vth,
            ptr=ptr,
            vdc_mean=vdc_mean, vdc_std=vdc_std,
            current_mean=current_mean, current_std=current_std,
            ss_gm_mean=ss_gm_mean, ss_gm_std=ss_gm_std,
            mosfet_ptr=mosfet_ptr,
        )

    if use_clm and ss_gds_pred is None:
        return torch.tensor(0.0, device=device)

    num_mosfets = len(mosfet_info)
    num_graphs = len(ptr) - 1

    # Build global mosfet indices with batch offsets
    mosfet_graph_idx = get_device_graph_idx(num_mosfets, num_graphs, mosfet_ptr, device)
    node_offsets = ptr[mosfet_graph_idx]

    # Terminal indices (for I_D and gm)
    drain_term_idx = mosfet_info[:, 1] + node_offsets
    # Net node indices (for Vgs, Vds — voltages predicted at net nodes)
    gate_net_idx = mosfet_info[:, 3] + node_offsets
    drain_net_idx = mosfet_info[:, 4] + node_offsets
    source_net_idx = mosfet_info[:, 5] + node_offsets

    # Saturation mask + Vth validity
    sat_mask = (mosfet_region_labels.to(device) == 2)
    vth_at_drain = node_mosfet_vth[drain_term_idx]
    valid_mask = sat_mask & (vth_at_drain.abs() > 1e-6)

    # Apply min_vov filter using GT Vov (stable) or predicted Vov (fallback)
    has_gt_vov_filter = mosfet_gt_vov is not None and min_vov > 0
    if has_gt_vov_filter:
        gt_vov = mosfet_gt_vov.to(device)
        valid_mask = valid_mask & (gt_vov >= min_vov)

    if not valid_mask.any():
        return torch.tensor(0.0, device=device)

    # gm from SS head (denormalize to real value)
    # ss_gm_pred is per-MOSFET (from concat head), not per-node
    log10_gm_pred = ss_gm_pred[valid_mask] * ss_gm_std + ss_gm_mean

    # I_D from current head (denormalize to real amps)
    Id_real = torch.pow(10, pred_currents[drain_term_idx[valid_mask]] * current_std + current_mean).clamp(min=1e-12)

    # Voltages for Vov computation: use GT voltages when available (accurate), else predicted (noisy)
    if node_voltage_targets is not None:
        V_gate = node_voltage_targets[gate_net_idx[valid_mask]]
        V_source = node_voltage_targets[source_net_idx[valid_mask]]
    else:
        V_gate = full_voltage_pred[gate_net_idx[valid_mask]] * vdc_std + vdc_mean
        V_source = full_voltage_pred[source_net_idx[valid_mask]] * vdc_std + vdc_mean
    Vgs = V_gate - V_source

    # Vth from SPICE (already |Vth| in volts)
    Vth = vth_at_drain[valid_mask]

    # Compute overdrive with correct sign per device type
    is_nmos = mosfet_info[:, 6][valid_mask].bool()
    Vov = torch.where(is_nmos, Vgs - Vth, -Vgs - Vth)

    # Apply min_vov filter only if explicitly set > 0 (legacy behavior)
    if min_vov > 0:
        vov_threshold = max(min_vov, 1e-3)
        if has_gt_vov_filter:
            Vov_filtered = Vov.clamp(min=vov_threshold)
        else:
            vov_mask = Vov >= vov_threshold
            if not vov_mask.any():
                return torch.tensor(0.0, device=device)
            log10_gm_pred = log10_gm_pred[vov_mask]
            Id_real = Id_real[vov_mask]
            Vov = Vov[vov_mask]
            is_nmos = is_nmos[vov_mask]
            Vov_filtered = Vov.clamp(min=vov_threshold)
    else:
        Vov_filtered = Vov  # no filtering, formula handles all Vov

    if use_clm:
        # CLM-corrected: gm·Vov + 2·gds·Vds = 2·Id
        gm_real = torch.pow(10, log10_gm_pred).clamp(min=1e-15)
        log10_gds_pred = ss_gds_pred[valid_mask] * ss_gds_std + ss_gds_mean
        if min_vov > 0 and not has_gt_vov_filter:
            log10_gds_pred = log10_gds_pred[vov_mask]
        gds_real = torch.pow(10, log10_gds_pred).clamp(min=1e-15)

        if node_voltage_targets is not None:
            V_drain = node_voltage_targets[drain_net_idx[valid_mask]]
            V_src = node_voltage_targets[source_net_idx[valid_mask]]
        else:
            V_drain = full_voltage_pred[drain_net_idx[valid_mask]] * vdc_std + vdc_mean
            V_src = full_voltage_pred[source_net_idx[valid_mask]] * vdc_std + vdc_mean
        if min_vov > 0 and not has_gt_vov_filter:
            V_drain = V_drain[vov_mask]
            V_src = V_src[vov_mask]
        Vds_raw = V_drain - V_src
        Vds = torch.where(is_nmos, Vds_raw, -Vds_raw).clamp(min=1e-6)

        lhs = (gm_real * Vov_filtered + 2.0 * gds_real * Vds).clamp(min=1e-12)
        rhs = (2.0 * Id_real).clamp(min=1e-12)
        return F.mse_loss(torch.log10(lhs), torch.log10(rhs))
    else:
        # Smooth all-region formula: blend strong and weak inversion
        Vt = 0.026   # thermal voltage (kT/q at room temp)
        delta = 0.06  # 60mV smoothing for strong inversion
        n_sub = 2.0   # subthreshold slope factor

        blend = torch.sigmoid(Vov_filtered / Vt)
        gm_strong = 2.0 * Id_real / (Vov_filtered + delta).clamp(min=1e-12)
        gm_weak = Id_real / (n_sub * Vt)
        gm_physics = (blend * gm_strong + (1.0 - blend) * gm_weak).clamp(min=1e-12)
        log10_gm_physics = torch.log10(gm_physics)
        return F.mse_loss(log10_gm_pred, log10_gm_physics)


def compute_smaxt_gm_loss(
    ss_gm_pred: torch.Tensor,
    pred_currents: torch.Tensor,
    full_voltage_pred: torch.Tensor,
    mosfet_info: torch.Tensor,
    node_mosfet_vth: torch.Tensor,
    ptr: torch.Tensor,
    vdc_mean: float = 0.0,
    vdc_std: float = 1.0,
    current_mean: float = 0.0,
    current_std: float = 1.0,
    ss_gm_mean: float = 0.0,
    ss_gm_std: float = 1.0,
    mosfet_ptr: torch.Tensor = None,
) -> torch.Tensor:
    """All-region gm self-consistency via smooth-min/max formulation.

    n_vt     = n · V_t                                  # n=1.5 NMOS, 2.0 PMOS;  V_t=25.85 mV
    V_ds_eff = V_ds - n_vt · softplus((V_ds - V_ov) / n_vt)            # ≈ min(V_ds, V_ov)
    denom    = n_vt + n_vt · softplus((V_ov - V_ds_eff/2 - n_vt) / n_vt)  # ≈ max(V_ov - V_ds_eff/2, n_vt)
    gm_smaxt = I_d / denom

    Asymptotes:
      strong-inv saturation (V_ds ≥ V_ov >> n·V_t) :  gm = 2·I_d / V_ov
      strong-inv triode     (V_ds < V_ov, V_ov >> n·V_t) :  gm = I_d / (V_ov - V_ds/2)
      weak inversion        (V_ov << n·V_t)        :  gm = I_d / (n·V_t)

    Compares log10(gm_pred) to log10(gm_smaxt) — works in all regions, no min_vov / region filter.
    Drops devices with |Vth| ≤ 1e-6 (invalid SPICE OP).
    Uses PREDICTED V/I/gm (no node_voltage_targets fallback) — couples all heads.
    """
    device = ss_gm_pred.device
    if (mosfet_info is None or len(mosfet_info) == 0
        or pred_currents is None or full_voltage_pred is None
        or node_mosfet_vth is None):
        return torch.tensor(0.0, device=device)

    num_mosfets = len(mosfet_info)
    num_graphs = len(ptr) - 1
    mosfet_graph_idx = get_device_graph_idx(num_mosfets, num_graphs, mosfet_ptr, device)
    node_offsets = ptr[mosfet_graph_idx]

    drain_term_idx = mosfet_info[:, 1] + node_offsets
    gate_net_idx = mosfet_info[:, 3] + node_offsets
    drain_net_idx = mosfet_info[:, 4] + node_offsets
    source_net_idx = mosfet_info[:, 5] + node_offsets

    vth_at_drain = node_mosfet_vth[drain_term_idx]
    valid_mask = vth_at_drain.abs() > 1e-6
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)

    is_nmos = mosfet_info[:, 6][valid_mask].bool()

    # Denormalize predicted V at gate / drain / source nets
    V_g = full_voltage_pred[gate_net_idx[valid_mask]] * vdc_std + vdc_mean
    V_d = full_voltage_pred[drain_net_idx[valid_mask]] * vdc_std + vdc_mean
    V_s = full_voltage_pred[source_net_idx[valid_mask]] * vdc_std + vdc_mean

    # Polarity-flipped V_gs, V_ds (NMOS positive; for PMOS, flip sign)
    Vgs = torch.where(is_nmos, V_g - V_s, V_s - V_g)
    Vds = torch.where(is_nmos, V_d - V_s, V_s - V_d)

    # |Vth| from SPICE; Vov can be negative in cutoff
    Vth = vth_at_drain[valid_mask]
    Vov = Vgs - Vth

    # Per-device n_vt (n=1.5 NMOS, 2.0 PMOS; V_t = kT/q at 300K)
    Vt = 0.02585
    n = torch.where(is_nmos, torch.full_like(Vth, 1.5), torch.full_like(Vth, 2.0))
    n_vt = n * Vt

    # Predicted I_d at drain terminal (z-score log10 → real amps)
    log_Id = pred_currents[drain_term_idx[valid_mask]] * current_std + current_mean
    I_d = torch.pow(10.0, log_Id).clamp(min=1e-15)

    # Smooth min(V_ds, V_ov)
    arg1 = ((Vds - Vov) / n_vt).clamp(-50.0, 50.0)
    Vds_eff = Vds - n_vt * F.softplus(arg1)

    # Smooth max(V_ov - V_ds_eff/2, n_vt)
    arg2 = ((Vov - Vds_eff / 2.0 - n_vt) / n_vt).clamp(-50.0, 50.0)
    denom = n_vt + n_vt * F.softplus(arg2)

    gm_smaxt = (I_d / denom.clamp(min=1e-12)).clamp(min=1e-15)

    # SS head gm prediction (de-normalize z-score log10 → log10 real)
    log10_gm_pred = ss_gm_pred[valid_mask] * ss_gm_std + ss_gm_mean
    log10_gm_smaxt = torch.log10(gm_smaxt)

    return F.mse_loss(log10_gm_pred, log10_gm_smaxt)


def compute_cutoff_physics_loss(
    ss_gm_pred: torch.Tensor,
    pred_currents: torch.Tensor,
    mosfet_info: torch.Tensor,
    mosfet_region_labels: torch.Tensor,
    ptr: torch.Tensor,
    current_mean: float = 0.0,
    current_std: float = 1.0,
    ss_gm_mean: float = 0.0,
    ss_gm_std: float = 1.0,
    n_nmos: float = 1.5,
    n_pmos: float = 2.0,
    mosfet_ptr: torch.Tensor = None,
) -> torch.Tensor:
    """
    Subthreshold gm physics loss for cutoff devices: gm = Id / (n * Vt).

    In subthreshold, gm/Id = 1/(n*Vt) where n is the subthreshold slope
    factor (process-dependent, ~1.5 for NMOS and ~2.0 for PMOS in SKY130)
    and Vt = kT/q = 25.85mV at 27°C.

    Compared in log10 space: log10(gm_pred) vs log10(Id / (n * Vt)).

    Args:
        ss_gm_pred: [num_nodes] SS head gm prediction (z-score normalized log10)
        pred_currents: [num_nodes] current head predictions (z-score normalized log10)
        mosfet_info: [num_mosfets, 7] cols: gate_t, drain_t, source_t, gate_n, drain_n, source_n, is_nmos
        mosfet_region_labels: [num_mosfets] 0=cutoff, 1=triode, 2=saturation
        ptr: [batch_size+1] node boundaries
        current_mean/std: current denormalization (I_real = 10^(pred * std + mean))
        ss_gm_mean/std: SS gm denormalization (log10_gm = pred * std + mean)
        n_nmos: subthreshold slope factor for NMOS (SKY130 default: 1.5)
        n_pmos: subthreshold slope factor for PMOS (SKY130 default: 2.0)
    """
    device = ss_gm_pred.device
    Vt = 0.02585  # kT/q at 27°C

    if (mosfet_info is None or len(mosfet_info) == 0 or
        pred_currents is None or mosfet_region_labels is None):
        return torch.tensor(0.0, device=device)

    num_mosfets = len(mosfet_info)
    num_graphs = len(ptr) - 1

    mosfet_graph_idx = get_device_graph_idx(num_mosfets, num_graphs, mosfet_ptr, device)
    node_offsets = ptr[mosfet_graph_idx]

    drain_term_idx = mosfet_info[:, 1] + node_offsets

    # Cutoff mask (region == 0)
    cutoff_mask = (mosfet_region_labels.to(device) == 0)
    if not cutoff_mask.any():
        return torch.tensor(0.0, device=device)

    # gm from SS head (per-MOSFET, denormalize to log10 scale)
    log10_gm_pred = ss_gm_pred[cutoff_mask] * ss_gm_std + ss_gm_mean

    # Id from current head (denormalize to real amps)
    Id_real = torch.pow(10, pred_currents[drain_term_idx[cutoff_mask]] * current_std + current_mean).clamp(min=1e-15)

    # Build per-device n*Vt
    is_nmos = mosfet_info[:, 6][cutoff_mask].bool()
    n_vt = torch.where(is_nmos,
                       torch.tensor(n_nmos * Vt, device=device),
                       torch.tensor(n_pmos * Vt, device=device))

    # gm_physics = Id / (n * Vt)
    gm_physics = (Id_real / n_vt).clamp(min=1e-15)
    log10_gm_physics = torch.log10(gm_physics)

    return F.mse_loss(log10_gm_pred, log10_gm_physics)


def compute_triode_physics_loss(
    ss_gm_pred: torch.Tensor,
    ss_gds_pred: torch.Tensor,
    pred_currents: torch.Tensor,
    full_voltage_pred: torch.Tensor,
    mosfet_info: torch.Tensor,
    node_mosfet_vth: torch.Tensor,
    mosfet_region_labels: torch.Tensor,
    ptr: torch.Tensor,
    vdc_mean: float = 0.0,
    vdc_std: float = 1.0,
    current_mean: float = 0.0,
    current_std: float = 1.0,
    ss_gm_mean: float = 0.0,
    ss_gm_std: float = 1.0,
    ss_gds_mean: float = 0.0,
    ss_gds_std: float = 1.0,
    min_vov: float = 0.0,
    eq1_enabled: bool = True,
    eq1_weight: float = 1.0,
    eq2_enabled: bool = True,
    eq2_weight: float = 1.0,
    eq3_enabled: bool = True,
    eq3_weight: float = 1.0,
    eq2_max_vds_vov: float = 1.0,
    mosfet_gt_vov: torch.Tensor = None,
    mosfet_ptr: torch.Tensor = None,
    node_voltage_targets: torch.Tensor = None,
) -> tuple:
    """
    Triode physics regularizer losses (training only).

    Three equations for MOSFETs in triode region (region_label == 1):
      Eq1: gm = I_DS / (Vov - Vds/2)
      Eq2: gds = (I_DS/Vds) * (Vov - Vds) / (Vov - Vds/2)
      Eq3: gm * (Vov - Vds) = gds * Vds   [self-consistency]

    All comparisons are in log10 space (MSE of log10 values).

    Args:
        ss_gm_pred: [num_nodes] SS head gm prediction (z-score normalized log10)
        ss_gds_pred: [num_nodes] SS head gds prediction (z-score normalized log10)
        pred_currents: [num_nodes] current head predictions (z-score normalized log10)
        full_voltage_pred: [num_nodes] voltage predictions (z-score normalized)
        mosfet_info: [num_mosfets, 7] cols: gate_t, drain_t, source_t, gate_n, drain_n, source_n, is_nmos
        node_mosfet_vth: [num_nodes] SPICE |Vth| at drain terminal positions
        mosfet_region_labels: [num_mosfets] 0=cutoff, 1=triode, 2=saturation
        ptr: [batch_size+1] node boundaries
        min_vov: minimum overdrive voltage filter (V)
        eq1/2/3_enabled: enable each equation
        eq1/2/3_weight: relative weight for each equation
        mosfet_gt_vov: [num_mosfets] pre-computed ground truth Vov (from SPICE gm/I_D).
                       If provided, used for min_vov filtering instead of predicted Vov.

    Returns:
        (combined_loss, eq1_loss, eq2_loss, eq3_loss) as tensors
    """
    device = ss_gm_pred.device
    zero = torch.tensor(0.0, device=device)

    if (mosfet_info is None or len(mosfet_info) == 0 or
        node_mosfet_vth is None or mosfet_region_labels is None):
        return zero, zero, zero, zero

    # Need at least one equation enabled
    if not (eq1_enabled or eq2_enabled or eq3_enabled):
        return zero, zero, zero, zero

    # Eq1/Eq2 need current predictions
    needs_current = (eq1_enabled or eq2_enabled) and pred_currents is not None
    # Eq3 only needs gm+gds (no current), but all need voltages for Vov/Vds
    if full_voltage_pred is None:
        return zero, zero, zero, zero
    if ss_gm_pred is None and (eq1_enabled or eq3_enabled):
        return zero, zero, zero, zero
    if ss_gds_pred is None and (eq2_enabled or eq3_enabled):
        return zero, zero, zero, zero

    num_mosfets = len(mosfet_info)
    num_graphs = len(ptr) - 1

    # Build global mosfet indices with batch offsets
    mosfet_graph_idx = get_device_graph_idx(num_mosfets, num_graphs, mosfet_ptr, device)
    node_offsets = ptr[mosfet_graph_idx]

    # Terminal indices (for I_D, gm, gds)
    drain_term_idx = mosfet_info[:, 1] + node_offsets
    # Net node indices (for voltages)
    gate_net_idx = mosfet_info[:, 3] + node_offsets
    drain_net_idx = mosfet_info[:, 4] + node_offsets
    source_net_idx = mosfet_info[:, 5] + node_offsets

    # Triode mask + Vth validity
    triode_mask = (mosfet_region_labels.to(device) == 1)
    vth_at_drain = node_mosfet_vth[drain_term_idx]
    valid_mask = triode_mask & (vth_at_drain.abs() > 1e-6)

    # Apply min_vov filter using GT Vov (stable) or predicted Vov (fallback)
    if mosfet_gt_vov is not None and min_vov > 0:
        gt_vov = mosfet_gt_vov.to(device)
        valid_mask = valid_mask & (gt_vov >= min_vov)

    if not valid_mask.any():
        return zero, zero, zero, zero

    # Voltages for physics equations: use GT when available (accurate), else predicted (detached)
    if node_voltage_targets is not None:
        V_gate = node_voltage_targets[gate_net_idx[valid_mask]]
        V_drain = node_voltage_targets[drain_net_idx[valid_mask]]
        V_source = node_voltage_targets[source_net_idx[valid_mask]]
    else:
        V_gate = full_voltage_pred[gate_net_idx[valid_mask]].detach() * vdc_std + vdc_mean
        V_drain = full_voltage_pred[drain_net_idx[valid_mask]].detach() * vdc_std + vdc_mean
        V_source = full_voltage_pred[source_net_idx[valid_mask]].detach() * vdc_std + vdc_mean

    # Compute Vgs, Vds, Vov with correct NMOS/PMOS polarity
    Vgs_raw = V_gate - V_source
    Vds_raw = V_drain - V_source
    Vth = vth_at_drain[valid_mask]
    is_nmos = mosfet_info[:, 6][valid_mask].bool()

    Vov = torch.where(is_nmos, Vgs_raw - Vth, -Vgs_raw - Vth)
    Vds = torch.where(is_nmos, Vds_raw, -Vds_raw)

    # Dynamic filter: predicted Vov > min_vov, Vds > 0, and Vov > Vds (true triode condition)
    vov_threshold = min_vov if min_vov > 0 else 1e-3
    vov_mask = (Vov > vov_threshold) & (Vds > 1e-6) & (Vov > Vds)
    if not vov_mask.any():
        return zero, zero, zero, zero

    # Apply filter
    Vov_f = Vov[vov_mask].clamp(min=1e-6)
    Vds_f = Vds[vov_mask].clamp(min=1e-6)
    drain_term_f = drain_term_idx[valid_mask][vov_mask]

    # Denormalize gm/gds predictions (per-MOSFET, log10 scale)
    log10_gm_pred_f = ss_gm_pred[valid_mask][vov_mask] * ss_gm_std + ss_gm_mean
    log10_gds_pred_f = ss_gds_pred[valid_mask][vov_mask] * ss_gds_std + ss_gds_mean

    # Denormalize current (real amps, still per-node)
    if needs_current:
        Id_real_f = torch.pow(10, pred_currents[drain_term_f] * current_std + current_mean).clamp(min=1e-12)

    # --- Eq1: gm = I_DS / (Vov - Vds/2) ---
    eq1_loss = zero
    if eq1_enabled and needs_current:
        denom1 = (Vov_f - Vds_f / 2.0).clamp(min=1e-6)
        gm_physics = (Id_real_f / denom1).clamp(min=1e-12)
        log10_gm_physics = torch.log10(gm_physics)
        eq1_loss = F.mse_loss(log10_gm_pred_f, log10_gm_physics)

    # --- Eq2: gds = (I_DS/Vds) * (Vov - Vds) / (Vov - Vds/2) ---
    # Near the saturation boundary (Vds → Vov), (Vov-Vds) → 0 and the equation
    # becomes numerically unstable. Filter by max Vds/Vov ratio.
    eq2_loss = zero
    if eq2_enabled and needs_current:
        eq2_mask = torch.ones_like(Vov_f, dtype=torch.bool)
        if eq2_max_vds_vov < 1.0:
            eq2_mask = (Vds_f / Vov_f) < eq2_max_vds_vov
        if eq2_mask.any():
            numer2 = (Vov_f[eq2_mask] - Vds_f[eq2_mask]).clamp(min=1e-6)
            denom2 = (Vov_f[eq2_mask] - Vds_f[eq2_mask] / 2.0).clamp(min=1e-6)
            gds_physics = (Id_real_f[eq2_mask] / Vds_f[eq2_mask] * numer2 / denom2).clamp(min=1e-12)
            log10_gds_physics = torch.log10(gds_physics)
            eq2_loss = F.mse_loss(log10_gds_pred_f[eq2_mask], log10_gds_physics)

    # --- Eq3: gm * (Vov - Vds) = gds * Vds  [self-consistency] ---
    # In log space: log10(gm) + log10(Vov - Vds) = log10(gds) + log10(Vds)
    eq3_loss = zero
    if eq3_enabled:
        vov_minus_vds = (Vov_f - Vds_f).clamp(min=1e-6)
        lhs = log10_gm_pred_f + torch.log10(vov_minus_vds)
        rhs = log10_gds_pred_f + torch.log10(Vds_f)
        eq3_loss = F.mse_loss(lhs, rhs)

    combined = eq1_weight * eq1_loss + eq2_weight * eq2_loss + eq3_weight * eq3_loss
    return combined, eq1_loss, eq2_loss, eq3_loss
