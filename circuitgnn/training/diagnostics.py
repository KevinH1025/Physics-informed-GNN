"""Training diagnostics for train_v3.

Hosts the per-metric evaluation helpers that used to be closures defined
inside train_v3's epoch loop (``_eval_ss``, ``_eval_gm_id``, ``_eval_iv_id``,
``_eval_region``, ``_eval_ac``, ``_eval_dc_gain``) so the every-50-epochs
detail report and the end-of-training report share one definition, plus the
two report printers themselves.

Every printed string in this module is a frozen surface: 15+ figure scripts
regex-parse training.log. Do not reword, respace, or change precision.

Note on the near-duplicates that are deliberately kept apart: the mid-training
region evaluation (``eval_region``) handles both the CORAL probability head and
the ordinal head and reports per-class recall, while the end-of-training one
(``collect_region_predictions``) only handles the ordinal head and reports
counts. They differed in the original code and still do.
"""

import numpy as np
import torch

from circuitgnn.training.loops import validate
from circuitgnn.training.losses import compute_kcl_loss, compute_kcl_per_net_debug


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers (formerly closures inside train_v3.main)
# ─────────────────────────────────────────────────────────────────────────────

def eval_ss(model, loader, device, ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std):
    """SS (gm/gds) log-MAE and relative errors over a loader."""
    _gm_e, _gds_e, _gm_rel, _gds_rel = [], [], [], []
    _use_iv = getattr(model, '_needs_autograd', False)
    with torch.no_grad():
        for _b in loader:
            if hasattr(_b, 'to'):
                _b = _b.to(device)
            if _use_iv:
                with torch.enable_grad():
                    _out = model(_b)
            else:
                _out = model(_b)
            _gm_p, _gds_p = _out.get('mosfet_gm_pred'), _out.get('mosfet_gds_pred')
            if _gm_p is None: break
            _mosfet_gm = getattr(_b, 'mosfet_gm', None)
            _mosfet_gds = getattr(_b, 'mosfet_gds', None)
            if _mosfet_gm is not None:
                _valid = _mosfet_gm > 1e-12
                if _valid.any():
                    gm_pred_log = _gm_p[_valid] * ss_gm_std + ss_gm_mean
                    gm_tgt_log = torch.log10(_mosfet_gm[_valid].to(_gm_p.device))
                    gds_pred_log = _gds_p[_valid] * ss_gds_std + ss_gds_mean
                    gds_tgt_log = torch.log10(_mosfet_gds[_valid].to(_gds_p.device).clamp(min=1e-20))
                    _gm_e.extend((gm_pred_log - gm_tgt_log).abs().cpu().tolist())
                    _gds_e.extend((gds_pred_log - gds_tgt_log).abs().cpu().tolist())
                    _gm_rel.extend(((10**gm_pred_log - 10**gm_tgt_log).abs() / (10**gm_tgt_log).clamp(min=1e-15) * 100).cpu().tolist())
                    _gds_rel.extend(((10**gds_pred_log - 10**gds_tgt_log).abs() / (10**gds_tgt_log).clamp(min=1e-15) * 100).cpu().tolist())
    return (np.array(_gm_e) if _gm_e else None, np.array(_gds_e) if _gds_e else None,
            np.array(_gm_rel) if _gm_rel else None, np.array(_gds_rel) if _gds_rel else None)


def eval_gm_id(model, loader, device, gm_id_mean, gm_id_std, current_mean, current_std):
    """gm/Id log error and relative error over a loader."""
    _errs, _rel = [], []
    with torch.no_grad():
        for _b in loader:
            if hasattr(_b, 'to'):
                _b = _b.to(device)
            _out = model(_b)
            _gm_id_p = _out.get('mosfet_gm_id_pred')
            if _gm_id_p is None: break
            _mosfet_gm = getattr(_b, 'mosfet_gm', None)
            if _mosfet_gm is None: break
            _valid = _mosfet_gm.to(device) > 1e-12
            if _valid.any():
                _pred_log = _gm_id_p[_valid] * gm_id_std + gm_id_mean
                _mi = _b.mosfet_info.long()
                _nm = _mi.shape[0]; _ng = _b.ptr.shape[0] - 1
                _mp = getattr(_b, 'mosfet_ptr', None)
                if _mp is not None:
                    _mg = torch.bucketize(torch.arange(_nm, device=device), _mp[1:].to(device), right=True)
                else:
                    _mg = torch.arange(_nm, device=device) // (_nm // _ng)
                _offs = _b.ptr.to(device)[_mg]
                _gt_log_gm = torch.log10(_mosfet_gm.to(device)[_valid].clamp(min=1e-20))
                _gt_log_id = _b.node_current_targets.to(device)[_mi[:, 1] + _offs] * current_std + current_mean
                _gt = _gt_log_gm - _gt_log_id[_valid]
                _errs.extend((_pred_log - _gt).abs().cpu().tolist())
                _rel.extend(((10**_pred_log - 10**_gt).abs() / (10**_gt).clamp(min=1e-15) * 100).cpu().tolist())
    return np.array(_errs) if _errs else None, np.array(_rel) if _rel else None


def eval_iv_id(model, loader, device, current_mean, current_std):
    """IV-model drain current log error over a loader."""
    _id_errs = []
    with torch.no_grad():
        for _b in loader:
            if hasattr(_b, 'to'):
                _b = _b.to(device)
            with torch.enable_grad():
                _out = model(_b)
            _iv_log_id = _out.get('mosfet_iv_log_abs_id')
            if _iv_log_id is None: break
            _mi = _b.mosfet_info
            _ptr = _b.ptr
            _mptr = getattr(_b, 'mosfet_ptr', None)
            if _mptr is not None:
                _offsets = _ptr[:-1].repeat_interleave(_mptr[1:] - _mptr[:-1])
            else:
                _n_graphs = len(_ptr) - 1
                _offsets = _ptr[:-1].repeat_interleave(len(_mi) // _n_graphs)
            _drain_idx = _mi[:, 1].long() + _offsets
            _gt_zscore = _b.node_current_targets[_drain_idx]
            _gt_log_id = _gt_zscore * current_std + current_mean
            _valid = torch.isfinite(_gt_log_id) & (_gt_log_id > -15)
            if _valid.any():
                _id_errs.extend((_iv_log_id[_valid] - _gt_log_id[_valid].to(_iv_log_id.device)).abs().cpu().tolist())
    return np.array(_id_errs) if _id_errs else None


def eval_region(model, loader, device):
    """Region classification accuracy over a loader: (overall, [cutoff, triode, sat])."""
    _preds, _labels = [], []
    _use_iv = getattr(model, '_needs_autograd', False)
    with torch.no_grad():
        for _b in loader:
            if hasattr(_b, 'to'):
                _b = _b.to(device)
            if _use_iv:
                with torch.enable_grad():
                    _out = model(_b)
            else:
                _out = model(_b)
            _rpred = _out.get('mosfet_region_pred')
            _rprobs = _out.get('mosfet_region_probs')
            if _rpred is None and _rprobs is None: break
            if _rprobs is not None:
                # CORAL: use region probs argmax
                _rlabels = getattr(_b, 'mosfet_region_labels', None)
                if _rlabels is not None:
                    _preds.append(_rprobs.argmax(dim=-1).cpu())
                    _labels.append(_rlabels.cpu())
            elif _rpred is not None:
                _m = _b.mosfet_drain_mask & (_b.node_region_labels >= 0)
                if _m.any():
                    # Ordinal: <0.5 → cutoff(0), 0.5-1.5 → triode(1), >1.5 → sat(2)
                    _preds.append(_rpred[_m].round().clamp(0, 2).long().cpu())
                    _labels.append(_b.node_region_labels[_m].cpu())
    if not _preds: return None
    _p = torch.cat(_preds); _l = torch.cat(_labels)
    _acc = 100 * (_p == _l).float().mean().item()
    _accs = []
    for _c in range(3):
        _cm = _l == _c
        _accs.append(100 * ((_p == _c) & _cm).sum().item() / _cm.sum().item() if _cm.any() else 0)
    return _acc, _accs  # overall, [cutoff, triode, sat]


def eval_ac(model, loader, device, ac_components, ac_mean, ac_std):
    """Denormalized AC errors over a loader.

    Returns ``(errors, rel_errors)``, both dicts keyed by component holding raw
    Python lists. ``rel_errors`` is only filled for 'ugbw' (relative % error in
    linear Hz space), matching the original code in both report sites.
    """
    _errs = {c: [] for c in ac_components}
    _rel = {c: [] for c in ac_components}
    _use_iv = getattr(model, '_needs_autograd', False)
    with torch.no_grad():
        for _b in loader:
            if hasattr(_b, 'to'):
                _b = _b.to(device)
            if _use_iv:
                with torch.enable_grad():
                    _out = model(_b)
            else:
                _out = model(_b)
            _ac_p = _out.get('ac_pred')
            if _ac_p is None: break
            _v = _b.ac_valid if hasattr(_b, 'ac_valid') else None
            if _v is None or not _v.any(): continue
            _pd = _ac_p[_v] * ac_std.to(device) + ac_mean.to(device)
            for _i, _c in enumerate(ac_components):
                if _c == 'ugbw':
                    _t = torch.log10(_b.ac_ugbw[_v].clamp(min=1.0).to(device))
                    _rel[_c].extend(((10**_pd[:, _i] - 10**_t).abs() / (10**_t).clamp(min=1e-15) * 100).cpu().tolist())
                elif _c == 'pm':
                    _t = _b.ac_pm[_v].to(device)
                elif _c == 'am':
                    _t = _b.ac_am[_v].to(device)
                _errs[_c].extend((_pd[:, _i] - _t).abs().cpu().tolist())
    return _errs, _rel


def eval_dc_gain(model, loader, device, dc_gain_mean, dc_gain_std):
    """Denormalized DC gain absolute errors (dB) over a loader."""
    _errs = []
    _use_iv = getattr(model, '_needs_autograd', False)
    with torch.no_grad():
        for _b in loader:
            if hasattr(_b, 'to'):
                _b = _b.to(device)
            if _use_iv:
                with torch.enable_grad():
                    _out = model(_b)
            else:
                _out = model(_b)
            _dc_p = _out.get('dc_gain_pred')
            if _dc_p is None: break
            _dc_t = _b.ac_dc_gain.to(device)
            # Denormalize
            _pred_db = _dc_p * dc_gain_std + dc_gain_mean
            _errs.extend((_pred_db - _dc_t).abs().cpu().tolist())
    return np.array(_errs) if _errs else None


def collect_dc_gain_predictions(model, loader, device, dc_gain_mean, dc_gain_std):
    """Denormalized DC gain predictions and targets (dB) for the validity classifier."""
    all_preds_db = []
    all_targets_db = []
    _use_iv = getattr(model, '_needs_autograd', False)
    with torch.no_grad():
        for batch in loader:
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            if _use_iv:
                with torch.enable_grad():
                    out_dict = model(batch)
            else:
                out_dict = model(batch)
            dc_pred = out_dict.get('dc_gain_pred')
            if dc_pred is None:
                break
            pred_db = dc_pred * dc_gain_std + dc_gain_mean
            all_preds_db.extend(pred_db.cpu().tolist())
            all_targets_db.extend(batch.ac_dc_gain.cpu().tolist())
    return all_preds_db, all_targets_db


def collect_region_predictions(model, loader, device):
    """Ordinal region predictions and labels over a loader (end-of-training form)."""
    all_preds = []
    all_labels = []
    _use_iv = getattr(model, '_needs_autograd', False)
    with torch.no_grad():
        for batch in loader:
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            if _use_iv:
                with torch.enable_grad():
                    out_dict = model(batch)
            else:
                out_dict = model(batch)
            rpred = out_dict.get('mosfet_region_pred')
            if rpred is None:
                break
            mask = batch.mosfet_drain_mask & (batch.node_region_labels >= 0)
            if mask.any():
                preds = rpred[mask].round().clamp(0, 2).long()
                labels = batch.node_region_labels[mask]
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
    return all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Mid-training detail report (every `detail_interval` epochs)
# ─────────────────────────────────────────────────────────────────────────────

def print_detail_report(model, train_loader, val_loader, args, epoch, lr,
                        cfg, metrics, uncertainty_weights):
    """Run validate() over the train loader and print the aligned detail table.

    ``cfg`` carries the loss weights, normalization stats and physics config
    that are in effect for this epoch; ``metrics`` carries the val-side results
    of this epoch's validate() plus the two train_epoch values the table shows
    directly (``train_gm_physics_loss``, ``train_vth_loss``).
    """
    # ── configuration in effect this epoch
    predict_currents = cfg['predict_currents']
    current_weight = cfg['current_weight']
    kcl_weight = cfg['kcl_weight']
    loss_type = cfg['loss_type']
    huber_delta = cfg['huber_delta']
    constraint_weight = cfg['constraint_weight']
    stage2_nodes = cfg['stage2_nodes']
    stage2_weight = cfg['stage2_weight']
    node_weights = cfg['node_weights']
    use_terminal_voltage_loss = cfg['use_terminal_voltage_loss']
    gm_physics_loss_weight = cfg['gm_physics_loss_weight']
    gm_physics_min_vov = cfg['gm_physics_min_vov']
    gm_physics_use_clm = cfg['gm_physics_use_clm']
    gm_physics_use_smaxt = cfg['gm_physics_use_smaxt']
    has_ss = cfg['has_ss']
    ss_gm_mean = cfg['ss_gm_mean']
    ss_gm_std = cfg['ss_gm_std']
    ss_gds_mean = cfg['ss_gds_mean']
    ss_gds_std = cfg['ss_gds_std']
    ac_loss_weight = cfg['ac_loss_weight']
    ac_mean = cfg['ac_mean']
    ac_std = cfg['ac_std']
    ac_components = cfg['ac_components']
    ss_gm_loss_weight = cfg['ss_gm_loss_weight']
    ss_gds_loss_weight = cfg['ss_gds_loss_weight']
    ss_region_stats = cfg['ss_region_stats']
    triode_physics_loss_weight = cfg['triode_physics_loss_weight']
    triode_physics_config = cfg['triode_physics_config']
    cutoff_physics_loss_weight = cfg['cutoff_physics_loss_weight']
    cutoff_physics_n_nmos = cfg['cutoff_physics_n_nmos']
    cutoff_physics_n_pmos = cfg['cutoff_physics_n_pmos']
    region_loss_weight = cfg['region_loss_weight']
    amp_dtype = cfg['amp_dtype']
    device_consistency_weight = cfg['device_consistency_weight']
    vov_loss_weight = cfg['vov_loss_weight']
    vth_loss_weight = cfg['vth_loss_weight']
    dc_gain_loss_weight = cfg['dc_gain_loss_weight']
    dc_gain_mean = cfg['dc_gain_mean']
    dc_gain_std = cfg['dc_gain_std']
    gm_id_mean = cfg['gm_id_mean']
    gm_id_std = cfg['gm_id_std']
    mirror_pair_indices = cfg['mirror_pair_indices']
    mirror_pair_ratios = cfg['mirror_pair_ratios']
    mirror_pair_names = cfg['mirror_pair_names']
    vdc_mean = cfg['vdc_mean']
    vdc_std = cfg['vdc_std']
    current_mean = cfg['current_mean']
    current_std = cfg['current_std']
    kcl_mode = cfg['kcl_mode']
    kcl_mask_unsupervised = cfg['kcl_mask_unsupervised']

    # ── this epoch's metrics
    best_val_loss = metrics['best_val_loss']
    best_epoch = metrics['best_epoch']
    composite_score = metrics['composite_score']
    best_composite_score = metrics['best_composite_score']
    max_grad_norm = metrics['max_grad_norm']
    avg_grad_norm = metrics['avg_grad_norm']
    gm_physics_loss = metrics['train_gm_physics_loss']
    train_vth_loss = metrics['train_vth_loss']
    val_loss = metrics['val_loss']
    val_mae_mv = metrics['val_mae_mv']
    val_v_loss = metrics['val_v_loss']
    val_c_loss = metrics['val_c_loss']
    val_c_mae = metrics['val_c_mae']
    acc80 = metrics['acc80']
    acc50 = metrics['acc50']
    acc20 = metrics['acc20']
    acc10 = metrics['acc10']
    current_acc50 = metrics['current_acc50']
    current_acc20 = metrics['current_acc20']
    current_acc10 = metrics['current_acc10']
    current_acc5 = metrics['current_acc5']
    val_kcl_loss = metrics['val_kcl_loss']
    val_dp_loss = metrics['val_dp_loss']
    val_mirror_loss = metrics['val_mirror_loss']
    val_os_loss = metrics['val_os_loss']
    val_lm_loss = metrics['val_lm_loss']
    val_gm_physics_loss = metrics['val_gm_physics_loss']
    val_ac_loss = metrics['val_ac_loss']
    val_ss_gm_loss = metrics['val_ss_gm_loss']
    val_ss_gds_loss = metrics['val_ss_gds_loss']
    val_triode_physics_loss = metrics['val_triode_physics_loss']
    val_triode_eq1_loss = metrics['val_triode_eq1_loss']
    val_triode_eq2_loss = metrics['val_triode_eq2_loss']
    val_triode_eq3_loss = metrics['val_triode_eq3_loss']
    val_cutoff_physics_loss = metrics['val_cutoff_physics_loss']
    val_region_loss = metrics['val_region_loss']
    val_rel_metrics = metrics['val_rel_metrics']
    val_vth_loss = metrics['val_vth_loss']
    val_ac_comp = metrics['val_ac_comp']
    val_dc_gain_loss = metrics['val_dc_gain_loss']
    val_hc_mirror_loss = metrics['val_hc_mirror_loss']
    val_mirror_pair_detail = metrics['val_mirror_pair_detail']

    tr_loss, tr_mae_mv, tr_v_loss, tr_c_loss, tr_c_mae, tr_acc80, tr_acc50, tr_acc20, tr_acc10, tr_current_acc50, tr_current_acc20, tr_current_acc10, tr_current_acc5, tr_kcl_loss, tr_dp_loss, tr_mirror_loss, tr_os_loss, tr_lm_loss, tr_gm_physics_loss, tr_ac_loss, tr_ss_gm_loss, tr_ss_gds_loss, tr_triode_physics_loss, tr_triode_eq1, tr_triode_eq2, tr_triode_eq3, tr_cutoff_physics_loss, tr_region_loss, _tr_rel_metrics, _tr_vov_loss, _tr_vth_loss, tr_ac_comp_detail, tr_dc_gain_loss, _tr_gm_id_loss, _tr_gm_id_aux_loss, tr_hc_mirror, tr_mirror_pair_detail = validate(
        model, train_loader, args.device, vdc_mean, vdc_std, current_mean, current_std,
        predict_currents=predict_currents, current_weight=current_weight,
        voltage_weight=getattr(args, 'voltage_weight', 1.0),
        kcl_weight=kcl_weight,
        loss_type=loss_type, huber_delta=huber_delta, constraint_weight=constraint_weight,
        stage2_nodes=stage2_nodes, stage2_weight=stage2_weight, node_weights=node_weights,
        use_terminal_voltage_loss=use_terminal_voltage_loss,
        gm_physics_loss_weight=gm_physics_loss_weight, gm_physics_min_vov=gm_physics_min_vov, gm_physics_use_clm=gm_physics_use_clm, gm_physics_use_smaxt=gm_physics_use_smaxt,
        ss_gm_mean=ss_gm_mean if has_ss else 0.0, ss_gm_std=ss_gm_std if has_ss else 1.0,
        ss_gds_mean=ss_gds_mean if has_ss else 0.0, ss_gds_std=ss_gds_std if has_ss else 1.0,
        ac_loss_weight=ac_loss_weight, ac_mean=ac_mean, ac_std=ac_std, ac_components=ac_components,
        ss_gm_loss_weight=ss_gm_loss_weight, ss_gds_loss_weight=ss_gds_loss_weight,
        ss_huber_delta=getattr(args, 'ss_huber_delta', 0.0),
        ss_region_stats=ss_region_stats,
        triode_physics_loss_weight=triode_physics_loss_weight, triode_physics_config=triode_physics_config,
        cutoff_physics_loss_weight=cutoff_physics_loss_weight, cutoff_physics_n_nmos=cutoff_physics_n_nmos, cutoff_physics_n_pmos=cutoff_physics_n_pmos,
        region_loss_weight=region_loss_weight,
        amp_dtype=amp_dtype,
        device_consistency_weight=device_consistency_weight,
        vov_loss_weight=vov_loss_weight,
        vth_loss_weight=vth_loss_weight,
        iv_id_loss_weight=getattr(args, 'iv_id_loss_weight', 0.0),
        dc_gain_loss_weight=dc_gain_loss_weight,
        dc_gain_mean=dc_gain_mean,
        dc_gain_std=dc_gain_std,
        gm_id_consistency_weight=getattr(args, 'gm_id_consistency_weight', 0.0),
        gm_id_aux_weight=getattr(args, 'gm_id_aux_weight', 0.0),
        gm_id_mean=gm_id_mean,
        gm_id_std=gm_id_std,
        mirror_pair_indices=mirror_pair_indices,
        mirror_pair_ratios=mirror_pair_ratios,
        mirror_pair_names=mirror_pair_names,
        ac_pred_filter=getattr(args, 'ac_pred_filter', False),
    )
    # Build rows dynamically: (label, train_value, val_value)
    rows = []
    rows.append(("Voltage", f"Loss={tr_v_loss:.4f}  MAE={tr_mae_mv:.1f}mV", f"Loss={val_v_loss:.4f}  MAE={val_mae_mv:.1f}mV"))
    rows.append(("  Acc", f"@80={tr_acc80:.0f}% @50={tr_acc50:.0f}% @20={tr_acc20:.0f}% @10={tr_acc10:.0f}%", f"@80={acc80:.0f}% @50={acc50:.0f}% @20={acc20:.0f}% @10={acc10:.0f}%"))
    # Relative voltage accuracy @1%
    if _tr_rel_metrics and val_rel_metrics:
        tr_vr = _tr_rel_metrics.get('v_rel_acc', {})
        vl_vr = val_rel_metrics.get('v_rel_acc', {})
        if tr_vr and vl_vr:
            rows.append(("  RelAcc", f"@1%={tr_vr[1]:.1f}% @5%={tr_vr[5]:.1f}% @10%={tr_vr[10]:.1f}%", f"@1%={vl_vr[1]:.1f}% @5%={vl_vr[5]:.1f}% @10%={vl_vr[10]:.1f}%"))
    if predict_currents:
        rows.append(("Current", f"Loss={tr_c_loss:.5f}  MAE={tr_c_mae:.1f}uA", f"Loss={val_c_loss:.5f}  MAE={val_c_mae:.1f}uA"))
        rows.append(("  Acc", f"@50%={tr_current_acc50:.0f}% @20%={tr_current_acc20:.0f}% @10%={tr_current_acc10:.0f}% @5%={tr_current_acc5:.0f}%", f"@50%={current_acc50:.0f}% @20%={current_acc20:.0f}% @10%={current_acc10:.0f}% @5%={current_acc5:.0f}%"))
        # Relative current accuracy @1%
        if _tr_rel_metrics and val_rel_metrics:
            tr_ir = _tr_rel_metrics.get('i_rel_acc', {})
            vl_ir = val_rel_metrics.get('i_rel_acc', {})
            if tr_ir and vl_ir:
                rows.append(("  RelAcc", f"@1%={tr_ir[1]:.1f}% @5%={tr_ir[5]:.1f}% @10%={tr_ir[10]:.1f}%", f"@1%={vl_ir[1]:.1f}% @5%={vl_ir[5]:.1f}% @10%={vl_ir[10]:.1f}%"))
    if predict_currents:
        rows.append(("KCL", f"{tr_kcl_loss:.2e}", f"{val_kcl_loss:.2e}"))
        # KCL diagnostics: per-component breakdown, GT floor, drop stats
        try:
            _use_iv = getattr(model, '_needs_autograd', False)
            with torch.no_grad():
                _vb = next(iter(val_loader))
                if hasattr(_vb, 'to'):
                    _vb = _vb.to(args.device)
                if _use_iv:
                    with torch.enable_grad():
                        _out = model(_vb)
                else:
                    _out = model(_vb)
                _kcl_args = dict(
                    node_currents=_out['node_currents'],
                    edge_index=_vb.edge_index,
                    num_terminals=_vb.num_terminals,
                    train_mask=_vb.train_mask,
                    ptr=_vb.ptr,
                    terminal_current_sign=getattr(_vb, 'terminal_current_sign', None),
                    current_mean=current_mean,
                    current_std=current_std,
                    gt_currents=_vb.node_current_targets if hasattr(_vb, 'node_current_targets') else None,
                    kcl_include_mask=(getattr(_vb, 'kcl_include_mask', None) & _vb.has_current_mask) if (getattr(_vb, 'kcl_include_mask', None) is not None and kcl_mask_unsupervised) else getattr(_vb, 'kcl_include_mask', None),
                    return_stats=True,
                    kcl_mode=kcl_mode,
                )
                _, _, _ks = compute_kcl_loss(**_kcl_args)
            # Per-net KCL relative violations (averaged across batch)
            if hasattr(_vb, 'node_names') and _vb.node_names is not None:
                _n_graphs = len(_vb.ptr) - 1
                _pred_accum = {}
                _kcl_inc_mask = getattr(_vb, 'kcl_include_mask', None)
                if _kcl_inc_mask is not None and kcl_mask_unsupervised:
                    _kcl_inc_mask = _kcl_inc_mask & _vb.has_current_mask
                _common_args = dict(
                    edge_index=_vb.edge_index,
                    num_terminals=_vb.num_terminals,
                    train_mask=_vb.train_mask,
                    ptr=_vb.ptr,
                    terminal_current_sign=getattr(_vb, 'terminal_current_sign', None),
                    current_mean=current_mean,
                    current_std=current_std,
                    kcl_include_mask=_kcl_inc_mask,
                    node_names=_vb.node_names,
                )
                for _gi in range(_n_graphs):
                    _pv = compute_kcl_per_net_debug(node_currents=_out['node_currents'], graph_idx=_gi, **_common_args)
                    for _name, _val in _pv.items():
                        _pred_accum.setdefault(_name, []).append(_val)
                _pred_mean = {k: sum(v)/len(v) for k, v in _pred_accum.items()}
                _pred_str = " ".join(f"{k}:{v*100:.1f}%" for k, v in sorted(_pred_mean.items(), key=lambda x: -x[1]))
                rows.append(("  KCL/net", "", _pred_str))
        except Exception as _e:
            rows.append(("  detail", f"err: {_e}", ""))
    if ss_gm_loss_weight > 0 or ss_gds_loss_weight > 0:
        rows.append(("SS/gm", f"{tr_ss_gm_loss:.2e}", f"{val_ss_gm_loss:.2e}"))
        rows.append(("SS/gds", f"{tr_ss_gds_loss:.2e}", f"{val_ss_gds_loss:.2e}"))
        # SS accuracy on train and val
        _tr_gm, _tr_gds, _tr_gm_r, _tr_gds_r = eval_ss(model, train_loader, args.device, ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std)
        _vl_gm, _vl_gds, _vl_gm_r, _vl_gds_r = eval_ss(model, val_loader, args.device, ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std)
        if _vl_gm is not None:
            rows.append(("  gm", f"MAE={_tr_gm.mean():.3f}log  <10%={100*np.mean(_tr_gm_r<10):.0f}%  <20%={100*np.mean(_tr_gm_r<20):.0f}%  <50%={100*np.mean(_tr_gm_r<50):.0f}%", f"MAE={_vl_gm.mean():.3f}log  <10%={100*np.mean(_vl_gm_r<10):.0f}%  <20%={100*np.mean(_vl_gm_r<20):.0f}%  <50%={100*np.mean(_vl_gm_r<50):.0f}%"))
            rows.append(("  gds", f"MAE={_tr_gds.mean():.3f}log  <10%={100*np.mean(_tr_gds_r<10):.0f}%  <20%={100*np.mean(_tr_gds_r<20):.0f}%  <50%={100*np.mean(_tr_gds_r<50):.0f}%", f"MAE={_vl_gds.mean():.3f}log  <10%={100*np.mean(_vl_gds_r<10):.0f}%  <20%={100*np.mean(_vl_gds_r<20):.0f}%  <50%={100*np.mean(_vl_gds_r<50):.0f}%"))
    # gm/Id accuracy (when predict_gm_id mode or aux head)
    if getattr(args, 'gm_id_aux_weight', 0.0) > 0:
        _tr_gmid, _tr_gmid_r = eval_gm_id(model, train_loader, args.device, gm_id_mean, gm_id_std, current_mean, current_std)
        _vl_gmid, _vl_gmid_r = eval_gm_id(model, val_loader, args.device, gm_id_mean, gm_id_std, current_mean, current_std)
        if _vl_gmid is not None:
            rows.append(("  gm/Id", f"MAE={_tr_gmid.mean():.3f}log  <10%={100*np.mean(_tr_gmid_r<10):.0f}%  <20%={100*np.mean(_tr_gmid_r<20):.0f}%  <50%={100*np.mean(_tr_gmid_r<50):.0f}%",
                                   f"MAE={_vl_gmid.mean():.3f}log  <10%={100*np.mean(_vl_gmid_r<10):.0f}%  <20%={100*np.mean(_vl_gmid_r<20):.0f}%  <50%={100*np.mean(_vl_gmid_r<50):.0f}%"))

        # IV model drain current accuracy (if separate IV model active)
        if getattr(model, 'use_iv_model', False):
            _tr_id = eval_iv_id(model, train_loader, args.device, current_mean, current_std)
            _vl_id = eval_iv_id(model, val_loader, args.device, current_mean, current_std)
            if _vl_id is not None:
                rows.append(("  IV_ID", f"MAE={_tr_id.mean():.3f}log", f"MAE={_vl_id.mean():.3f}log"))
    if gm_physics_loss_weight > 0:
        rows.append(("GmPhy", f"{gm_physics_loss:.2e}", f"{val_gm_physics_loss:.2e}"))
    if triode_physics_loss_weight > 0:
        rows.append(("TriPhy", f"{tr_triode_physics_loss:.2e}", f"{val_triode_physics_loss:.2e}"))
        rows.append(("  eq1gm", f"{tr_triode_eq1:.2e}", f"{val_triode_eq1_loss:.2e}"))
        rows.append(("  eq2gds", f"{tr_triode_eq2:.2e}", f"{val_triode_eq2_loss:.2e}"))
        rows.append(("  eq3sc", f"{tr_triode_eq3:.2e}", f"{val_triode_eq3_loss:.2e}"))
    if cutoff_physics_loss_weight > 0:
        rows.append(("CutPhy", f"{tr_cutoff_physics_loss:.2e}", f"{val_cutoff_physics_loss:.2e}"))
    if region_loss_weight > 0:
        rows.append(("Region", f"{tr_region_loss:.2e}", f"{val_region_loss:.2e}"))
        # Quick region accuracy on train and val
        _tr_r = eval_region(model, train_loader, args.device)
        _vl_r = eval_region(model, val_loader, args.device)
        if _tr_r and _vl_r:
            rows.append(("", f"Acc={_tr_r[0]:.0f}%  cut={_tr_r[1][0]:.0f}% tri={_tr_r[1][1]:.0f}% sat={_tr_r[1][2]:.0f}%",
                            f"Acc={_vl_r[0]:.0f}%  cut={_vl_r[1][0]:.0f}% tri={_vl_r[1][1]:.0f}% sat={_vl_r[1][2]:.0f}%"))
    if ac_loss_weight > 0:
        # Show per-component AC losses
        ac_tr_parts = "  ".join(f"{c}={tr_ac_comp_detail.get(c, 0):.2e}" for c in ac_components)
        ac_vl_parts = "  ".join(f"{c}={val_ac_comp.get(c, 0):.2e}" for c in ac_components)
        rows.append(("AC", f"{tr_ac_loss:.2e}  ({ac_tr_parts})", f"{val_ac_loss:.2e}  ({ac_vl_parts})"))
        # AC accuracy on train and val
        _tr_ac_raw, _tr_rel_raw = eval_ac(model, train_loader, args.device, ac_components, ac_mean, ac_std)
        _tr_ac = {c: np.array(v) for c, v in _tr_ac_raw.items() if v}
        _tr_ugbw_rel = np.array(_tr_rel_raw['ugbw']) if _tr_rel_raw.get('ugbw') else None
        _vl_ac_raw, _vl_rel_raw = eval_ac(model, val_loader, args.device, ac_components, ac_mean, ac_std)
        _vl_ac = {c: np.array(v) for c, v in _vl_ac_raw.items() if v}
        _vl_ugbw_rel = np.array(_vl_rel_raw['ugbw']) if _vl_rel_raw.get('ugbw') else None
        for _c in ac_components:
            if _c in _vl_ac:
                _te, _ve = _tr_ac.get(_c, _vl_ac[_c]), _vl_ac[_c]
                if _c == 'ugbw':
                    _tr_ur = _tr_ugbw_rel if _tr_ugbw_rel is not None else _vl_ugbw_rel
                    rows.append(("  UGBW", f"MAE={_te.mean():.3f}log  <5%={100*np.mean(_tr_ur<5):.0f}%  <10%={100*np.mean(_tr_ur<10):.0f}%  <20%={100*np.mean(_tr_ur<20):.0f}%", f"MAE={_ve.mean():.3f}log  <5%={100*np.mean(_vl_ugbw_rel<5):.0f}%  <10%={100*np.mean(_vl_ugbw_rel<10):.0f}%  <20%={100*np.mean(_vl_ugbw_rel<20):.0f}%"))
                elif _c == 'pm':
                    rows.append(("  PM", f"MAE={_te.mean():.1f}d  <5={100*np.mean(_te<5):.0f}%  <10={100*np.mean(_te<10):.0f}%  <20={100*np.mean(_te<20):.0f}%", f"MAE={_ve.mean():.1f}d  <5={100*np.mean(_ve<5):.0f}%  <10={100*np.mean(_ve<10):.0f}%  <20={100*np.mean(_ve<20):.0f}%"))
                elif _c == 'am':
                    rows.append(("  AM", f"MAE={_te.mean():.1f}dB  <1={100*np.mean(_te<1):.0f}%  <3={100*np.mean(_te<3):.0f}%  <5={100*np.mean(_te<5):.0f}%", f"MAE={_ve.mean():.1f}dB  <1={100*np.mean(_ve<1):.0f}%  <3={100*np.mean(_ve<3):.0f}%  <5={100*np.mean(_ve<5):.0f}%"))
    if dc_gain_loss_weight > 0:
        rows.append(("DCGain", f"Loss={tr_dc_gain_loss:.2e}", f"Loss={val_dc_gain_loss:.2e}"))
        # Compute DC gain accuracy metrics
        _tr_dc = eval_dc_gain(model, train_loader, args.device, dc_gain_mean, dc_gain_std)
        _vl_dc = eval_dc_gain(model, val_loader, args.device, dc_gain_mean, dc_gain_std)
        if _tr_dc is not None and _vl_dc is not None:
            rows.append(("  Acc", f"MAE={_tr_dc.mean():.1f}dB  <3={100*np.mean(_tr_dc<3):.0f}%  <5={100*np.mean(_tr_dc<5):.0f}%  <10={100*np.mean(_tr_dc<10):.0f}%",
                                f"MAE={_vl_dc.mean():.1f}dB  <3={100*np.mean(_vl_dc<3):.0f}%  <5={100*np.mean(_vl_dc<5):.0f}%  <10={100*np.mean(_vl_dc<10):.0f}%"))
    if vth_loss_weight > 0:
        rows.append(("Vth", f"{train_vth_loss:.2e}", f"{val_vth_loss:.2e}"))
    if constraint_weight > 0:
        if tr_dp_loss > 0 or val_dp_loss > 0:
            rows.append(("DiffPr", f"{tr_dp_loss:.2e}", f"{val_dp_loss:.2e}"))
        if tr_mirror_loss > 0 or val_mirror_loss > 0:
            rows.append(("Mirror", f"{tr_mirror_loss:.2e}", f"{val_mirror_loss:.2e}"))
        if tr_os_loss > 0 or val_os_loss > 0:
            rows.append(("OutStg", f"{tr_os_loss:.2e}", f"{val_os_loss:.2e}"))
        if tr_lm_loss > 0 or val_lm_loss > 0:
            rows.append(("LamMir", f"{tr_lm_loss:.2e}", f"{val_lm_loss:.2e}"))
        if mirror_pair_indices is not None and len(mirror_pair_indices) > 0:
            rows.append(("HCMirr", f"{tr_hc_mirror:.2e}", f"{val_hc_mirror_loss:.2e}"))

    # Print with dynamic alignment
    label_w = max(len(r[0]) for r in rows)
    tr_w = max(len(r[1]) for r in rows)
    print(f"\nEpoch {epoch:3d} | LR={lr:.2e} | BestLoss={best_val_loss:.4f} | Score={composite_score:.1f} (best={best_composite_score:.1f}, ep={best_epoch}) | GradNorm max={max_grad_norm:.2f} avg={avg_grad_norm:.2f}")
    print(f"  {'':>{label_w}}   {'Train':^{tr_w}} | {'Val'}")
    for label, tr_val, val_val in rows:
        print(f"  {label:>{label_w}}   {tr_val:<{tr_w}} | {val_val}")
    tr_total = f"Loss={tr_loss:.4f}"
    val_total = f"Loss={val_loss:.4f}"
    print(f"  {'Total':>{label_w}}   {tr_total:<{tr_w}} | {val_total}")
    # Mirror per-pair detail (printed outside table to avoid column stretching)
    if constraint_weight > 0 and mirror_pair_indices is not None and len(mirror_pair_indices) > 0:
        if val_mirror_pair_detail:
            print(f"  Mirror pairs (val): {' '.join(f'{k}:{v:.2e}' for k, v in val_mirror_pair_detail.items())}")
    if uncertainty_weights is not None:
        norm_w = uncertainty_weights.get_weights()
        sigma_parts = []
        for name, log_var in uncertainty_weights.log_vars.items():
            sigma = torch.exp(log_var / 2).item()
            w = norm_w[name].item()
            sigma_parts.append(f"{name}:σ={sigma:.3f}/w={w:.2f}")
        print(f"  {'Uncert':>{label_w}}   {' | '.join(sigma_parts)}")
    # Log GENConv softmax temperature parameters
    t_vals = []
    for name, param in model.named_parameters():
        if 'aggr_module.t' in name and param.numel() == 1:
            # Extract layer identifier from name (e.g. backbone_layers.0.conv.aggr_module.t)
            parts = name.split('.')
            layer_name = '.'.join(parts[:2]) if len(parts) > 2 else name
            t_vals.append(f"{layer_name}={param.item():.3f}")
    if t_vals:
        print(f"  {'GENConv t':>{label_w}}   {' '.join(t_vals)}")


# ─────────────────────────────────────────────────────────────────────────────
# End-of-training report (AC / DC gain / region on the best model)
# ─────────────────────────────────────────────────────────────────────────────

def print_final_evaluation(model, val_loader, args, best_model_state,
                           ac_loss_weight_target, ac_mean, ac_std, ac_components,
                           dc_gain_loss_weight_target, dc_gain_mean, dc_gain_std,
                           region_loss_weight_target):
    """End-of-training AC and region evaluation on the best model."""
    model.load_state_dict(best_model_state)
    model.eval()

    # AC evaluation: denormalized MAE in real units
    if ac_loss_weight_target > 0 and ac_mean is not None and ac_components:
        ac_errors, ac_rel_errors = eval_ac(model, val_loader, args.device, ac_components, ac_mean, ac_std)

        if ac_errors[ac_components[0]]:
            print(f"\n--- AC Evaluation (best model, val set) ---")
            for comp in ac_components:
                errs = np.array(ac_errors[comp])
                if comp == 'ugbw':
                    print(f"  UGBW  MAE: {errs.mean():.3f} log10(Hz)  (median {np.median(errs):.3f})")
                    ugbw_rel = np.array(ac_rel_errors[comp])
                    print(f"  UGBW  within  5%:  {100*np.mean(ugbw_rel < 5):.1f}%")
                    print(f"  UGBW  within 10%:  {100*np.mean(ugbw_rel < 10):.1f}%")
                    print(f"  UGBW  within 20%:  {100*np.mean(ugbw_rel < 20):.1f}%")
                elif comp == 'pm':
                    print(f"  PM    MAE: {errs.mean():.1f} deg  (median {np.median(errs):.1f})")
                    print(f"  PM    within  5d:  {100*np.mean(errs < 5):.1f}%")
                    print(f"  PM    within 10d:  {100*np.mean(errs < 10):.1f}%")
                    print(f"  PM    within 20d:  {100*np.mean(errs < 20):.1f}%")
                elif comp == 'am':
                    print(f"  AM    MAE: {errs.mean():.1f} dB  (median {np.median(errs):.1f})")
                    print(f"  AM    within 1dB:  {100*np.mean(errs < 1):.1f}%")
                    print(f"  AM    within 3dB:  {100*np.mean(errs < 3):.1f}%")
                    print(f"  AM    within 5dB:  {100*np.mean(errs < 5):.1f}%")

    # DC gain evaluation
    if dc_gain_loss_weight_target > 0:
        errs = eval_dc_gain(model, val_loader, args.device, dc_gain_mean, dc_gain_std)

        if errs is not None:
            print(f"\n--- DC Gain Evaluation (best model, val set) ---")
            print(f"  MAE: {errs.mean():.2f} dB  (median {np.median(errs):.2f})")
            print(f"  within  3dB:  {100*np.mean(errs < 3):.1f}%")
            print(f"  within  5dB:  {100*np.mean(errs < 5):.1f}%")
            print(f"  within 10dB:  {100*np.mean(errs < 10):.1f}%")
            # Validity classifier accuracy: predict dc_gain > 0 as "valid"
            all_preds_db, all_targets_db = collect_dc_gain_predictions(
                model, val_loader, args.device, dc_gain_mean, dc_gain_std)
            if all_preds_db:
                preds_arr = np.array(all_preds_db)
                targets_arr = np.array(all_targets_db)
                pred_valid = preds_arr > 0
                actual_valid = targets_arr > 0
                accuracy = np.mean(pred_valid == actual_valid) * 100
                print(f"  Validity classifier (dc_gain>0): {accuracy:.1f}% accuracy")

    # Region classification evaluation
    if region_loss_weight_target > 0:
        region_names = ['cutoff', 'triode', 'saturation']
        all_preds, all_labels = collect_region_predictions(model, val_loader, args.device)

        if all_preds:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            total_correct = (all_preds == all_labels).sum().item()
            total_samples = len(all_labels)
            overall_acc = 100 * total_correct / total_samples

            print(f"\n--- Region Classification (best model, val set) ---")
            print(f"  Overall accuracy: {overall_acc:.1f}% ({total_correct}/{total_samples})")
            for c in range(3):
                c_mask = all_labels == c
                c_total = c_mask.sum().item()
                if c_total > 0:
                    c_correct = ((all_preds == c) & c_mask).sum().item()
                    c_acc = 100 * c_correct / c_total
                    print(f"  {region_names[c]:>10s}: {c_acc:5.1f}% ({c_correct}/{c_total})")
                else:
                    print(f"  {region_names[c]:>10s}: N/A (0 samples)")
