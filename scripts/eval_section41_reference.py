#!/usr/bin/env python
"""Compute §4.1 reference-model accuracy metrics and dump parity arrays.

Reference model = `usingnow` recipe = GINE+VN-MHA+loop+KCL+DC-head, scratch,
8 backbone layers, full fan_smc data (best.pt = epoch 4022, 6.42 mV V MAE).

Also evaluates the sau_cfcc full-data scratch+DC seeds (3 seeds) which use the
SAME architecture but slightly different optimizer/warmup (lr=1e-3, kcl_warmup=200).

Outputs:
  figures/thesis/section41_metrics.json           — per-topology metrics
  figures/thesis/section41_parity_fan_smc.npz     — parity arrays
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.pretrain_loader import PretrainCombinedLoader
from src.training.checkpoint import create_model_from_args
from src.training.config import parse_training_config
from eval_all_checkpoints import _drain_idx_batched, attach_norm

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
OUT_METRICS = REPO / 'figures/thesis/section41_metrics.json'
OUT_PARITY = REPO / 'figures/thesis/section41_parity_fan_smc.npz'


def evaluate(exp_dir: Path, dataset_pkl: str, topo_filter: str | None, device,
             dump_parity: bool = False):
    """Run model on val set, return per-topo metrics (and parity arrays if asked)."""
    # Pick checkpoint file
    ck_path = exp_dir / 'best.pt'
    if not ck_path.exists():
        ck_path = exp_dir / 'best_model.pt'
    ckpt = torch.load(ck_path, map_location=device, weights_only=False)
    with open(exp_dir / 'original_config.yaml') as f:
        cfg = yaml.safe_load(f)

    # Get norm stats — try both formats. Both vintages train heads to output in
    # NORMALIZED z-space; external denorm is always required.
    if 'norm_stats' in ckpt:
        stats = ckpt['norm_stats']
    elif 'stats' in ckpt:
        s = ckpt['stats']
        stats = {
            'v_mean': s['vdc']['mean'], 'v_std': s['vdc']['std'],
            'i_mean': s['current']['mean'], 'i_std': s['current']['std'],
            'gm_mean': s['ss_gm']['mean'], 'gm_std': s['ss_gm']['std'],
            'gds_mean': s['ss_gds']['mean'], 'gds_std': s['ss_gds']['std'],
            'dc_gain_mean': s['dc_gain']['mean'], 'dc_gain_std': s['dc_gain']['std'],
        }
    else:
        raise RuntimeError('No norm_stats found in checkpoint')
    head_output_is_physical = False   # heads are z-scored in both vintages

    val_loader = PretrainCombinedLoader(
        dataset_pkl, batch_size=200, device=device, shuffle=False,
        drop_last=False, topology_filter=topo_filter,
    )
    # PretrainCombinedLoader drops ac_dc_gain / ac_valid from batches. Reload the
    # raw samples in the same order so we can grab DC targets by index.
    import pickle as _pkl
    with open(dataset_pkl, 'rb') as _f:
        _raw = _pkl.load(_f)
    if topo_filter is not None:
        _raw = [s for s in _raw if s.get('topology') == topo_filter]
    # Sample order: PretrainCombinedLoader takes by_topo[t] = filtered list, batches by_topo
    # with batch_size/num_topos per batch. With shuffle=False and a single topo, it's
    # just sequential through _raw. Pull ac_dc_gain + ac_valid as parallel arrays.
    _ac_dc = torch.tensor([
        (s['graph'].ac_dc_gain.item() if hasattr(s['graph'].ac_dc_gain, 'item')
         else float(s['graph'].ac_dc_gain[0])) if hasattr(s['graph'], 'ac_dc_gain') else 0.0
        for s in _raw
    ], dtype=torch.float)
    _ac_valid = torch.tensor([
        bool(s['graph'].ac_valid) if hasattr(s['graph'], 'ac_valid') else True
        for s in _raw
    ], dtype=torch.bool)
    print(f'  [debug] raw samples: {len(_raw)}, ac_dc_gain valid: {(_ac_valid & (_ac_dc > 0)).sum().item()}/{len(_raw)}')
    sample = next(iter(val_loader))
    node_dim = sample.x.shape[-1]
    type_dim = sample.type_tens.shape[-1]

    import argparse as _ap
    cli = _ap.Namespace()
    for k, v in parse_training_config(cfg).items():
        if v is not None:
            setattr(cli, k, v)
    cli.device = str(device)
    cli.dataset = str(Path(dataset_pkl).parent)
    cli.predict_currents = True
    model, _ = create_model_from_args(cli, node_dim + type_dim, device)
    model.load_state_dict(ckpt['model_state_dict'])
    if hasattr(model, 'set_normalization_stats'):
        model.set_normalization_stats(
            vdc_mean=stats['v_mean'], vdc_std=stats['v_std'],
            ss_gm_mean=stats['gm_mean'], ss_gm_std=stats['gm_std'],
            ss_gds_mean=stats['gds_mean'], ss_gds_std=stats['gds_std'],
            current_mean=stats['i_mean'], current_std=stats['i_std'])
    if hasattr(model, 'current_epoch'):
        model.current_epoch = ckpt.get('epoch', 99999)
    model.eval()

    # Collect per-topo sums + (optionally) parity arrays
    metrics = {t: {'v_sum': 0.0, 'v_n': 0,
                    'i_sum_uA': 0.0, 'i_logsum': 0.0, 'i_logn': 0, 'i_n': 0,
                    'gm_sum': 0.0, 'gds_sum': 0.0, 'gm_n': 0,
                    'dc_sum_dB': 0.0, 'dc_n': 0}
               for t in val_loader.topo_names}
    parity = {'V_pred': [], 'V_tgt': [],
              'I_pred_log10': [], 'I_tgt_log10': [],
              'I_pred_uA': [], 'I_tgt_uA': [],
              'gm_pred_log10': [], 'gm_tgt_log10': [],
              'gds_pred_log10': [], 'gds_tgt_log10': [],
              'DC_pred_dB': [], 'DC_tgt_dB': [],
              'DC_physics_dB': []}    # analytical formula estimate (no MLP refinement)

    dc_gain_mean = stats.get('dc_gain_mean', 0.0)
    dc_gain_std = stats.get('dc_gain_std', 1.0)

    n_batches = 0
    dc_keys_seen = set()
    _raw_idx = 0  # running pointer into _raw to grab ac_dc_gain per batch
    with torch.no_grad():
        for batch in val_loader:
            attach_norm(batch, stats)
            out = model(batch)
            n_batches += 1
            slices = batch.topo_node_slices
            # ---- DC gain (per-graph). Use raw-sample ac_dc_gain since loader drops it.
            dc_key = 'dc_gain_pred' if 'dc_gain_pred' in out else ('dc_gain' if 'dc_gain' in out else None)
            if dc_key is not None: dc_keys_seen.add(dc_key)
            ng = int(batch.num_graphs)
            if dc_key is not None:
                tgt_dB = _ac_dc[_raw_idx:_raw_idx + ng]
                valid = _ac_valid[_raw_idx:_raw_idx + ng] & (tgt_dB > 0)
                dc_pred_raw = out[dc_key].cpu().flatten()
                if valid.any():
                    pred_dB = dc_pred_raw if head_output_is_physical else (dc_pred_raw * dc_gain_std + dc_gain_mean)
                    err = (pred_dB[valid] - tgt_dB[valid]).abs().sum().item()
                    n = int(valid.sum().item())
                    for t in val_loader.topo_names:
                        metrics[t]['dc_sum_dB'] += err
                        metrics[t]['dc_n'] += n
                    if dump_parity:
                        parity['DC_pred_dB'].append(pred_dB[valid].numpy())
                        parity['DC_tgt_dB'].append(tgt_dB[valid].numpy())
                        # Also dump the physics-formula-only estimate (no MLP refinement)
                        if 'dc_gain_physics_est_dB' in out:
                            phys = out['dc_gain_physics_est_dB'].cpu().flatten()
                            parity['DC_physics_dB'].append(phys[valid].numpy())
            _raw_idx += ng
            # ---- per-topo node-level metrics ----
            for t, (start, end, _n, _g) in slices.items():
                m = metrics[t]
                # V — net-node mask only (match train_v3's `get_prediction_mask`
                # which uses `batch.train_mask = output_node_mask & ~known_voltage_mask`).
                # The V head outputs a value for every node but the loss only supervises
                # internal nets — so MAE over terminals is meaningless / inflated.
                v_pred_raw = out['node_voltages'][start:end]
                v_tgt = batch.node_voltage_targets[start:end]
                tmask = batch.train_mask[start:end] if hasattr(batch, 'train_mask') else (~batch.known_voltage_mask[start:end])
                if tmask.any():
                    # Always z-space output → denorm externally
                    pv = v_pred_raw[tmask] * stats['v_std'] + stats['v_mean']
                    tv = v_tgt[tmask]
                    m['v_sum'] += (pv - tv).abs().sum().item()
                    m['v_n'] += int(tmask.sum().item())
                    if dump_parity and t == 'fan_smc':
                        parity['V_pred'].append(pv.cpu().numpy())
                        parity['V_tgt'].append(tv.cpu().numpy())
                # I (in log10 z-space)
                if 'node_currents' in out:
                    ip_z = out['node_currents'][start:end]
                    it = batch.node_current_targets[start:end]
                    im = batch.has_current_mask[start:end]
                    if im.any():
                        log_pred = ip_z[im] * stats['i_std'] + stats['i_mean']  # log10 amps
                        log_tgt = it[im].abs().clamp_min(1e-12).log10()
                        ip_A = torch.pow(10.0, log_pred)
                        it_A = it[im].abs()
                        m['i_sum_uA'] += (ip_A * 1e6 - it_A * 1e6).abs().sum().item()
                        m['i_n'] += int(im.sum().item())
                        m['i_logsum'] += (log_pred - log_tgt).abs().sum().item()
                        m['i_logn'] += int(im.sum().item())
                        if dump_parity and t == 'fan_smc':
                            parity['I_pred_log10'].append(log_pred.cpu().numpy())
                            parity['I_tgt_log10'].append(log_tgt.cpu().numpy())
                            parity['I_pred_uA'].append((ip_A * 1e6).cpu().numpy())
                            parity['I_tgt_uA'].append((it_A * 1e6).cpu().numpy())
                # gm / gds
                if 'mosfet_gm_pred' in out and 'mosfet_gds_pred' in out:
                    drain_idx = _drain_idx_batched(batch)
                    mos_in_t = (drain_idx >= start) & (drain_idx < end)
                    if mos_in_t.any():
                        gp = out['mosfet_gm_pred'][mos_in_t] * stats['gm_std'] + stats['gm_mean']
                        dp = out['mosfet_gds_pred'][mos_in_t] * stats['gds_std'] + stats['gds_mean']
                        gt = batch.node_log_gm[drain_idx[mos_in_t]]
                        dt = batch.node_log_gds[drain_idx[mos_in_t]]
                        m['gm_sum'] += (gp - gt).abs().sum().item()
                        m['gds_sum'] += (dp - dt).abs().sum().item()
                        m['gm_n'] += int(mos_in_t.sum().item())
                        if dump_parity and t == 'fan_smc':
                            parity['gm_pred_log10'].append(gp.cpu().numpy())
                            parity['gm_tgt_log10'].append(gt.cpu().numpy())
                            parity['gds_pred_log10'].append(dp.cpu().numpy())
                            parity['gds_tgt_log10'].append(dt.cpu().numpy())

    print(f'  [debug] DC keys seen in model output: {dc_keys_seen}; batches: {n_batches}; head_output_is_physical={head_output_is_physical}')
    out_metrics = {}
    for t in val_loader.topo_names:
        m = metrics[t]
        if m['v_n'] == 0: continue
        out_metrics[t] = {
            'v_mae_mV': 1000.0 * m['v_sum'] / m['v_n'],
            'i_mae_uA': m['i_sum_uA'] / max(1, m['i_n']),
            'i_logMAE': m['i_logsum'] / max(1, m['i_logn']),
            'gm_logMAE': m['gm_sum'] / max(1, m['gm_n']),
            'gds_logMAE': m['gds_sum'] / max(1, m['gm_n']),
            'dc_gain_MAE_dB': m['dc_sum_dB'] / max(1, m['dc_n']) if m['dc_n'] else None,
            'dc_gain_n_valid': m['dc_n'],
            'v_n': m['v_n'], 'i_n': m['i_n'], 'gm_n': m['gm_n'],
        }
    if dump_parity:
        for k in parity: parity[k] = np.concatenate(parity[k]) if parity[k] else np.zeros(0)
        return out_metrics, parity
    return out_metrics, None


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[setup] device={device}')
    results = {}

    # === fan_smc reference (usingnow) ===
    # Eval on the 5-topo combined val filtered to fan_smc (same val samples as the
    # per-topo fan_smc dataset, but PretrainCombinedLoader requires the combined
    # corpus format which carries the 'topology' field).
    print('\n[1/4] fan_smc — usingnow reference (eval on 5-topo val, filtered to fan_smc)')
    fan_dir = REPO / 'datasets/opamp_3stage_fan_smc_openloop_5k/experiments/ginebn_3layer_bb8_notower_vn_kcl_w10_edge6_netrole_clip03_openloop_5k_usingnow'
    m, parity = evaluate(
        fan_dir,
        str(REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/dataset_val.pkl'),
        topo_filter='fan_smc', device=device, dump_parity=True,
    )
    results['fan_smc_usingnow'] = m.get('fan_smc')
    if parity:
        np.savez(OUT_PARITY, **parity)
        print(f'  → parity arrays saved to {OUT_PARITY}')

    # === sau_cfcc — 3 seeds, scratch+DC ===
    for s in (42, 43, 44):
        print(f'\n[{1+s-41}/4] sau_cfcc — seed {s}')
        d = REPO / f'datasets/opamp_3stage_pretrain_combined_5topo/experiments/v5_5topo_pertopo_sau_cfcc_dcgain_n4000_s{s}'
        m, _ = evaluate(
            d,
            str(REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/dataset_val.pkl'),
            topo_filter='sau_cfcc', device=device, dump_parity=False,
        )
        results[f'sau_cfcc_s{s}'] = m.get('sau_cfcc')

    # Save and print
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_METRICS, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved metrics to {OUT_METRICS}')

    print('\n=== §4.1 REFERENCE METRICS ===')
    print(f"{'run':<28} {'V mV':>7} {'I µA':>8} {'I logMAE':>10} {'gm logMAE':>11} {'gds logMAE':>12} {'DC dB':>7}")
    for k, v in results.items():
        if v is None: print(f"  {k}: missing"); continue
        print(f"{k:<28} {v['v_mae_mV']:>7.2f} {v['i_mae_uA']:>8.2f} {v.get('i_logMAE',0):>10.4f} "
              f"{v['gm_logMAE']:>11.4f} {v['gds_logMAE']:>12.4f} {v.get('dc_gain_MAE_dB') or 0:>7.2f}")


if __name__ == '__main__':
    main()
