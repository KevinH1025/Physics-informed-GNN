#!/usr/bin/env python
"""Per-topology V/I/gm/gds MAE for each zero-shot pretrain checkpoint on its
held-out topology — to test whether device-physics (gm, gds) transfer cleanly
across topologies while voltage (which depends on global feedback) degrades.
"""
from __future__ import annotations
import sys, re, json
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from circuitgnn.data.pretrain_loader import PretrainCombinedLoader
from circuitgnn.training.checkpoint import create_model_from_args
from circuitgnn.training.config import parse_training_config
from circuitgnn.evaluation import attach_norm

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/experiments'
VAL_PKL = REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/dataset_val.pkl'
OUT_JSON = REPO / 'figures/thesis/section442_zeroshot_per_topo.json'

# (topology, zeroshot-run-name-suffix)
ZERO_SHOT_RUNS = [
    ('fan_smc',     'fansmc'),
    ('sau_cfcc',    'sau_cfcc'),
    ('peng_tcfc',   'peng_tcfc'),
    ('leung_nmcf',  'leung_nmcf'),
    ('leung_nmcnr', 'leung_nmcnr'),
]

# Same-topology baselines for comparison (from metric_table.json "scratch")
BASELINE = {
    'fan_smc':     {'V': 14.13, 'I': 9.43,  'gm': 0.0359, 'gds': 0.0555},
    'sau_cfcc':    {'V': 26.79, 'I': 26.57, 'gm': 0.0625, 'gds': 0.0812},
    'peng_tcfc':   {'V': 12.93, 'I': 4.91,  'gm': 0.0237, 'gds': 0.0435},
    'leung_nmcf':  {'V': 24.27, 'I': 10.27, 'gm': 0.0570, 'gds': 0.0885},
    'leung_nmcnr': {'V': 35.65, 'I': 16.96, 'gm': 0.0819, 'gds': 0.1161},
}


def load_model(exp_dir, device, sample):
    ck_path = exp_dir / 'best.pt'
    if not ck_path.exists():
        ck_path = exp_dir / 'best_model.pt'
    ckpt = torch.load(ck_path, map_location=device, weights_only=False)
    with open(exp_dir / 'original_config.yaml') as f:
        cfg = yaml.safe_load(f)
    if 'norm_stats' in ckpt:
        stats = ckpt['norm_stats']
    else:
        s = ckpt['stats']
        stats = {'v_mean': s['vdc']['mean'], 'v_std': s['vdc']['std'],
                 'i_mean': s['current']['mean'], 'i_std': s['current']['std'],
                 'gm_mean': s['ss_gm']['mean'], 'gm_std': s['ss_gm']['std'],
                 'gds_mean': s['ss_gds']['mean'], 'gds_std': s['ss_gds']['std']}
    import argparse as _ap
    cli = _ap.Namespace()
    for k, v in parse_training_config(cfg).items():
        if v is not None: setattr(cli, k, v)
    cli.device = str(device); cli.dataset = str(VAL_PKL.parent); cli.predict_currents = True
    node_dim = sample.x.shape[-1]; type_dim = sample.type_tens.shape[-1]
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
    return model, stats


@torch.no_grad()
def evaluate(exp_dir, held_out_topo, device):
    val = PretrainCombinedLoader(str(VAL_PKL), batch_size=50, device=device,
                                 shuffle=False, drop_last=False, topology_filter=held_out_topo)
    sample = next(iter(val))
    model, stats = load_model(exp_dir, device, sample)
    val = PretrainCombinedLoader(str(VAL_PKL), batch_size=50, device=device,
                                 shuffle=False, drop_last=False, topology_filter=held_out_topo)

    v_sum = 0.0; v_n = 0
    i_sum_uA = 0.0; i_log_sum = 0.0; i_n = 0
    gm_sum = 0.0; gds_sum = 0.0; gm_n = 0
    for batch in val:
        attach_norm(batch, stats)
        out = model(batch)
        # V (net nodes only — train_mask)
        v_pred = out['node_voltages'] * stats['v_std'] + stats['v_mean']
        v_tgt = batch.node_voltage_targets
        tmask = batch.train_mask if hasattr(batch, 'train_mask') else (~batch.known_voltage_mask)
        if tmask.any():
            v_sum += (v_pred[tmask] - v_tgt[tmask]).abs().sum().item()
            v_n += int(tmask.sum().item())
        # I (drain terminals)
        if 'node_currents' in out:
            ip_z = out['node_currents']
            it = batch.node_current_targets
            im = batch.has_current_mask
            if im.any():
                log_pred = ip_z[im] * stats['i_std'] + stats['i_mean']
                log_tgt = it[im].abs().clamp_min(1e-12).log10()
                ip_A = torch.pow(10.0, log_pred)
                it_A = it[im].abs()
                i_sum_uA += (ip_A * 1e6 - it_A * 1e6).abs().sum().item()
                i_log_sum += (log_pred - log_tgt).abs().sum().item()
                i_n += int(im.sum().item())
        # gm/gds at drain terminals
        if 'mosfet_gm_pred' in out and 'mosfet_gds_pred' in out:
            mi = batch.mosfet_info.long()
            ptr = batch.ptr
            num_mosfets = mi.shape[0]; num_graphs = len(ptr) - 1
            mosfets_per_graph = num_mosfets // num_graphs
            graph_idx = torch.arange(num_mosfets, device=mi.device) // mosfets_per_graph
            node_offsets = ptr[graph_idx]
            drain_idx = mi[:, 1] + node_offsets
            gp = out['mosfet_gm_pred'].squeeze() * stats['gm_std'] + stats['gm_mean']
            dp = out['mosfet_gds_pred'].squeeze() * stats['gds_std'] + stats['gds_mean']
            gt = batch.node_log_gm[drain_idx]
            dt = batch.node_log_gds[drain_idx]
            gm_sum += (gp - gt).abs().sum().item()
            gds_sum += (dp - dt).abs().sum().item()
            gm_n += num_mosfets

    return {
        'V_mae_mV':   1000.0 * v_sum / max(1, v_n),
        'I_mae_uA':   i_sum_uA / max(1, i_n),
        'I_logMAE':   i_log_sum / max(1, i_n),
        'gm_logMAE':  gm_sum / max(1, gm_n),
        'gds_logMAE': gds_sum / max(1, gm_n),
        'n_v_nodes':  v_n,
        'n_mosfets':  gm_n,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[setup] device={device}\n')

    results = {}
    for topo, alias in ZERO_SHOT_RUNS:
        exp_dir = EXP / f'v5_5topo_zeroshot_{alias}'
        if not exp_dir.exists():
            print(f'[skip] {exp_dir} not found')
            continue
        print(f'[eval zero-shot] held-out={topo}  ({exp_dir.name})')
        m = evaluate(exp_dir, topo, device)
        b = BASELINE[topo]
        results[topo] = {**m, 'baseline': b}
        print(f'  V   MAE = {m["V_mae_mV"]:7.2f} mV    (baseline {b["V"]:7.2f}, ratio {m["V_mae_mV"]/b["V"]:.2f}x)')
        print(f'  I   MAE = {m["I_mae_uA"]:7.2f} µA    (baseline {b["I"]:7.2f}, ratio {m["I_mae_uA"]/b["I"]:.2f}x)')
        print(f'  gm  log = {m["gm_logMAE"]:7.4f}        (baseline {b["gm"]:7.4f}, ratio {m["gm_logMAE"]/b["gm"]:.2f}x)')
        print(f'  gds log = {m["gds_logMAE"]:7.4f}        (baseline {b["gds"]:7.4f}, ratio {m["gds_logMAE"]/b["gds"]:.2f}x)')
        print()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved → {OUT_JSON}')

    print('\n' + '='*98)
    print(f"{'§4.4.2 ZERO-SHOT V/I/gm/gds degradation per held-out topology':^98}")
    print('='*98)
    print(f"{'topo':<14}|{'V (mV)':>26}|{'I (µA)':>22}|{'gm (logMAE)':>22}|{'gds (logMAE)':>22}")
    print(f"{'':14}|{'zero-shot':>13}{'baseline':>13}|{'ratio':>10}{'b-ratio':>12}|{'ratio':>10}{'b-ratio':>12}|{'ratio':>10}{'b-ratio':>12}")
    print('-'*100)
    for topo, _ in ZERO_SHOT_RUNS:
        if topo not in results: continue
        m = results[topo]; b = m['baseline']
        print(f"{topo:<14}|{m['V_mae_mV']:>13.2f}{b['V']:>13.2f}"
              f"|{m['I_mae_uA']:>10.2f}{m['I_mae_uA']/b['I']:>10.2f}x"
              f"|{m['gm_logMAE']:>10.4f}{m['gm_logMAE']/b['gm']:>10.2f}x"
              f"|{m['gds_logMAE']:>10.4f}{m['gds_logMAE']/b['gds']:>10.2f}x")


if __name__ == '__main__':
    main()
