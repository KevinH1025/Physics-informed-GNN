#!/usr/bin/env python
"""Measure physical KCL residual (current-conservation violation) for physics-trained
vs non-physics-trained models. This is the metric the KCL loss directly optimizes and
that V/I/gm/gds MAE do NOT capture: a surrogate can be accurate yet predict currents
that violate Kirchhoff's law. Lower relative violation = more physically consistent.

Compares the LOO pretrains (KCL+loop ON vs OFF, identical otherwise).

Output: figures/thesis/kcl_residual.json + console table.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from circuitgnn.data.pretrain_loader import PretrainCombinedLoader
from circuitgnn.training.checkpoint import create_model_from_args
from circuitgnn.training.config import parse_training_config
from circuitgnn.training.losses import compute_kcl_per_net_debug

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / 'datasets/opamp_3stage_pretrain_combined_5topo'
EXP = DATASET / 'experiments'
TOPOS = ['fan_smc', 'sau_cfcc', 'peng_tcfc', 'leung_nmcf', 'leung_nmcnr']

DEFAULT_STATS = {'v_mean': 0.9, 'v_std': 0.5, 'i_mean': -5.55, 'i_std': 1.42,
                 'gm_mean': -4.43, 'gm_std': 1.44, 'gds_mean': -5.89, 'gds_std': 1.89}


def attach_norm(batch, stats):
    for name, val in [('vdc_mean', stats['v_mean']), ('vdc_std', stats['v_std']),
                      ('current_mean', stats['i_mean']), ('current_std', stats['i_std']),
                      ('ss_gm_mean', stats['gm_mean']), ('ss_gm_std', stats['gm_std']),
                      ('ss_gds_mean', stats['gds_mean']), ('ss_gds_std', stats['gds_std'])]:
        setattr(batch, name, torch.tensor(val, device=batch.x.device))


def load_model(exp_dir, device):
    ckpt = torch.load(exp_dir / 'best.pt', map_location=device, weights_only=False)
    with open(exp_dir / 'original_config.yaml') as f:
        cfg = yaml.safe_load(f)
    stats = ckpt.get('norm_stats') or DEFAULT_STATS
    import argparse as _ap
    cli = _ap.Namespace()
    for k, v in parse_training_config(cfg).items():
        if v is not None:
            setattr(cli, k, v)
    cli.device = str(device); cli.dataset = str(DATASET); cli.predict_currents = True
    return ckpt, cfg, stats, cli


def kcl_violation(exp_dir, device, topo, max_graphs=120):
    """Mean relative KCL violation over internal nets, on `topo` val graphs."""
    val = PretrainCombinedLoader(f'{DATASET}/dataset_val.pkl', batch_size=1,
                                 device=device, shuffle=False, drop_last=False,
                                 topology_filter=topo)
    ckpt, cfg, stats, cli = load_model(exp_dir, device)
    sample = next(iter(val))
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

    viols = []
    n = 0
    with torch.no_grad():
        for batch in val:
            if n >= max_graphs:
                break
            n += 1
            attach_norm(batch, stats)
            out = model(batch)
            ic = out.get('node_currents')
            if ic is None:
                continue
            n_nodes = batch.x.shape[0]
            names = [f'n{i}' for i in range(n_nodes)]  # debug fn only uses for keys
            inc = getattr(batch, 'kcl_include_mask', None)
            res = compute_kcl_per_net_debug(
                node_currents=ic, edge_index=batch.edge_index,
                num_terminals=batch.num_terminals,
                train_mask=~batch.known_voltage_mask, ptr=batch.ptr,
                terminal_current_sign=batch.terminal_current_sign,
                current_mean=stats['i_mean'], current_std=stats['i_std'],
                node_names=names, kcl_include_mask=inc, graph_idx=0)
            viols.extend(res.values())
    if not viols:
        return None
    return float(np.mean(viols)), float(np.median(viols)), len(viols)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[setup] device={device}')
    # IN-DISTRIBUTION test: KCL-during-FT (col1) vs no-KCL-FT (col2), same pretrain,
    # both fine-tuned on the target topo. Measures whether the KCL loss actually makes
    # predictions more current-conserving ON THE TOPOLOGY IT WAS TRAINED ON.
    SHORT = {'fan_smc': 'fansmc', 'sau_cfcc': 'sau', 'peng_tcfc': 'peng',
             'leung_nmcf': 'leungf', 'leung_nmcnr': 'leungnr'}
    T2 = {t: ('fansmc' if t == 'fan_smc' else t) for t in TOPOS}
    N = 1000

    out = {}
    print(f"\nIn-distribution KCL violation (FT on target, N={N})")
    print(f"{'topo':<13} | {'KCL-FT mean/med':<20} | {'noKCL-FT mean/med':<20} | winner")
    print('-' * 78)
    for topo in TOPOS:
        # col1 = phys pretrain + phys (KCL) FT  →  default ftzeroshot run
        c1 = EXP / f'v5_5topo_ftzeroshot_{T2[topo]}_n{N}_v2'
        if not (c1 / 'best.pt').exists():
            c1 = EXP / f'v5_5topo_ftzeroshot_{T2[topo]}_n{N}'
        # col2 = phys pretrain + nophys FT (KCL off in FT)
        c2 = EXP / f'v5_5topo_ftzs_{SHORT[topo]}_phys_nophysFT_n{N}'
        rk = kcl_violation(c1, device, topo) if (c1 / 'best.pt').exists() else None
        rn = kcl_violation(c2, device, topo) if (c2 / 'best.pt').exists() else None
        out[topo] = {'kcl_ft': rk, 'nokcl_ft': rn}
        if rk and rn:
            win = 'KCL-FT' if rk[0] < rn[0] else 'noKCL-FT'
            print(f"{topo:<13} | {rk[0]*100:6.2f}% / {rk[1]*100:6.2f}%   | "
                  f"{rn[0]*100:6.2f}% / {rn[1]*100:6.2f}%   | {win}")
        else:
            print(f"{topo:<13} | kcl_ft={rk} nokcl_ft={rn}")

    OUT = REPO / 'figures/thesis/kcl_residual_indist.json'
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {OUT}')


if __name__ == '__main__':
    main()
