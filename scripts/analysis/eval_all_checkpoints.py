#!/usr/bin/env python
"""Run eval_per_topo across ALL N values for scratch/ftzeroshot/joint.

Loads each checkpoint in-process (no slurm/subprocess overhead) and dumps a
single JSON of:
    method → topo → N → {v_mae_mV, i_mae_uA, gm_log_mae, gds_log_mae, ...}

Output: figures/thesis/metric_table_per_n.json

Usage:
    python scripts/analysis/eval_all_checkpoints.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from circuitgnn.data.pretrain_loader import PretrainCombinedLoader
from circuitgnn.training.checkpoint import create_model_from_args
from circuitgnn.training.config import parse_training_config


from circuitgnn.evaluation import (
    DATASET,
    EXP,
    REPO,
    TOPOS,
    _drain_idx_batched,
    attach_norm,
    eval_one,
)
from circuitgnn.evaluation.checkpoints import _LOG_FLOOR

OUT_PATH = REPO / 'figures/thesis/metric_table_per_n.json'


def eval_mlp_one(exp_dir: Path, device, topo: str, max_per_topo):
    """Evaluate a per-topology MLP baseline checkpoint with the same metric
    definitions as eval_one. Recomputes norm stats from the (deterministic
    lst[:max_per_topo]) training subset, exactly as train_mlp_baseline does."""
    from scripts.train_mlp_baseline import MLP
    ckpt = torch.load(exp_dir / 'best.pt', map_location=device, weights_only=False)
    sd = ckpt['model_state_dict']
    in_dim = sd['backbone.0.weight'].shape[1]
    hidden = sd['backbone.0.weight'].shape[0]
    n_internal = sd['v_head.weight'].shape[0]
    n_mosfets = sd['i_head.weight'].shape[0]
    num_layers = sum(1 for k in sd if k.startswith('backbone.') and k.endswith('.weight')
                     and 'norm' not in k.lower()) // 1
    # backbone layers = count of Linear layers in backbone (every 3rd module is Linear)
    num_layers = len([k for k in sd if k.startswith('backbone.') and k.endswith('.weight')
                      and sd[k].dim() == 2])

    train_loader = PretrainCombinedLoader(
        f'{DATASET}/dataset_train.pkl', batch_size=64, device=device,
        shuffle=False, topology_filter=topo, max_per_topo=max_per_topo)
    val_loader = PretrainCombinedLoader(
        f'{DATASET}/dataset_val.pkl', batch_size=200, device=device,
        shuffle=False, drop_last=False, topology_filter=topo)

    s = next(iter(train_loader))
    n_nodes = (s.ptr[1] - s.ptr[0]).item()
    n_term = s.num_terminals[0].item() if hasattr(s.num_terminals, '__len__') else int(s.num_terminals)
    n_mos = (s.mosfet_ptr[1] - s.mosfet_ptr[0]).item()

    # ---- norm stats from train subset (order-independent mean/std) ----
    Vs, log_Is, log_gms, log_gdss, Ws, Ls, Ms = [], [], [], [], [], [], []
    for b in train_loader:
        Vs.append(b.node_voltage_targets.flatten())
        if getattr(b, 'has_current_mask', None) is not None:
            mk = b.has_current_mask
            log_Is.append(b.node_current_targets[mk].abs().clamp_min(_LOG_FLOOR).log10())
        dm = b.mosfet_drain_mask
        log_gms.append(b.node_log_gm[dm]); log_gdss.append(b.node_log_gds[dm])
        Ws.append(b.mosfet_wl_um[:, 0]); Ls.append(b.mosfet_wl_um[:, 1]); Ms.append(b.mosfet_m)
    def ms_(x): c = torch.cat(x); return c.mean().item(), max(c.std().item(), 1e-6)
    v_mean, v_std = ms_(Vs); i_mean, i_std = ms_(log_Is)
    gm_mean, gm_std = ms_(log_gms); gds_mean, gds_std = ms_(log_gdss)
    w_mean, w_std = ms_(Ws); l_mean, l_std = ms_(Ls)
    mlog = torch.cat(Ms).clamp_min(0.5).log10(); m_mean, m_std = mlog.mean().item(), max(mlog.std().item(), 1e-6)

    model = MLP(in_dim, hidden, num_layers, n_internal, n_mosfets).to(device)
    model.load_state_dict(sd); model.eval()

    v_sum = v_n = i_sum = i_n = gm_sum = gds_sum = gm_n = 0.0
    with torch.no_grad():
        for b in val_loader:
            ng = b.num_graphs
            wl = b.mosfet_wl_um.view(ng, n_mos, 2); mm = b.mosfet_m.view(ng, n_mos)
            w_n = (wl[..., 0] - w_mean) / w_std; l_n = (wl[..., 1] - l_mean) / l_std
            m_n = (mm.clamp_min(0.5).log10() - m_mean) / m_std
            per_mos = torch.stack([w_n, l_n, m_n], -1).view(ng, -1)
            gfeat = b.x.view(ng, n_nodes, -1)[:, 0, 4:9]
            x = torch.cat([per_mos, gfeat], -1)
            out = model(x)
            # V (internal nodes only)
            v_full = b.node_voltage_targets.view(ng, n_nodes)
            v_tgt = v_full[:, n_term:]
            pv = out['v'] * v_std + v_mean
            v_sum += (pv - v_tgt).abs().sum().item(); v_n += v_tgt.numel()
            # I at drain
            mi = b.mosfet_info.view(ng, n_mos, -1); dloc = mi[:, :, 1].long()
            i_full = b.node_current_targets.view(ng, n_nodes)
            i_tgt_A = i_full.gather(1, dloc).abs()
            log_pred = out['i'] * i_std + i_mean
            i_pred_A = torch.pow(10.0, log_pred)
            i_sum += (i_pred_A - i_tgt_A).abs().sum().item(); i_n += i_tgt_A.numel()
            # gm/gds at drain (log10 units)
            gm_full = b.node_log_gm.view(ng, n_nodes); gds_full = b.node_log_gds.view(ng, n_nodes)
            gm_tgt = gm_full.gather(1, dloc); gds_tgt = gds_full.gather(1, dloc)
            gp = out['gm'] * gm_std + gm_mean; dp = out['gds'] * gds_std + gds_mean
            gm_sum += (gp - gm_tgt).abs().sum().item(); gds_sum += (dp - gds_tgt).abs().sum().item()
            gm_n += gm_tgt.numel()

    return {topo: {
        'v_mae_mV': 1000.0 * v_sum / max(1, v_n),
        'i_mae_uA': 1e6 * i_sum / max(1, i_n),
        'gm_log_mae': gm_sum / max(1, gm_n),
        'gds_log_mae': gds_sum / max(1, gm_n),
    }}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[setup] device={device}')

    # Build the run list:
    # scratch: 5 topos × {100, 250, 500, 1000, 2500, 4000}
    # ftzeroshot: 5 topos × {10, 50, 100, 250, 500, 1000, 2500, 4000}, prefer v2
    # joint: {100, 250, 500, 800, 1000, 4000} — model returns all 5 topos
    runs = []
    for t in TOPOS:
        t2 = 'fansmc' if t == 'fan_smc' else t
        for N in (100, 250, 500, 1000, 2500):
            runs.append(('scratch', t, N, EXP / f'v5_5topo_pertopo_{t2}_n{N}'))
        runs.append(('scratch', t, 4000, EXP / f'v5_5topo_pertopo_{t}'))
        for N in (10, 50, 100, 250, 500, 1000, 2500, 4000):
            v2 = EXP / f'v5_5topo_ftzeroshot_{t2}_n{N}_v2'
            v1 = EXP / f'v5_5topo_ftzeroshot_{t2}_n{N}'
            runs.append(('ftzeroshot', t, N, v2 if v2.exists() else v1))
    for N in (100, 250, 500, 800, 1000):
        runs.append(('joint', None, N, EXP / f'v5_5topo_joint_n{N}'))
    runs.append(('joint', None, 4000, EXP / 'v5_5topo_joint'))
    # joint_total: apples-to-apples — x = TOTAL training budget (per-topo cap × 5).
    # Same joint runs (plus low-N reseeds) placed at their true total-sample budget.
    joint_total_runs = {
        100:  'v5_5topo_joint_n20',      # 20/topo × 5
        250:  'v5_5topo_joint_n50',      # 50/topo × 5
        500:  'v5_5topo_joint_n100_v2',  # 100/topo × 5 (re-run; monotone fix)
        1000: 'v5_5topo_joint_n200',     # 200/topo × 5
        2500: 'v5_5topo_joint_n500',     # 500/topo × 5
        4000: 'v5_5topo_joint_n800',     # 800/topo × 5
        5000: 'v5_5topo_joint_n1000',    # 1000/topo × 5
    }
    for total, run in joint_total_runs.items():
        runs.append(('joint_total', None, total, EXP / run))
    # MLP baseline (per-topology, 8-layer). max_per_topo carried in N (None=full).
    for t in TOPOS:
        t2 = 'fansmc' if t == 'fan_smc' else t
        for N in (100, 250, 500, 1000, 2500):
            runs.append(('mlp', t, N, EXP / f'mlp_{t2}_n{N}'))
        runs.append(('mlp', t, 4000, EXP / f'mlp_{t2}_full'))

    # Drop missing
    runs = [(m, t, N, p) for m, t, N, p in runs if (p / 'best.pt').exists()]
    print(f'[setup] {len(runs)} checkpoints to evaluate')

    table = {'scratch': {t: {} for t in TOPOS},
             'ftzeroshot': {t: {} for t in TOPOS},
             'joint': {t: {} for t in TOPOS},
             'joint_total': {t: {} for t in TOPOS},
             'mlp': {t: {} for t in TOPOS}}

    for i, (method, topo, N, p) in enumerate(runs, 1):
        print(f'[{i}/{len(runs)}] {method} {topo or "all"} N={N} → {p.name}', flush=True)
        try:
            if method == 'mlp':
                m = eval_mlp_one(p, device, topo, max_per_topo=(None if N == 4000 else N))
            else:
                m = eval_one(p, device, topo_filter=topo)
        except Exception as e:
            print(f'  ERROR: {e}')
            continue
        if method in ('joint', 'joint_total'):
            for tn, mt in m.items():
                if tn in TOPOS:
                    table[method][tn][N] = mt
        else:
            if topo in m:
                table[method][topo][N] = m[topo]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(table, f, indent=2)
    print(f'\nSaved per-N metrics → {OUT_PATH}')


if __name__ == '__main__':
    main()
