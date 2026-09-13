#!/usr/bin/env python
"""MLP baseline for per-topology V/I/gm/gds prediction.

For a fixed topology, all samples have the same graph structure — so we can
flatten input features into a fixed-size vector and use a vanilla MLP.

This is the architecture-justification baseline: shows that the GNN's inductive
bias is necessary, especially at low data. MLPs can't easily handle multi-
topology training (different graph sizes), which is itself a point.

Per-MOSFET features:
  - W (µm), L (µm), M (multiplier)
Global features:
  - VDD, Vcm, Vinp, Vinn, log10(I_ref)

Outputs:
  - V at all internal nodes
  - log10(|I|) at all drain terminals
  - log10(gm), log10(gds) at all drains

Usage:
    python scripts/train_mlp_baseline.py --topology fan_smc --max-per-topo 1000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pretrain_loader import PretrainCombinedLoader

_LOG_FLOOR = 1e-12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--topology', required=True,
                   choices=['sau_cfcc', 'peng_tcfc', 'leung_nmcf', 'leung_nmcnr', 'fan_smc'])
    p.add_argument('--dataset-dir', default='datasets/opamp_3stage_pretrain_combined_5topo')
    p.add_argument('--max-per-topo', type=int, default=None)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=3000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--patience', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, n_internal, n_mosfets):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.v_head = nn.Linear(hidden, n_internal)        # V at internal nodes
        self.i_head = nn.Linear(hidden, n_mosfets)         # log10|I| at drain
        self.gm_head = nn.Linear(hidden, n_mosfets)
        self.gds_head = nn.Linear(hidden, n_mosfets)

    def forward(self, x):
        h = self.backbone(x)
        return {
            'v': self.v_head(h),      # [B, n_internal]
            'i': self.i_head(h),      # [B, M]
            'gm': self.gm_head(h),
            'gds': self.gds_head(h),
        }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data filtered to one topology
    train_loader = PretrainCombinedLoader(
        f'{args.dataset_dir}/dataset_train.pkl',
        batch_size=args.batch_size, device=device, shuffle=True,
        topology_filter=args.topology, max_per_topo=args.max_per_topo,
    )
    val_loader = PretrainCombinedLoader(
        f'{args.dataset_dir}/dataset_val.pkl',
        batch_size=args.batch_size, device=device, shuffle=False, drop_last=False,
        topology_filter=args.topology,
    )
    print(f'Loaded: train={len(train_loader)} batches, val={len(val_loader)} (topology={args.topology})')

    # Probe shapes from one batch
    sample = next(iter(train_loader))
    n_nodes_per_g = (sample.ptr[1] - sample.ptr[0]).item()
    n_term_per_g = sample.num_terminals[0].item() if hasattr(sample.num_terminals, '__len__') else int(sample.num_terminals)
    n_mosfets_per_g = (sample.mosfet_ptr[1] - sample.mosfet_ptr[0]).item()
    n_graphs = sample.num_graphs
    n_internal = n_nodes_per_g - n_term_per_g
    print(f'Per-graph: nodes={n_nodes_per_g}, terminals={n_term_per_g}, internal nets={n_internal}, mosfets={n_mosfets_per_g}')

    # Build features: per-MOSFET (W, L, M) + global (vdd from x[0,4..8])
    # We extract W/L/M from mosfet_wl_um and mosfet_m exposed by the loader.
    if not hasattr(sample, 'mosfet_wl_um') or sample.mosfet_wl_um is None:
        raise RuntimeError('mosfet_wl_um not in batch — patched corpus required')

    in_dim = n_mosfets_per_g * 3 + 5  # W/L/M per MOSFET + 5 global
    print(f'MLP input dim: {in_dim} (= {n_mosfets_per_g}×3 W/L/M + 5 global)')

    # Normalization stats from train set
    print('Computing norm stats...')
    Vs, log_Is, log_gms, log_gdss = [], [], [], []
    Ws, Ls, Ms = [], [], []
    for b in train_loader:
        Vs.append(b.node_voltage_targets.flatten().cpu())
        if hasattr(b, 'has_current_mask') and b.has_current_mask is not None:
            mask = b.has_current_mask
            log_Is.append(b.node_current_targets[mask].abs().clamp_min(_LOG_FLOOR).log10().cpu())
        dm = b.mosfet_drain_mask
        log_gms.append(b.node_log_gm[dm].cpu())
        log_gdss.append(b.node_log_gds[dm].cpu())
        Ws.append(b.mosfet_wl_um[:, 0].cpu())
        Ls.append(b.mosfet_wl_um[:, 1].cpu())
        Ms.append(b.mosfet_m.cpu())

    v_mean, v_std = torch.cat(Vs).mean().item(), max(torch.cat(Vs).std().item(), 1e-6)
    i_mean, i_std = torch.cat(log_Is).mean().item(), max(torch.cat(log_Is).std().item(), 1e-6)
    gm_mean, gm_std = torch.cat(log_gms).mean().item(), max(torch.cat(log_gms).std().item(), 1e-6)
    gds_mean, gds_std = torch.cat(log_gdss).mean().item(), max(torch.cat(log_gdss).std().item(), 1e-6)
    w_mean, w_std = torch.cat(Ws).mean().item(), max(torch.cat(Ws).std().item(), 1e-6)
    l_mean, l_std = torch.cat(Ls).mean().item(), max(torch.cat(Ls).std().item(), 1e-6)
    m_log = torch.cat(Ms).clamp_min(0.5).log10()
    m_mean, m_std = m_log.mean().item(), max(m_log.std().item(), 1e-6)
    print(f'  V({v_mean:.3f},{v_std:.3f}), I({i_mean:.3f},{i_std:.3f}), gm({gm_mean:.3f},{gm_std:.3f}), gds({gds_mean:.3f},{gds_std:.3f})')

    def extract_features(b):
        """Return per-graph feature matrix [n_graphs, in_dim]."""
        wl = b.mosfet_wl_um.view(n_graphs, n_mosfets_per_g, 2)
        ms = b.mosfet_m.view(n_graphs, n_mosfets_per_g)
        # Normalize
        w_n = (wl[..., 0] - w_mean) / w_std
        l_n = (wl[..., 1] - l_mean) / l_std
        m_n = (ms.clamp_min(0.5).log10() - m_mean) / m_std
        per_mos = torch.stack([w_n, l_n, m_n], dim=-1).view(n_graphs, -1)  # [B, M*3]
        # Global features: take first row of x at terminal 0 (assumes per-graph constants
        # are stored there). Use cols [4..9] (vdd, vcm, vinp, vinn, log10(i_ref)+5 from
        # graph_builder).
        x_per_graph = b.x.view(n_graphs, n_nodes_per_g, -1)
        global_feat = x_per_graph[:, 0, 4:9]  # [B, 5]
        return torch.cat([per_mos, global_feat], dim=-1)  # [B, in_dim]

    def extract_targets(b):
        """Return per-graph V (internal) and I/gm/gds (per-MOSFET) targets normalized."""
        # V: take internal nodes only (after terminals)
        v_full = b.node_voltage_targets.view(n_graphs, n_nodes_per_g)
        v_internal = v_full[:, n_term_per_g:]  # [B, n_internal]
        v_z = (v_internal - v_mean) / v_std
        # I per MOSFET: take from drain terminal in node_current_targets
        # Drain idx = mosfet_info[:,1] + per-graph offset (which is 0 within sample)
        # Per-graph: mosfet_info per graph has drain at col 1.
        mi_per_g = b.mosfet_info.view(n_graphs, n_mosfets_per_g, -1)
        drain_local = mi_per_g[:, :, 1].long()  # [B, M], indices into per-graph nodes
        graph_idx = torch.arange(n_graphs, device=b.x.device).unsqueeze(1).expand(-1, n_mosfets_per_g)
        i_full = b.node_current_targets.view(n_graphs, n_nodes_per_g)
        i_drain = i_full.gather(1, drain_local)
        i_z = (i_drain.abs().clamp_min(_LOG_FLOOR).log10() - i_mean) / i_std
        # gm/gds: use node_log_gm at drain
        gm_full = b.node_log_gm.view(n_graphs, n_nodes_per_g)
        gds_full = b.node_log_gds.view(n_graphs, n_nodes_per_g)
        gm_z = (gm_full.gather(1, drain_local) - gm_mean) / gm_std
        gds_z = (gds_full.gather(1, drain_local) - gds_mean) / gds_std
        return v_z, i_z, gm_z, gds_z

    model = MLP(in_dim, args.hidden, args.layers, n_internal, n_mosfets_per_g).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'MLP params: {n_params/1e6:.2f}M')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=200, factor=0.5, min_lr=1e-6)

    out_dir = Path(f'{args.dataset_dir}/experiments/mlp_{args.name}')
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []
    log_path = out_dir / 'training.log'
    best_val = float('inf')
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        n_steps = 0
        for b in train_loader:
            x = extract_features(b)
            v_t, i_t, gm_t, gds_t = extract_targets(b)
            out = model(x)
            loss = (F.mse_loss(out['v'], v_t) + F.mse_loss(out['i'], i_t)
                    + F.mse_loss(out['gm'], gm_t) + F.mse_loss(out['gds'], gds_t))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            opt.step()
            tr_loss += loss.item()
            n_steps += 1
        tr_loss /= n_steps

        model.eval()
        val_loss = 0.0
        v_mae_sum = 0.0
        n_val_internal = 0
        n_v = 0
        with torch.no_grad():
            for b in val_loader:
                x = extract_features(b)
                v_t, i_t, gm_t, gds_t = extract_targets(b)
                out = model(x)
                vl = (F.mse_loss(out['v'], v_t) + F.mse_loss(out['i'], i_t)
                      + F.mse_loss(out['gm'], gm_t) + F.mse_loss(out['gds'], gds_t))
                val_loss += vl.item()
                n_v += 1
                # V MAE in mV
                v_pred_v = out['v'] * v_std + v_mean
                v_t_v = v_t * v_std + v_mean
                v_mae_sum += (v_pred_v - v_t_v).abs().sum().item()
                n_val_internal += v_t.numel()
        val_loss /= n_v
        v_mae_mv = 1000.0 * v_mae_sum / n_val_internal

        sch.step(val_loss)
        cur_lr = opt.param_groups[0]['lr']
        msg = f'epoch {epoch:4d}: lr={cur_lr:.2e} tr={tr_loss:.4e} val={val_loss:.4e} V MAE={v_mae_mv:.2f}mV'
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            msg += ' [BEST]'
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': val_loss, 'v_mae_mv': v_mae_mv}, out_dir / 'best.pt')
        if epoch % 50 == 0 or '[BEST]' in msg or epoch == args.epochs - 1:
            print(msg)
        log_lines.append(msg)
        if epoch % 50 == 0:
            with open(log_path, 'w') as f: f.write('\n'.join(log_lines) + '\n')
        if (epoch - best_epoch) >= args.patience:
            print(f'Early stopping at epoch {epoch} (best at {best_epoch}, val={best_val:.4e})')
            log_lines.append(f'Early stopping at epoch {epoch} (best at {best_epoch}, val={best_val:.4e})')
            break

    with open(log_path, 'w') as f: f.write('\n'.join(log_lines) + '\n')
    print(f'Done. Best val: {best_val:.4e} at epoch {best_epoch}')


if __name__ == '__main__':
    main()
