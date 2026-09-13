#!/usr/bin/env python
"""Physics-Informed Neural Network (PINN) pretraining — multi-topology.

Canonical PINN formulation: supervised V loss + physics constraints (KCL on
LUT-derived MOSFET currents). Trains a TowerGENConv-compatible backbone on
the 4-topology pretrain corpus.

Stability features:
  - Architectural V bound: V_pred = sigmoid(raw) * VDD, always in [0, VDD]
  - Bias-init: V head outputs ~VDD/2 at start (sensible mid-rail guess)
  - KCL warmup: physics loss ramps from 0 → 1 over `kcl_warmup_epochs`
  - Boundary anchor: known V's (VDD, GND, Vin) injected via known_voltage_mask
  - V supervision: anchor on internal node V's

Loss:
    L = w_v · MSE(V_pred_internal, V_target_internal_zscored)
      + w_kcl(t) · ||KCL_residual||²    (warmed up over time)

where KCL_residual at each internal net node = sum of LUT-derived MOSFET
currents flowing into that node. Currents are NOT predicted — they come
directly from BSIM physics applied to the predicted V.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuitgnn.data.pretrain_loader import PretrainCombinedLoader
from circuitgnn.gnn.architectures.pretrain_backbone import PretrainBackbone
from circuitgnn.gnn.components.lut_id_query import LUTIdQuery


class PhysicsPretrainModel(nn.Module):
    """Backbone + V head with architectural bound and bias init."""

    def __init__(self, backbone: PretrainBackbone, hidden_dim: int, vdd: float):
        super().__init__()
        self.backbone = backbone
        self.vdd = vdd
        # V head outputs raw scalar, then sigmoid*vdd ensures V_pred ∈ [0, vdd].
        self.voltage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Init last layer so sigmoid(raw) ≈ 0.5 at init → V_pred ≈ vdd/2.
        last = self.voltage_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, data):
        x_in = self.backbone._get_input_features(data)
        x = self.backbone.input_linear(x_in)
        batch_vec = data.batch
        num_graphs = data.num_graphs

        _loop_ei = _loop_dtm = None
        if (self.backbone.use_loop_attention
                and self.backbone.loop_attn_layers is not None):
            _loop_ei, _loop_dtm, _ = self.backbone._get_loop_data(data)

        vn_is_mha = (
            self.backbone.virtual_node is not None
            and self.backbone.virtual_node.mode == 'mha'
        )
        topo_slices = getattr(data, 'topo_node_slices', None)

        for layer_idx, layer in enumerate(self.backbone.backbone):
            if vn_is_mha:
                if topo_slices is not None:
                    mha_out = torch.zeros_like(x)
                    for (start, end, _, _) in topo_slices.values():
                        x_slice = x[start:end]
                        bs = batch_vec[start:end] - batch_vec[start]
                        mha_out[start:end] = self.backbone.virtual_node.global_mha(
                            x_slice, bs, layer_idx=layer_idx,
                        )
                    x = x + mha_out
                else:
                    x = x + self.backbone.virtual_node.global_mha(x, batch_vec, layer_idx=layer_idx)
            x = layer(x, data.edge_index)
            if (self.backbone.loop_attn_layers is not None
                    and self.backbone.current_epoch >= self.backbone.loop_attn_warmup_epochs):
                la_layer = self.backbone.loop_attn_layers[layer_idx]
                if _loop_dtm is not None and _loop_ei is not None:
                    x_loop = la_layer(x, _loop_dtm, _loop_ei)
                    if self.backbone.loop_attn_warmup_duration > 0:
                        prog = self.backbone.current_epoch - self.backbone.loop_attn_warmup_epochs
                        alpha = min(1.0, prog / self.backbone.loop_attn_warmup_duration)
                        x = x + alpha * x_loop
                    else:
                        x = x + x_loop

        raw = self.voltage_head(x).squeeze(-1)
        v_pred_volts = torch.sigmoid(raw) * self.vdd  # bounded
        return {'node_voltages_volts': v_pred_volts, 'backbone_hidden': x}


def supervised_v_loss(v_pred_volts, v_target, known_mask):
    """MSE on internal-node V's only (boundary nodes excluded — they're forced)."""
    internal = ~known_mask
    if not internal.any():
        return torch.tensor(0.0, device=v_pred_volts.device)
    return F.mse_loss(v_pred_volts[internal], v_target[internal])


def kcl_physics_loss(v_pred_volts, data, lut: LUTIdQuery):
    """KCL on LUT-derived MOSFET currents at each internal net node."""
    N = v_pred_volts.shape[0]

    # 1. Inject known boundary V's (no gradient through these)
    known_mask = data.known_voltage_mask
    v_known = data.node_voltage_targets
    v_used = torch.where(known_mask, v_known.detach(), v_pred_volts)

    # 2. Per-MOSFET BSIM lookup
    mi = data.mosfet_info
    gate_idx = mi[:, 0]
    drain_idx = mi[:, 1]
    source_idx = mi[:, 2]
    is_nmos = mi[:, 6].long()
    bulk_idx = drain_idx + 3
    bulk_safe = torch.where(bulk_idx < N, bulk_idx, source_idx)

    Vgs = v_used[gate_idx] - v_used[source_idx]
    Vds = v_used[drain_idx] - v_used[source_idx]
    Vbs = v_used[bulk_safe] - v_used[source_idx]

    W_um = data.mosfet_wl_um[:, 0]
    L_um = data.mosfet_wl_um[:, 1]
    M = data.mosfet_m
    id_lut = lut(W_um, L_um, Vgs, Vds, Vbs, M, is_nmos)  # [M_total] amps

    # 3. KCL: scatter currents to net nodes
    # Sign convention: current INTO drain (NMOS conducts from drain to source);
    # equivalent magnitude flows OUT of source. Use signed accounting per polarity.
    nmos_factor = 2.0 * is_nmos.float() - 1.0  # nmos=+1, pmos=-1
    drain_in = id_lut * nmos_factor       # signed current INTO drain node
    source_out = id_lut * nmos_factor     # signed current OUT of source node (i.e. away)
    # KCL = sum of currents flowing INTO each net.
    net_currents = torch.zeros(N, device=v_pred_volts.device, dtype=id_lut.dtype)
    net_currents.index_add_(0, drain_idx, drain_in)
    net_currents.index_add_(0, source_idx, -source_out)
    # (Bulk is essentially an inert terminal at DC for KCL purposes — skip.)

    # 4. KCL only on internal nodes (boundary nodes absorb mismatch via supplies).
    internal = ~known_mask
    kcl_residual = net_currents[internal]
    # Convert to a unit-normalized scale: typical opamp transistor I ~ 1e-6 A
    # so squared residuals are ~1e-12. Multiply by 1e12 for SGD-friendly range.
    kcl = (kcl_residual ** 2).mean() * 1e12
    return kcl, id_lut.abs().mean().item()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def build_backbone(cfg, node_feature_dim, type_feature_dim) -> PretrainBackbone:
    m = cfg['model']
    tower = m.get('tower', {})
    vn = m.get('virtual_node', {})
    loop = tower.get('loop_attention', {})
    return PretrainBackbone(
        node_feature_dim=node_feature_dim,
        type_feature_dim=type_feature_dim,
        hidden_dim=m['hidden_dim'],
        backbone_layers=tower.get('backbone_layers', 8),
        genconv_num_layers=m.get('genconv_num_layers', 2),
        norm_type=m.get('norm_type', 'layer'),
        act_type=m.get('act_type', 'gelu'),
        dropout=m.get('dropout', 0.0),
        skip_connection=m.get('skip_connection', False),
        conv_type=m.get('conv_type', 'gin'),
        mlp_depth=m.get('mlp_depth', 3),
        use_virtual_node=vn.get('enabled', True),
        vn_mode=vn.get('mode', 'mha'),
        vn_num_heads=vn.get('num_heads', 4),
        vn_head_dim=vn.get('head_dim', 32),
        vn_gate_broadcast=vn.get('gate_broadcast', True),
        use_loop_attention=loop.get('enabled', False),
        loop_num_heads=loop.get('num_heads', 4),
        loop_head_dim=loop.get('head_dim', 32),
        loop_fusion=loop.get('fusion', 'gate'),
        loop_warmup_epochs=loop.get('warmup_epochs', 0),
        loop_warmup_duration=loop.get('warmup_duration', 0),
        mask_target_dim=4, mask_head_hidden=64,
    )


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_path = cfg['data']['train_path']
    val_path = cfg['data']['val_path']
    bs = cfg['training']['batch_size']
    print(f'Loading train: {train_path}')
    t0 = time.time()
    train_loader = PretrainCombinedLoader(train_path, batch_size=bs, device=device, shuffle=True)
    print(f'  in {time.time()-t0:.1f}s; {len(train_loader)} batches/epoch')
    val_loader = PretrainCombinedLoader(val_path, batch_size=bs, device=device, shuffle=False, drop_last=False)
    print(f'  val: {len(val_loader)} batches')

    sample = next(iter(train_loader))
    node_dim = sample.x.shape[-1]
    type_dim = sample.type_tens.shape[-1]
    print(f'node_feature_dim={node_dim}, type_feature_dim={type_dim}')

    physics_cfg = cfg.get('physics', {})
    vdd = float(physics_cfg.get('vdd', 1.8))
    v_w = float(physics_cfg.get('voltage_weight', 1.0))
    kcl_w = float(physics_cfg.get('kcl_weight', 1.0))
    kcl_warmup = int(physics_cfg.get('kcl_warmup_epochs', 30))
    lut_path = physics_cfg.get('lut_path', 'datasets/lut/lut_v2/sky130_mosfet_lut_v2_id_gm_gds.h5')
    lut = LUTIdQuery(lut_path).to(device)
    print(f'PINN setup: voltage_weight={v_w}, kcl_weight={kcl_w}, kcl_warmup={kcl_warmup} ep, vdd={vdd}V')
    print(f'Loaded LUT: {lut_path}')

    backbone = build_backbone(cfg, node_dim, type_dim)
    model = PhysicsPretrainModel(backbone, cfg['model']['hidden_dim'], vdd=vdd).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model params: {n_params/1e6:.2f}M')

    opt = torch.optim.Adam(model.parameters(),
                           lr=cfg['optimizer']['lr'],
                           weight_decay=cfg['optimizer'].get('weight_decay', 0.0))
    sch = None
    sch_cfg = cfg.get('scheduler', {})
    if sch_cfg.get('type') == 'plateau':
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min',
            patience=sch_cfg.get('patience', 30),
            factor=sch_cfg.get('factor', 0.5),
            min_lr=sch_cfg.get('min_lr', 1e-6),
        )

    epochs = cfg['training']['epochs']
    grad_clip = cfg['training'].get('gradient_clip', 1.0)
    out_dir = Path(f"datasets/{Path(train_path).parent.name}/experiments/{args.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'original_config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f)
    log_path = out_dir / 'training.log'
    log_lines = []
    best_val = float('inf')

    pbar = tqdm(total=epochs * len(train_loader), desc='PINN', dynamic_ncols=True)

    for epoch in range(epochs):
        # KCL warmup ramp 0→1 over kcl_warmup epochs
        kcl_alpha = min(1.0, epoch / max(1, kcl_warmup))
        effective_kcl_w = kcl_w * kcl_alpha

        model.train()
        backbone.current_epoch = epoch
        tr_v = tr_kcl = 0.0
        tr_id_uA = 0.0
        tr_v_mean = tr_v_std = 0.0
        n_steps = 0
        for batch in train_loader:
            out = model(batch)
            v_pred = out['node_voltages_volts']
            loss_v = supervised_v_loss(v_pred, batch.node_voltage_targets,
                                        batch.known_voltage_mask)
            loss_kcl, mean_id = kcl_physics_loss(v_pred, batch, lut)
            loss = v_w * loss_v + effective_kcl_w * loss_kcl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_v += loss_v.item()
            tr_kcl += loss_kcl.item()
            tr_id_uA += mean_id * 1e6
            tr_v_mean += v_pred.mean().item()
            tr_v_std += v_pred.std().item()
            n_steps += 1
            pbar.set_postfix(
                v=f'{loss_v.item():.3e}',
                kcl=f'{loss_kcl.item():.3e}',
                a_kcl=f'{kcl_alpha:.2f}',
                id_uA=f'{mean_id*1e6:.1f}',
                refresh=False,
            )
            pbar.update(1)
        tr_v /= n_steps; tr_kcl /= n_steps; tr_id_uA /= n_steps
        tr_v_mean /= n_steps; tr_v_std /= n_steps

        model.eval()
        backbone.current_epoch = epoch
        val_v = val_kcl = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch)
                v_pred = out['node_voltages_volts']
                lv = supervised_v_loss(v_pred, batch.node_voltage_targets, batch.known_voltage_mask).item()
                lk, _ = kcl_physics_loss(v_pred, batch, lut)
                val_v += lv
                val_kcl += lk.item()
                n_val += 1
        val_v /= n_val; val_kcl /= n_val
        # Use V loss for best-checkpoint tracking (kcl scale fluctuates with warmup)
        val_total = v_w * val_v + effective_kcl_w * val_kcl

        cur_lr = opt.param_groups[0]['lr']
        msg = (f'epoch {epoch:4d}: lr={cur_lr:.2e} a_kcl={kcl_alpha:.2f} '
               f'tr_v={tr_v:.3e} tr_kcl={tr_kcl:.3e} V[mean,std]=[{tr_v_mean:.3f},{tr_v_std:.3f}] '
               f'id_uA={tr_id_uA:.1f} '
               f'| val_v={val_v:.3e} val_kcl={val_kcl:.3e}')
        if sch is not None:
            sch.step(val_v)  # schedule on V loss only (more stable signal)
        if val_v < best_val:
            best_val = val_v
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.backbone.state_dict(),
                'val_v': val_v,
                'val_kcl': val_kcl,
                'config': cfg,
            }, out_dir / 'best.pt')
            msg += ' [BEST]'
        tqdm.write(msg, file=sys.stdout)
        log_lines.append(msg)
        with open(log_path, 'w') as f:
            f.write('\n'.join(log_lines) + '\n')

    pbar.close()
    print(f'Done. Best val_v: {best_val:.4e}')
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.backbone.state_dict(),
        'config': cfg,
    }, out_dir / 'final.pt')


if __name__ == '__main__':
    main()
