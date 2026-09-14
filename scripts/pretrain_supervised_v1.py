#!/usr/bin/env python
"""Supervised multi-topology pretraining for circuit GNNs.

Trains a TowerGENConv (no towers, full heads) on the 4-topology pretrain
corpus with V + I + gm + gds supervision (DC gain head disabled — no per-
topology formula available in this corpus). Saves the FULL state_dict
(backbone + all heads) so all ~2.79M params get pretrained.

Architecture: identical to configs/gnn/pretrain/pretrain_supervised_gine_v1.yaml
Data:         datasets/opamp_3stage_pretrain_combined (4 topologies, 16k train + 4k val)
Loss:         V (z-score) + I (log10 z-score) + gm (log10 z-score) + gds (log10 z-score)

This is the "ImageNet-style" baseline: pretrain on a labeled source corpus,
fine-tune on a labeled target. Standard transfer learning for circuits.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuitgnn.data.pretrain_loader import PretrainCombinedLoader
from circuitgnn.training.checkpoint import create_model_from_args
from circuitgnn.training.config import parse_training_config
from circuitgnn.training.losses import compute_kcl_loss
from circuitgnn.training.pretrain import (
    make_progress_bar,
    prepare_out_dir,
    write_log_line,
)


_LOG_FLOOR = 1e-12


def attach_normalization_to_batch(batch, vdc_mean, vdc_std,
                                  current_mean, current_std,
                                  ss_gm_mean, ss_gm_std,
                                  ss_gds_mean, ss_gds_std):
    device = batch.x.device
    for name, val in [
        ('vdc_mean', vdc_mean), ('vdc_std', vdc_std),
        ('current_mean', current_mean), ('current_std', current_std),
        ('ss_gm_mean', ss_gm_mean), ('ss_gm_std', ss_gm_std),
        ('ss_gds_mean', ss_gds_mean), ('ss_gds_std', ss_gds_std),
    ]:
        setattr(batch, name, torch.tensor(val, device=device))


def compute_norm_stats(loader):
    """Single pass through train to compute V, I (log), gm/gds (log) stats."""
    Vs, log_Is, log_gms, log_gdss = [], [], [], []
    for b in loader:
        if getattr(b, 'node_voltage_targets', None) is not None:
            Vs.append(b.node_voltage_targets.flatten().cpu())
        # Currents: log10(|I| + eps), only at terminals where has_current_mask is true
        if (getattr(b, 'node_current_targets', None) is not None
                and getattr(b, 'has_current_mask', None) is not None):
            i_t = b.node_current_targets
            mask = b.has_current_mask
            if mask.any():
                vals = i_t[mask].abs().clamp_min(_LOG_FLOOR).log10()
                log_Is.append(vals.cpu())
        # gm/gds: stored per-node already as log10 — only meaningful on drain
        if (getattr(b, 'node_log_gm', None) is not None
                and getattr(b, 'mosfet_drain_mask', None) is not None):
            dm = b.mosfet_drain_mask
            if dm.any():
                log_gms.append(b.node_log_gm[dm].cpu())
                log_gdss.append(b.node_log_gds[dm].cpu())

    def stats(parts, fallback_mean, fallback_std):
        if not parts:
            return fallback_mean, fallback_std
        x = torch.cat(parts)
        return x.mean().item(), max(x.std().item(), 1e-6)

    v_mean, v_std = stats(Vs, 0.9, 0.5)
    i_mean, i_std = stats(log_Is, -5.55, 1.42)
    gm_mean, gm_std = stats(log_gms, -4.43, 1.44)
    gds_mean, gds_std = stats(log_gdss, -5.89, 1.89)
    return dict(
        v_mean=v_mean, v_std=v_std,
        i_mean=i_mean, i_std=i_std,
        gm_mean=gm_mean, gm_std=gm_std,
        gds_mean=gds_mean, gds_std=gds_std,
    )


def supervised_v_loss(out, batch, v_mean, v_std, mask, loss_type='mse', huber_delta=1.0):
    pred = out['node_voltages']
    target = (batch.node_voltage_targets - v_mean) / v_std
    if mask is not None and mask.any():
        p, t = pred[mask], target[mask]
    else:
        p, t = pred, target
    if loss_type == 'huber':
        return F.huber_loss(p, t, delta=huber_delta)
    elif loss_type == 'mae':
        return (p - t).abs().mean()
    return F.mse_loss(p, t)


def supervised_i_loss(out, batch, i_mean, i_std):
    """log10(|I|) z-score MSE, masked by has_current_mask & where pred exists."""
    if 'node_currents' not in out:
        return None
    if getattr(batch, 'node_current_targets', None) is None:
        return None
    if getattr(batch, 'has_current_mask', None) is None:
        return None
    mask = batch.has_current_mask
    if not mask.any():
        return None
    pred = out['node_currents']
    i_t = batch.node_current_targets
    target = (i_t.abs().clamp_min(_LOG_FLOOR).log10() - i_mean) / i_std
    return F.mse_loss(pred[mask], target[mask])


def _drain_idx_batched(batch):
    """Compute drain node indices in batched space from per-graph mosfet_info."""
    mi = batch.mosfet_info
    mp = batch.mosfet_ptr
    ptr = batch.ptr
    num_mos = mi.shape[0]
    # graph_idx[k] = which graph MOSFET k belongs to
    graph_idx = torch.bucketize(
        torch.arange(num_mos, device=mi.device),
        mp[1:].to(mi.device), right=True,
    )
    offsets = ptr[graph_idx]
    return mi[:, 1].long() + offsets


def supervised_ss_loss(out, batch, gm_mean, gm_std, gds_mean, gds_std):
    """gm + gds log10 z-score MSE on drain nodes."""
    losses = {}
    dm = getattr(batch, 'mosfet_drain_mask', None)
    if dm is None or not dm.any():
        return losses
    drain_idx = _drain_idx_batched(batch)
    if 'mosfet_gm_pred' in out and getattr(batch, 'node_log_gm', None) is not None:
        pred = out['mosfet_gm_pred']
        target_node = (batch.node_log_gm - gm_mean) / gm_std
        if pred.dim() == 1 and pred.shape[0] != target_node.shape[0]:
            target_per_mos = target_node[drain_idx]
            losses['gm'] = F.mse_loss(pred, target_per_mos)
        else:
            losses['gm'] = F.mse_loss(pred[dm], target_node[dm])
    if 'mosfet_gds_pred' in out and getattr(batch, 'node_log_gds', None) is not None:
        pred = out['mosfet_gds_pred']
        target_node = (batch.node_log_gds - gds_mean) / gds_std
        if pred.dim() == 1 and pred.shape[0] != target_node.shape[0]:
            target_per_mos = target_node[drain_idx]
            losses['gds'] = F.mse_loss(pred, target_per_mos)
        else:
            losses['gds'] = F.mse_loss(pred[dm], target_node[dm])
    return losses


def _save_curves(history: dict, out_path) -> None:
    """Save a multi-panel training curve PNG. Lazy import matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib not installed
        print(f'  [curve] skip — matplotlib not available: {e}')
        return
    if not history.get('epoch'):
        return
    eps = history['epoch']
    panels = [
        ('Total loss', [('train', history.get('tr_total')),
                        ('val', history.get('val_total')),
                        ('val_sched (no KCL)', history.get('val_sched'))]),
        ('Voltage loss', [('train', history.get('tr_v')),
                          ('val', history.get('val_v'))]),
        ('Current loss', [('train', history.get('tr_i')),
                          ('val', history.get('val_i'))]),
        ('SS gm loss', [('train', history.get('tr_gm')),
                        ('val', history.get('val_gm'))]),
        ('SS gds loss', [('train', history.get('tr_gds')),
                         ('val', history.get('val_gds'))]),
        ('KCL loss', [('train', history.get('tr_kcl')),
                      ('val', history.get('val_kcl'))]),
        ('LR', [('lr', history.get('lr'))]),
    ]
    # Per-topo V MAE — one panel
    topo_keys = sorted(k for k in history if k.startswith('val_v_mae_mV_'))
    if topo_keys:
        topo_panel = ('Per-topo V MAE (mV)',
                      [(k.replace('val_v_mae_mV_', ''), history[k]) for k in topo_keys])
        panels.append(topo_panel)

    npanels = sum(1 for (_, lines) in panels if any(v for _, v in lines))
    cols = 4
    rows = (npanels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    pi = 0
    for title, lines in panels:
        if not any(v for _, v in lines):
            continue
        ax = axes[pi // cols][pi % cols]
        for label, ys in lines:
            if ys is None or len(ys) == 0:
                continue
            ax.plot(eps[: len(ys)], ys, label=label, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.set_yscale('log' if 'loss' in title.lower() or title == 'LR' else 'linear')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
        pi += 1
    # Hide unused axes
    for j in range(pi, rows * cols):
        axes[j // cols][j % cols].axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--warmup-epochs', type=int, default=50)
    p.add_argument('--topology', type=str, default=None,
                   help='Restrict training/val to a single topology from the '
                        'combined corpus (per-topology baseline mode). '
                        'Default: None = use all 4 topologies (joint training).')
    p.add_argument('--max-per-topo', type=int, default=None,
                   help='Few-shot mode: cap each topology to the first N samples. '
                        'For joint, gives 4N total. For per-topo, gives N total.')
    p.add_argument('--dataset-dir', type=str,
                   default='datasets/opamp_3stage_pretrain_combined',
                   help='Directory containing dataset_train.pkl / dataset_val.pkl. '
                        'Use opamp_3stage_pretrain_combined_5topo for the 5-topology version.')
    p.add_argument('--exclude-topology', type=str, default=None,
                   help='Hold out one topology from TRAINING. Val set still includes '
                        'all topologies so zero-shot performance on the held-out topo '
                        'is tracked via per-topo V MAE in the val log.')
    p.add_argument('--init-from', type=str, default=None,
                   help='Path to a best.pt checkpoint to initialize weights from '
                        '(for fine-tuning from a zero-shot or pretrained model). '
                        'BN running stats are reset on load to match new data distribution.')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Loss weights from config
    loss_cfg = cfg.get('loss', {})
    w_v = float(loss_cfg.get('v_weight', 1.0))
    v_loss_type = str(loss_cfg.get('v_loss_type', loss_cfg.get('type', 'mse')))
    v_huber_delta = float(loss_cfg.get('huber_delta', 1.0))
    w_i = float(loss_cfg.get('current_weight', 1.0))
    ss_cfg = loss_cfg.get('ss_loss', {})
    w_gm = float(ss_cfg.get('gm_weight', 1.0))
    w_gds = float(ss_cfg.get('gds_weight', 1.0))
    w_kcl = float(loss_cfg.get('kcl_weight', 0.0))
    kcl_start = int(loss_cfg.get('kcl_start_epoch', 0))
    kcl_warmup = int(loss_cfg.get('kcl_warmup_epochs', 0))
    # When fine-tuning from a checkpoint, KCL was already active during the
    # init model's training. Restarting the warmup forces the model to "forget"
    # KCL for the first kcl_start epochs, then re-learn it — degrades best
    # val_sched. Skip the warmup when init_from is set.
    if getattr(args, 'init_from', None) is not None:
        if kcl_start > 0 or kcl_warmup > 0:
            print(f'  [init_from] disabling KCL warmup: kcl_start {kcl_start}->0, kcl_warmup {kcl_warmup}->0')
        kcl_start = 0
        kcl_warmup = 0
    kcl_gt_filter = float(loss_cfg.get('kcl_gt_filter', 0.0))

    # Smooth-min/max gm physics loss (uses PREDICTED V — needs warmup)
    smaxt_cfg = loss_cfg.get('smaxt_gm_loss', {})
    w_smaxt = float(smaxt_cfg.get('weight', 0.0))
    smaxt_start = int(smaxt_cfg.get('start_epoch', 0))
    smaxt_warmup = int(smaxt_cfg.get('warmup_epochs', 0))
    print(f'  smaxt_gm physics: weight={w_smaxt}, start_epoch={smaxt_start}, warmup={smaxt_warmup}')

    print(f'Loss weights: V={w_v} ({v_loss_type}, δ={v_huber_delta}), I={w_i}, gm={w_gm}, gds={w_gds}, KCL={w_kcl} '
          f'(start_ep={kcl_start}, warmup={kcl_warmup})')

    # Build args namespace mimicking train_v3.py for create_model_from_args
    cli_args = argparse.Namespace()
    parsed = parse_training_config(cfg)
    for k, v in parsed.items():
        if v is not None:
            setattr(cli_args, k, v)
    cli_args.device = str(device)
    cli_args.dataset = args.dataset_dir
    cli_args.predict_currents = True

    # ── Data loaders
    if args.topology:
        print(f'Loading SINGLE-topology mode: {args.topology}')
    else:
        print('Loading multi-topology corpus (joint training)...')
    t0 = time.time()
    train_loader = PretrainCombinedLoader(
        f'{args.dataset_dir}/dataset_train.pkl',
        batch_size=args.batch_size, device=device, shuffle=True,
        topology_filter=args.topology,
        max_per_topo=args.max_per_topo,
        exclude_topology=args.exclude_topology,  # zero-shot: drop this from train
    )
    # Val: always include ALL topologies (so zero-shot perf on the held-out
    # topo is tracked via per-topo V MAE). Val batch size adjusted if joint
    # train excluded one topo: train sees N topos, val sees N+1 topos →
    # need val batch_size divisible by val's num_topos.
    val_batch_size = args.batch_size
    if args.exclude_topology is not None and args.topology is None:
        # Joint train minus 1 topo. Train per-topo = batch_size / (n-1).
        # Val has n topos → adjust to keep per-topo similar.
        train_per_topo = args.batch_size // train_loader.num_topos
        val_num_topos = train_loader.num_topos + 1
        val_batch_size = train_per_topo * val_num_topos
        print(f'  val: adjusting batch_size {args.batch_size} -> {val_batch_size} '
              f'for {val_num_topos}-topo val (per_topo={train_per_topo})')
    val_loader = PretrainCombinedLoader(
        f'{args.dataset_dir}/dataset_val.pkl',
        batch_size=val_batch_size, device=device, shuffle=False, drop_last=False,
        topology_filter=args.topology,
        # NOTE: val is NOT capped, NOT excluded — full eval including held-out.
    )
    print(f'  loaded in {time.time()-t0:.1f}s; train={len(train_loader)} val={len(val_loader)} '
          f'(topos: {train_loader.topo_names})')

    # ── Normalization stats from train
    print('Computing normalization stats from train (V, I, gm, gds)...')
    stats = compute_norm_stats(train_loader)
    print(f'  V: mean={stats["v_mean"]:.4f}, std={stats["v_std"]:.4f}')
    print(f'  I (log10): mean={stats["i_mean"]:.4f}, std={stats["i_std"]:.4f}')
    print(f'  gm (log10): mean={stats["gm_mean"]:.4f}, std={stats["gm_std"]:.4f}')
    print(f'  gds (log10): mean={stats["gds_mean"]:.4f}, std={stats["gds_std"]:.4f}')

    # DC gain norm stats from training set (in dB; filter by ac_valid)
    w_dc = float(loss_cfg.get('dc_gain_loss', {}).get('weight', 0.0))
    dc_gain_mean = 0.0; dc_gain_std = 1.0
    if w_dc > 0:
        dc_vals = []
        for _b in train_loader:
            if hasattr(_b, 'ac_dc_gain') and hasattr(_b, 'ac_valid'):
                v = _b.ac_valid.bool()
                if v.any():
                    dc_vals.append(_b.ac_dc_gain[v].detach().cpu())
        if dc_vals:
            dc_t = torch.cat(dc_vals)
            dc_gain_mean = float(dc_t.mean().item())
            dc_gain_std = float(dc_t.std().clamp(min=1e-3).item())
            print(f'  dc_gain (dB): mean={dc_gain_mean:.3f}, std={dc_gain_std:.3f}, n_valid={dc_t.numel()}')
        else:
            print('  WARN: no valid DC gain samples in train set; dc_gain_loss disabled')
            w_dc = 0.0

    # ── Build model
    sample = next(iter(train_loader))
    node_dim = sample.x.shape[-1]
    type_dim = sample.type_tens.shape[-1]
    total_input_dim = node_dim + type_dim
    print(f'node_feature_dim={node_dim}, type_feature_dim={type_dim}, total={total_input_dim}')

    model, model_class = create_model_from_args(cli_args, total_input_dim, device)
    print(f'Built {model_class}, params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    # Optional: initialize weights from a previously trained checkpoint.
    # Used for fine-tuning from a zero-shot or pretrained model. BN running
    # stats reset because the new dataset's activation distribution may differ.
    init_epoch_offset = 0  # Used to make warmup-gated components see post-warmup state
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        init_epoch_offset = int(ck.get('epoch', 99999))
        sd = ck['model_state_dict']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_bn = 0
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                if m.running_mean is not None:
                    m.running_mean.zero_()
                if m.running_var is not None:
                    m.running_var.fill_(1.0)
                if m.num_batches_tracked is not None:
                    m.num_batches_tracked.zero_()
                n_bn += 1
        print(f'Loaded init weights from {args.init_from} (epoch {ck.get("epoch", "?")}, '
              f'val_loss {ck.get("val_loss", "?")})')
        print(f'  missing keys: {len(missing)}, unexpected: {len(unexpected)}, BN reset: {n_bn} layers')

    if hasattr(model, 'set_normalization_stats'):
        model.set_normalization_stats(
            vdc_mean=stats['v_mean'], vdc_std=stats['v_std'],
            ss_gm_mean=stats['gm_mean'], ss_gm_std=stats['gm_std'],
            ss_gds_mean=stats['gds_mean'], ss_gds_std=stats['gds_std'],
            current_mean=stats['i_mean'], current_std=stats['i_std'],
        )
        print('Normalization stats set on model.')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    warmup_epochs = int(args.warmup_epochs)
    sch_cfg = cfg.get('scheduler', {})
    sch_type = str(sch_cfg.get('type', 'plateau')).lower()
    sch_min_lr = float(sch_cfg.get('min_lr', 1e-6))

    if sch_type == 'cosine':
        # Linear warmup + cosine decay to min_lr over args.epochs
        import math
        min_lr_ratio = sch_min_lr / args.lr
        def cosine_lr(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs       # linear ramp 0 → 1
            progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
            cos = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))  # 1 → 0
            return max(min_lr_ratio, cos)                # floor at min_lr
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=cosine_lr)
        print(f'LR schedule: linear warmup {warmup_epochs} ep → cosine to min_lr={sch_min_lr} over {args.epochs} epochs (peak={args.lr})')
    else:
        sch_patience = int(sch_cfg.get('patience', 150))
        sch_factor = float(sch_cfg.get('factor', 0.5))
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', patience=sch_patience,
            factor=sch_factor, min_lr=sch_min_lr,
        )
        print(f'LR warmup: {warmup_epochs} epochs (linear 0 → {args.lr})')
        print(f'Plateau scheduler: patience={sch_patience} factor={sch_factor} min_lr={sch_min_lr}')

    out_dir = Path(f"{args.dataset_dir}/experiments/{args.name}")
    log_path = prepare_out_dir(out_dir, cfg)
    log_lines = []
    history: dict = {}  # epoch-indexed metric arrays for curve plotting
    best_val = float('inf')
    best_epoch = -1
    es_patience = int(cfg.get('training', {}).get('early_stopping_patience', 500))
    print(f'Early stopping patience: {es_patience} epochs')

    pbar = make_progress_bar(args.epochs, len(train_loader), desc='SupPretrain')

    def attach_norm(b):
        attach_normalization_to_batch(
            b, stats['v_mean'], stats['v_std'],
            current_mean=stats['i_mean'], current_std=stats['i_std'],
            ss_gm_mean=stats['gm_mean'], ss_gm_std=stats['gm_std'],
            ss_gds_mean=stats['gds_mean'], ss_gds_std=stats['gds_std'],
        )

    def kcl_active_weight(epoch):
        if w_kcl <= 0 or epoch < kcl_start:
            return 0.0
        if kcl_warmup <= 0:
            return w_kcl
        ramp = min(1.0, (epoch - kcl_start) / kcl_warmup)
        return w_kcl * ramp

    def smaxt_active_weight(epoch):
        if w_smaxt <= 0 or epoch < smaxt_start:
            return 0.0
        if smaxt_warmup <= 0:
            return w_smaxt
        ramp = min(1.0, (epoch - smaxt_start) / smaxt_warmup)
        return w_smaxt * ramp

    def step_losses(out, batch, epoch):
        internal_mask = ~batch.known_voltage_mask
        v_l = supervised_v_loss(out, batch, stats['v_mean'], stats['v_std'], internal_mask,
                                loss_type=v_loss_type, huber_delta=v_huber_delta)
        i_l = supervised_i_loss(out, batch, stats['i_mean'], stats['i_std'])
        ss_l = supervised_ss_loss(
            out, batch, stats['gm_mean'], stats['gm_std'],
            stats['gds_mean'], stats['gds_std'],
        )
        total = w_v * v_l
        parts = {'v': v_l.item()}
        if i_l is not None:
            total = total + w_i * i_l
            parts['i'] = i_l.item()
        if 'gm' in ss_l:
            total = total + w_gm * ss_l['gm']
            parts['gm'] = ss_l['gm'].item()
        if 'gds' in ss_l:
            total = total + w_gds * ss_l['gds']
            parts['gds'] = ss_l['gds'].item()
        kw = kcl_active_weight(epoch)
        if kw > 0 and 'node_currents' in out:
            # Build gt_currents in normalized log10 space (matches model output)
            i_t = batch.node_current_targets
            gt_norm = (i_t.abs().clamp_min(_LOG_FLOOR).log10() - stats['i_mean']) / stats['i_std']
            kcl_l, _ = compute_kcl_loss(
                node_currents=out['node_currents'],
                edge_index=batch.edge_index,
                num_terminals=batch.num_terminals,
                train_mask=batch.train_mask,
                ptr=batch.ptr,
                terminal_current_sign=batch.terminal_current_sign,
                current_mean=stats['i_mean'],
                current_std=stats['i_std'],
                gt_currents=gt_norm,
                kcl_include_mask=batch.kcl_include_mask,
                kcl_gt_filter=kcl_gt_filter,
            )
            total = total + kw * kcl_l
            parts['kcl'] = kcl_l.item()
        if w_dc > 0 and 'dc_gain_pred' in out and hasattr(batch, 'ac_dc_gain'):
            dc_pred = out['dc_gain_pred']  # z-scored by convention
            tgt = batch.ac_dc_gain.to(dc_pred.device)
            mask = batch.ac_valid.to(dc_pred.device) if hasattr(batch, 'ac_valid') else torch.ones_like(tgt, dtype=torch.bool)
            mask = mask & (tgt > 0)  # exclude unstable / negative DC gain samples
            if mask.any():
                tgt_z = (tgt - dc_gain_mean) / max(dc_gain_std, 1e-3)
                dc_l = F.mse_loss(dc_pred[mask], tgt_z[mask])
                total = total + w_dc * dc_l
                parts['dc'] = dc_l.item()

        sw = smaxt_active_weight(epoch)
        if sw > 0 and 'mosfet_gm_pred' in out and 'node_currents' in out and getattr(batch, 'node_mosfet_vth', None) is not None:
            from circuitgnn.training.losses import compute_smaxt_gm_loss
            smaxt_l = compute_smaxt_gm_loss(
                ss_gm_pred=out['mosfet_gm_pred'],
                pred_currents=out['node_currents'],
                full_voltage_pred=out['node_voltages'],
                mosfet_info=batch.mosfet_info,
                node_mosfet_vth=batch.node_mosfet_vth,
                ptr=batch.ptr,
                vdc_mean=stats['v_mean'], vdc_std=stats['v_std'],
                current_mean=stats['i_mean'], current_std=stats['i_std'],
                ss_gm_mean=stats['gm_mean'], ss_gm_std=stats['gm_std'],
                mosfet_ptr=getattr(batch, 'mosfet_ptr', None),
            )
            total = total + sw * smaxt_l
            parts['smaxt'] = smaxt_l.item()
        return total, parts

    for epoch in range(args.epochs):
        if epoch < warmup_epochs:
            wf = (epoch + 1) / warmup_epochs
            for pg in opt.param_groups:
                pg['lr'] = args.lr * wf
        if hasattr(model, 'current_epoch'):
            # NOTE: tested setting current_epoch += init_epoch_offset for fine-tune
            # to skip loop attention warmup. Made things WORSE (24.7mV vs 17.5mV
            # with KCL-only fix). Likely because BN running stats are reset on load
            # but loop attention layers contain BN; activating them immediately
            # propagates bad features. Keep natural warmup for forward components.
            model.current_epoch = epoch
        model.train()
        tr_total = 0.0
        tr_parts = {}
        n_steps = 0
        for batch in train_loader:
            attach_norm(batch)
            out = model(batch)
            loss, parts = step_losses(out, batch, epoch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_total += loss.item()
            for k, v in parts.items():
                tr_parts[k] = tr_parts.get(k, 0.0) + v
            n_steps += 1
            pbar.set_postfix(loss=f'{loss.item():.3e}', refresh=False)
            pbar.update(1)
        tr_total /= n_steps
        tr_parts = {k: v / n_steps for k, v in tr_parts.items()}

        # Validate
        model.eval()
        val_total = 0.0
        val_parts = {}
        # Per-topology V MAE in mV for diagnostic. Keyed by topology name.
        per_topo_v_sum = {}
        per_topo_v_count = {}
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                attach_norm(batch)
                out = model(batch)
                lv, parts = step_losses(out, batch, epoch)
                val_total += lv.item()
                for k, v in parts.items():
                    val_parts[k] = val_parts.get(k, 0.0) + v
                n_val += 1
                # Per-topo V MAE: slice predictions/targets by topology range.
                slices = getattr(batch, 'topo_node_slices', None)
                if slices is not None and getattr(batch, 'node_voltage_targets', None) is not None:
                    v_pred = out['node_voltages']
                    v_tgt = batch.node_voltage_targets
                    internal_mask = ~batch.known_voltage_mask
                    for t, (start, end, _n, _g) in slices.items():
                        m = internal_mask[start:end]
                        if not m.any():
                            continue
                        # Denormalize back to volts: pred is z-scored, target is raw V.
                        pred_v = v_pred[start:end][m] * stats['v_std'] + stats['v_mean']
                        tgt_v = v_tgt[start:end][m]
                        mae = (pred_v - tgt_v).abs().sum().item()
                        per_topo_v_sum[t] = per_topo_v_sum.get(t, 0.0) + mae
                        per_topo_v_count[t] = per_topo_v_count.get(t, 0) + int(m.sum().item())
        val_total /= n_val
        val_parts = {k: v / n_val for k, v in val_parts.items()}
        per_topo_v_mae = {
            t: 1000.0 * per_topo_v_sum[t] / per_topo_v_count[t]
            for t in per_topo_v_sum if per_topo_v_count[t] > 0
        }

        cur_lr = opt.param_groups[0]['lr']
        parts_str = ' '.join(f'{k}={v:.3e}' for k, v in val_parts.items())
        topo_str = ''
        if per_topo_v_mae:
            topo_str = ' | per-topo V MAE (mV): ' + ', '.join(
                f'{t}={per_topo_v_mae[t]:.1f}' for t in sorted(per_topo_v_mae)
            )
        msg = (f'epoch {epoch:4d}: lr={cur_lr:.2e} tr={tr_total:.4e} | val={val_total:.4e} '
               f'[{parts_str}]{topo_str}')
        # NOTE: epoch-end handling deliberately does NOT use the shared
        # scheduler_step_and_save_best() skeleton from
        # circuitgnn.training.pretrain because this script diverges: the
        # best/scheduler metric is avg per-topo V MAE (not a val loss), the
        # scheduler may be cosine (step() takes no metric), best_epoch is
        # tracked for early stopping and the checkpoint schema adds
        # val_loss_sched/avg_v_mae_mV/per_topo_v_mae/val_parts/norm_stats.
        # Best-model save and LR scheduler both use avg per-topo V MAE — directly
        # tracks the headline metric, immune to i_loss/KCL trade-off that fooled
        # val_sched into triggering ES too early.
        val_sched = sum(
            val_parts.get(k, 0.0) * w
            for k, w in [('v', w_v), ('i', w_i), ('gm', w_gm), ('gds', w_gds)]
        )
        avg_v_mae = (sum(per_topo_v_mae.values()) / len(per_topo_v_mae)
                     if per_topo_v_mae else val_parts.get('v', val_sched))
        if sch_type == 'cosine':
            sch.step()
        else:
            sch.step(avg_v_mae)
        if avg_v_mae < best_val:
            best_val = avg_v_mae
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_total,
                'val_loss_sched': val_sched,
                'avg_v_mae_mV': avg_v_mae,
                'per_topo_v_mae': per_topo_v_mae,
                'val_parts': val_parts,
                'config': cfg,
                'norm_stats': stats,
            }, out_dir / 'best.pt')
            msg += ' [BEST]'
        # Track per-epoch metrics for the curve plot regardless of logging cadence.
        history.setdefault('epoch', []).append(epoch)
        history.setdefault('lr', []).append(cur_lr)
        history.setdefault('tr_total', []).append(tr_total)
        history.setdefault('val_total', []).append(val_total)
        history.setdefault('val_sched', []).append(val_sched)
        for k, v in tr_parts.items():
            history.setdefault(f'tr_{k}', []).append(v)
        for k, v in val_parts.items():
            history.setdefault(f'val_{k}', []).append(v)
        for t, mv in per_topo_v_mae.items():
            history.setdefault(f'val_v_mae_mV_{t}', []).append(mv)

        # Console + log file: only every 50 epochs OR when a new best was found.
        # No need to write the log file every epoch — once per 50 is enough.
        if epoch % 50 == 0 or epoch == args.epochs - 1 or '[BEST]' in msg:
            write_log_line(msg, log_lines, log_path)
            _save_curves(history, out_dir / 'training_curve.png')

        if (epoch - best_epoch) >= es_patience:
            stop_msg = (f'Early stopping at epoch {epoch}: no improvement '
                        f'for {es_patience} epochs (best at {best_epoch}, '
                        f'val={best_val:.4e})')
            write_log_line(stop_msg, log_lines, log_path)
            break

    pbar.close()
    print(f'Done. Best val: {best_val:.4e} at epoch {best_epoch}')


if __name__ == '__main__':
    main()
