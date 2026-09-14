#!/usr/bin/env python
"""Pretrain TowerGENConv-style backbone via masked MOSFET-feature prediction.

Loads the combined 4-topology dataset, runs the stratified pretraining loader,
and trains a backbone-only model with a small mask head. Saves backbone-prefix
weights for fine-tuning into TowerGENConv (load with strict=False).

Usage:
    python scripts/pretrain_v1.py --config configs/gnn/pretrain/pretrain_combined.yaml \\
                                  --name pretrain_combined_v1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# Make `src` importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuitgnn.data.pretrain_loader import PretrainCombinedLoader
from circuitgnn.gnn.architectures.pretrain_backbone import build_pretrain_backbone
from circuitgnn.training.pretrain import (
    build_adam,
    build_plateau_scheduler,
    make_progress_bar,
    prepare_out_dir,
    pretrain_epoch,
    pretrain_validate,
    save_final_checkpoint,
    scheduler_step_and_save_best,
    write_log_line,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_path = cfg['data']['train_path']
    val_path = cfg['data']['val_path']
    batch_size = cfg['training']['batch_size']

    print(f'Loading train from {train_path} ...')
    t0 = time.time()
    train_loader = PretrainCombinedLoader(
        train_path, batch_size=batch_size, device=device,
        shuffle=True, drop_last=True,
    )
    print(f'  loaded in {time.time()-t0:.1f}s; {len(train_loader)} batches/epoch')

    print(f'Loading val from {val_path} ...')
    t0 = time.time()
    val_loader = PretrainCombinedLoader(
        val_path, batch_size=batch_size, device=device,
        shuffle=False, drop_last=False,
    )
    print(f'  loaded in {time.time()-t0:.1f}s; {len(val_loader)} batches/epoch')

    # Build model — node_feature_dim from x.shape[-1], type_feature_dim from type_tens
    sample_batch = next(iter(train_loader))
    node_dim = sample_batch.x.shape[-1]
    type_dim = sample_batch.type_tens.shape[-1] if sample_batch.type_tens is not None else 0
    print(f'node_feature_dim={node_dim}, type_feature_dim={type_dim}')

    model = build_pretrain_backbone(
        cfg, node_dim, type_dim,
        backbone_layers_default=6,
        mask_target_dim=cfg['pretrain'].get('mask_target_dim', 4),
        mask_head_hidden=cfg['pretrain'].get('mask_head_hidden', 128),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model params: {n_params/1e6:.2f}M')

    optimizer = build_adam(model, cfg)
    scheduler = build_plateau_scheduler(optimizer, cfg, default_patience=100)

    pre_cfg = cfg['pretrain']
    mask_ratio = pre_cfg.get('mask_ratio', 0.15)
    mask_strategy = pre_cfg.get('strategy', 'random')   # 'random' | 'mirror' | 'stage'
    print(f'Mask strategy: {mask_strategy}, ratio: {mask_ratio}')
    if mask_strategy == 'mirror':
        print(f'  mirror_groups: {len(train_loader.batched_mirror_groups)} groups')
    elif mask_strategy == 'stage':
        print(f'  stage_groups: {len(train_loader.batched_stage_groups)} groups')

    epochs = cfg['training']['epochs']
    grad_clip = cfg['training'].get('gradient_clip', 1.0)
    val_freq = cfg['training'].get('val_freq', 1)
    log_interval = cfg['output'].get('log_interval', 1)

    # Output dir under datasets/<dataset_name>/experiments/<name>/
    dataset_name = Path(cfg['data']['train_path']).parent.name
    out_dir = Path(f'datasets/{dataset_name}/experiments/{args.name}')
    log_path = prepare_out_dir(out_dir, cfg)

    best_val_loss = float('inf')

    print(f'Starting pretraining: {epochs} epochs, mask_ratio={mask_ratio}')
    pbar = make_progress_bar(epochs, len(train_loader), desc='Pretrain')
    log_lines = []

    for epoch in range(epochs):
        train_loss = pretrain_epoch(
            model, train_loader, optimizer,
            mask_ratio=mask_ratio,
            grad_clip=grad_clip,
            device=device, epoch=epoch,
            progress=pbar,
            strategy=mask_strategy,
        )

        cur_lr = optimizer.param_groups[0]['lr']
        msg = f'epoch {epoch:4d}: train_loss={train_loss:.4e} lr={cur_lr:.2e}'
        if (epoch + 1) % val_freq == 0:
            val = pretrain_validate(model, val_loader, mask_ratio, epoch=epoch,
                                     strategy=mask_strategy)
            msg += f' | val_loss={val["loss"]:.4e} val_mae={val["mae"]:.4e}'
            best_val_loss, msg = scheduler_step_and_save_best(
                val['loss'], best_val_loss, epoch, model, cfg, out_dir, msg,
                scheduler=scheduler, extra_ckpt={'val_loss': val['loss']},
            )

        if (epoch + 1) % log_interval == 0:
            write_log_line(msg, log_lines, log_path)

    pbar.close()
    print(f'Done. Best val loss: {best_val_loss:.4e}')
    save_final_checkpoint(out_dir, epochs, model, cfg)

    # Auto-plot curves
    try:
        from scripts.plot_pretrain import plot as plot_curves
        plot_curves(log_path, out_dir / 'training_curves.png')
    except Exception as e:
        print(f'[warn] plot failed: {e}')


if __name__ == '__main__':
    main()
