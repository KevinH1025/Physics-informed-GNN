#!/usr/bin/env python
"""MAE-style self-supervised pretraining on a single topology (openloop).

Pattern from He et al. 2022 "Masked Autoencoders Are Scalable Vision Learners":
  - Pretrain on the SAME dataset as downstream
  - Mask high fraction (75%) of input features
  - Predict masked features (no labels needed)
  - Fine-tune supervised on same dataset (with labels)

For circuits: mask 75% of MOSFETs in openloop netlists, reconstruct W/L/M.
No SPICE needed for pretrain. Only netlists.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuitgnn.data.pretrain_loader import PretrainBatch
from circuitgnn.gnn.architectures.pretrain_backbone import build_pretrain_backbone
from circuitgnn.training.pretrain import (
    apply_mosfet_mask,
    build_adam,
    build_plateau_scheduler,
    make_progress_bar,
    prepare_out_dir,
    pretrain_loss,
    scheduler_step_and_save_best,
    write_log_line,
)


class SingleTopologyMaskLoader:
    """Load prebatched openloop dataset for MAE-style masked pretraining.

    Loads variant_*.pkl batches, attaches mosfet_terminal_idx-derived terminal
    indices, applies masking on the fly during iteration.
    """
    def __init__(self, data_dir: str, device: torch.device, shuffle: bool = True):
        self.device = device
        self.shuffle = shuffle
        # Load all variants
        data_dir = Path(data_dir)
        variants = sorted(data_dir.glob('variant_*.pkl'))
        self.batches = []
        for vp in variants:
            with open(vp, 'rb') as f:
                batches = pickle.load(f)
            for b in batches:
                # Move to device, expose batched_mosfet_term in node-batched space
                b = b.to(device)
                # Build batched_mosfet_term: [4 * M_total]
                #   Per MOSFET: drain, gate, source, bulk node indices in batched space
                mi = b.mosfet_info  # [M_total, 7]
                term = torch.stack([mi[:, 1], mi[:, 0], mi[:, 2], mi[:, 1] + 3], dim=1)  # [M, 4]
                # Bulk = drain + 3 in single-graph local indexing — but in batched space,
                # we need to add per-graph node offsets. mi[:, 1] is already in batched
                # node space. However, mi[:, 1]+3 only works if drain+3 is the bulk
                # IN THE SAME GRAPH. Since each graph's terminals are contiguous, +3
                # within a graph stays in the same graph. Verify with num_terminals.
                b.batched_mosfet_term = term.flatten()
                self.batches.append(b)
        print(f'  loaded {len(self.batches)} batches from {len(variants)} variants')

    def __iter__(self):
        import random
        order = list(range(len(self.batches)))
        if self.shuffle: random.shuffle(order)
        for i in order:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_dir = cfg['data']['train_dir']
    val_dir = cfg['data']['val_dir']
    print(f'Loading train from {train_dir}')
    train_loader = SingleTopologyMaskLoader(train_dir, device=device, shuffle=True)
    val_loader = SingleTopologyMaskLoader(val_dir, device=device, shuffle=False)

    # Probe shapes
    sample = next(iter(train_loader))
    node_dim = sample.x.shape[-1]
    type_dim = sample.type_tens.shape[-1] if sample.type_tens is not None else 0
    print(f'node_feature_dim={node_dim}, type_feature_dim={type_dim}')
    print(f'sample x: {tuple(sample.x.shape)}, mosfet_term: {tuple(sample.batched_mosfet_term.shape)}')

    backbone = build_pretrain_backbone(
        cfg, node_dim, type_dim,
        mask_target_dim=cfg.get('pretrain', {}).get('mask_target_dim', 4),
        mask_head_hidden=cfg.get('pretrain', {}).get('mask_head_hidden', 128),
    ).to(device)
    n_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f'PretrainBackbone params: {n_params/1e6:.2f}M')

    pre_cfg = cfg['pretrain']
    mask_ratio = pre_cfg.get('mask_ratio', 0.75)  # MAE default
    edge_drop_ratio = pre_cfg.get('edge_drop_ratio', 0.0)  # 0 = no edge masking
    print(f'MAE mask ratio (nodes): {mask_ratio}, edge drop ratio: {edge_drop_ratio}')

    opt = build_adam(backbone, cfg)
    sch = build_plateau_scheduler(opt, cfg)
    epochs = cfg['training']['epochs']
    grad_clip = cfg['training'].get('gradient_clip', 1.0)

    out_dir = Path(f"datasets/{Path(train_dir).parent.name}/experiments/{args.name}")
    log_path = prepare_out_dir(out_dir, cfg)
    log_lines = []
    best_val = float('inf')

    pbar = make_progress_bar(epochs, len(train_loader), desc='MAE')

    for epoch in range(epochs):
        backbone.train()
        backbone.current_epoch = epoch
        tr_loss = 0.0
        n_steps = 0
        for batch in train_loader:
            # Optional: drop a fraction of edges (graph augmentation, like
            # GraphCL / SimCLR-graph). Keep at least 50% to prevent disconnect.
            orig_edge_index = batch.edge_index
            orig_edge_attr = getattr(batch, 'edge_attr', None)
            if edge_drop_ratio > 0:
                n_edges = orig_edge_index.shape[1]
                keep_n = max(int(n_edges * (1 - edge_drop_ratio)), n_edges // 2)
                perm = torch.randperm(n_edges, device=device)
                keep = perm[:keep_n].sort()[0]
                batch.edge_index = orig_edge_index[:, keep]
                if orig_edge_attr is not None:
                    batch.edge_attr = orig_edge_attr[keep]
            targets, masked_idx = apply_mosfet_mask(batch, mask_ratio)
            out = backbone(batch, mask_indices=masked_idx)
            loss = pretrain_loss(out['mask_pred'], targets)
            # Restore edges for next batch (we modified in place)
            batch.edge_index = orig_edge_index
            if orig_edge_attr is not None:
                batch.edge_attr = orig_edge_attr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), grad_clip)
            opt.step()
            tr_loss += loss.item() * targets.shape[0]
            n_steps += targets.shape[0]
            pbar.set_postfix(loss=f'{loss.item():.4e}', refresh=False)
            pbar.update(1)
        tr_loss /= max(1, n_steps)

        backbone.eval()
        val_loss = 0.0
        val_mae = 0.0
        n_val = 0
        rng = torch.Generator(device=device)
        rng.manual_seed(epoch + 12345)
        with torch.no_grad():
            for batch in val_loader:
                targets, masked_idx = apply_mosfet_mask(batch, mask_ratio, rng=rng)
                out = backbone(batch, mask_indices=masked_idx)
                lv = pretrain_loss(out['mask_pred'], targets).item()
                mae = (out['mask_pred'] - targets).abs().mean().item()
                bs = targets.shape[0]
                val_loss += lv * bs
                val_mae += mae * bs
                n_val += bs
        val_loss /= max(1, n_val)
        val_mae /= max(1, n_val)

        cur_lr = opt.param_groups[0]['lr']
        msg = f'epoch {epoch:4d}: train_loss={tr_loss:.4e} lr={cur_lr:.2e} | val_loss={val_loss:.4e} val_mae={val_mae:.4e}'
        best_val, msg = scheduler_step_and_save_best(
            val_loss, best_val, epoch, backbone, cfg, out_dir, msg,
            scheduler=sch, extra_ckpt={'val_loss': val_loss},
        )
        write_log_line(msg, log_lines, log_path)

    pbar.close()
    print(f'Done. Best val: {best_val:.4e}')


if __name__ == '__main__':
    main()
