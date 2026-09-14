#!/usr/bin/env python
"""Contrastive (SimCLR-style) pretraining for circuit GNNs.

For each graph in a batch, create TWO augmented views (different random
perturbations of features + dropped edges), compute graph-level embeddings,
and apply NT-Xent / InfoNCE loss to make views of the same graph similar
and views of different graphs dissimilar.

Backbone state-dict prefixes match TowerGENConv (input_linear, backbone,
virtual_node, loop_attn_layers) so weights transfer via strict=False.

Augmentations applied per view:
  - x feature noise: Gaussian σ=0.02 on cols 0-3 (W/L/wl/M)
  - edge dropping: 10% of edges randomly removed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuitgnn.data.pretrain_loader import PretrainCombinedLoader, PretrainBatch
from circuitgnn.gnn.architectures.pretrain_backbone import (
    PretrainBackbone,
    build_pretrain_backbone,
)
from circuitgnn.training.pretrain import (
    build_adam,
    build_plateau_scheduler,
    make_progress_bar,
    prepare_out_dir,
    save_final_checkpoint,
    scheduler_step_and_save_best,
    write_log_line,
)


class ContrastivePretrainModel(nn.Module):
    """Backbone + projection head (SimCLR style).

    Backbone state-dict matches TowerGENConv prefixes; projection head is
    used only during pretraining and discarded at fine-tune time.
    """
    def __init__(self, backbone: PretrainBackbone, hidden_dim: int, proj_dim: int = 64):
        super().__init__()
        self.backbone = backbone
        # Projection head: 2-layer MLP (SimCLR convention).
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def encode_graph(self, data) -> torch.Tensor:
        """Run backbone → mean-pool over each graph → projection.

        NOTE: the layer loop below is a manual copy of
        PretrainBackbone.encode() that has DRIFTED from it: it omits the
        default-mode virtual node path (init_embedding, per-layer broadcast,
        post-layer VN state update). It matches encode() bitwise only when
        the VN is disabled or vn mode == 'mha'; with vn mode 'default' this
        loop silently skips the VN while encode() applies it. Kept as is to
        preserve behavior; not switched to backbone.encode().

        Returns:
            [num_graphs, proj_dim] L2-normalized embeddings.
        """
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

        # Graph-level mean pooling
        from torch_geometric.nn import global_mean_pool
        graph_emb = global_mean_pool(x, batch_vec, size=num_graphs)  # [B, hidden]
        proj = self.proj_head(graph_emb)
        return F.normalize(proj, dim=-1)


def augment_view(data: PretrainBatch, feature_noise: float = 0.02,
                 edge_drop: float = 0.10, rng=None) -> PretrainBatch:
    """Return a perturbed copy of the batch:
      - cols 0-3 of x get additive Gaussian noise (σ=feature_noise)
      - edge_drop fraction of edges removed (random subset)
    """
    device = data.x.device
    # Copy x and inject noise on cols 0-3
    x_noisy = data.x.clone()
    if feature_noise > 0:
        noise = torch.randn(x_noisy[:, :4].shape, device=device, generator=rng) * feature_noise
        x_noisy[:, :4] = x_noisy[:, :4] + noise

    # Drop random subset of edges (always keep at least 50%)
    ei = data.edge_index
    n_edges = ei.shape[1]
    if edge_drop > 0 and n_edges > 0:
        keep_n = max(int(n_edges * (1 - edge_drop)), n_edges // 2)
        perm = torch.randperm(n_edges, device=device, generator=rng)
        keep = perm[:keep_n].sort()[0]  # sorted to preserve any structure
        ei_dropped = ei[:, keep]
    else:
        ei_dropped = ei

    # Build perturbed batch — share most attrs from the original
    attrs = dict(data.__dict__)
    attrs['x'] = x_noisy
    attrs['edge_index'] = ei_dropped
    return PretrainBatch(attrs)


def infonce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent / InfoNCE for two L2-normalized embeddings of size [B, D].
    Positive pair = same index across z1, z2. Negatives = all other indices.
    """
    B = z1.shape[0]
    # Concatenate to a 2B × D matrix
    z = torch.cat([z1, z2], dim=0)               # [2B, D]
    sim = (z @ z.T) / temperature                # [2B, 2B]
    # Mask self-similarity
    sim = sim.masked_fill(torch.eye(2 * B, dtype=torch.bool, device=z.device), float('-inf'))
    # Positives: i in [0..B-1] pairs with i+B; i in [B..2B-1] pairs with i-B
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)
    return F.cross_entropy(sim, targets)


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
    bs = cfg['training']['batch_size']

    print(f'Loading: {train_path}')
    t0 = time.time()
    train_loader = PretrainCombinedLoader(train_path, batch_size=bs, device=device, shuffle=True)
    print(f'  in {time.time()-t0:.1f}s; {len(train_loader)} batches/epoch')
    val_loader = PretrainCombinedLoader(val_path, batch_size=bs, device=device, shuffle=False, drop_last=False)

    sample = next(iter(train_loader))
    node_dim = sample.x.shape[-1]
    type_dim = sample.type_tens.shape[-1]
    print(f'node_feature_dim={node_dim}, type_feature_dim={type_dim}')

    backbone = build_pretrain_backbone(cfg, node_dim, type_dim)
    proj_dim = cfg['contrastive'].get('proj_dim', 64)
    model = ContrastivePretrainModel(backbone, cfg['model']['hidden_dim'], proj_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model params: {n_params/1e6:.2f}M (backbone + proj head {proj_dim}-d)')

    contrast_cfg = cfg['contrastive']
    feat_noise = float(contrast_cfg.get('feature_noise', 0.02))
    edge_drop = float(contrast_cfg.get('edge_drop', 0.10))
    temperature = float(contrast_cfg.get('temperature', 0.5))
    print(f'Augmentation: feat_noise={feat_noise}, edge_drop={edge_drop}, temp={temperature}')

    opt = build_adam(model, cfg)
    sch = build_plateau_scheduler(opt, cfg)

    epochs = cfg['training']['epochs']
    grad_clip = cfg['training'].get('gradient_clip', 1.0)
    out_dir = Path(f"datasets/{Path(train_path).parent.name}/experiments/{args.name}")
    log_path = prepare_out_dir(out_dir, cfg)
    log_lines = []
    best_val = float('inf')

    pbar = make_progress_bar(epochs, len(train_loader), desc='Contrastive')

    for epoch in range(epochs):
        model.train()
        backbone.current_epoch = epoch
        tr_loss = 0.0
        n_steps = 0
        for batch in train_loader:
            v1 = augment_view(batch, feat_noise, edge_drop)
            v2 = augment_view(batch, feat_noise, edge_drop)
            z1 = model.encode_graph(v1)
            z2 = model.encode_graph(v2)
            loss = infonce_loss(z1, z2, temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_loss += loss.item()
            n_steps += 1
            pbar.set_postfix(loss=f'{loss.item():.4e}', refresh=False)
            pbar.update(1)
        tr_loss /= n_steps

        model.eval()
        backbone.current_epoch = epoch
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                v1 = augment_view(batch, feat_noise, edge_drop)
                v2 = augment_view(batch, feat_noise, edge_drop)
                z1 = model.encode_graph(v1)
                z2 = model.encode_graph(v2)
                val_loss += infonce_loss(z1, z2, temperature).item()
                n_val += 1
        val_loss /= n_val

        cur_lr = opt.param_groups[0]['lr']
        msg = f'epoch {epoch:4d}: lr={cur_lr:.2e} tr_loss={tr_loss:.4e} | val_loss={val_loss:.4e}'
        best_val, msg = scheduler_step_and_save_best(
            val_loss, best_val, epoch, model.backbone,  # backbone-only state dict
            cfg, out_dir, msg,
            scheduler=sch, extra_ckpt={'val_loss': val_loss},
        )
        write_log_line(msg, log_lines, log_path)

    pbar.close()
    print(f'Done. Best val: {best_val:.4e}')
    save_final_checkpoint(out_dir, epochs, model.backbone, cfg)


if __name__ == '__main__':
    main()
