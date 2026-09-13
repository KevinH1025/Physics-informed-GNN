"""Pretraining utilities: masking + epoch loops."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


# Indices of (W, L, wl_ratio, M) within data.x. These are the design
# parameters that vary per MOSFET; cols 4-8 hold type/encoding/bias.
MOSFET_PROP_COLS = (0, 1, 2, 3)


def apply_mosfet_mask(
    data,
    mask_ratio: float,
    rng: torch.Generator | None = None,
    strategy: str = 'random',
    mirror_groups=None,
    stage_groups=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask MOSFET design features.

    Strategies:
      'random' (default): pick `mask_ratio * M_total` MOSFETs uniformly at random.
      'mirror': pick mirror PAIRS (gate-net groups) at random until ~mask_ratio
                of MOSFETs are masked. Mask both partners of each chosen pair
                together — kills the mirror-copy shortcut.
      'stage':  pick whole parameter groups (transistors sharing W/L/M)
                at random until ~mask_ratio of MOSFETs masked. Forces
                cross-stage reasoning.

    Args:
        data: PretrainBatch — `data.x` (float32) modified IN PLACE.
              Requires `data.batched_mosfet_term`: [4 * M_total].
        mask_ratio: target fraction of MOSFETs masked (0..1).
        rng: optional torch.Generator for deterministic masking.
        strategy: 'random' | 'mirror' | 'stage'
        mirror_groups: list of lists of MOSFET indices (per topology); needed for 'mirror'.
                       Each batch concatenates 4 topos × per_topo graphs of
                       same topology, so for mirror/stage we need batched_groups
                       (already in batched MOSFET-index space).
        stage_groups: same idea — list of lists of MOSFET indices.

    Returns:
        targets: [K, 4] original (W, L, wl_ratio, M) values at masked nodes
        masked_indices: [K] node indices of masked nodes
    """
    term = data.batched_mosfet_term  # [4 * M_total]
    n_per = 4
    M_total = term.shape[0] // n_per
    if M_total == 0:
        empty_t = torch.empty((0, len(MOSFET_PROP_COLS)),
                              dtype=data.x.dtype, device=data.x.device)
        empty_i = torch.empty((0,), dtype=torch.long, device=data.x.device)
        return empty_t, empty_i

    device = data.x.device

    if strategy == 'random':
        n_mask = max(1, int(round(mask_ratio * M_total)))
        perm = torch.randperm(M_total, generator=rng, device=device)
        chosen = perm[:n_mask]
    elif strategy == 'mirror':
        if mirror_groups is None or len(mirror_groups) == 0:
            raise ValueError("mirror_groups required for strategy='mirror'")
        # Shuffle group order and accumulate until we hit mask_ratio.
        target = max(1, int(round(mask_ratio * M_total)))
        group_perm = torch.randperm(len(mirror_groups), generator=rng, device=device)
        chosen_list = []
        count = 0
        for gi in group_perm:
            g = mirror_groups[gi.item()]
            chosen_list.extend(g)
            count += len(g)
            if count >= target:
                break
        chosen = torch.tensor(chosen_list, dtype=torch.long, device=device)
    elif strategy == 'stage':
        if stage_groups is None or len(stage_groups) == 0:
            raise ValueError("stage_groups required for strategy='stage'")
        target = max(1, int(round(mask_ratio * M_total)))
        group_perm = torch.randperm(len(stage_groups), generator=rng, device=device)
        chosen_list = []
        count = 0
        for gi in group_perm:
            g = stage_groups[gi.item()]
            chosen_list.extend(g)
            count += len(g)
            if count >= target:
                break
        chosen = torch.tensor(chosen_list, dtype=torch.long, device=device)
    else:
        raise ValueError(f'Unknown masking strategy: {strategy}')

    # Collect all 4 terminal indices per chosen MOSFET → [n_mask * 4]
    term_view = term.view(M_total, n_per)
    masked_idx = term_view[chosen].reshape(-1)

    cols = torch.tensor(MOSFET_PROP_COLS, dtype=torch.long, device=device)
    targets = data.x[masked_idx][:, cols].clone()
    data.x[masked_idx[:, None], cols[None, :]] = 0.0
    return targets, masked_idx


def pretrain_loss(
    pred: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """MSE loss on masked positions (all cols equally weighted)."""
    return F.mse_loss(pred, targets)


def pretrain_epoch(
    model,
    loader,
    optimizer,
    mask_ratio: float,
    grad_clip: float = 1.0,
    device: torch.device = torch.device('cuda'),
    epoch: int = 0,
    progress=None,
    strategy: str = 'random',
):
    """Run one pretraining epoch; returns mean loss."""
    model.train()
    model.current_epoch = epoch
    total_loss = 0.0
    total_items = 0
    rng = None
    mirror_groups = getattr(loader, 'batched_mirror_groups', None)
    stage_groups = getattr(loader, 'batched_stage_groups', None)
    for step, batch in enumerate(loader):
        targets, masked_idx = apply_mosfet_mask(
            batch, mask_ratio, rng=rng,
            strategy=strategy,
            mirror_groups=mirror_groups,
            stage_groups=stage_groups,
        )
        out = model(batch, mask_indices=masked_idx)
        loss = pretrain_loss(out['mask_pred'], targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = targets.shape[0]
        total_loss += loss.item() * bs
        total_items += bs
        if progress is not None:
            progress.set_postfix(loss=f'{loss.item():.4e}', refresh=False)
            progress.update(1)
    return total_loss / max(1, total_items)


@torch.no_grad()
def pretrain_validate(
    model,
    loader,
    mask_ratio: float,
    epoch: int = 0,
    strategy: str = 'random',
):
    model.eval()
    model.current_epoch = epoch
    total_loss = 0.0
    total_mae = 0.0
    total_items = 0
    rng = torch.Generator(device=loader.device)
    rng.manual_seed(epoch + 1234567)
    mirror_groups = getattr(loader, 'batched_mirror_groups', None)
    stage_groups = getattr(loader, 'batched_stage_groups', None)
    for batch in loader:
        targets, masked_idx = apply_mosfet_mask(
            batch, mask_ratio, rng=rng,
            strategy=strategy,
            mirror_groups=mirror_groups,
            stage_groups=stage_groups,
        )
        out = model(batch, mask_indices=masked_idx)
        loss = pretrain_loss(out['mask_pred'], targets)
        mae = (out['mask_pred'] - targets).abs().mean()
        bs = targets.shape[0]
        total_loss += loss.item() * bs
        total_mae += mae.item() * bs
        total_items += bs
    return {
        'loss': total_loss / max(1, total_items),
        'mae': total_mae / max(1, total_items),
    }
