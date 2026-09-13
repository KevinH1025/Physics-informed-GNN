#!/usr/bin/env python3
"""Train the SKY130 MOSFET IV-surface convolutional autoencoder (cluster edition).

Source: v2 "id-only" LUT. Each (W, L, Vbs, polarity) is one training surface,
log10|I_D| over (Vgs, Vds). All 10 Vbs slices are used by default (32,000
surfaces total: 20 W × 80 L × 10 Vbs × 2 polarities). Pass --no-all-vbs to
train on the Vbs=0 slice only (3,200 surfaces, legacy behaviour).

Pipeline:
 1. Load surfaces into one pre-allocated float32 array, log10-transform in place.
 2. Group-level 80/20 split by (polarity, W_idx, L_idx): all 10 Vbs slices for a
    given (W, L, pol) go to the same split so near-duplicates can't leak.
 3. Normalize with train-only mean/std of log10|id|.
 4. Upload normalized tensor to GPU once; batches are free index gathers.
 5. Encoder + decoder from circuitgnn.physics.iv_autoencoder. Adam + linear warmup +
    ReduceLROnPlateau. bfloat16 autocast on CUDA. Early stop on val plateau.
 6. Validation each epoch: MSE (normalized) + linear-id relative error +
    log-space absolute error. Quantiles from a 1 M random pixel sample.
 7. Best weights kept on GPU and cloned to CPU on every improvement; flushed to
    disk every --save-every epochs AND on every improvement (cheap on the
    cluster FS).
 8. Best checkpoint reloaded at the end for plots (loss curves, reconstruction
    samples, t-SNE by polarity and L).

Outputs (under datasets/lut/lut_v2/iv_autoencoder/<run_name>/):
  encoder.pt, decoder.pt, normalization.pt, config.json,
  training_log.json, training.log, plots/

Default run name: plateau_b{bottleneck}_bs{batch}_ep{epochs}_wm{warmup}_allvbs
(collision-safe; suffixed _01, _02, ...).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from circuitgnn.physics.iv_autoencoder import IVAutoencoder


LUT_PATH = 'datasets/lut/lut_v2/sky130_mosfet_lut_v2_id_only.h5'
OUT_BASE = Path('datasets/lut/lut_v2/iv_autoencoder')
LOG_FLOOR = 1e-20
QUANTILE_SAMPLE = 1_000_000   # torch.quantile max ~2^24; cap for val metrics


def resolve_run_dir(base: Path, run_name: str) -> Path:
    candidate = base / run_name
    if not candidate.exists():
        return candidate
    for i in range(1, 1000):
        candidate = base / f'{run_name}_{i:02d}'
        if not candidate.exists():
            return candidate
    raise RuntimeError(f'Too many collisions for {run_name}')


# ───────────────────────────── Data loading ─────────────────────────────
def load_surfaces(lut_path: str, all_vbs: bool = True):
    """Pre-allocate, fill polarity-by-polarity, log10-transform in place.

    Returns:
        surfaces:   (N, nVgs, nVds) float32, log10|id|
        pol_lbl:    (N,) int64, 0=nmos, 1=pmos
        w_idx:      (N,) int64, W grid index
        l_idx:      (N,) int64, L grid index
        vbs_idx:    (N,) int64, Vbs grid index (0 when all_vbs=False)
        W_um, L_um: grid axes for plotting
    """
    with h5py.File(lut_path, 'r') as h5:
        W_um = h5['W_um'][:]
        L_um = h5['L_um'][:]
        nW, nL = len(W_um), len(L_um)
        nVgs = len(h5['Vgs_V'][:])
        nVds = len(h5['Vds_V'][:])
        nVbs = h5['n/id'].shape[2] if all_vbs else 1
        n_per_pol = nW * nL * nVbs
        N = n_per_pol * 2

        print(f'  Allocating {N} x {nVgs} x {nVds} float32 '
              f'({N * nVgs * nVds * 4 / 1e9:.2f} GB)...')
        surfaces = np.empty((N, nVgs, nVds), dtype=np.float32)
        w_idx = np.empty(N, dtype=np.int64)
        l_idx = np.empty(N, dtype=np.int64)
        vbs_idx = np.empty(N, dtype=np.int64)
        pol_lbl = np.empty(N, dtype=np.int64)

        wi, li, vi = np.meshgrid(
            np.arange(nW), np.arange(nL), np.arange(nVbs), indexing='ij'
        )
        wi = wi.ravel()
        li = li.ravel()
        vi = vi.ravel()

        for p_idx, pol in enumerate(['n', 'p']):
            dst = slice(p_idx * n_per_pol, (p_idx + 1) * n_per_pol)
            print(f'  Reading {pol}/id...')
            if all_vbs:
                # Axes: (W, L, Vbs, Vds, Vgs) -> (W, L, Vbs, Vgs, Vds)
                arr = np.transpose(h5[f'{pol}/id'][:], (0, 1, 2, 4, 3)).copy()
                arr = arr.reshape(n_per_pol, nVgs, nVds)
            else:
                arr = h5[f'{pol}/id'][:, :, 0, :, :]              # (W, L, Vds, Vgs)
                arr = np.transpose(arr, (0, 1, 3, 2)).copy()       # (W, L, Vgs, Vds)
                arr = arr.reshape(n_per_pol, nVgs, nVds)
            arr = arr.astype(np.float32, copy=False)
            np.abs(arr, out=arr)
            np.maximum(arr, LOG_FLOOR, out=arr)
            np.log10(arr, out=arr)
            surfaces[dst] = arr
            w_idx[dst] = wi
            l_idx[dst] = li
            vbs_idx[dst] = vi
            pol_lbl[dst] = p_idx
            del arr

    return surfaces, pol_lbl, w_idx, l_idx, vbs_idx, W_um, L_um


def group_split(pol_lbl, w_idx, l_idx, train_frac=0.8, seed=42):
    """Split by unique (pol, w, l) groups so all Vbs slices for a (W, L, pol)
    stay in the same bucket."""
    rng = np.random.default_rng(seed)
    nW = int(w_idx.max()) + 1
    nL = int(l_idx.max()) + 1
    group_key = (pol_lbl.astype(np.int64) * nW * nL
                 + w_idx.astype(np.int64) * nL
                 + l_idx.astype(np.int64))
    unique_groups = np.unique(group_key)
    rng.shuffle(unique_groups)
    n_train_groups = int(round(train_frac * len(unique_groups)))
    train_groups = set(unique_groups[:n_train_groups].tolist())
    train_mask = np.array([int(g) in train_groups for g in group_key])
    return np.where(train_mask)[0], np.where(~train_mask)[0]


# ──────────────────────────────── LR sched ──────────────────────────────
def warmup_lr(epoch: int, warmup_epochs: int, lr_max: float) -> float:
    if warmup_epochs <= 0:
        return lr_max
    return lr_max * min(1.0, (epoch + 1) / warmup_epochs)


# ─────────────────────────────── Val metrics ─────────────────────────────
@torch.no_grad()
def compute_val_metrics(model, X_val_gpu, mean, std, batch_size, amp):
    """Pass full val set in batches; accumulate MSE exactly, sample for quantiles."""
    model.eval()
    device = X_val_gpu.device
    n = X_val_gpu.shape[0]

    mse_accum = torch.zeros(1, device=device, dtype=torch.float32)
    total_pixels = 0
    per_batch = max(1, QUANTILE_SAMPLE // max(1, (n + batch_size - 1) // batch_size))
    rel_samples = []
    log_samples = []

    for i in range(0, n, batch_size):
        xb = X_val_gpu[i:i + batch_size].unsqueeze(1)
        if amp:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                recon = model(xb)
            recon = recon.float()
        else:
            recon = model(xb)

        diff = recon - xb
        mse_accum += (diff * diff).sum()
        total_pixels += xb.numel()

        log_err = diff.abs() * std

        orig_log = xb * std + mean
        rec_log = recon * std + mean
        orig_lin = torch.pow(10.0, orig_log)
        rec_lin = torch.pow(10.0, rec_log)
        rel = (rec_lin - orig_lin).abs() / (orig_lin.abs() + 1e-15)

        flat_rel = rel.reshape(-1)
        flat_log = log_err.reshape(-1)
        if flat_rel.numel() > per_batch:
            idx = torch.randint(0, flat_rel.numel(), (per_batch,), device=device)
            rel_samples.append(flat_rel[idx])
            log_samples.append(flat_log[idx])
        else:
            rel_samples.append(flat_rel)
            log_samples.append(flat_log)

    val_mse = float(mse_accum.item()) / max(total_pixels, 1)

    rel_cat = torch.cat(rel_samples)
    log_cat = torch.cat(log_samples)
    if rel_cat.numel() > QUANTILE_SAMPLE:
        idx = torch.randint(0, rel_cat.numel(), (QUANTILE_SAMPLE,), device=device)
        rel_cat = rel_cat[idx]
        log_cat = log_cat[idx]
    rel_cat = rel_cat.float() * 100.0
    log_cat = log_cat.float()
    median_rel = float(torch.quantile(rel_cat, 0.5))
    p95_rel = float(torch.quantile(rel_cat, 0.95))
    median_log = float(torch.quantile(log_cat, 0.5))
    p95_log = float(torch.quantile(log_cat, 0.95))
    return val_mse, median_rel, p95_rel, median_log, p95_log


# ──────────────────────────────── Training ──────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        # H100 perf: autotune convs for the fixed batch/shape we train with,
        # allow TF32 matmuls (safe for bfloat16 pipelines), set higher
        # matmul precision so fp32 ops still use TensorCores when used.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        print(f'  {torch.cuda.get_device_name()} '
              f'({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)')
        print(f'  cudnn.benchmark=True  TF32=True  matmul_precision=high')

    vbs_tag = 'allvbs' if args.all_vbs else 'vbs0'
    default_name = (f'plateau_b{args.bottleneck}_bs{args.batch_size}'
                    f'_ep{args.epochs}_wm{args.warmup_epochs}_{vbs_tag}')
    run_name = args.run_name or default_name
    out_dir = resolve_run_dir(OUT_BASE, run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'plots').mkdir(exist_ok=True)
    print(f'Run dir: {out_dir}')

    log_file = open(out_dir / 'training.log', 'w', buffering=1)
    def _log(msg: str):
        print(msg, flush=True)
        log_file.write(msg + '\n')

    _log(f'Loading surfaces from {LUT_PATH}  (all_vbs={args.all_vbs})')
    t0 = time.time()
    surfaces, pol_lbl, w_idx, l_idx, vbs_idx, W_um, L_um = load_surfaces(
        LUT_PATH, all_vbs=args.all_vbs,
    )
    _log(f'  {surfaces.shape[0]} surfaces, shape={surfaces.shape[1:]}  '
         f'(nmos={int((pol_lbl==0).sum())}, pmos={int((pol_lbl==1).sum())})  '
         f'[{time.time()-t0:.1f}s]')

    tr_idx, va_idx = group_split(pol_lbl, w_idx, l_idx,
                                 train_frac=0.8, seed=42)
    n_unique_groups = len(np.unique(pol_lbl.astype(np.int64) * 10000
                                    + w_idx.astype(np.int64) * 100
                                    + l_idx.astype(np.int64)))
    _log(f'  Group split: train={len(tr_idx)}  val={len(va_idx)}  '
         f'(groups: {n_unique_groups})')

    mean = float(surfaces[tr_idx].mean())
    std = float(surfaces[tr_idx].std())
    _log(f'  Train-only normalization: mean={mean:.4f}  std={std:.4f}')

    surfaces -= mean
    surfaces /= std

    _log('Uploading normalized surfaces to GPU...')
    t0 = time.time()
    X_all_gpu = torch.from_numpy(surfaces).to(device, non_blocking=True)
    del surfaces
    tr_idx_t = torch.from_numpy(tr_idx).to(device)
    va_idx_t = torch.from_numpy(va_idx).to(device)
    X_tr_gpu = X_all_gpu[tr_idx_t].contiguous()
    X_va_gpu = X_all_gpu[va_idx_t].contiguous()
    del X_all_gpu
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    tr_gb = X_tr_gpu.element_size() * X_tr_gpu.numel() / 1e9
    va_gb = X_va_gpu.element_size() * X_va_gpu.numel() / 1e9
    _log(f'  GPU bytes: train={tr_gb:.2f} GB  val={va_gb:.2f} GB  [{time.time()-t0:.1f}s]')

    raw_model = IVAutoencoder(bottleneck=args.bottleneck).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())
    _log(f'  Model: {n_params/1e6:.2f}M params  bottleneck={args.bottleneck}')
    if args.compile and device.type == 'cuda':
        _log('  torch.compile(mode="reduce-overhead") — first batch will be slow')
        model = torch.compile(raw_model, mode='reduce-overhead')
    else:
        model = raw_model

    opt = torch.optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=0.0)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=args.plateau_patience,
        min_lr=args.lr_min, threshold=1e-4,
    )

    amp_enabled = device.type == 'cuda' and not args.no_amp
    _log(f'  AMP: bfloat16={amp_enabled}')

    log = {'train_loss': [], 'val_loss': [],
           'val_median_rel': [], 'val_p95_rel': [],
           'val_median_log': [], 'val_p95_log': [],
           'lr': [], 'epoch': []}

    best_val = float('inf')
    best_state_enc = None
    best_state_dec = None
    epochs_since_improve = 0
    t_train = time.time()
    n_train = X_tr_gpu.shape[0]
    epoch = -1

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            for g in opt.param_groups:
                g['lr'] = warmup_lr(epoch, args.warmup_epochs, args.lr_max)
        lr = opt.param_groups[0]['lr']

        perm = torch.randperm(n_train, device=device)

        model.train()
        # Accumulate on GPU to avoid per-step CPU sync (which would stall
        # the H100 since the model is small — Python dispatch would dominate).
        tr_loss_accum = torch.zeros(1, device=device, dtype=torch.float32)
        tr_pixels = 0
        for i in range(0, n_train, args.batch_size):
            sel = perm[i:i + args.batch_size]
            xb = X_tr_gpu[sel].unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    recon = model(xb)
                    loss = F.mse_loss(recon, xb)
            else:
                recon = model(xb)
                loss = F.mse_loss(recon, xb)
            loss.backward()
            opt.step()
            tr_loss_accum += loss.detach() * xb.numel()
            tr_pixels += xb.numel()

        tr_mean = float(tr_loss_accum.item()) / tr_pixels

        val_mse, median_rel, p95_rel, median_log, p95_log = compute_val_metrics(
            model, X_va_gpu, mean, std, args.batch_size, amp_enabled,
        )

        log['epoch'].append(epoch)
        log['train_loss'].append(tr_mean)
        log['val_loss'].append(val_mse)
        log['val_median_rel'].append(median_rel)
        log['val_p95_rel'].append(p95_rel)
        log['val_median_log'].append(median_log)
        log['val_p95_log'].append(p95_log)
        log['lr'].append(lr)

        improved = val_mse < best_val - 1e-6
        if improved:
            best_val = val_mse
            epochs_since_improve = 0
            # Pull from raw_model to dodge torch.compile's OptimizedModule wrapper
            best_state_enc = {k: v.detach().clone() for k, v in raw_model.encoder.state_dict().items()}
            best_state_dec = {k: v.detach().clone() for k, v in raw_model.decoder.state_dict().items()}
            torch.save(best_state_enc, out_dir / 'encoder.pt')
            torch.save(best_state_dec, out_dir / 'decoder.pt')
        else:
            epochs_since_improve += 1

        if epoch >= args.warmup_epochs:
            plateau.step(val_mse)

        if epoch % args.log_interval == 0 or epoch == args.epochs - 1:
            elapsed = (time.time() - t_train) / 60.0
            _log(f'  ep {epoch:4d}  lr={lr:.2e}  '
                 f'train={tr_mean:.5f}  val={val_mse:.5f}  '
                 f'med_rel={median_rel:.2f}%  p95_rel={p95_rel:.2f}%  '
                 f'med_log={median_log:.3f}  p95_log={p95_log:.3f}  '
                 f'no_imp={epochs_since_improve:3d}  '
                 f't={elapsed:.1f}m')

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0 and not improved \
                and best_state_enc is not None:
            torch.save(best_state_enc, out_dir / 'encoder.pt')
            torch.save(best_state_dec, out_dir / 'decoder.pt')

        if epochs_since_improve >= args.early_stop_patience:
            _log(f'  [early stop] no improvement for {epochs_since_improve} epochs '
                 f'(best val={best_val:.5f})')
            break

    torch.save({'mean': mean, 'std': std}, out_dir / 'normalization.pt')
    with open(out_dir / 'config.json', 'w') as f:
        json.dump({
            'bottleneck': args.bottleneck,
            'input_h': X_va_gpu.shape[1], 'input_w': X_va_gpu.shape[2],
            'padded_h': 192, 'padded_w': 384,
            'log_floor': LOG_FLOOR,
            'log_transform': True,
            'normalization': {'mean': mean, 'std': std},
            'lut_path': LUT_PATH,
            'all_vbs': bool(args.all_vbs),
            'n_train': int(len(tr_idx)),
            'n_val': int(len(va_idx)),
            'epochs_requested': args.epochs,
            'epochs_run': epoch + 1,
            'batch_size': args.batch_size,
            'lr_max': args.lr_max, 'lr_min': args.lr_min,
            'warmup_epochs': args.warmup_epochs,
            'scheduler': 'warmup+ReduceLROnPlateau',
            'plateau_patience': args.plateau_patience,
            'plateau_factor': 0.5,
            'early_stop_patience': args.early_stop_patience,
            'amp': bool(amp_enabled),
            'best_val_loss': best_val,
        }, f, indent=2)
    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    _log(f'Reloading best model (val {best_val:.5f}) for plots...')
    raw_model.encoder.load_state_dict(torch.load(out_dir / 'encoder.pt',
                                                  map_location=device, weights_only=True))
    raw_model.decoder.load_state_dict(torch.load(out_dir / 'decoder.pt',
                                                  map_location=device, weights_only=True))
    raw_model.eval()

    _log('Generating plots...')
    plot_loss_curve(log, out_dir / 'plots' / 'loss_curve.png')
    plot_reconstruction_samples(raw_model, X_va_gpu, mean, std, pol_lbl[va_idx],
                                 device, out_dir / 'plots' / 'reconstruction_samples.png')
    plot_tsne(raw_model, X_tr_gpu, X_va_gpu, tr_idx, va_idx,
              pol_lbl, l_idx, L_um, device,
              out_dir / 'plots' / 'embedding_tsne.png')

    _log(f'\nDone.  Outputs in {out_dir}/')
    _log(f'  Best val MSE:       {best_val:.5f}')
    _log(f'  Last median rel:    {log["val_median_rel"][-1]:.2f}%')
    _log(f'  Last p95 rel:       {log["val_p95_rel"][-1]:.2f}%')
    _log(f'  Last median log:    {log["val_median_log"][-1]:.3f} dec')
    _log(f'  Last p95 log:       {log["val_p95_log"][-1]:.3f} dec')
    log_file.close()


# ───────────────────────────────── Plots ────────────────────────────────
def plot_loss_curve(log, path):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    ep = log['epoch']
    axes[0].plot(ep, log['train_loss'], label='train')
    axes[0].plot(ep, log['val_loss'], label='val')
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('MSE (normalized)')
    axes[0].set_yscale('log'); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[0].set_title('Reconstruction MSE')
    axes[1].plot(ep, log['val_median_rel'], label='median')
    axes[1].plot(ep, log['val_p95_rel'], label='p95')
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('rel err (%)')
    axes[1].set_yscale('log'); axes[1].grid(alpha=0.3); axes[1].legend()
    axes[1].set_title('Linear rel err on |I_D|')
    axes[2].plot(ep, log['val_median_log'], label='median')
    axes[2].plot(ep, log['val_p95_log'], label='p95')
    axes[2].set_xlabel('epoch'); axes[2].set_ylabel('decades')
    axes[2].set_yscale('log'); axes[2].grid(alpha=0.3); axes[2].legend()
    axes[2].set_title('Log-space abs err (decades)')
    axes[3].plot(ep, log['lr'])
    axes[3].set_xlabel('epoch'); axes[3].set_ylabel('lr')
    axes[3].set_yscale('log'); axes[3].grid(alpha=0.3, which='both')
    axes[3].set_title('Learning rate')
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches='tight'); plt.close(fig)


def plot_reconstruction_samples(model, X_va_gpu, mean, std, pol_va, device, path, n_samples=5):
    model.eval()
    idx = np.random.RandomState(0).choice(X_va_gpu.shape[0], n_samples, replace=False)
    with torch.no_grad():
        xb = X_va_gpu[torch.from_numpy(idx).to(device)].unsqueeze(1)
        rec = model(xb)
    X = xb.cpu().numpy()[:, 0]
    R = rec.cpu().numpy()[:, 0]
    X_log = X * std + mean
    R_log = R * std + mean

    fig, axes = plt.subplots(n_samples, 3, figsize=(14, 3.2 * n_samples))
    for i in range(n_samples):
        pol = 'nmos' if pol_va[idx[i]] == 0 else 'pmos'
        vmin, vmax = X_log[i].min(), X_log[i].max()
        a = axes[i, 0].imshow(X_log[i], aspect='auto', origin='lower',
                              vmin=vmin, vmax=vmax, cmap='viridis')
        axes[i, 0].set_title(f'{pol}  original  log10|I_D|'); plt.colorbar(a, ax=axes[i, 0])
        b = axes[i, 1].imshow(R_log[i], aspect='auto', origin='lower',
                              vmin=vmin, vmax=vmax, cmap='viridis')
        axes[i, 1].set_title('reconstruction'); plt.colorbar(b, ax=axes[i, 1])
        diff = R_log[i] - X_log[i]
        m = float(np.abs(diff).max())
        c = axes[i, 2].imshow(diff, aspect='auto', origin='lower',
                              vmin=-m, vmax=m, cmap='RdBu_r')
        axes[i, 2].set_title(f'residual (Δlog10 max |{m:.2f}|)'); plt.colorbar(c, ax=axes[i, 2])
        for ax in axes[i]:
            ax.set_xlabel('Vds idx'); ax.set_ylabel('Vgs idx')
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches='tight'); plt.close(fig)


def plot_tsne(model, X_tr_gpu, X_va_gpu, tr_idx, va_idx,
              pol_all, l_idx_all, L_um, device, path):
    from sklearn.manifold import TSNE
    model.eval()
    max_points = 6000
    half = max_points // 2
    tr_sub = np.random.RandomState(1).choice(len(tr_idx), min(half, len(tr_idx)), replace=False)
    va_sub = np.random.RandomState(2).choice(len(va_idx), min(half, len(va_idx)), replace=False)
    X_sub = torch.cat([
        X_tr_gpu[torch.from_numpy(tr_sub).to(device)],
        X_va_gpu[torch.from_numpy(va_sub).to(device)],
    ], dim=0)
    original_indices = np.concatenate([tr_idx[tr_sub], va_idx[va_sub]])

    embs = []
    with torch.no_grad():
        for i in range(0, X_sub.shape[0], 256):
            embs.append(model.encoder(X_sub[i:i + 256].unsqueeze(1)).cpu().numpy())
    E = np.concatenate(embs, axis=0)
    print(f'  t-SNE on {E.shape[0]} embeddings (dim={E.shape[1]})...')
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=0)
    E2 = tsne.fit_transform(E)

    pol_sub = pol_all[original_indices]
    l_sub = l_idx_all[original_indices]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for pol, (name, col) in enumerate([('nmos', 'tab:blue'), ('pmos', 'tab:red')]):
        m = pol_sub == pol
        axes[0].scatter(E2[m, 0], E2[m, 1], s=6, alpha=0.5, c=col, label=name)
    axes[0].set_title('t-SNE of embeddings — by polarity')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    scat = axes[1].scatter(E2[:, 0], E2[:, 1], s=6, alpha=0.6,
                           c=L_um[l_sub], cmap='plasma',
                           norm=matplotlib.colors.LogNorm())
    axes[1].set_title('t-SNE — by L (µm, log color)')
    axes[1].grid(alpha=0.3)
    plt.colorbar(scat, ax=axes[1], label='L (µm)')
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches='tight'); plt.close(fig)


# ─────────────────────────────── Entrypoint ─────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=2000)
    ap.add_argument('--warmup-epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr-max', type=float, default=1e-3)
    ap.add_argument('--lr-min', type=float, default=1e-5)
    ap.add_argument('--bottleneck', type=int, default=64)
    ap.add_argument('--plateau-patience', type=int, default=45)
    ap.add_argument('--early-stop-patience', type=int, default=150)
    ap.add_argument('--save-every', type=int, default=20,
                    help='Flush best weights every N epochs AND on every improvement '
                         '(0 = only on improvement).')
    ap.add_argument('--log-interval', type=int, default=5)
    ap.add_argument('--no-amp', action='store_true', help='Disable bfloat16 AMP')
    ap.add_argument('--compile', action='store_true',
                    help='torch.compile(model, mode="reduce-overhead"). First batch '
                         'takes ~30-60s to compile; steady-state gain ~15-30 pct on H100.')
    ap.add_argument('--all-vbs', dest='all_vbs', action='store_true', default=True,
                    help='Use all 10 Vbs slices (default).')
    ap.add_argument('--no-all-vbs', dest='all_vbs', action='store_false',
                    help='Debug: Vbs=0 slice only (3200 surfaces).')
    ap.add_argument('--run-name', type=str, default=None)
    args = ap.parse_args()
    train(args)
