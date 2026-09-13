"""Pretrained IV-surface autoencoder embedder for MOSFETs.

Given (W_um, L_um, is_nmos), looks up the SKY130 v2 LUT at the Vbs=0 slice,
bilinearly interpolates the log10|I_D| surface in (log W, log L) space,
normalizes with training-time stats, and encodes through the frozen 64-dim
convolutional autoencoder (src.physics.iv_autoencoder.Encoder).

The encoder is loaded frozen, always in eval mode so BatchNorm uses running
stats. No backprop through LUT or encoder.

Typical usage (inside a GNN):

    embedder = IVEmbedder('datasets/lut/lut_v2/iv_autoencoder/plateau_b64_bs128_ep2000_wm10')
    z = embedder(W_um, L_um, is_nmos)  # -> [N, 64]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn

from circuitgnn.physics.iv_autoencoder import Encoder

from .io import log10_abs_floor, sorted_vbs_orders, stack_polarities, swap_vgs_vds


class _NullCtx:
    """No-op context manager — used so forward() can pick torch.no_grad OR
    nothing at runtime without a branchy `if` around the actual work."""
    def __enter__(self): return None
    def __exit__(self, *a): return False


class IVEmbedder(nn.Module):
    LOG_FLOOR = 1e-20

    def __init__(self, run_dir: str | Path, lut_path: Optional[str | Path] = None,
                 freeze: bool = True, use_vbs: bool = False):
        """Load a pretrained IV surface autoencoder.

        Args:
            run_dir: Directory containing encoder.pt, config.json, normalization.pt.
            lut_path: Optional override for the LUT file (else reads from config).
            freeze: If True (default) the encoder is frozen and runs in eval mode.
            use_vbs: If True, load the full Vbs axis and interpolate over |Vbs|
                per device. If False (default), use the Vbs=0 slice only.
                Set True only when the encoder was trained on all_vbs (32 k
                surfaces) — otherwise the encoder has never seen non-Vbs=0
                surfaces and will produce garbage embeddings.
        """
        super().__init__()
        run_dir = Path(run_dir)
        with open(run_dir / 'config.json') as f:
            cfg = json.load(f)
        self.bottleneck = int(cfg.get('bottleneck', 64))
        self.embed_dim = self.bottleneck
        self._freeze = bool(freeze)
        self.use_vbs = bool(use_vbs)

        norm = cfg['normalization']
        self.register_buffer('surf_mean', torch.tensor(float(norm['mean']), dtype=torch.float32))
        self.register_buffer('surf_std', torch.tensor(float(norm['std']), dtype=torch.float32))

        self.encoder = Encoder(bottleneck=self.bottleneck)
        state = torch.load(run_dir / 'encoder.pt', map_location='cpu', weights_only=True)
        self.encoder.load_state_dict(state)
        if self._freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

        lut_path = Path(lut_path) if lut_path is not None else Path(cfg['lut_path'])
        with h5py.File(lut_path, 'r') as h5:
            W_um = h5['W_um'][:]
            L_um = h5['L_um'][:]
            if self.use_vbs:
                # Load full Vbs axis — shape (W, L, Vbs, Vds, Vgs).
                n_id = h5['n/id'][:]
                p_id = h5['p/id'][:]
                n_vbs = h5['n/Vbs_V'][:]
                p_vbs = h5['p/Vbs_V'][:]
            else:
                n_id = h5['n/id'][:, :, 0, :, :]
                p_id = h5['p/id'][:, :, 0, :, :]
                n_vbs = None
                p_vbs = None

        if self.use_vbs:
            # Reorient to (W, L, Vbs, Vgs, Vds) — matches encoder input layout.
            n_log = log10_abs_floor(swap_vgs_vds(n_id), self.LOG_FLOOR)
            p_log = log10_abs_floor(swap_vgs_vds(p_id), self.LOG_FLOOR)

            # Sort both polarities by |Vbs| ascending so a single unified
            # |Vbs| axis serves both. NMOS uses negative Vbs, PMOS positive —
            # magnitudes match by construction.
            order_n, order_p, vbs_mag = sorted_vbs_orders(
                n_vbs, p_vbs,
                'NMOS and PMOS |Vbs| axes differ — expected mirrored grids')
            n_log = n_log[:, :, order_n, :, :]
            p_log = p_log[:, :, order_p, :, :]

            lut = stack_polarities(n_log, p_log)  # [2, W, L, Vbs, Vgs, Vds]
            self.register_buffer('lut', torch.from_numpy(lut))
            self.register_buffer('vbs_axis', torch.from_numpy(vbs_mag))
            self._nVbs = int(self.vbs_axis.numel())
        else:
            n_log = log10_abs_floor(swap_vgs_vds(n_id), self.LOG_FLOOR)
            p_log = log10_abs_floor(swap_vgs_vds(p_id), self.LOG_FLOOR)
            lut = stack_polarities(n_log, p_log)  # [2, W, L, Vgs, Vds]
            self.register_buffer('lut', torch.from_numpy(lut))
            self._nVbs = 0

        self.register_buffer('log_W', torch.from_numpy(np.log(W_um).astype(np.float32)))
        self.register_buffer('log_L', torch.from_numpy(np.log(L_um).astype(np.float32)))
        self._nW = int(self.log_W.numel())
        self._nL = int(self.log_L.numel())

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze:
            # Frozen: never update BN running stats.
            self.encoder.eval()
        return self

    # MOSFET-dim chunk size for _forward_core. 4096 in bf16 peak
    # ~8 GB per chunk (4 corner tensors × ~2 GB) + ~1 GB encoder activations.
    # 6 chunks per 24 k-MOSFET batch → half the launches of 2048/chunk → less
    # Python-loop overhead and better GPU utilization on small-channel convs.
    _chunk_size = 4096

    def forward(self, W_um: torch.Tensor, L_um: torch.Tensor,
                is_nmos: torch.Tensor,
                Vbs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Chunked dispatch. Full batches of MOSFETs (24 × 1000) would blow past
        94 GB on H100 because each corner tensor alone is N × 181 × 361 × 4 B
        (6.4 GB at N=24 000). Splitting into chunks bounds peak memory."""
        n = W_um.numel()
        if n == 0:
            return torch.zeros(0, self.embed_dim, device=self.lut.device)

        if self.use_vbs and Vbs is None:
            raise ValueError('use_vbs=True but Vbs tensor not supplied')

        if self._freeze:
            torch_ctx = torch.no_grad
        else:
            torch_ctx = _NullCtx

        def _slice(x, sl):
            return None if x is None else x[sl]

        if n <= self._chunk_size:
            with torch_ctx():
                return self._forward_core(W_um, L_um, is_nmos, Vbs)

        chunks = []
        step = self._chunk_size
        with torch_ctx():
            for i in range(0, n, step):
                sl = slice(i, i + step)
                chunks.append(self._forward_core(
                    W_um[sl], L_um[sl], is_nmos[sl], _slice(Vbs, sl),
                ))
        return torch.cat(chunks, dim=0)

    def _forward_core(self, W_um: torch.Tensor, L_um: torch.Tensor,
                      is_nmos: torch.Tensor,
                      Vbs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute bottleneck-dim IV embeddings for a single chunk."""
        device = self.lut.device
        if W_um.numel() == 0:
            return torch.zeros(0, self.embed_dim, device=device)

        W_um = W_um.to(device).float().clamp_min(1e-9)
        L_um = L_um.to(device).float().clamp_min(1e-9)
        pol_idx = (1 - is_nmos.to(device).long()).clamp(0, 1)  # nmos→0, pmos→1

        log_w = torch.log(W_um).clamp(self.log_W[0], self.log_W[-1])
        log_l = torch.log(L_um).clamp(self.log_L[0], self.log_L[-1])

        iw = (torch.bucketize(log_w, self.log_W) - 1).clamp(0, self._nW - 2)
        il = (torch.bucketize(log_l, self.log_L) - 1).clamp(0, self._nL - 2)

        w_lo = self.log_W[iw]
        w_hi = self.log_W[iw + 1]
        l_lo = self.log_L[il]
        l_hi = self.log_L[il + 1]
        fw = ((log_w - w_lo) / (w_hi - w_lo).clamp_min(1e-12)).view(-1, 1, 1)
        fl = ((log_l - l_lo) / (l_hi - l_lo).clamp_min(1e-12)).view(-1, 1, 1)

        if self.use_vbs:
            vbs_mag = Vbs.to(device).float().abs().clamp(
                float(self.vbs_axis[0]), float(self.vbs_axis[-1]))
            iv = (torch.bucketize(vbs_mag, self.vbs_axis) - 1).clamp(0, self._nVbs - 2)
            v_lo = self.vbs_axis[iv]
            v_hi = self.vbs_axis[iv + 1]
            fv = ((vbs_mag - v_lo) / (v_hi - v_lo).clamp_min(1e-12)).view(-1, 1, 1)

            # Interp one Vbs slice at a time to keep peak memory bounded to
            # 4 corners (same as the no-Vbs path). Each slice returns a full
            # [N, Vgs, Vds] surface; blend by fv.
            # lut[pol, iw, il, iv] → [N, Vgs, Vds]
            def surf_at_vbs(iv_k):
                c00 = self.lut[pol_idx, iw,     il,     iv_k]
                c01 = self.lut[pol_idx, iw,     il + 1, iv_k]
                c10 = self.lut[pol_idx, iw + 1, il,     iv_k]
                c11 = self.lut[pol_idx, iw + 1, il + 1, iv_k]
                return (
                    (1.0 - fw) * (1.0 - fl) * c00
                    + (1.0 - fw) * fl       * c01
                    + fw        * (1.0 - fl) * c10
                    + fw        * fl        * c11
                )
            surf_lo = surf_at_vbs(iv)
            surf_hi = surf_at_vbs(iv + 1)
            surf = surf_lo * (1.0 - fv) + surf_hi * fv
            surf_n = (surf - self.surf_mean) / self.surf_std
            return self.encoder(surf_n.unsqueeze(1))

        # Vbs=0 slice path — 2D interp only.
        c00 = self.lut[pol_idx, iw,     il    ]
        c01 = self.lut[pol_idx, iw,     il + 1]
        c10 = self.lut[pol_idx, iw + 1, il    ]
        c11 = self.lut[pol_idx, iw + 1, il + 1]

        surf = (
            (1.0 - fw) * (1.0 - fl) * c00
            + (1.0 - fw) * fl       * c01
            + fw        * (1.0 - fl) * c10
            + fw        * fl        * c11
        )  # [N, nVgs, nVds] — log10|I_D|

        surf_n = (surf - self.surf_mean) / self.surf_std
        return self.encoder(surf_n.unsqueeze(1))  # [N, 64]
