"""Differentiable id lookup from the SKY130 v2 MOSFET LUT.

Given (W_um, L_um, Vgs, Vds, Vbs, M, is_nmos), returns linear |id| via
quintlinear interpolation over the log10|id| tables (interpolating in log
space is physically appropriate — id spans >10 orders of magnitude).

Gradients flow through Vgs, Vds, Vbs (differentiable bilinear-in-W/L +
trilinear-in-V weights). W, L are design parameters (no grad path via them
— they're gathered by integer indices with linear mixing, but the fractions
are detached from autograd).

Axes:
  log_W, log_L: log-spaced axes (20 W × 80 L values)
  Vgs_axis, Vds_axis: uniform grids (181 × 361)
  Vbs_axis: 10 values in ascending magnitude per polarity (same magnitude set
    for both pols: 0 to 1.8V; sign is inferred from polarity)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
import torch.nn as nn


class LUTIdQuery(nn.Module):
    LOG_FLOOR = 1e-20

    def __init__(self, lut_path: Union[str, Path]):
        super().__init__()
        lut_path = Path(lut_path)
        with h5py.File(lut_path, 'r') as h5:
            W_um = h5['W_um'][:]
            L_um = h5['L_um'][:]
            Vgs = h5['Vgs_V'][:]
            Vds = h5['Vds_V'][:]
            n_id = h5['n/id'][:]          # (W, L, Vbs, Vds, Vgs)
            p_id = h5['p/id'][:]
            n_vbs = h5['n/Vbs_V'][:]      # negative: [0, -0.025, ..., -1.8]
            p_vbs = h5['p/Vbs_V'][:]      # positive: [0,  0.025, ...,  1.8]

        # Reorient so last two axes are (Vgs, Vds) to match the usual
        # "surface" layout. log10(|id|) for physically appropriate interp.
        n_id = np.transpose(n_id, (0, 1, 2, 4, 3))   # (W, L, Vbs, Vgs, Vds)
        p_id = np.transpose(p_id, (0, 1, 2, 4, 3))
        n_log = np.log10(np.maximum(np.abs(n_id), self.LOG_FLOOR)).astype(np.float32)
        p_log = np.log10(np.maximum(np.abs(p_id), self.LOG_FLOOR)).astype(np.float32)

        # Sort both Vbs axes to ascending magnitude so a single unified
        # |Vbs| axis can be used for both polarities.
        n_vbs_mag = np.abs(n_vbs)
        p_vbs_mag = np.abs(p_vbs)
        if not np.allclose(n_vbs_mag, p_vbs_mag):
            raise ValueError('NMOS and PMOS Vbs magnitudes differ; '
                             'expected mirrored axes.')
        vbs_mag = n_vbs_mag.copy()
        order_n = np.argsort(n_vbs_mag)
        order_p = np.argsort(p_vbs_mag)
        n_log = n_log[:, :, order_n, :, :]
        p_log = p_log[:, :, order_p, :, :]
        vbs_mag = vbs_mag[order_n]   # ascending magnitudes: [0, 0.025, ..., 1.8]

        # Stack polarities: lut[pol, W, L, Vbs, Vgs, Vds]
        lut = np.stack([n_log, p_log], axis=0)
        self.register_buffer('lut', torch.from_numpy(lut))
        self.register_buffer('log_W', torch.from_numpy(np.log(W_um).astype(np.float32)))
        self.register_buffer('log_L', torch.from_numpy(np.log(L_um).astype(np.float32)))
        self.register_buffer('Vgs_axis', torch.from_numpy(Vgs.astype(np.float32)))
        self.register_buffer('Vds_axis', torch.from_numpy(Vds.astype(np.float32)))
        self.register_buffer('Vbs_axis', torch.from_numpy(vbs_mag.astype(np.float32)))

        self._nW = int(self.log_W.numel())
        self._nL = int(self.log_L.numel())
        self._nVbs = int(self.Vbs_axis.numel())
        self._nVgs = int(self.Vgs_axis.numel())
        self._nVds = int(self.Vds_axis.numel())

    # ─────────────────────────── Helpers ───────────────────────────
    @staticmethod
    def _bracket(axis: torch.Tensor, values: torch.Tensor) -> tuple:
        """Return (idx_lo, frac) where values ≈ axis[idx_lo](1-frac) + axis[idx_lo+1]*frac.

        Values clipped to [axis[0], axis[-1]]. Fractions are differentiable
        wrt values; indices are detached.
        """
        axis = axis.detach()
        n = int(axis.numel())
        v_c = values.clamp(float(axis[0]), float(axis[-1]))
        # bucketize requires 1D axis; values can be any shape
        flat = v_c.reshape(-1)
        idx = torch.bucketize(flat, axis) - 1
        idx = idx.clamp(0, n - 2).reshape(values.shape)
        lo = axis[idx]
        hi = axis[idx + 1]
        frac = (v_c - lo) / (hi - lo).clamp_min(1e-12)
        return idx, frac

    # ─────────────────────────── Forward ───────────────────────────
    def forward(self, W_um: torch.Tensor, L_um: torch.Tensor,
                Vgs: torch.Tensor, Vds: torch.Tensor, Vbs: torch.Tensor,
                M: torch.Tensor, is_nmos: torch.Tensor) -> torch.Tensor:
        """Return per-MOSFET |id_total| = |id_per_finger(W, L, |Vgs|, |Vds|, |Vbs|, pol)| × M.

        All V inputs may be signed — we take magnitudes for LUT axes (matches
        LUT convention: axes are in positive magnitude).

        Args:
            W_um: [M] widths in µm
            L_um: [M] lengths in µm
            Vgs, Vds, Vbs: [M] terminal voltage *differences* in V (signed)
            M:       [M] multiplier (fingers)
            is_nmos: [M] 1=nmos, 0=pmos

        Returns:
            [M] |id| in amperes (linear), positive.
        """
        dev = self.lut.device
        if W_um.numel() == 0:
            return torch.zeros(0, device=dev)

        W_um = W_um.to(dev).float().clamp_min(1e-9)
        L_um = L_um.to(dev).float().clamp_min(1e-9)
        Vgs_mag = Vgs.to(dev).float().abs()
        Vds_mag = Vds.to(dev).float().abs()
        Vbs_mag = Vbs.to(dev).float().abs()
        M = M.to(dev).float().clamp_min(1.0)
        pol = (1 - is_nmos.to(dev).long()).clamp(0, 1)   # nmos→0, pmos→1

        log_w = torch.log(W_um)
        log_l = torch.log(L_um)

        iw, fw = self._bracket(self.log_W, log_w)
        il, fl = self._bracket(self.log_L, log_l)
        iv, fv = self._bracket(self.Vbs_axis, Vbs_mag)
        ig, fg = self._bracket(self.Vgs_axis, Vgs_mag)
        id_, fd = self._bracket(self.Vds_axis, Vds_mag)

        # Reshape fractions so broadcasting lines up (we'll keep scalar per
        # sample; each query is independent).
        # Gather the 2^5 = 32 corner values using advanced indexing. Shape
        # of each corner tensor: [N].
        # lut index order: [pol, W, L, Vbs, Vgs, Vds]
        def c(i_w, i_l, i_v, i_g, i_d):
            return self.lut[pol, i_w, i_l, i_v, i_g, i_d]

        # Order of nesting (innermost blended first): Vds, then Vgs, Vbs, L, W
        v000_00 = c(iw,     il,     iv,     ig,     id_    )
        v000_01 = c(iw,     il,     iv,     ig,     id_ + 1)
        v000_10 = c(iw,     il,     iv,     ig + 1, id_    )
        v000_11 = c(iw,     il,     iv,     ig + 1, id_ + 1)
        v001_00 = c(iw,     il,     iv + 1, ig,     id_    )
        v001_01 = c(iw,     il,     iv + 1, ig,     id_ + 1)
        v001_10 = c(iw,     il,     iv + 1, ig + 1, id_    )
        v001_11 = c(iw,     il,     iv + 1, ig + 1, id_ + 1)
        v010_00 = c(iw,     il + 1, iv,     ig,     id_    )
        v010_01 = c(iw,     il + 1, iv,     ig,     id_ + 1)
        v010_10 = c(iw,     il + 1, iv,     ig + 1, id_    )
        v010_11 = c(iw,     il + 1, iv,     ig + 1, id_ + 1)
        v011_00 = c(iw,     il + 1, iv + 1, ig,     id_    )
        v011_01 = c(iw,     il + 1, iv + 1, ig,     id_ + 1)
        v011_10 = c(iw,     il + 1, iv + 1, ig + 1, id_    )
        v011_11 = c(iw,     il + 1, iv + 1, ig + 1, id_ + 1)
        v100_00 = c(iw + 1, il,     iv,     ig,     id_    )
        v100_01 = c(iw + 1, il,     iv,     ig,     id_ + 1)
        v100_10 = c(iw + 1, il,     iv,     ig + 1, id_    )
        v100_11 = c(iw + 1, il,     iv,     ig + 1, id_ + 1)
        v101_00 = c(iw + 1, il,     iv + 1, ig,     id_    )
        v101_01 = c(iw + 1, il,     iv + 1, ig,     id_ + 1)
        v101_10 = c(iw + 1, il,     iv + 1, ig + 1, id_    )
        v101_11 = c(iw + 1, il,     iv + 1, ig + 1, id_ + 1)
        v110_00 = c(iw + 1, il + 1, iv,     ig,     id_    )
        v110_01 = c(iw + 1, il + 1, iv,     ig,     id_ + 1)
        v110_10 = c(iw + 1, il + 1, iv,     ig + 1, id_    )
        v110_11 = c(iw + 1, il + 1, iv,     ig + 1, id_ + 1)
        v111_00 = c(iw + 1, il + 1, iv + 1, ig,     id_    )
        v111_01 = c(iw + 1, il + 1, iv + 1, ig,     id_ + 1)
        v111_10 = c(iw + 1, il + 1, iv + 1, ig + 1, id_    )
        v111_11 = c(iw + 1, il + 1, iv + 1, ig + 1, id_ + 1)

        # Interp over Vds (innermost)
        def blend_d(a, b):
            return a * (1 - fd) + b * fd
        w000_0 = blend_d(v000_00, v000_01)
        w000_1 = blend_d(v000_10, v000_11)
        w001_0 = blend_d(v001_00, v001_01)
        w001_1 = blend_d(v001_10, v001_11)
        w010_0 = blend_d(v010_00, v010_01)
        w010_1 = blend_d(v010_10, v010_11)
        w011_0 = blend_d(v011_00, v011_01)
        w011_1 = blend_d(v011_10, v011_11)
        w100_0 = blend_d(v100_00, v100_01)
        w100_1 = blend_d(v100_10, v100_11)
        w101_0 = blend_d(v101_00, v101_01)
        w101_1 = blend_d(v101_10, v101_11)
        w110_0 = blend_d(v110_00, v110_01)
        w110_1 = blend_d(v110_10, v110_11)
        w111_0 = blend_d(v111_00, v111_01)
        w111_1 = blend_d(v111_10, v111_11)

        # Interp over Vgs
        def blend_g(a, b):
            return a * (1 - fg) + b * fg
        x000 = blend_g(w000_0, w000_1)
        x001 = blend_g(w001_0, w001_1)
        x010 = blend_g(w010_0, w010_1)
        x011 = blend_g(w011_0, w011_1)
        x100 = blend_g(w100_0, w100_1)
        x101 = blend_g(w101_0, w101_1)
        x110 = blend_g(w110_0, w110_1)
        x111 = blend_g(w111_0, w111_1)

        # Interp over Vbs
        def blend_v(a, b):
            return a * (1 - fv) + b * fv
        y00 = blend_v(x000, x001)
        y01 = blend_v(x010, x011)
        y10 = blend_v(x100, x101)
        y11 = blend_v(x110, x111)

        # Interp over L (in log space)
        z0 = y00 * (1 - fl) + y01 * fl
        z1 = y10 * (1 - fl) + y11 * fl

        # Interp over W (in log space)
        log_id = z0 * (1 - fw) + z1 * fw      # log10(|id_per_finger|)

        id_per_finger = torch.pow(10.0, log_id)
        return id_per_finger * M
