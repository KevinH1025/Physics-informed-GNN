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

from .interp import bracket, quintlinear
from .io import log10_abs_floor, sorted_vbs_orders, stack_polarities, swap_vgs_vds


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
        n_log = log10_abs_floor(swap_vgs_vds(n_id), self.LOG_FLOOR)   # (W, L, Vbs, Vgs, Vds)
        p_log = log10_abs_floor(swap_vgs_vds(p_id), self.LOG_FLOOR)

        # Sort both Vbs axes to ascending magnitude so a single unified
        # |Vbs| axis can be used for both polarities.
        order_n, order_p, vbs_mag = sorted_vbs_orders(
            n_vbs, p_vbs,
            'NMOS and PMOS Vbs magnitudes differ; expected mirrored axes.')
        n_log = n_log[:, :, order_n, :, :]
        p_log = p_log[:, :, order_p, :, :]
        # vbs_mag holds ascending magnitudes: [0, 0.025, ..., 1.8]

        # Stack polarities: lut[pol, W, L, Vbs, Vgs, Vds]
        lut = stack_polarities(n_log, p_log)
        self.register_buffer('lut', torch.from_numpy(lut))
        self.register_buffer('log_W', torch.from_numpy(np.log(W_um).astype(np.float32)))
        self.register_buffer('log_L', torch.from_numpy(np.log(L_um).astype(np.float32)))
        self.register_buffer('Vgs_axis', torch.from_numpy(Vgs.astype(np.float32)))
        self.register_buffer('Vds_axis', torch.from_numpy(Vds.astype(np.float32)))
        self.register_buffer('Vbs_axis', torch.from_numpy(vbs_mag))

        self._nW = int(self.log_W.numel())
        self._nL = int(self.log_L.numel())
        self._nVbs = int(self.Vbs_axis.numel())
        self._nVgs = int(self.Vgs_axis.numel())
        self._nVds = int(self.Vds_axis.numel())

    # ─────────────────────────── Helpers ───────────────────────────
    _bracket = staticmethod(bracket)

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

        # Blend the 2^5 = 32 corner values (each query is independent).
        # lut index order: [pol, W, L, Vbs, Vgs, Vds]
        log_id = quintlinear(self.lut, pol, iw, il, iv, ig, id_,
                             fw, fl, fv, fg, fd)   # log10(|id_per_finger|)

        id_per_finger = torch.pow(10.0, log_id)
        return id_per_finger * M
