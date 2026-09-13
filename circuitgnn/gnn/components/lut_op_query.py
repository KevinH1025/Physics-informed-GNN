"""Differentiable SKY130 LUT lookup for (id, gm, gds) at an operating point.

Used by the "iterative-refinement" flow (Option A3):
  1. A frozen pass-1 model predicts V₀.
  2. We compute per-MOSFET Vgs, Vds, Vbs from V₀.
  3. LUTOpQuery returns (id, gm, gds) at that exact bias — physics-exact.
  4. These three scalars become extra input features for a pass-2 GNN.

Quintlinear interpolation over (log W, log L, |Vbs|, Vds, Vgs). Log-space
id interp is physically appropriate (spans 10+ decades). gm and gds are
interpolated directly — they're smooth enough that linear interp is fine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
import torch.nn as nn


class LUTOpQuery(nn.Module):
    LOG_FLOOR = 1e-20

    def __init__(self, lut_path: Union[str, Path]):
        super().__init__()
        lut_path = Path(lut_path)
        with h5py.File(lut_path, 'r') as h5:
            W_um = h5['W_um'][:]
            L_um = h5['L_um'][:]
            Vgs = h5['Vgs_V'][:]
            Vds = h5['Vds_V'][:]

            # id in log10 (physics), gm/gds in linear
            slabs = {'id': [], 'gm': [], 'gds': []}
            for pol in ('n', 'p'):
                pol_vbs = h5[f'{pol}/Vbs_V'][:]
                for q in ('id', 'gm', 'gds'):
                    arr = h5[f'{pol}/{q}'][:]          # (W, L, Vbs, Vds, Vgs)
                    arr = np.transpose(arr, (0, 1, 2, 4, 3))  # (W, L, Vbs, Vgs, Vds)
                    if q == 'id':
                        arr = np.log10(np.maximum(np.abs(arr), self.LOG_FLOOR)).astype(np.float32)
                    else:
                        arr = arr.astype(np.float32)
                    slabs[q].append((arr, pol_vbs))

        # Sort each polarity's Vbs axis to ascending magnitude, stack.
        # All three quantities share the same axes.
        axes_sorted = None
        stacks = {}
        for q in ('id', 'gm', 'gds'):
            pol_slabs = []
            for (arr, pol_vbs) in slabs[q]:
                mag = np.abs(pol_vbs)
                order = np.argsort(mag)
                pol_slabs.append(arr[:, :, order, :, :])
                if axes_sorted is None:
                    axes_sorted = mag[order].astype(np.float32)
            stacks[q] = np.stack(pol_slabs, axis=0)   # [2, W, L, Vbs, Vgs, Vds]

        self.register_buffer('lut_id', torch.from_numpy(stacks['id']))
        self.register_buffer('lut_gm', torch.from_numpy(stacks['gm']))
        self.register_buffer('lut_gds', torch.from_numpy(stacks['gds']))
        self.register_buffer('log_W', torch.from_numpy(np.log(W_um).astype(np.float32)))
        self.register_buffer('log_L', torch.from_numpy(np.log(L_um).astype(np.float32)))
        self.register_buffer('Vgs_axis', torch.from_numpy(Vgs.astype(np.float32)))
        self.register_buffer('Vds_axis', torch.from_numpy(Vds.astype(np.float32)))
        self.register_buffer('Vbs_axis', torch.from_numpy(axes_sorted))

        self._nW = int(self.log_W.numel())
        self._nL = int(self.log_L.numel())
        self._nVbs = int(self.Vbs_axis.numel())
        self._nVgs = int(self.Vgs_axis.numel())
        self._nVds = int(self.Vds_axis.numel())

    @staticmethod
    def _bracket(axis: torch.Tensor, values: torch.Tensor):
        axis = axis.detach()
        n = int(axis.numel())
        v_c = values.clamp(float(axis[0]), float(axis[-1]))
        flat = v_c.reshape(-1)
        idx = torch.bucketize(flat, axis) - 1
        idx = idx.clamp(0, n - 2).reshape(values.shape)
        lo = axis[idx]
        hi = axis[idx + 1]
        frac = (v_c - lo) / (hi - lo).clamp_min(1e-12)
        return idx, frac

    def _interp_lut(self, lut, pol, iw, il, iv, ig, id_,
                    fw, fl, fv, fg, fd):
        """Quintlinear interp. All args are [N] tensors (fractions) or
        [N] long indices. lut is [2, W, L, Vbs, Vgs, Vds]."""
        def c(w, l_, v, g, d):
            return lut[pol, w, l_, v, g, d]
        # 32 corners: (w, l, v, g, d) each 0/1
        # Nested blending: Vds inner → Vgs → Vbs → L → W (outer)
        def blend(a, b, f): return a * (1.0 - f) + b * f
        acc = None
        for dw in (0, 1):
            for dl in (0, 1):
                for dv in (0, 1):
                    for dg in (0, 1):
                        x0 = c(iw + dw, il + dl, iv + dv, ig + dg, id_    )
                        x1 = c(iw + dw, il + dl, iv + dv, ig + dg, id_ + 1)
                        x_g = blend(x0, x1, fd)
                        tag = (dw, dl, dv, dg)
                        if tag == (0, 0, 0, 0):
                            stash = {tag: x_g}
                        else:
                            stash[tag] = x_g
        # Blend Vgs
        for dw in (0, 1):
            for dl in (0, 1):
                for dv in (0, 1):
                    a = stash[(dw, dl, dv, 0)]
                    b = stash[(dw, dl, dv, 1)]
                    stash[(dw, dl, dv)] = blend(a, b, fg)
        # Blend Vbs
        for dw in (0, 1):
            for dl in (0, 1):
                a = stash[(dw, dl, 0)]
                b = stash[(dw, dl, 1)]
                stash[(dw, dl)] = blend(a, b, fv)
        # Blend L
        for dw in (0, 1):
            a = stash[(dw, 0)]
            b = stash[(dw, 1)]
            stash[(dw,)] = blend(a, b, fl)
        # Blend W
        return blend(stash[(0,)], stash[(1,)], fw)

    def forward(self, W_um: torch.Tensor, L_um: torch.Tensor,
                Vgs: torch.Tensor, Vds: torch.Tensor, Vbs: torch.Tensor,
                M: torch.Tensor, is_nmos: torch.Tensor):
        """Return (id_linear, gm, gds), each [N], all in SI units.

        V inputs are SIGNED; LUT uses magnitudes internally per polarity.
        id and gm, gds scale linearly with M.
        """
        device = self.lut_id.device
        if W_um.numel() == 0:
            z = torch.zeros(0, device=device)
            return z, z, z

        W_um = W_um.to(device).float().clamp_min(1e-9)
        L_um = L_um.to(device).float().clamp_min(1e-9)
        Vgs_mag = Vgs.to(device).float().abs()
        Vds_mag = Vds.to(device).float().abs()
        Vbs_mag = Vbs.to(device).float().abs()
        M = M.to(device).float().clamp_min(1.0)
        pol = (1 - is_nmos.to(device).long()).clamp(0, 1)

        log_w = torch.log(W_um)
        log_l = torch.log(L_um)
        iw, fw = self._bracket(self.log_W, log_w)
        il, fl = self._bracket(self.log_L, log_l)
        iv, fv = self._bracket(self.Vbs_axis, Vbs_mag)
        ig, fg = self._bracket(self.Vgs_axis, Vgs_mag)
        id_, fd = self._bracket(self.Vds_axis, Vds_mag)

        log_id = self._interp_lut(self.lut_id,  pol, iw, il, iv, ig, id_,
                                   fw, fl, fv, fg, fd)
        id_linear = torch.pow(10.0, log_id) * M
        gm  = self._interp_lut(self.lut_gm,  pol, iw, il, iv, ig, id_,
                                fw, fl, fv, fg, fd) * M
        gds = self._interp_lut(self.lut_gds, pol, iw, il, iv, ig, id_,
                                fw, fl, fv, fg, fd) * M
        return id_linear, gm, gds
