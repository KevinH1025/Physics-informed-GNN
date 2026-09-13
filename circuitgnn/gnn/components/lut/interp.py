"""Differentiable interpolation helpers for the SKY130 MOSFET LUT.

bracket() finds interval indices and fractional positions on a 1D axis;
quintlinear() blends the 2^5 corner values over (W, L, Vbs, Vgs, Vds).
"""
from __future__ import annotations

import torch


def bracket(axis: torch.Tensor, values: torch.Tensor) -> tuple:
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


def quintlinear(lut, pol, iw, il, iv, ig, id_,
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
