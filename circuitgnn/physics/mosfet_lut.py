"""SKY130 MOSFET DC operating-point lookup table — query helper.

Supports both v1 (4-D: W, L, Vds, Vgs) and v2 (5-D: adds Vbs) LUTs.

Use like:

    from circuitgnn.physics.mosfet_lut import MosfetLUT

    lut = MosfetLUT.load('datasets/lut/lut_v2/sky130_mosfet_lut_v2.h5')
    params = lut.query('n', W=5.0, L=1.0, Vgs=0.8, Vds=0.9, Vbs=-0.1, M=2)
    id_uA = params['id'] * 1e6

Inputs use positive magnitudes (|Vgs|, |Vds|) for both polarities.
Vbs is signed: NMOS uses negative values (reverse bias), PMOS uses positive.
W, L in micrometers. Currents are per-finger; multiplier `M` scales I, gm, gds.

Loading modes:
  - eager (default for small LUTs): all quantities loaded into RAM, then
    scipy.RegularGridInterpolator is used for fast vectorized queries.
  - lazy (default for files > 10 GB): keeps h5py handles open, reads a tiny
    2^D corner hypercube per query and does manual multilinear interp.
    Slower per-query but avoids the 70 GB RAM blow-up on v2.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Union

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator


ArrayLike = Union[float, np.ndarray]
_LAZY_SIZE_THRESHOLD_BYTES = 10 * 1024 ** 3   # files > 10 GB → lazy mode


class MosfetLUT:
    """Lookup table of SKY130 MOSFET DC operating-point parameters.

    v1 layout: arrays shape (nW, nL, nVds, nVgs), no Vbs_V.
    v2 layout: arrays shape (nW, nL, nVbs, nVds, nVgs) + Vbs_V axis per polarity.
    """

    # Quantities that scale linearly with multiplier M
    _M_SCALED = {'id', 'ig', 'is', 'ib', 'gm', 'gds', 'gmbs',
                 'cgs', 'cgd', 'cgb', 'cds', 'cdb', 'csb'}

    # Quantities interpolated in log10|x| space. Stored linearly, transformed
    # per-query: corners -> log10(max(abs(x), FLOOR)) -> lerp -> 10^x.
    # vth, vdsat, (and von if added) stay linear.
    _LOG_INTERP = {'id', 'ig', 'is', 'ib', 'gm', 'gds', 'gmbs',
                   'cgs', 'cgd', 'cgb', 'cds', 'cdb', 'csb'}
    _LOG_FLOOR = 1e-20  # min |x| before log — caps can go slightly negative

    # Quantities eligible for cubic interpolation. When enabled via
    # enable_cubic(quantity), we eagerly load that quantity's array for
    # both polarities into RAM and build a scipy RGI with method='cubic'.
    _CUBIC_ELIGIBLE = {'id', 'gm', 'gds', 'gmbs'}

    def __init__(self, W_um, L_um, Vgs_V, Vds_V, quantities,
                 data=None, datasets=None, Vbs_V_per_pol=None,
                 h5_file=None):
        self.W_um = np.asarray(W_um, dtype=np.float32)
        self.L_um = np.asarray(L_um, dtype=np.float32)
        self.Vgs_V = np.asarray(Vgs_V, dtype=np.float32)
        self.Vds_V = np.asarray(Vds_V, dtype=np.float32)
        self.quantities = list(quantities)
        self._data = data or {'n': {}, 'p': {}}       # eager arrays
        self._datasets = datasets or {'n': {}, 'p': {}}  # lazy h5py handles
        self._h5_file = h5_file  # keep file open in lazy mode
        self._Vbs_per_pol = Vbs_V_per_pol or {}
        self._has_vbs = {pol: pol in self._Vbs_per_pol for pol in ('n', 'p')}
        self._interp: Dict[str, Dict[str, RegularGridInterpolator]] = {'n': {}, 'p': {}}
        self.lazy = h5_file is not None
        self._cubic_interp: Dict[str, Dict[str, RegularGridInterpolator]] = {'n': {}, 'p': {}}
        self._cubic_enabled: set = set()  # quantities promoted to cubic

    def __del__(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass

    @classmethod
    def load(cls, path: Union[str, Path], lazy: Union[bool, None] = None) -> 'MosfetLUT':
        """Load LUT from HDF5.

        lazy=None (default): auto — eager for files < 10 GB, lazy above.
        lazy=True: always lazy (stream from disk).
        lazy=False: always eager (load all arrays to RAM).
        """
        path = Path(path)
        size = path.stat().st_size
        if lazy is None:
            lazy = size > _LAZY_SIZE_THRESHOLD_BYTES

        if lazy:
            return cls._load_lazy(path)
        return cls._load_eager(path)

    @classmethod
    def _load_eager(cls, path: Path) -> 'MosfetLUT':
        with h5py.File(path, 'r') as h5:
            W = h5['W_um'][:]
            L = h5['L_um'][:]
            Vgs = h5['Vgs_V'][:]
            Vds = h5['Vds_V'][:]
            quantities = list(h5.attrs.get('quantities', []))
            data = {}
            Vbs_per_pol = {}
            for pol in ['n', 'p']:
                grp = h5[pol]
                data[pol] = {q: grp[q][:] for q in quantities if q in grp}
                if 'Vbs_V' in grp:
                    Vbs_per_pol[pol] = grp['Vbs_V'][:]
        return cls(W, L, Vgs, Vds, quantities, data=data,
                   Vbs_V_per_pol=Vbs_per_pol)

    @classmethod
    def _load_lazy(cls, path: Path) -> 'MosfetLUT':
        # 2 GB chunk cache so repeated nearby queries hit RAM.
        h5 = h5py.File(path, 'r', rdcc_nbytes=2 * 1024**3, rdcc_nslots=50021)
        W = h5['W_um'][:]
        L = h5['L_um'][:]
        Vgs = h5['Vgs_V'][:]
        Vds = h5['Vds_V'][:]
        quantities = list(h5.attrs.get('quantities', []))
        datasets = {'n': {}, 'p': {}}
        Vbs_per_pol = {}
        for pol in ['n', 'p']:
            grp = h5[pol]
            for q in quantities:
                if q in grp:
                    datasets[pol][q] = grp[q]
            if 'Vbs_V' in grp:
                Vbs_per_pol[pol] = grp['Vbs_V'][:]
        return cls(W, L, Vgs, Vds, quantities, datasets=datasets,
                   Vbs_V_per_pol=Vbs_per_pol, h5_file=h5)

    # ─────────────────────── Cubic promotion ─────────────────────────
    def enable_cubic(self, quantity: str) -> None:
        """Promote a quantity to cubic interpolation.

        Eagerly loads that quantity's array for BOTH polarities into RAM and
        builds scipy RGIs with method='cubic'. Use for quantities where linear
        is known to be too coarse in smooth regions (e.g. gds in saturation).

        ~4.2 GB RAM per polarity per quantity for the v2 grid shape.
        """
        if quantity not in self._CUBIC_ELIGIBLE:
            raise ValueError(f'Cubic not supported for {quantity!r}. '
                             f'Eligible: {self._CUBIC_ELIGIBLE}')
        log_W = np.log(self.W_um)
        log_L = np.log(self.L_um)
        for pol in ('n', 'p'):
            # Source the array: may be in _data (eager) or _datasets (lazy)
            if quantity in self._data[pol]:
                arr = self._data[pol][quantity]
            elif quantity in self._datasets[pol]:
                arr = self._datasets[pol][quantity][:]
            else:
                continue
            if self._has_vbs[pol]:
                Vbs_axis = self._Vbs_per_pol[pol]
                if Vbs_axis[0] > Vbs_axis[-1]:
                    Vbs_axis = Vbs_axis[::-1]
                    arr = arr[:, :, ::-1, :, :]
                grid = (log_W, log_L, Vbs_axis, self.Vds_V, self.Vgs_V)
            else:
                grid = (log_W, log_L, self.Vds_V, self.Vgs_V)
            if quantity in self._LOG_INTERP:
                arr = np.log10(np.maximum(np.abs(arr), self._LOG_FLOOR))
            self._cubic_interp[pol][quantity] = RegularGridInterpolator(
                grid, arr, method='cubic', bounds_error=False, fill_value=np.nan,
            )
        self._cubic_enabled.add(quantity)

    # ─────────────────────────── Eager path ───────────────────────────
    def _interpolator(self, polarity: str, quantity: str) -> RegularGridInterpolator:
        cache = self._interp[polarity]
        if quantity not in cache:
            # W, L are log-spaced on the grid → interpolate in log space.
            log_W = np.log(self.W_um)
            log_L = np.log(self.L_um)
            if self._has_vbs[polarity]:
                Vbs_axis = self._Vbs_per_pol[polarity]
                data = self._data[polarity][quantity]
                if Vbs_axis[0] > Vbs_axis[-1]:
                    Vbs_axis = Vbs_axis[::-1]
                    data = data[:, :, ::-1, :, :]
                grid = (log_W, log_L, Vbs_axis, self.Vds_V, self.Vgs_V)
            else:
                data = self._data[polarity][quantity]
                grid = (log_W, log_L, self.Vds_V, self.Vgs_V)
            if quantity in self._LOG_INTERP:
                data = np.log10(np.maximum(np.abs(data), self._LOG_FLOOR))
            cache[quantity] = RegularGridInterpolator(
                grid, data, method='linear', bounds_error=False, fill_value=np.nan,
            )
        return cache[quantity]

    # ─────────────────────────── Lazy path ────────────────────────────
    @staticmethod
    def _bracket(axis: np.ndarray, value: float, log: bool = False) -> tuple:
        """Return (idx_lo, frac) s.t. value ≈ axis[idx_lo]*(1-frac)+axis[idx_lo+1]*frac.

        Assumes axis is monotonically increasing. value must already be clipped
        into [axis[0], axis[-1]]. If log=True, the fractional position is
        computed in log space (correct for log-spaced W, L grids).
        """
        n = len(axis)
        if value <= axis[0]:
            return 0, 0.0
        if value >= axis[-1]:
            return n - 2, 1.0
        idx = int(np.searchsorted(axis, value, side='right') - 1)
        idx = max(0, min(n - 2, idx))
        a, b = float(axis[idx]), float(axis[idx + 1])
        if log:
            la, lb, lv = np.log(a), np.log(b), np.log(value)
            frac = 0.0 if lb == la else (lv - la) / (lb - la)
        else:
            frac = 0.0 if b == a else (value - a) / (b - a)
        return idx, float(frac)

    def _query_lazy_scalar(self, polarity, W, L, Vgs, Vds, Vbs, quantities):
        """Scalar query via 2^D corner read + manual multilinear interp."""
        out = {}
        # axis brackets — W, L are log-spaced grids → log fractional position.
        # Vgs, Vds, Vbs are linear-spaced → linear fractional position.
        iw, fw = self._bracket(self.W_um, W, log=True)
        il, fl = self._bracket(self.L_um, L, log=True)
        id_, fd = self._bracket(self.Vds_V, Vds)
        ig, fg = self._bracket(self.Vgs_V, Vgs)

        has_vbs = self._has_vbs[polarity]
        if has_vbs:
            Vbs_axis = self._Vbs_per_pol[polarity]
            # Ensure ascending for bracket lookup
            if Vbs_axis[0] > Vbs_axis[-1]:
                Vbs_asc = Vbs_axis[::-1]
                iv_asc, fv = self._bracket(Vbs_asc, Vbs)
                # map ascending idx back to native (descending) dataset idx
                n_vbs = len(Vbs_axis)
                iv = n_vbs - 2 - iv_asc
                # fv then refers to native idx iv vs iv+1, but ascending picks
                # axis[iv+1] and axis[iv] (descending ordering). Swap frac:
                fv = 1.0 - fv
                # Now in native dataset: value = ds[iv]*(1-fv) + ds[iv+1]*fv
            else:
                iv, fv = self._bracket(Vbs_axis, Vbs)

        for q in quantities:
            ds = self._datasets[polarity].get(q)
            if ds is None:
                continue
            if has_vbs:
                # shape (2, 2, 2, 2, 2) along (W, L, Vbs, Vds, Vgs)
                corners = ds[iw:iw+2, il:il+2, iv:iv+2, id_:id_+2, ig:ig+2]
            else:
                corners = ds[iw:iw+2, il:il+2, id_:id_+2, ig:ig+2]
            log_mode = q in self._LOG_INTERP
            if log_mode:
                corners = np.log10(np.maximum(np.abs(corners), self._LOG_FLOOR))
            if has_vbs:
                c = corners[..., 0] * (1 - fg) + corners[..., 1] * fg
                c = c[..., 0] * (1 - fd) + c[..., 1] * fd
                c = c[..., 0] * (1 - fv) + c[..., 1] * fv
                c = c[..., 0] * (1 - fl) + c[..., 1] * fl
                c = c[0] * (1 - fw) + c[1] * fw
            else:
                c = corners[..., 0] * (1 - fg) + corners[..., 1] * fg
                c = c[..., 0] * (1 - fd) + c[..., 1] * fd
                c = c[..., 0] * (1 - fl) + c[..., 1] * fl
                c = c[0] * (1 - fw) + c[1] * fw
            out[q] = float(10.0 ** c) if log_mode else float(c)
        return out

    # ────────────────────────────── Query ─────────────────────────────
    def query(
        self,
        polarity: str,
        W: ArrayLike, L: ArrayLike,
        Vgs: ArrayLike, Vds: ArrayLike,
        Vbs: ArrayLike = 0.0,
        M: ArrayLike = 1.0,
        quantities: List[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Interpolate MOSFET parameters.

        Args:
            polarity: 'n' or 'p'
            W, L: device width/length in µm
            Vgs, Vds: positive magnitudes [V]
            Vbs: signed value [V]. NMOS: 0 to -1.8; PMOS: 0 to +1.8.
                 Clipped to LUT grid range. For v1 LUT (no Vbs dim), ignored
                 with a warning if non-zero.
            M: multiplier (parallel fingers); scales I, gm, gds
        """
        if polarity not in ('n', 'p'):
            raise ValueError(f'polarity must be "n" or "p", got {polarity!r}')

        quantities = quantities or self.quantities
        shape = np.broadcast_shapes(
            np.shape(W), np.shape(L), np.shape(Vgs), np.shape(Vds),
            np.shape(Vbs), np.shape(M))
        W_a = np.broadcast_to(np.asarray(W, dtype=np.float32), shape)
        L_a = np.broadcast_to(np.asarray(L, dtype=np.float32), shape)
        Vgs_a = np.broadcast_to(np.asarray(Vgs, dtype=np.float32), shape)
        Vds_a = np.broadcast_to(np.asarray(Vds, dtype=np.float32), shape)
        Vbs_a = np.broadcast_to(np.asarray(Vbs, dtype=np.float32), shape)
        M_a = np.broadcast_to(np.asarray(M, dtype=np.float32), shape)

        W_c = np.clip(W_a, self.W_um[0], self.W_um[-1])
        L_c = np.clip(L_a, self.L_um[0], self.L_um[-1])
        Vgs_c = np.clip(Vgs_a, self.Vgs_V[0], self.Vgs_V[-1])
        Vds_c = np.clip(Vds_a, self.Vds_V[0], self.Vds_V[-1])
        if self._has_vbs[polarity]:
            Vbs_axis = self._Vbs_per_pol[polarity]
            vbs_lo, vbs_hi = float(min(Vbs_axis)), float(max(Vbs_axis))
            Vbs_c = np.clip(Vbs_a, vbs_lo, vbs_hi)
        else:
            Vbs_c = Vbs_a
            if not self.lazy:
                pass  # already handled by broadcast; no axis to clip against

        # Precompute cubic query points if any cubic quantities requested
        cubic_qs = [q for q in quantities if q in self._cubic_enabled
                    and q in self._cubic_interp[polarity]]
        cubic_out = {}
        if cubic_qs:
            log_W_c = np.log(W_c); log_L_c = np.log(L_c)
            if self._has_vbs[polarity]:
                pts = np.stack([log_W_c.ravel(), log_L_c.ravel(), Vbs_c.ravel(),
                                Vds_c.ravel(), Vgs_c.ravel()], axis=-1)
            else:
                pts = np.stack([log_W_c.ravel(), log_L_c.ravel(),
                                Vds_c.ravel(), Vgs_c.ravel()], axis=-1)
            for q in cubic_qs:
                vals = self._cubic_interp[polarity][q](pts).reshape(shape).astype(np.float32)
                if q in self._LOG_INTERP:
                    vals = np.power(10.0, vals)
                cubic_out[q] = vals

        if self.lazy:
            flat_W = W_c.ravel(); flat_L = L_c.ravel()
            flat_Vgs = Vgs_c.ravel(); flat_Vds = Vds_c.ravel()
            flat_Vbs = Vbs_c.ravel(); flat_M = M_a.ravel()
            n = flat_W.size
            lazy_qs = [q for q in quantities if q in self._datasets[polarity]
                       and q not in cubic_out]
            out_flat = {q: np.full(n, np.nan, dtype=np.float32) for q in lazy_qs}
            for i in range(n):
                vals = self._query_lazy_scalar(
                    polarity,
                    float(flat_W[i]), float(flat_L[i]),
                    float(flat_Vgs[i]), float(flat_Vds[i]),
                    float(flat_Vbs[i]),
                    list(out_flat.keys()),
                )
                for q, v in vals.items():
                    out_flat[q][i] = v
            out = dict(cubic_out)
            for q, arr in out_flat.items():
                vals = arr.reshape(shape)
                if q in self._M_SCALED:
                    vals = vals * M_a
                out[q] = vals
            # Apply M scaling to cubic quantities too
            for q in list(cubic_out.keys()):
                if q in self._M_SCALED:
                    out[q] = out[q] * M_a
            return out

        # Eager path — W, L passed in log space to match interpolator grid.
        log_W_c = np.log(W_c)
        log_L_c = np.log(L_c)
        if self._has_vbs[polarity]:
            points = np.stack([log_W_c.ravel(), log_L_c.ravel(), Vbs_c.ravel(),
                               Vds_c.ravel(), Vgs_c.ravel()], axis=-1)
        else:
            points = np.stack([log_W_c.ravel(), log_L_c.ravel(),
                               Vds_c.ravel(), Vgs_c.ravel()], axis=-1)

        out = {}
        for q in quantities:
            if q not in self._data[polarity]:
                continue
            interp = self._interpolator(polarity, q)
            vals = interp(points).reshape(shape).astype(np.float32)
            if q in self._LOG_INTERP:
                vals = np.power(10.0, vals)
            if q in self._M_SCALED:
                vals = vals * M_a
            out[q] = vals
        return out
