"""Shared array transforms for the SKY130 MOSFET LUT pipeline.

The HDF5 tables store per-polarity arrays with axes (W, L, Vbs, Vds, Vgs).
All consumers (IVEmbedder, LUTIdQuery, LUTOpQuery) reorient to a
(..., Vgs, Vds) surface layout, floor |id| before taking log10, sort the
Vbs axis to ascending magnitude and stack the two polarities into a single
table indexed [pol, ...] with pol 0=nmos, 1=pmos.
"""
from __future__ import annotations

import numpy as np


def swap_vgs_vds(arr: np.ndarray) -> np.ndarray:
    """Swap the trailing (Vds, Vgs) axes into the (Vgs, Vds) surface layout."""
    return np.swapaxes(arr, -1, -2)


def log10_abs_floor(arr: np.ndarray, floor: float) -> np.ndarray:
    """log10(max(|arr|, floor)) as float32.

    Log-space id interpolation is physically appropriate since id spans
    more than 10 orders of magnitude.
    """
    return np.log10(np.maximum(np.abs(arr), floor)).astype(np.float32)


def sorted_vbs_orders(n_vbs: np.ndarray, p_vbs: np.ndarray,
                      mismatch_message: str):
    """Sort orders putting both polarities' Vbs axes in ascending magnitude.

    NMOS uses negative Vbs, PMOS positive; the magnitude grids must mirror
    each other so a single unified |Vbs| axis serves both polarities.

    Returns:
        (order_n, order_p, vbs_mag): argsort order per polarity and the
        unified ascending |Vbs| axis as float32.

    Raises:
        ValueError(mismatch_message) if the magnitude grids differ.
    """
    n_vbs_mag = np.abs(n_vbs)
    p_vbs_mag = np.abs(p_vbs)
    if not np.allclose(n_vbs_mag, p_vbs_mag):
        raise ValueError(mismatch_message)
    order_n = np.argsort(n_vbs_mag)
    order_p = np.argsort(p_vbs_mag)
    return order_n, order_p, n_vbs_mag[order_n].astype(np.float32)


def sort_slab_by_vbs(arr: np.ndarray, pol_vbs: np.ndarray):
    """Sort one polarity slab (W, L, Vbs, Vgs, Vds) to ascending |Vbs|.

    Returns:
        (arr_sorted, vbs_mag): the reordered slab and its ascending |Vbs|
        axis as float32.
    """
    mag = np.abs(pol_vbs)
    order = np.argsort(mag)
    return arr[:, :, order, :, :], mag[order].astype(np.float32)


def stack_polarities(n_arr: np.ndarray, p_arr: np.ndarray) -> np.ndarray:
    """Stack NMOS and PMOS tables into one array indexed [pol, ...]."""
    return np.stack([n_arr, p_arr], axis=0)
