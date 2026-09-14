#!/usr/bin/env python3
"""Build SKY130 MOSFET DC operating-point LUT — v2 (with Vbs dimension).

5-D LUT keyed by (W, L, Vbs, Vds, Vgs) for each polarity.

Pattern: workers return results through IPC, main accumulates in RAM and
flushes to temp pickle files every save_interval tasks (mirrors the proven
pattern in scripts/generate_dataset.py). At the end, temp pickles are
combined into one HDF5 file.

Usage:
    python scripts/migrations/build_mosfet_lut.py
    python scripts/migrations/build_mosfet_lut.py --quick
"""

import argparse
import gc
import multiprocessing
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import h5py
from tqdm import tqdm

os.environ.setdefault('PYSPICE_LIBRARY_PATH', '/usr/local/lib')

SKY130_ROOT = '/home/kevin/tech/sky130/libraries/sky130_fd_pr/latest'
NFET_MODEL = f'{SKY130_ROOT}/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.corner.spice'
PFET_MODEL = f'{SKY130_ROOT}/cells/pfet_01v8/sky130_fd_pr__pfet_01v8__tt.corner.spice'

MISMATCH_PARAMS = """.param mc_mm_switch=0
.param sky130_fd_pr__nfet_01v8__toxe_slope=0
.param sky130_fd_pr__nfet_01v8__toxe_slope_spectre=0
.param sky130_fd_pr__nfet_01v8__vth0_slope=0
.param sky130_fd_pr__nfet_01v8__vth0_slope1=0
.param sky130_fd_pr__nfet_01v8__vth0_slope_spectre=0
.param sky130_fd_pr__nfet_01v8__voff_slope=0
.param sky130_fd_pr__nfet_01v8__voff_slope_spectre=0
.param sky130_fd_pr__nfet_01v8__nfactor_slope=0
.param sky130_fd_pr__pfet_01v8__toxe_slope=0
.param sky130_fd_pr__pfet_01v8__toxe_slope1=0
.param sky130_fd_pr__pfet_01v8__toxe_slope_spectre=0
.param sky130_fd_pr__pfet_01v8__vth0_slope=0
.param sky130_fd_pr__pfet_01v8__vth0_slope1=0
.param sky130_fd_pr__pfet_01v8__vth0_slope_spectre=0
.param sky130_fd_pr__pfet_01v8__voff_slope=0
.param sky130_fd_pr__pfet_01v8__voff_slope1=0
.param sky130_fd_pr__pfet_01v8__voff_slope_spectre=0
.param sky130_fd_pr__pfet_01v8__nfactor_slope=0
.param sky130_fd_pr__pfet_01v8__nfactor_slope1=0
.param sky130_fd_pr__pfet_01v8__wlod_diff=0
.param sky130_fd_pr__pfet_01v8__kvth0_diff=0
.param sky130_fd_pr__pfet_01v8__lkvth0_diff=0
.param sky130_fd_pr__pfet_01v8__wkvth0_diff=0
.param sky130_fd_pr__pfet_01v8__ku0_diff=0
.param sky130_fd_pr__pfet_01v8__lku0_diff=0
.param sky130_fd_pr__pfet_01v8__wku0_diff=0
.param sky130_fd_pr__pfet_01v8__kvsat_diff=0
"""

QUANTITIES = [
    'id', 'ig', 'is', 'ib',
    'vth', 'vdsat',
    'gm', 'gds', 'gmbs',
    'cgs', 'cgd', 'cgb', 'cds', 'cdb', 'csb',
]

VBS_GRID_NMOS_V = np.array([0, -0.025, -0.050, -0.100, -0.200, -0.400, -0.800, -1.200, -1.600, -1.800], dtype=np.float32)
VBS_GRID_PMOS_V = np.array([0, +0.025, +0.050, +0.100, +0.200, +0.400, +0.800, +1.200, +1.600, +1.800], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────
# Worker — runs ALL Vbs sweeps for one (W, L) × polarity in a single
# ngspice session. This amortizes the ~500 ms ngspice startup cost
# across all Vbs values instead of paying it per sweep.
# ──────────────────────────────────────────────────────────────────
def worker_run_wl_bundle(bundle):
    """Run all Vbs sweeps for a single (polarity, W, L) in one ngspice session.

    Reuses a single NgSpiceShared instance across every Vbs sweep in the
    bundle — amortizes the ~500 ms ngspice startup cost.

    bundle = (w_idx, l_idx, polarity, W_um, L_um, vbs_list,
              vgs_start, vgs_stop, vgs_step, n_vgs,
              vds_start, vds_stop, vds_step, n_vds)

    Returns (w_idx, l_idx, polarity, results_list, error_count)
    where results_list = [(vbs_idx, result_dict_or_None), ...]
    """
    import logging
    logging.getLogger('PySpice.Spice.NgSpice.Shared').setLevel(logging.ERROR)
    os.environ['PYSPICE_LIBRARY_PATH'] = '/usr/local/lib'
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    from contextlib import contextmanager

    @contextmanager
    def _suppress():
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        saved_out = os.dup(stdout_fd)
        saved_err = os.dup(stderr_fd)
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, stdout_fd); os.dup2(devnull, stderr_fd)
            os.close(devnull)
            yield
        finally:
            os.dup2(saved_out, stdout_fd); os.dup2(saved_err, stderr_fd)
            os.close(saved_out); os.close(saved_err)

    (w_idx, l_idx, polarity, w_um, l_um, vbs_list,
     vgs_start, vgs_stop, vgs_step, n_vgs,
     vds_start, vds_stop, vds_step, n_vds) = bundle

    model_name = f'sky130_fd_pr__{polarity}fet_01v8'
    model_path = NFET_MODEL if polarity == 'n' else PFET_MODEL

    sign = 1.0 if polarity == 'n' else -1.0
    vg_start, vg_stop, vg_step = sign * vgs_start, sign * vgs_stop, sign * vgs_step
    vd_start, vd_stop, vd_step = sign * vds_start, sign * vds_stop, sign * vds_step
    save_items = ' '.join(f'@m.xm1.m{model_name}[{q}]' for q in QUANTITIES)

    results_list = []
    errors = 0
    ngspice = None
    # Build netlist ONCE with Vb at first vbs value; use `alter` to change
    # Vb between runs. Each run creates a new plot: dc1, dc2, dc3, …
    first_vbs = float(vbs_list[0][1])
    netlist = f"""* MOSFET LUT sweep
{MISMATCH_PARAMS}
.include "{model_path}"

Vd d 0 DC {vd_start}
Vg g 0 DC {vg_start}
Vs s 0 DC 0
Vb b 0 DC {first_vbs}

XM1 d g s b {model_name} W={w_um}u L={l_um}u

.save {save_items}
.dc Vg {vg_start} {vg_stop} {vg_step} Vd {vd_start} {vd_stop} {vd_step}
.end
"""
    try:
        with _suppress():
            ngspice = NgSpiceShared.new_instance()
            ngspice.load_circuit(netlist)
            for run_i, (vbs_idx, vbs_V) in enumerate(vbs_list):
                try:
                    ngspice.exec_command(f'alter vb = {float(vbs_V)}')
                    try: ngspice.run()
                    except Exception: pass

                    # Each successive .run() creates dc1, dc2, …
                    plot_dc = None
                    for pn in [f'dc{run_i + 1}', 'dc1', 'dc', 'sweep']:
                        try:
                            plot_dc = ngspice.plot('dc', pn)
                            break
                        except Exception:
                            continue
                    if plot_dc is None:
                        results_list.append((vbs_idx, None))
                        errors += 1
                        continue

                    keys = list(plot_dc.keys())
                    out = {}
                    for q in QUANTITIES:
                        matches = [k for k in keys if k.lower().endswith(f'[{q}]')]
                        if not matches:
                            continue
                        wf = np.asarray(plot_dc[matches[0]].to_waveform())
                        if wf.size != n_vgs * n_vds:
                            arr = np.full(n_vgs * n_vds, np.nan)
                            n = min(wf.size, n_vgs * n_vds)
                            arr[:n] = wf[:n]
                            wf = arr
                        arr2d = wf.reshape(n_vds, n_vgs).astype(np.float32)
                        if polarity == 'p' and q in ('id', 'ig', 'is', 'ib', 'gm', 'gds'):
                            arr2d = np.abs(arr2d)
                        out[q] = arr2d
                    results_list.append((vbs_idx, out))
                except Exception:
                    results_list.append((vbs_idx, None))
                    errors += 1
        return (w_idx, l_idx, polarity, results_list, errors)
    except Exception:
        return (w_idx, l_idx, polarity, results_list, errors + 1)
    finally:
        if ngspice is not None:
            try: ngspice.destroy()
            except Exception: pass


# ──────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='Tiny grid smoke test')
    ap.add_argument('--out', default='datasets/lut/sky130_mosfet_lut_v2.h5')
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--save-interval', type=int, default=1000,
                    help='Flush RAM batch to temp pickle every N tasks, then restart workers')
    ap.add_argument('--h5-compression', default='gzip', choices=['gzip', 'lzf', 'none'])
    args = ap.parse_args()

    if args.quick:
        W_grid = np.logspace(np.log10(0.42), np.log10(50), 3)
        L_grid = np.logspace(np.log10(0.15), np.log10(20), 4)
        Vgs_grid = np.round(np.arange(0, 1.81, 0.1), 4)
        Vds_grid = np.round(np.arange(0, 1.81, 0.05), 4)
        Vbs_NMOS = VBS_GRID_NMOS_V[:3]
        Vbs_PMOS = VBS_GRID_PMOS_V[:3]
    else:
        W_grid = np.logspace(np.log10(0.42), np.log10(50), 20)
        L_grid = np.logspace(np.log10(0.15), np.log10(20), 80)
        Vgs_grid = np.round(np.arange(0, 1.801, 0.01), 4)
        Vds_grid = np.round(np.arange(0, 1.801, 0.005), 4)
        Vbs_NMOS = VBS_GRID_NMOS_V
        Vbs_PMOS = VBS_GRID_PMOS_V

    n_W, n_L = len(W_grid), len(L_grid)
    n_Vgs, n_Vds = len(Vgs_grid), len(Vds_grid)
    n_Vbs_n, n_Vbs_p = len(Vbs_NMOS), len(Vbs_PMOS)

    vgs_start, vgs_stop = float(Vgs_grid[0]), float(Vgs_grid[-1])
    vds_start, vds_stop = float(Vds_grid[0]), float(Vds_grid[-1])
    vgs_step = float(Vgs_grid[1] - Vgs_grid[0])
    vds_step = float(Vds_grid[1] - Vds_grid[0])

    n_workers = args.workers
    total_points_n = n_W * n_L * n_Vbs_n * n_Vds * n_Vgs
    total_points_p = n_W * n_L * n_Vbs_p * n_Vds * n_Vgs
    print(f'Grid: W={n_W}, L={n_L}, Vgs={n_Vgs} (step {vgs_step*1000:.0f} mV), '
          f'Vds={n_Vds} (step {vds_step*1000:.0f} mV)')
    print(f'  Vbs NMOS: {n_Vbs_n} pts, range [{Vbs_NMOS.min()*1000:.0f}, {Vbs_NMOS.max()*1000:.0f}] mV')
    print(f'  Vbs PMOS: {n_Vbs_p} pts, range [{Vbs_PMOS.min()*1000:.0f}, {Vbs_PMOS.max()*1000:.0f}] mV')
    print(f'  NMOS: {total_points_n:,} points ({n_W*n_L*n_Vbs_n:,} sweeps)')
    print(f'  PMOS: {total_points_p:,} points ({n_W*n_L*n_Vbs_p:,} sweeps)')
    print(f'  Workers: {n_workers}  save_interval: {args.save_interval}')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.with_suffix('.tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Build bundle task list — ONE task per (polarity, W, L) running all Vbs
    # values in a single ngspice session. This amortizes ngspice startup
    # (~500 ms) across all 10 Vbs sweeps, so we pay startup once per bundle
    # instead of per sweep.
    tasks = []
    vbs_list_n = [(k, float(vbs)) for k, vbs in enumerate(Vbs_NMOS)]
    vbs_list_p = [(k, float(vbs)) for k, vbs in enumerate(Vbs_PMOS)]
    for i, w in enumerate(W_grid):
        for j, l in enumerate(L_grid):
            tasks.append((i, j, 'n', float(w), float(l), vbs_list_n,
                          vgs_start, vgs_stop, vgs_step, n_Vgs,
                          vds_start, vds_stop, vds_step, n_Vds))
            tasks.append((i, j, 'p', float(w), float(l), vbs_list_p,
                          vgs_start, vgs_stop, vgs_step, n_Vgs,
                          vds_start, vds_stop, vds_step, n_Vds))
    total_bundles = len(tasks)
    sweeps_per_bundle_n = len(vbs_list_n)
    sweeps_per_bundle_p = len(vbs_list_p)
    total_sweeps = n_W * n_L * (sweeps_per_bundle_n + sweeps_per_bundle_p)
    print(f'  Total bundles: {total_bundles:,}  (≈{total_sweeps:,} sweeps)')

    # ── Run with periodic worker restart (mirrors scripts/generate_dataset.py) ──
    t0 = time.time()
    errors = 0
    bundle_idx = 0
    temp_files = []
    batch = []  # list of (polarity, w_idx, l_idx, vbs_idx, results)
    sweeps_since_restart = 0
    executor = ProcessPoolExecutor(max_workers=n_workers)

    def flush_batch():
        nonlocal batch, temp_files
        if not batch: return
        f_idx = len(temp_files)
        tf = tmp_dir / f'batch_{f_idx:04d}.pkl'
        with open(tf, 'wb') as f:
            pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)
        temp_files.append(tf)
        batch = []
        gc.collect()

    with tqdm(total=total_sweeps, desc='Sweeps') as pbar:
        # Keep a rolling in-flight window so all 20 workers stay saturated.
        in_flight = {}
        def _submit_one(idx):
            fut = executor.submit(worker_run_wl_bundle, tasks[idx])
            in_flight[fut] = idx

        # Prime with 2× workers worth of bundles
        while bundle_idx < total_bundles and len(in_flight) < n_workers * 2:
            _submit_one(bundle_idx); bundle_idx += 1

        while in_flight:
            done_iter = as_completed(in_flight)
            fut = next(done_iter)
            del in_flight[fut]
            try:
                w_idx, l_idx, polarity, results_list, bundle_errors = fut.result(timeout=600)
                errors += bundle_errors
                for vbs_idx, result in results_list:
                    if result is not None:
                        batch.append((polarity, w_idx, l_idx, vbs_idx, result))
                    sweeps_since_restart += 1
                    pbar.update(1)
                pbar.set_postfix({'err': errors, 'buf': len(batch)})
            except Exception as e:
                pbar.write(f'[err] bundle future: {e}')
                # Still need to advance counters so progress doesn't stall
                est = sweeps_per_bundle_n  # unknown polarity → approx
                errors += est
                sweeps_since_restart += est
                pbar.update(est)

            # Keep pool fed
            if bundle_idx < total_bundles:
                _submit_one(bundle_idx); bundle_idx += 1

            # Flush & restart when we've processed ≥ save_interval SWEEPS
            if sweeps_since_restart >= args.save_interval:
                # Drain currently in-flight before restart to avoid orphaned results
                pbar.write(f'[flush {len(batch)} results; restart workers after '
                           f'{sweeps_since_restart} sweeps]')
                # Wait for remaining in-flight futures to complete cleanly
                for pending_fut in list(in_flight):
                    try:
                        w_idx, l_idx, polarity, results_list, bundle_errors = pending_fut.result(timeout=600)
                        errors += bundle_errors
                        for vbs_idx, result in results_list:
                            if result is not None:
                                batch.append((polarity, w_idx, l_idx, vbs_idx, result))
                            sweeps_since_restart += 1
                            pbar.update(1)
                    except Exception as e:
                        pbar.write(f'[err] drain: {e}')
                in_flight.clear()
                flush_batch()
                executor.shutdown(wait=True)
                executor = ProcessPoolExecutor(max_workers=n_workers)
                sweeps_since_restart = 0
                gc.collect()
                # Re-prime the pool
                while bundle_idx < total_bundles and len(in_flight) < n_workers * 2:
                    _submit_one(bundle_idx); bundle_idx += 1

    executor.shutdown(wait=True)
    flush_batch()  # flush final partial batch
    dt = time.time() - t0
    print(f'\nAll {total_sweeps:,} sweeps ({total_bundles:,} bundles) done in {dt:.1f}s '
          f'({total_sweeps/dt:.1f} sweep/s, {total_bundles/dt:.2f} bundle/s), {errors} errors')
    print(f'Wrote {len(temp_files)} temp batch files totalling '
          f'{sum(f.stat().st_size for f in temp_files)/1e9:.2f} GB')

    # ── Assemble HDF5 from temp pickles, one batch at a time ──
    print(f'\nAssembling HDF5 ({args.h5_compression} compression)...')
    t_asm = time.time()
    comp_kwargs = {}
    if args.h5_compression == 'gzip':
        comp_kwargs = {'compression': 'gzip', 'compression_opts': 4}
    elif args.h5_compression == 'lzf':
        comp_kwargs = {'compression': 'lzf'}

    with h5py.File(out_path, 'w') as h5:
        h5.create_dataset('W_um', data=W_grid.astype(np.float32))
        h5.create_dataset('L_um', data=L_grid.astype(np.float32))
        h5.create_dataset('Vgs_V', data=Vgs_grid.astype(np.float32))
        h5.create_dataset('Vds_V', data=Vds_grid.astype(np.float32))
        h5.attrs['quantities'] = QUANTITIES
        h5.attrs['convention'] = 'positive Vgs/Vds magnitudes; Vbs signed (NMOS≤0, PMOS≥0)'
        h5.attrs['version'] = 2

        ds = {'n': {}, 'p': {}}
        for pol, vbs_axis, n_vbs in [('n', Vbs_NMOS, n_Vbs_n), ('p', Vbs_PMOS, n_Vbs_p)]:
            grp = h5.create_group(pol)
            grp.create_dataset('Vbs_V', data=vbs_axis.astype(np.float32))
            for q in QUANTITIES:
                ds[pol][q] = grp.create_dataset(
                    q, shape=(n_W, n_L, n_vbs, n_Vds, n_Vgs),
                    dtype=np.float32, fillvalue=np.nan,
                    chunks=(1, 1, 1, n_Vds, n_Vgs),
                    **comp_kwargs,
                )

        with tqdm(total=len(temp_files), desc='H5 assemble') as pbar2:
            for tf in temp_files:
                with open(tf, 'rb') as f:
                    records = pickle.load(f)
                for polarity, w_idx, l_idx, vbs_idx, result in records:
                    for q, arr in result.items():
                        if arr is not None:
                            ds[polarity][q][w_idx, l_idx, vbs_idx, :, :] = arr
                del records
                gc.collect()
                pbar2.update(1)

    size_mb = out_path.stat().st_size / 1e6
    print(f'\nHDF5 saved: {out_path}  ({size_mb:.1f} MB) in {time.time()-t_asm:.1f}s')

    # Remove temp files
    print(f'Removing {tmp_dir}...')
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    main()
