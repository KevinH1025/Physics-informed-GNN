#!/usr/bin/env python
"""Benchmark SPICE simulation vs GNN forward-pass speed.

For each of the 5 topologies, samples N parameter sets from the val set, then
times:
  - SPICE wall time per design (DC-only and DC+AC modes)
  - GNN wall time per design (batched and unbatched)

Outputs a per-topology table + saves figure/timing.json.

Usage:
    python scripts/benchmark_speed.py <experiment_name> [--n-samples 50]
    python scripts/benchmark_speed.py v5_5topo_joint --n-samples 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.circuits.simulator import CircuitSimulator
from src.data.pretrain_loader import PretrainCombinedLoader
from src.data.sampling import generate_netlist
from src.training.checkpoint import create_model_from_args
from src.training.config import parse_training_config


TOPOS = ['fan_smc', 'sau_cfcc', 'peng_tcfc', 'leung_nmcf', 'leung_nmcnr']


def _spice_worker(args):
    """Worker function: time a single SPICE simulate() call in an isolated process.
    Returns (time_seconds, n_voltages) — n_voltages > 0 means convergence."""
    netlist, mode = args
    import os
    os.environ.setdefault('PYSPICE_LIBRARY_PATH', '/usr/local/lib')
    import sys, time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.circuits.simulator import CircuitSimulator
    sim = CircuitSimulator(analysis_types=['dc'] if mode == 'dc' else ['dc', 'ac'])
    t0 = time.perf_counter()
    out = sim.simulate(netlist)
    dt = time.perf_counter() - t0
    return dt, len(out.get('_all_net_voltages', {}))


def time_spice(template_path: str, ac_template_path: str | None, params_list,
               tmp_dir: Path) -> tuple[float, float, int, int]:
    """Return (total_seconds_dc_only, total_seconds_dc_ac, n_dc_ok, n_ac_ok).

    Uses multiprocessing with maxtasksperchild=1 to give each SPICE call a
    fresh process — the only reliable way to avoid the PySpice cffi
    duplicate-struct issue that breaks every sim after the first.
    """
    import multiprocessing as mp
    # Prepare DC netlists
    dc_jobs = []
    for i, params in enumerate(params_list):
        p = dict(params)
        if 'VIN_P' not in p and 'VCM' in p:
            p['VIN_P'] = p['VCM'] + p.get('VDIFF', 0) / 2
            p['VIN_N'] = p['VCM'] - p.get('VDIFF', 0) / 2
        np_path = tmp_dir / f'dc_{i}.sp'
        generate_netlist(template_path, p, str(np_path))
        with open(np_path) as f:
            nl = f.read()
        np_path.unlink(missing_ok=True)
        dc_jobs.append((nl, 'dc'))

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=4, maxtasksperchild=1) as pool:
        dc_results = pool.map(_spice_worker, dc_jobs)
    t_dc = sum(r[0] for r in dc_results)
    n_dc_ok = sum(1 for r in dc_results if r[1] > 0)

    t_ac = 0.0; n_ac_ok = 0
    if ac_template_path:
        ac_jobs = []
        for i, params in enumerate(params_list):
            p = dict(params)
            if 'VIN_P' not in p and 'VCM' in p:
                p['VIN_P'] = p['VCM'] + p.get('VDIFF', 0) / 2
                p['VIN_N'] = p['VCM'] - p.get('VDIFF', 0) / 2
            np_path = tmp_dir / f'ac_{i}.sp'
            generate_netlist(ac_template_path, p, str(np_path))
            with open(np_path) as f:
                nl = f.read()
            np_path.unlink(missing_ok=True)
            ac_jobs.append((nl, 'ac'))
        with ctx.Pool(processes=4, maxtasksperchild=1) as pool:
            ac_results = pool.map(_spice_worker, ac_jobs)
        t_ac = sum(r[0] for r in ac_results)
        n_ac_ok = sum(1 for r in ac_results if r[1] > 0)

    return t_dc, t_ac, n_dc_ok, n_ac_ok


def time_gnn_batched(model, batch, n_warmup: int = 5, n_runs: int = 20) -> float:
    """Return average seconds per batch."""
    device = next(model.parameters()).device
    use_cuda = device.type == 'cuda'
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(batch)
            if use_cuda:
                torch.cuda.synchronize()
        ts = []
        for _ in range(n_runs):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(batch)
            if use_cuda:
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return float(np.mean(ts))


def attach_norm(batch, stats):
    device = batch.x.device
    for name, val in [
        ('vdc_mean', stats['v_mean']), ('vdc_std', stats['v_std']),
        ('current_mean', stats['i_mean']), ('current_std', stats['i_std']),
        ('ss_gm_mean', stats['gm_mean']), ('ss_gm_std', stats['gm_std']),
        ('ss_gds_mean', stats['gds_mean']), ('ss_gds_std', stats['gds_std']),
    ]:
        setattr(batch, name, torch.tensor(val, device=device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp', help='Experiment name (under datasets/.../experiments/)')
    ap.add_argument('--dataset-dir', default='datasets/opamp_3stage_pretrain_combined_5topo')
    ap.add_argument('--netlist-dir', default='netlists')
    ap.add_argument('--n-samples', type=int, default=50)
    ap.add_argument('--skip-ac', action='store_true', help='Skip the slower DC+AC mode')
    ap.add_argument('--out-dir', default='figures/thesis')
    args = ap.parse_args()

    p = Path(args.exp)
    if not p.exists():
        p = Path(args.dataset_dir) / 'experiments' / args.exp
    if not p.exists():
        raise FileNotFoundError(f'Experiment not found: {args.exp}')
    ckpt_path = p / 'best.pt'
    cfg_path = p / 'original_config.yaml'

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[setup] device={device}, exp={p.name}, n_samples={args.n_samples}/topo')

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    stats = ckpt.get('norm_stats') or ckpt.get('config', {}).get('norm_stats') or {}
    if not stats:
        stats = {'v_mean': 0.9, 'v_std': 0.5, 'i_mean': -5.55, 'i_std': 1.42,
                 'gm_mean': -4.43, 'gm_std': 1.44, 'gds_mean': -5.89, 'gds_std': 1.89}

    val_loader = PretrainCombinedLoader(
        f'{args.dataset_dir}/dataset_val.pkl',
        batch_size=args.n_samples * len(TOPOS),
        device=device, shuffle=False, drop_last=False,
    )
    sample_batch = next(iter(val_loader))
    node_dim = sample_batch.x.shape[-1]
    type_dim = sample_batch.type_tens.shape[-1]

    import argparse as _argparse
    cli_args = _argparse.Namespace()
    for k, v in parse_training_config(cfg).items():
        if v is not None:
            setattr(cli_args, k, v)
    cli_args.device = str(device)
    cli_args.dataset = args.dataset_dir
    cli_args.predict_currents = True
    model, _ = create_model_from_args(cli_args, node_dim + type_dim, device)
    model.load_state_dict(ckpt['model_state_dict'])
    if hasattr(model, 'set_normalization_stats'):
        model.set_normalization_stats(
            vdc_mean=stats['v_mean'], vdc_std=stats['v_std'],
            ss_gm_mean=stats['gm_mean'], ss_gm_std=stats['gm_std'],
            ss_gds_mean=stats['gds_mean'], ss_gds_std=stats['gds_std'],
            current_mean=stats['i_mean'], current_std=stats['i_std'],
        )
    if hasattr(model, 'current_epoch'):
        model.current_epoch = ckpt.get('epoch', 99999)
    model.eval()

    # Group val samples by topology
    import pickle
    with open(f'{args.dataset_dir}/dataset_val.pkl', 'rb') as f:
        all_samples = pickle.load(f)
    by_topo = {t: [] for t in TOPOS}
    for s in all_samples:
        if s['topology'] in by_topo:
            by_topo[s['topology']].append(s)

    tmp_dir = Path('/tmp/spice_bench_tmp')
    tmp_dir.mkdir(exist_ok=True)

    results = {}
    for topo in TOPOS:
        samples = by_topo[topo][:args.n_samples]
        if not samples:
            print(f'[{topo}] no samples — skipping')
            continue
        params_list = [s['params'] for s in samples]

        dc_template = f'{args.netlist_dir}/opamp_3stage_{topo}_template.sp'
        ac_template = None if args.skip_ac else f'{args.netlist_dir}/opamp_3stage_{topo}_ac_template.sp'
        if ac_template and not Path(ac_template).exists():
            ac_template = None

        print(f'\n[{topo}] running SPICE on {len(samples)} designs...')
        t_dc, t_ac, n_dc_ok, n_ac_ok = time_spice(
            dc_template, ac_template, params_list, tmp_dir)
        spice_per_dc = t_dc / max(1, len(samples))
        spice_per_ac = t_ac / max(1, len(samples)) if ac_template else None
        print(f'  SPICE DC-only:  {t_dc:.2f}s total, {1e3*spice_per_dc:.1f} ms/design ({n_dc_ok}/{len(samples)} ok)')
        if ac_template:
            print(f'  SPICE DC+AC:    {t_ac:.2f}s total, {1e3*spice_per_ac:.1f} ms/design ({n_ac_ok}/{len(samples)} ok)')

        # Build a properly-collated GNN batch with topology slices via PretrainCombinedLoader
        loader_topo = PretrainCombinedLoader(
            f'{args.dataset_dir}/dataset_val.pkl',
            batch_size=args.n_samples, device=device, shuffle=False, drop_last=False,
            topology_filter=topo,
        )
        batch = next(iter(loader_topo))

        print(f'[{topo}] running GNN on batch of {batch.num_graphs}...')
        attach_norm(batch, stats)
        t_batch = time_gnn_batched(model, batch, n_warmup=5, n_runs=30)
        gnn_per_batched = t_batch / batch.num_graphs

        # Single-sample timing — use a 1-sample loader
        loader_single = PretrainCombinedLoader(
            f'{args.dataset_dir}/dataset_val.pkl',
            batch_size=1, device=device, shuffle=False, drop_last=False,
            topology_filter=topo,
        )
        single_batch = next(iter(loader_single))
        attach_norm(single_batch, stats)
        t_single = time_gnn_batched(model, single_batch, n_warmup=5, n_runs=30)
        print(f'  GNN batched:    {1e3*t_batch:.1f} ms/batch ({len(samples)} designs), '
              f'{1e3*gnn_per_batched:.3f} ms/design')
        print(f'  GNN single:     {1e3*t_single:.1f} ms/design')

        results[topo] = {
            'n_samples': len(samples),
            'spice_dc_total_s': t_dc,
            'spice_dc_per_s': spice_per_dc,
            'spice_dc_ok': n_dc_ok,
            'spice_ac_total_s': t_ac if ac_template else None,
            'spice_ac_per_s': spice_per_ac,
            'spice_ac_ok': n_ac_ok if ac_template else None,
            'gnn_batched_total_s': t_batch,
            'gnn_batched_per_s': gnn_per_batched,
            'gnn_single_per_s': t_single,
            'speedup_batched_dc': spice_per_dc / max(1e-12, gnn_per_batched),
            'speedup_single_dc': spice_per_dc / max(1e-12, t_single),
            'speedup_batched_ac': (spice_per_ac / max(1e-12, gnn_per_batched)) if spice_per_ac else None,
        }

    # Print summary table
    print('\n' + '=' * 100)
    print(f'{"Topology":<14} {"SPICE DC ms":>12} {"SPICE AC ms":>12} '
          f'{"GNN batch ms":>13} {"GNN single ms":>14} {"Speedup (batched, DC)":>22}')
    print('-' * 100)
    for t in TOPOS:
        r = results.get(t)
        if not r: continue
        ac_ms = f'{1e3*r["spice_ac_per_s"]:.1f}' if r['spice_ac_per_s'] else 'n/a'
        print(f'{t:<14} {1e3*r["spice_dc_per_s"]:>12.1f} {ac_ms:>12} '
              f'{1e3*r["gnn_batched_per_s"]:>13.3f} {1e3*r["gnn_single_per_s"]:>14.2f} '
              f'{r["speedup_batched_dc"]:>20.0f}x')

    # Save raw json
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'speed_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {out_dir / "speed_benchmark.json"}')


if __name__ == '__main__':
    main()
