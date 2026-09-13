#!/usr/bin/env python3
"""Add mosfet_wl_um + mosfet_terminal_idx to an existing dataset, then re-prebatch.

The old `CircuitGraphBuilder` didn't attach raw (W, L) per MOSFET or the bulk
terminal index, so prebatched batches produced before the IV-embedder wiring
cannot drive the LUT. This script patches an existing dataset in place:

  1. For each sample in `dataset_{train,val,test}.pkl`:
       - Reads `sample['params']` (raw W, L in meters, per parameter group).
       - Maps each MOSFET device name -> group -> raw (W, L) via the
         `device_to_group` mapping in the original dataset config YAML.
       - Writes `graph.mosfet_wl_um` [M, 2] in µm aligned with `mosfet_info`.
       - Writes `graph.mosfet_terminal_idx` [M, 4] (G, D, S, B) derived from
         `mosfet_info` + `node_names` (bulk looked up as "<device>_bulk").
  2. Re-saves the patched `dataset_{split}.pkl`.
  3. Re-runs `create_prebatched_dataset` for every split that was present, so
     the `{split}/variant_*.pkl` and metadata are regenerated in lockstep with
     the patched graphs.

Usage:
    python scripts/patch_dataset_mosfet_wl.py \\
        --dataset-dir datasets/opamp_3stage_fan_smc_v9_5k_nofil \\
        --config configs/opamp_dataset/opamp_3stage_fan_smc_wide_5k_v7_nofil.yaml
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.data.batching import create_prebatched_dataset


def load_device_to_group(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    d2g = cfg.get('device_to_group')
    if not d2g:
        raise ValueError(f'{config_path} has no device_to_group mapping')
    return d2g


def _device_id(device_name: str) -> str:
    """Match CircuitGraphBuilder._normalize_mosfet_props convention."""
    return device_name.upper().replace('X', '')


def _lookup_wl_um(device_name: str, params: dict, device_to_group: dict) -> tuple[float, float]:
    """Return (W_um, L_um) for a MOSFET by consulting params + device_to_group."""
    dev_id = _device_id(device_name)
    group = device_to_group.get(dev_id)
    if group is None:
        w_key, l_key = f'W_{dev_id}', f'L_{dev_id}'
    else:
        w_key, l_key = f'W_{group}', f'L_{group}'
    w_m = params[w_key]
    l_m = params[l_key]
    return float(w_m) * 1e6, float(l_m) * 1e6


def _lookup_m(device_name: str, params: dict, device_to_group: dict) -> float:
    """Return multiplier M (fingers) for a MOSFET."""
    dev_id = _device_id(device_name)
    group = device_to_group.get(dev_id)
    key = f'M_{group}' if group is not None else f'M_{dev_id}'
    return float(params.get(key, 1.0))


def patch_graph(graph, params: dict, device_to_group: dict) -> None:
    """Attach mosfet_wl_um and mosfet_terminal_idx to a single graph in place."""
    mosfet_info = graph.mosfet_info
    node_names = graph.node_names

    M = mosfet_info.shape[0]
    if M == 0:
        graph.mosfet_wl_um = torch.zeros((0, 2), dtype=torch.float)
        graph.mosfet_terminal_idx = torch.zeros((0, 4), dtype=torch.long)
        graph.mosfet_m = torch.zeros((0,), dtype=torch.float)
        graph.mosfet_vbs = torch.zeros((0,), dtype=torch.float)
        return

    # MOSFET device name order — derived from the gate terminal of each row.
    # node_names are like "Xm0_gate", so strip "_gate".
    device_names = []
    for row in range(M):
        g_idx = int(mosfet_info[row, 0])
        name = node_names[g_idx]
        if not name.endswith('_gate'):
            raise ValueError(f'Unexpected node_name for gate: {name!r}')
        device_names.append(name[:-len('_gate')])

    bulk_idx: dict = {}
    for i, n in enumerate(node_names):
        if n.endswith('_bulk'):
            bulk_idx[n[:-len('_bulk')]] = i

    wl_rows, term_rows, m_rows = [], [], []
    for row, dev_name in enumerate(device_names):
        g, d, s = int(mosfet_info[row, 0]), int(mosfet_info[row, 1]), int(mosfet_info[row, 2])
        b = bulk_idx.get(dev_name, -1)
        term_rows.append([g, d, s, b])
        wl_rows.append(list(_lookup_wl_um(dev_name, params, device_to_group)))
        m_rows.append(_lookup_m(dev_name, params, device_to_group))

    graph.mosfet_wl_um = torch.tensor(wl_rows, dtype=torch.float)
    graph.mosfet_terminal_idx = torch.tensor(term_rows, dtype=torch.long)
    graph.mosfet_m = torch.tensor(m_rows, dtype=torch.float)

    # Vbs = V[bulk] - V[source], signed.
    # NMOS: Vbs <= 0 (source above bulk). PMOS: Vbs >= 0 (source below bulk).
    # Uses GT voltages from node_voltage_targets (raw volts).
    v = graph.node_voltage_targets
    vbs_rows = []
    for row in range(M):
        s_idx = term_rows[row][2]
        b_idx = term_rows[row][3]
        if b_idx < 0:
            vbs_rows.append(0.0)
        else:
            vbs_rows.append(float(v[b_idx] - v[s_idx]))
    graph.mosfet_vbs = torch.tensor(vbs_rows, dtype=torch.float)


def patch_split(split_pkl: Path, device_to_group: dict) -> list:
    with open(split_pkl, 'rb') as f:
        samples = pickle.load(f)
    print(f'  Patching {len(samples)} graphs in {split_pkl.name}...')
    for s in samples:
        patch_graph(s['graph'], s['params'], device_to_group)
    with open(split_pkl, 'wb') as f:
        pickle.dump(samples, f)
    print(f'  Saved patched {split_pkl.name}')
    return samples


def reprebatch(samples: list, split_dir: Path, batch_size: int, num_variants: int,
               seed: int, is_validation: bool) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    create_prebatched_dataset(
        samples=samples,
        output_dir=split_dir,
        batch_size=batch_size,
        num_variants=num_variants,
        seed=seed,
        is_validation=is_validation,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True, type=Path)
    ap.add_argument('--config', required=True, type=Path,
                    help='YAML used to generate the dataset (for device_to_group).')
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--num-variants', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device_to_group = load_device_to_group(args.config)
    print(f'Loaded device_to_group with {len(device_to_group)} entries')

    for split in ('train', 'val', 'test'):
        pkl = args.dataset_dir / f'dataset_{split}.pkl'
        if not pkl.exists():
            continue
        print(f'\n=== {split} ===')
        samples = patch_split(pkl, device_to_group)
        reprebatch(
            samples=samples,
            split_dir=args.dataset_dir / split,
            batch_size=args.batch_size,
            num_variants=args.num_variants,
            seed=args.seed,
            is_validation=(split != 'train'),
        )

    # Also refresh the un-split rollup if it's present.
    full_pkl = args.dataset_dir / 'dataset.pkl'
    if full_pkl.exists():
        print('\n=== dataset.pkl (rollup) ===')
        patch_split(full_pkl, device_to_group)

    print('\nDone.')


if __name__ == '__main__':
    main()
