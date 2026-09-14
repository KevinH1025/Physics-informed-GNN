#!/usr/bin/env python
"""Add mosfet_wl_um, mosfet_m, mosfet_terminal_idx to opamp_3stage_pretrain_combined.

The combined pretrain dataset has 4 topologies (sau_cfcc, peng_tcfc, leung_nmcf,
leung_nmcnr) but no `mosfet_wl_um` field, so the LUT can't be queried for
physics pretraining. This script:

  1. Parses each topology's netlist template (.sp) to extract device→group
     mapping + per-device M multiplier prefix.
  2. For each sample, uses the sample's `topology` field to pick the right
     mapping, then computes (W_um, L_um, M_total) per MOSFET from `params`.
  3. Saves patched dataset_{train,val}.pkl.

Usage:
    python scripts/migrations/patch_pretrain_combined.py
"""
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch


_MOSFET_LINE_RE = re.compile(
    r'^X(m\d+)\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+'
    r"W=\{?(W_\w+?)\}?\s+L=\{?(L_\w+?)\}?\s+M=([^\s]+)",
    re.IGNORECASE,
)


def parse_template(template_path: Path) -> dict:
    """Parse a netlist template file → dict per MOSFET name (Xm0, Xm1, ...).

    Returns:
        {device_name: {'group': str, 'm_factor': float}}
    """
    devices = {}
    with open(template_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            m = _MOSFET_LINE_RE.match(line)
            if not m:
                continue
            dev_name_lower = m.group(1)  # m0, m1, ...
            dev_name = 'X' + dev_name_lower  # Xm0, Xm1, ... matching mosfet_device_names
            w_key = m.group(2)             # W_BIASCM_P
            l_key = m.group(3)             # L_BIASCM_P
            m_expr = m.group(4)            # {M_BIASCM_P} or '4*{M_BIASCM_P}' or '10*{M_BIASCM_P}'

            # Strip braces and extract group name + multiplier
            # Patterns: "{M_BIASCM_P}" → group BIASCM_P, factor 1
            #           "'4*{M_BIASCM_P}'" → group BIASCM_P, factor 4
            m_expr = m_expr.replace("'", '').replace('"', '').strip()
            mult_match = re.match(r'^(\d+)\s*\*\s*\{?M_(\w+)\}?$', m_expr)
            if mult_match:
                m_factor = float(mult_match.group(1))
                group = mult_match.group(2)
            else:
                simple_match = re.match(r'^\{?M_(\w+)\}?$', m_expr)
                if not simple_match:
                    raise ValueError(f'Cannot parse M expression: {m_expr!r}')
                m_factor = 1.0
                group = simple_match.group(1)

            # Verify W and L groups match the M group (should always be true in our templates)
            w_group = w_key.replace('W_', '')
            assert w_group == group, f'{dev_name}: W={w_key} vs M group={group}'

            devices[dev_name] = {'group': group, 'm_factor': m_factor}
    return devices


def build_topology_mappings(netlists_dir: Path) -> dict:
    """Build {topology_name: {device_name: {'group', 'm_factor'}}} for the 4 topos."""
    topo_to_template = {
        'sau_cfcc':    'opamp_3stage_sau_cfcc_template.sp',
        'peng_tcfc':   'opamp_3stage_peng_tcfc_template.sp',
        'leung_nmcf':  'opamp_3stage_leung_nmcf_template.sp',
        'leung_nmcnr': 'opamp_3stage_leung_nmcnr_template.sp',
    }
    mappings = {}
    for topo, fname in topo_to_template.items():
        path = netlists_dir / fname
        if not path.exists():
            raise FileNotFoundError(f'Missing template: {path}')
        mappings[topo] = parse_template(path)
        print(f'  {topo}: parsed {len(mappings[topo])} MOSFETs from {fname}')
    return mappings


def patch_sample(sample: dict, topo_mapping: dict) -> None:
    """Add mosfet_wl_um [M, 2] (µm), mosfet_m [M] (effective multiplier)
    to sample['graph'] in place. mosfet_info row order matches mosfet_device_names.
    """
    graph = sample['graph']
    params = sample['params']
    device_names = graph['mosfet_device_names']  # list of strings like "Xm0"

    M = len(device_names)
    wl_um = torch.zeros((M, 2), dtype=torch.float32)
    m_eff = torch.zeros(M, dtype=torch.float32)

    for i, dev in enumerate(device_names):
        if dev not in topo_mapping:
            raise KeyError(
                f'Device {dev!r} not found in topology mapping for sample id '
                f'{sample.get("sample_id")} topo={sample.get("topology")}; '
                f'available: {list(topo_mapping.keys())[:6]}...'
            )
        info = topo_mapping[dev]
        group = info['group']
        factor = info['m_factor']
        w_m = float(params[f'W_{group}'])
        l_m = float(params[f'L_{group}'])
        m_base = float(params[f'M_{group}'])
        wl_um[i, 0] = w_m * 1e6   # µm
        wl_um[i, 1] = l_m * 1e6   # µm
        m_eff[i] = m_base * factor

    graph['mosfet_wl_um'] = wl_um
    graph['mosfet_m'] = m_eff


def patch_split(input_path: Path, mappings: dict) -> None:
    print(f'\nPatching {input_path} ...')
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    print(f'  {len(data)} samples loaded')

    # Sanity check first sample
    s0 = data[0]
    topo0 = s0['topology']
    patch_sample(s0, mappings[topo0])
    g0 = s0['graph']
    print(f'  first sample: topo={topo0}, mosfet_wl_um[0]={g0["mosfet_wl_um"][0].tolist()}, '
          f'M_eff[0]={g0["mosfet_m"][0].item()}')

    # Patch the rest
    for i, s in enumerate(data[1:], start=1):
        patch_sample(s, mappings[s['topology']])

    with open(input_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'  saved patched {input_path}')


def main():
    repo = Path(__file__).resolve().parents[1]
    netlists_dir = repo / 'netlists'
    print(f'Parsing netlist templates from {netlists_dir}...')
    mappings = build_topology_mappings(netlists_dir)

    pretrain_dir = repo / 'datasets' / 'opamp_3stage_pretrain_combined'
    for split in ('train', 'val'):
        path = pretrain_dir / f'dataset_{split}.pkl'
        if path.exists():
            patch_split(path, mappings)


if __name__ == '__main__':
    main()
