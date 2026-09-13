#!/usr/bin/env python3
"""Validate the MOSFET LUT against SPICE ground truth on the 5k dataset.

Splits operating points into two groups:
  - Group A: |Vbs| < 10 mV — pure interpolation quality check
  - Group B: |Vbs| >= 10 mV — quantifies body-effect limitation of Vbs=0 LUT

Reports per-group relative error for id, gm, gds, vth.
"""
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from circuitgnn.physics.mosfet_lut import MosfetLUT


LUT_PATH = 'datasets/lut/lut_v2/sky130_mosfet_lut_v2.h5'
DATASET_PATH = 'datasets/opamp_3stage_fan_smc_v9_5k_nofil/dataset.pkl'
N_SAMPLES = 500  # subset of val for speed
VBS_THRESHOLD = 0.01  # 10 mV

# Device group → W/L/M parameter names + optional M multiplier from netlist
# (xm<N> → (group, M_mult_prefix)).  M_mult=1 means M=M_{GROUP}; M_mult=4 means 4*M_{GROUP}.
DEVICE_GROUPS = {
    **{f'xm{i}': ('BIASCM_P', 1) for i in range(8)},
    **{'xm4': ('BIASCM_P', 4)},                       # M4 = 4× bias
    **{f'xm{i}': ('GM1', 1) for i in (8, 9)},
    'xm10': ('GM2', 1),
    'xm11': ('GMF2', 1),
    'xm14': ('BIASCM_N', 1),
    **{f'xm{i}': ('BIASCM_N', 4) for i in (12, 13, 15, 16, 17, 18)},
    **{f'xm{i}': ('BIASCM_N', 8) for i in (19, 20)},
    **{f'xm{i}': ('LOAD2', 1) for i in (21, 22)},
    'xm23': ('GM3', 1),
}


def main():
    print(f'Loading LUT from {LUT_PATH}...')
    lut = MosfetLUT.load(LUT_PATH)
    print(f'  LUT grid: W×L×Vgs×Vds = {len(lut.W_um)}×{len(lut.L_um)}×{len(lut.Vgs_V)}×{len(lut.Vds_V)}')

    print(f'Loading dataset from {DATASET_PATH}...')
    with open(DATASET_PATH, 'rb') as f:
        ds = pickle.load(f)
    print(f'  {len(ds)} samples, using first {N_SAMPLES}')
    ds = ds[:N_SAMPLES]

    # Build graph-index → device-name mapping from sample 0
    g0 = ds[0]['graph']
    names = g0.node_names
    mi = g0.mosfet_info
    graph_to_dev = {}
    dev_to_graph = {}
    for gidx in range(24):
        gate_name = names[mi[gidx, 0].item()]
        m = re.match(r'(X\w+)_gate', gate_name)
        if m:
            dev = m.group(1).lower()
            graph_to_dev[gidx] = dev
            dev_to_graph[dev] = gidx
    src_net_per_dev = {dev: names[mi[gi, 5].item()] for gi, dev in graph_to_dev.items()}

    # Accumulate errors per quantity, split by group
    records = {  # (dev, quantity) -> list of (rel_err, abs_err, Vbs)
    }
    n_a = n_b = 0

    for s in ds:
        params = s['params']
        nv = s['specs']['_all_net_voltages']
        regs = s['specs']['_mosfet_regions']
        ss = s['specs']['_mosfet_ss_params']
        tc = s['specs']['_mosfet_terminal_currents']
        VDD = nv.get('vdda', s.get('vdd', 1.8))

        for dev in graph_to_dev.values():
            if dev not in DEVICE_GROUPS: continue
            group, m_mult = DEVICE_GROUPS[dev]
            W_m = params.get(f'W_{group}')
            L_m = params.get(f'L_{group}')
            M_base = params.get(f'M_{group}', 1)
            if W_m is None or L_m is None: continue

            W_um = W_m * 1e6
            L_um = L_m * 1e6
            M_val = float(M_base) * m_mult

            # SPICE operating point
            reg = regs.get(dev)
            if reg is None: continue
            Vgs_mag = abs(reg['vgs'])
            Vds_mag = abs(reg['vds'])
            is_pmos = reg['is_pmos']
            polarity = 'p' if is_pmos else 'n'

            # Vbs from source voltage.
            # xm8/xm9 (PMOS diff pair GM1) have bulk tied to source in the
            # netlist (Xm8 dm_2 vinn net31 net31 ...), so Vbs=0 despite being
            # PMOS. All other PMOS have bulk=VDD; all NMOS have bulk=GND.
            src = src_net_per_dev.get(dev)
            Vs = nv.get(src, {'vdda': VDD, 'gnda': 0.0}.get(src))
            if Vs is None: continue
            if dev in ('xm8', 'xm9'):
                Vbs = 0.0  # bulk-to-source shorted
            else:
                Vbulk = VDD if is_pmos else 0.0
                Vbs = Vbulk - Vs

            # SPICE ground truth
            sp_id = abs(tc.get(dev, {}).get('id', 0.0))
            sp_gm = abs(ss.get(dev, {}).get('gm', 0.0))
            sp_gds = abs(ss.get(dev, {}).get('gds', 0.0))
            sp_vth = abs(reg['vth'])

            # LUT query — pass Vbs with proper sign per polarity
            # (our grid stores signed Vbs: NMOS <= 0, PMOS >= 0)
            # Our Vbs variable already has correct sign from Vb - Vs
            lut_pred = lut.query(polarity, W=W_um, L=L_um, Vgs=Vgs_mag, Vds=Vds_mag,
                                 Vbs=Vbs, M=M_val,
                                 quantities=['id', 'gm', 'gds', 'vth'])
            for q, sp_val in [('id', sp_id), ('gm', sp_gm), ('gds', sp_gds), ('vth', sp_vth)]:
                pred = float(lut_pred[q])
                if q == 'vth':
                    # vth is per-finger, M doesn't scale it — undo the M scaling (vth is NOT in _M_SCALED)
                    pass
                abs_err = abs(pred - sp_val)
                rel_err = abs_err / max(sp_val, 1e-15) * 100
                records.setdefault((q,), [])
                records[(q,)].append((rel_err, abs_err, abs(Vbs), dev))

            if abs(Vbs) < VBS_THRESHOLD:
                n_a += 1
            else:
                n_b += 1

    print(f'\nTotal device-samples: Group A (Vbs≈0) = {n_a:,},  Group B (Vbs≠0) = {n_b:,}')

    print('\n' + '=' * 90)
    print(f'{"Quantity":<10s} {"Group":<8s} {"N":>8s} {"median %":>12s} {"95th %":>12s} {"max %":>12s}')
    print('=' * 90)
    for q in ['id', 'gm', 'gds', 'vth']:
        rows = records[(q,)]
        rel_arr = np.array([r[0] for r in rows])
        vbs_arr = np.array([r[2] for r in rows])
        for group_name, mask in [('A (|Vbs|<10mV)', vbs_arr < VBS_THRESHOLD),
                                  ('B (|Vbs|>=10mV)', vbs_arr >= VBS_THRESHOLD)]:
            r = rel_arr[mask]
            if len(r) == 0:
                continue
            med = np.median(r); p95 = np.percentile(r, 95); mx = r.max()
            print(f'{q:<10s} {group_name:<16s} {len(r):>8d} {med:>12.3f} {p95:>12.3f} {mx:>12.3f}')
        print()

    # Per-device breakdown for Group B devices
    print('=' * 90)
    print('Per-device relative error on Group B (Vbs ≠ 0) only — id')
    print('=' * 90)
    rows = records[('id',)]
    dev_errs = {}
    for rel_err, abs_err, vbs, dev in rows:
        if vbs >= VBS_THRESHOLD:
            dev_errs.setdefault(dev, []).append((rel_err, vbs))
    for dev in sorted(dev_errs.keys()):
        arr = dev_errs[dev]
        rels = np.array([a[0] for a in arr])
        vbss = np.array([a[1] for a in arr])
        print(f'  {dev:<6s} n={len(rels):>5d}  median_rel={np.median(rels):>6.2f}%  '
              f'p95_rel={np.percentile(rels, 95):>6.1f}%  '
              f'typical |Vbs|≈{np.median(vbss)*1000:.0f} mV')


if __name__ == '__main__':
    main()
