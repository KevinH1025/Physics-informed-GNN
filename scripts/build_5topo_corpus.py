#!/usr/bin/env python
"""Build a 5-topology pretrain corpus by merging:
  - datasets/opamp_3stage_pretrain_combined (4 topos: sau_cfcc, peng_tcfc, leung_nmcf, leung_nmcnr)
  - datasets/opamp_3stage_fan_smc_openloop_5k (1 topo: fan_smc) — adds 'topology' field

Output: datasets/opamp_3stage_pretrain_combined_5topo/{dataset_train.pkl, dataset_val.pkl}
"""
from __future__ import annotations

import pickle
from pathlib import Path


SRC_4TOPO = Path('datasets/opamp_3stage_pretrain_combined')
SRC_FANSMC = Path('datasets/opamp_3stage_fan_smc_openloop_5k')
DST = Path('datasets/opamp_3stage_pretrain_combined_5topo')


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val'):
        src4 = SRC_4TOPO / f'dataset_{split}.pkl'
        srcf = SRC_FANSMC / f'dataset_{split}.pkl'
        with open(src4, 'rb') as f:
            d4 = pickle.load(f)
        with open(srcf, 'rb') as f:
            df = pickle.load(f)
        # Tag fan_smc samples with topology field if missing
        for s in df:
            if 'topology' not in s:
                s['topology'] = 'fan_smc'
        merged = d4 + df
        # Sanity: count per topo
        topos = {}
        for s in merged:
            topos[s['topology']] = topos.get(s['topology'], 0) + 1
        print(f'{split}: {len(merged)} total samples; per-topo:')
        for t, n in sorted(topos.items()):
            print(f'  {t}: {n}')
        with open(DST / f'dataset_{split}.pkl', 'wb') as f:
            pickle.dump(merged, f)
        print(f'  saved → {DST / f"dataset_{split}.pkl"}')


if __name__ == '__main__':
    main()
