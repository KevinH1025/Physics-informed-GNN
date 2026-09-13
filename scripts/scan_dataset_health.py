#!/usr/bin/env python
"""Health check across all 5 topologies in the combined corpus.
Reports: unphysical V, NaN/Inf, extreme outliers per quantity.
Output: figures/thesis/dataset_health.json"""
import pickle, json, gc
from pathlib import Path
import numpy as np

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
OUT = REPO / 'figures/thesis/dataset_health.json'
VDD = 1.8
TOL = 0.001


def scan(samples, topo_filter=None):
    n_total = 0; bad_v = []; bad_i = []; bad_gm = []; bad_gds = []; bad_dc = []
    nan_count = {'V': 0, 'I': 0, 'gm': 0, 'gds': 0, 'DC': 0}
    inf_count = dict(nan_count)
    extreme_v_min = +1e18; extreme_v_max = -1e18
    bad_v_net_distrib = {}
    for i, s in enumerate(samples):
        if topo_filter is not None and s.get('topology') != topo_filter:
            continue
        n_total += 1
        g = s['graph']
        mask = g.train_mask if hasattr(g,'train_mask') else ~g.known_voltage_mask
        Vs = g.node_voltage_targets.cpu().numpy()
        v_arr = Vs[mask.cpu().numpy()]
        if np.isnan(v_arr).any(): nan_count['V'] += 1
        if np.isinf(v_arr).any(): inf_count['V'] += 1
        vmin, vmax = v_arr.min(), v_arr.max()
        if vmin < extreme_v_min: extreme_v_min = float(vmin)
        if vmax > extreme_v_max: extreme_v_max = float(vmax)
        if vmin < -TOL or vmax > VDD + TOL:
            bad_v.append({'i': i, 'vmin': float(vmin), 'vmax': float(vmax)})
            # which net
            v_idx = mask.nonzero(as_tuple=True)[0].cpu().numpy()
            for k, p in enumerate(v_idx):
                if Vs[p] < -TOL or Vs[p] > VDD + TOL:
                    bad_v_net_distrib[k] = bad_v_net_distrib.get(k, 0) + 1
        # Currents
        if hasattr(g, 'node_current_targets') and hasattr(g, 'has_current_mask'):
            ic = g.node_current_targets[g.has_current_mask].cpu().numpy()
            if np.isnan(ic).any(): nan_count['I'] += 1
            if np.isinf(ic).any(): inf_count['I'] += 1
            if (np.abs(ic) > 10).any():  # > 10 A = clearly broken
                bad_i.append({'i': i, 'max_abs': float(np.abs(ic).max())})
        # gm/gds
        if hasattr(g, 'node_log_gm') and hasattr(g, 'mosfet_drain_mask'):
            gm = g.node_log_gm[g.mosfet_drain_mask].cpu().numpy()
            gds = g.node_log_gds[g.mosfet_drain_mask].cpu().numpy()
            if np.isnan(gm).any(): nan_count['gm'] += 1
            if np.isnan(gds).any(): nan_count['gds'] += 1
            if (gm > 5).any() or (gm < -20).any():  # log S
                bad_gm.append({'i': i, 'min': float(gm.min()), 'max': float(gm.max())})
            if (gds > 5).any() or (gds < -20).any():
                bad_gds.append({'i': i, 'min': float(gds.min()), 'max': float(gds.max())})
        # DC gain
        if hasattr(g, 'ac_dc_gain'):
            v = g.ac_dc_gain.item() if hasattr(g.ac_dc_gain,'item') else float(g.ac_dc_gain[0])
            if np.isnan(v): nan_count['DC'] += 1
            if np.isinf(v): inf_count['DC'] += 1
            if abs(v) > 500:
                bad_dc.append({'i': i, 'val': float(v)})
    return {
        'n_total': n_total,
        'V': {
            'extreme_min': extreme_v_min, 'extreme_max': extreme_v_max,
            'unphysical_samples': len(bad_v), 'unphysical_pct': 100*len(bad_v)/max(n_total,1),
            'bad_v_net_distrib': bad_v_net_distrib,
            'first_5_bad_v': bad_v[:5],
        },
        'I': {'extreme_samples': len(bad_i), 'first_3': bad_i[:3]},
        'gm': {'extreme_samples': len(bad_gm), 'first_3': bad_gm[:3]},
        'gds': {'extreme_samples': len(bad_gds), 'first_3': bad_gds[:3]},
        'DC_gain': {'extreme_samples': len(bad_dc), 'first_3': bad_dc[:3]},
        'NaN_counts': nan_count,
        'Inf_counts': inf_count,
    }


def main():
    health = {}
    # Pass 1: fan_smc standalone
    print('=== fan_smc ===')
    fan = pickle.load(open(REPO/'datasets/opamp_3stage_fan_smc_openloop_5k/dataset_train.pkl','rb')) \
        + pickle.load(open(REPO/'datasets/opamp_3stage_fan_smc_openloop_5k/dataset_val.pkl','rb'))
    health['fan_smc'] = scan(fan)
    del fan; gc.collect()
    # Pass 2: 4-topo combined train+val
    print('=== loading 4-topo train+val ===')
    c4 = pickle.load(open(REPO/'datasets/opamp_3stage_pretrain_combined/dataset_train.pkl','rb')) \
       + pickle.load(open(REPO/'datasets/opamp_3stage_pretrain_combined/dataset_val.pkl','rb'))
    for t in ('sau_cfcc', 'peng_tcfc', 'leung_nmcf', 'leung_nmcnr'):
        print(f'  {t} ...')
        health[t] = scan(c4, topo_filter=t)
    del c4; gc.collect()

    # Save + print summary
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(health, f, indent=2)

    print('\n=== DATASET HEALTH SUMMARY ===')
    print(f'{"topology":<14} {"N":>5} {"V min":>10} {"V max":>8} {"bad V":>8} {"bad I":>8} {"NaN V":>6} {"NaN I":>6} {"NaN DC":>7}')
    for t, h in health.items():
        v = h['V']
        nv = h['NaN_counts']
        print(f"{t:<14} {h['n_total']:>5} {v['extreme_min']:>10.3f} {v['extreme_max']:>8.3f} "
              f"{v['unphysical_samples']:>4} ({v['unphysical_pct']:>4.1f}%) {h['I']['extreme_samples']:>8} "
              f"{nv['V']:>6} {nv['I']:>6} {nv['DC']:>7}")
        if v['bad_v_net_distrib']:
            print(f"   {t} bad-V net distribution: {v['bad_v_net_distrib']}")

    print(f'\nsaved → {OUT}')


if __name__ == '__main__':
    main()
