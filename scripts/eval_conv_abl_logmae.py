#!/usr/bin/env python
"""Compute I log10-MAE for the conv-type ablation checkpoints on fan_smc val.
Adapts eval_section41_reference.py logic but evaluates all 4 (GINE+3 ablations)."""
import sys, yaml, argparse, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from src.data.pretrain_loader import PretrainCombinedLoader
from src.training.checkpoint import create_model_from_args
from src.training.config import parse_training_config
from eval_all_checkpoints import attach_norm

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
EXP = REPO / 'datasets/opamp_3stage_fan_smc_openloop_5k/experiments'
DATA = REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/dataset_val.pkl'

CHECKPOINTS = {
    'GINE (usingnow)': EXP / 'ginebn_3layer_bb8_notower_vn_kcl_w10_edge6_netrole_clip03_openloop_5k_usingnow' / 'best_model.pt',
    'GIN (no edge)':   EXP / 'ginebn_3layer_bb8_notower_vn_kcl_w10_edge6_netrole_clip03_openloop_5k_ABL_gin_noedge' / 'best_model.pt',
    'GATv2':           EXP / 'ginebn_3layer_bb8_notower_vn_kcl_w10_edge6_netrole_clip03_openloop_5k_ABL_gatv2' / 'best_model.pt',
    'GENConv':         EXP / 'ginebn_3layer_bb8_notower_vn_kcl_w10_edge6_netrole_clip03_openloop_5k_ABL_genconv' / 'best_model.pt',
}


def eval_one(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = yaml.safe_load(open(ckpt_path.parent / 'original_config.yaml'))
    s = ck['stats']
    stats = {'v_mean': s['vdc']['mean'], 'v_std': s['vdc']['std'],
             'i_mean': s['current']['mean'], 'i_std': s['current']['std'],
             'gm_mean': s['ss_gm']['mean'], 'gm_std': s['ss_gm']['std'],
             'gds_mean': s['ss_gds']['mean'], 'gds_std': s['ss_gds']['std']}
    i_mean, i_std = stats['i_mean'], stats['i_std']
    dl = PretrainCombinedLoader(str(DATA), batch_size=200, device='cpu', shuffle=False,
                                drop_last=False, topology_filter='fan_smc')
    b0 = next(iter(dl))
    cli = argparse.Namespace()
    for k, v in parse_training_config(cfg).items():
        if v is not None: setattr(cli, k, v)
    cli.device = 'cpu'; cli.dataset = str(DATA.parent); cli.predict_currents = True
    model, _ = create_model_from_args(cli, b0.x.shape[-1] + b0.type_tens.shape[-1], 'cpu')
    model.load_state_dict(ck['model_state_dict'])
    model.set_normalization_stats(
        vdc_mean=stats['v_mean'], vdc_std=stats['v_std'],
        ss_gm_mean=stats['gm_mean'], ss_gm_std=stats['gm_std'],
        ss_gds_mean=stats['gds_mean'], ss_gds_std=stats['gds_std'],
        current_mean=stats['i_mean'], current_std=stats['i_std'])
    model.current_epoch = ck.get('epoch', 4022)
    model.eval()
    i_errs_log, i_errs_uA = [], []
    with torch.no_grad():
        for b in dl:
            attach_norm(b, stats)
            out = model(b)
            ic = out['node_currents']  # z-scored log10|I|
            it = b.node_current_targets
            hcm = b.has_current_mask
            log_p = ic * i_std + i_mean
            log_t = it.abs().clamp_min(1e-12).log10()
            err_log = (log_p[hcm] - log_t[hcm]).abs().cpu().numpy()
            i_errs_log.append(err_log)
            pred_uA = (10 ** log_p[hcm]) * 1e6
            tgt_uA  = (10 ** log_t[hcm]) * 1e6
            i_errs_uA.append((pred_uA - tgt_uA).abs().cpu().numpy())
    errs = np.concatenate(i_errs_log)
    uA = np.concatenate(i_errs_uA)
    return {
        'epoch': int(ck.get('epoch', -1)),
        'I_logMAE': float(errs.mean()),
        'I_log_median': float(np.median(errs)),
        'I_uA_MAE': float(uA.mean()),
        'I_uA_median': float(np.median(uA)),
        'n_terminals': int(errs.size),
    }


def main():
    results = {}
    for label, path in CHECKPOINTS.items():
        if not path.exists():
            print(f'  MISSING: {path}'); continue
        print(f'\n=== {label} ===')
        print(f'  path: {path}')
        r = eval_one(path)
        results[label] = {**r, 'checkpoint_path': str(path)}
        print(f'  epoch     : {r["epoch"]}')
        print(f'  I logMAE  : {r["I_logMAE"]:.4f}')
        print(f'  I log med : {r["I_log_median"]:.4f}')
        print(f'  I µA MAE  : {r["I_uA_MAE"]:.2f}')
        print(f'  I µA med  : {r["I_uA_median"]:.2f}')
        print(f'  N terms   : {r["n_terminals"]}')
    out_json = REPO / 'figures/thesis/conv_abl_logmae.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nsaved → {out_json}')


if __name__ == '__main__':
    main()
