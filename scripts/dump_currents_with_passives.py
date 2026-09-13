#!/usr/bin/env python
"""Re-dump fan_smc I parity arrays to ALSO include passive terminals
(V sources, I sources) — not just MOSFET drain/source.
Adds: I_pred_pass_log10, I_tgt_pass_log10 to section41_parity_fan_smc.npz."""
import sys, pickle, yaml, argparse
import numpy as np, torch
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from circuitgnn.data.pretrain_loader import PretrainCombinedLoader
from circuitgnn.training.checkpoint import create_model_from_args
from circuitgnn.training.config import parse_training_config
from eval_all_checkpoints import attach_norm

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
EXP = REPO / 'datasets/opamp_3stage_fan_smc_openloop_5k/experiments/ginebn_3layer_bb8_notower_vn_kcl_w10_edge6_netrole_clip03_openloop_5k_usingnow'
DATA = REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/dataset_val.pkl'
OUT = REPO / 'figures/thesis/section41_parity_fan_smc.npz'

def main():
    ck = torch.load(EXP/'best_model.pt', map_location='cpu', weights_only=False)
    cfg = yaml.safe_load(open(EXP/'original_config.yaml'))
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
    for k,v in parse_training_config(cfg).items():
        if v is not None: setattr(cli, k, v)
    cli.device='cpu'; cli.dataset=str(DATA.parent); cli.predict_currents=True
    model, _ = create_model_from_args(cli, b0.x.shape[-1]+b0.type_tens.shape[-1], 'cpu')
    model.load_state_dict(ck['model_state_dict'])
    model.set_normalization_stats(vdc_mean=stats['v_mean'], vdc_std=stats['v_std'],
        ss_gm_mean=stats['gm_mean'], ss_gm_std=stats['gm_std'],
        ss_gds_mean=stats['gds_mean'], ss_gds_std=stats['gds_std'],
        current_mean=stats['i_mean'], current_std=stats['i_std'])
    model.current_epoch = 4022; model.eval()

    mos_p, mos_t, pass_p, pass_t = [], [], [], []
    with torch.no_grad():
        for batch in dl:
            attach_norm(batch, stats)
            out = model(batch)
            ic = out['node_currents']                   # z-scored log10|I|
            it = batch.node_current_targets             # raw |I| (signed actually; abs() needed)
            hcm = batch.has_current_mask                # True at MOSFET drain/source
            # Positions with NON-ZERO target but NOT in has_current_mask = passives (V/I sources)
            nonzero = (it.abs() > 1e-15)
            passive_mask = nonzero & (~hcm)
            # MOSFET-supervised positions
            log_p_all = ic * i_std + i_mean             # log10|I_pred|
            log_t_all = it.abs().clamp_min(1e-12).log10()
            mos_p.append(log_p_all[hcm].cpu().numpy())
            mos_t.append(log_t_all[hcm].cpu().numpy())
            pass_p.append(log_p_all[passive_mask].cpu().numpy())
            pass_t.append(log_t_all[passive_mask].cpu().numpy())
    mos_p = np.concatenate(mos_p); mos_t = np.concatenate(mos_t)
    pass_p = np.concatenate(pass_p); pass_t = np.concatenate(pass_t)
    print(f'MOSFET positions: {mos_p.size}')
    print(f'Passive positions (V/I sources): {pass_p.size}')

    # Load existing npz, augment, save
    d = dict(np.load(OUT))
    d['I_pred_pass_log10'] = pass_p
    d['I_tgt_pass_log10']  = pass_t
    # also dump µA for convenience
    d['I_pred_pass_uA'] = (10**pass_p) * 1e6
    d['I_tgt_pass_uA']  = (10**pass_t) * 1e6
    np.savez(OUT, **d)
    print(f'saved → {OUT}')
    err = np.abs(mos_p - mos_t).mean(); errp = np.abs(pass_p - pass_t).mean()
    print(f'\nlog10 MAE: MOSFET={err:.4f}, passive={errp:.4f}')

if __name__ == '__main__':
    main()
