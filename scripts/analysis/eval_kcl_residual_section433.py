#!/usr/bin/env python
"""§4.3.3 KCL residual: KCL ON (usingnow) vs KCL OFF (DATAEFF n=5000, kclOff, s88).

KCL residual is defined per internal net (predicted-voltage net) as:
    raw_residual_A  = | Σ_t  sign(t) · |I_t|  |          [amps]
    norm_residual   = | Σ_t  sign(t) · |I_t|  | / Σ_t |I_t|   [unitless]

where t ranges over kcl_include_mask terminals connected to the net,
sign(t) = batch.terminal_current_sign[t], and |I_t| is the predicted
(or GT) current magnitude at terminal t. Nets with <2 included terminals,
total current <1 nA, or single sign at every terminal are skipped — those
nets cannot satisfy KCL by construction. Matches compute_kcl_loss semantics.

Stats reported:
  mean / median / p90 / p95 / p99 of raw & normalized residuals
  same on GT currents (the irreducible floor — should be ~ngspice noise)
  per-sample correlation between mean residual and current log-MAE.

Output: figures/thesis/section433_kcl_residual.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from circuitgnn.data.pretrain_loader import PretrainCombinedLoader
from circuitgnn.training.checkpoint import create_model_from_args
from circuitgnn.training.config import parse_training_config
from eval_all_checkpoints import attach_norm

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
EXP = REPO / 'datasets/opamp_3stage_fan_smc_openloop_5k/experiments'
KCL_ON_DIR  = EXP / 'ginebn_3layer_bb8_notower_vn_kcl_w10_edge6_netrole_clip03_openloop_5k_usingnow'
KCL_OFF_DIR = EXP / 'ginebn_3layer_bb8_notower_vn_kcl_w10_edge6_netrole_clip03_openloop_5k_DATAEFF_n5000_kclOff_s88'
VAL_PKL = REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/dataset_val.pkl'
OUT_JSON = REPO / 'figures/thesis/section433_kcl_residual.json'


def load_model(exp_dir, device, sample):
    ck_path = exp_dir / 'best.pt'
    if not ck_path.exists():
        ck_path = exp_dir / 'best_model.pt'
    ckpt = torch.load(ck_path, map_location=device, weights_only=False)
    with open(exp_dir / 'original_config.yaml') as f:
        cfg = yaml.safe_load(f)
    if 'norm_stats' in ckpt:
        stats = ckpt['norm_stats']
    elif 'stats' in ckpt:
        s = ckpt['stats']
        stats = {'v_mean': s['vdc']['mean'], 'v_std': s['vdc']['std'],
                 'i_mean': s['current']['mean'], 'i_std': s['current']['std'],
                 'gm_mean': s['ss_gm']['mean'], 'gm_std': s['ss_gm']['std'],
                 'gds_mean': s['ss_gds']['mean'], 'gds_std': s['ss_gds']['std']}
    else:
        raise RuntimeError(f'no norm stats in {ck_path}')
    import argparse as _ap
    cli = _ap.Namespace()
    for k, v in parse_training_config(cfg).items():
        if v is not None:
            setattr(cli, k, v)
    cli.device = str(device); cli.dataset = str(VAL_PKL.parent); cli.predict_currents = True
    node_dim = sample.x.shape[-1]; type_dim = sample.type_tens.shape[-1]
    model, _ = create_model_from_args(cli, node_dim + type_dim, device)
    model.load_state_dict(ckpt['model_state_dict'])
    if hasattr(model, 'set_normalization_stats'):
        model.set_normalization_stats(
            vdc_mean=stats['v_mean'], vdc_std=stats['v_std'],
            ss_gm_mean=stats['gm_mean'], ss_gm_std=stats['gm_std'],
            ss_gds_mean=stats['gds_mean'], ss_gds_std=stats['gds_std'],
            current_mean=stats['i_mean'], current_std=stats['i_std'])
    if hasattr(model, 'current_epoch'):
        model.current_epoch = ckpt.get('epoch', 99999)
    model.eval()
    return model, stats


def per_net_residuals(I_signed_A, batch, kcl_include, sample_id_offset):
    """Vectorized per-net residual sums. I_signed_A is (num_nodes,) signed amps
    (already multiplied by terminal_current_sign for terminals; non-terminals → 0)."""
    device = I_signed_A.device
    src, dst = batch.edge_index
    ptr = batch.ptr
    num_nodes = I_signed_A.numel()

    graph_sizes = ptr[1:] - ptr[:-1]
    batch_idx = torch.repeat_interleave(
        torch.arange(len(graph_sizes), device=device), graph_sizes)
    local_idx = torch.arange(num_nodes, device=device) - ptr[batch_idx]
    num_terminals = batch.num_terminals
    if isinstance(num_terminals, int):
        terminal_mask = local_idx < num_terminals
    else:
        terminal_mask = local_idx < num_terminals[batch_idx]

    tmask = batch.train_mask if hasattr(batch, 'train_mask') else (~batch.known_voltage_mask)
    internal_net_mask = tmask & ~terminal_mask

    valid_edges = terminal_mask[src] & internal_net_mask[dst]
    if kcl_include is not None:
        valid_edges &= kcl_include[src]
    if not valid_edges.any():
        return [], []
    vsrc = src[valid_edges]
    vdst = dst[valid_edges]

    signed_per_t = I_signed_A[vsrc]
    abs_per_t    = signed_per_t.abs()

    signed_sum = torch.zeros(num_nodes, device=device, dtype=torch.float64)
    abs_sum    = torch.zeros(num_nodes, device=device, dtype=torch.float64)
    n_terms    = torch.zeros(num_nodes, device=device, dtype=torch.long)
    pos_count  = torch.zeros(num_nodes, device=device, dtype=torch.long)
    neg_count  = torch.zeros(num_nodes, device=device, dtype=torch.long)

    signed_sum.scatter_add_(0, vdst, signed_per_t.double())
    abs_sum.scatter_add_(0, vdst, abs_per_t.double())
    n_terms.scatter_add_(0, vdst, torch.ones_like(vdst))
    pos_count.scatter_add_(0, vdst, (signed_per_t > 0).long())
    neg_count.scatter_add_(0, vdst, (signed_per_t < 0).long())

    valid_net = (internal_net_mask
                 & (n_terms >= 2)
                 & (pos_count > 0) & (neg_count > 0)
                 & (abs_sum > 1e-9))

    if not valid_net.any():
        return [], []

    keep_idx = torch.nonzero(valid_net, as_tuple=False).flatten()
    raw_arr  = signed_sum[keep_idx].abs().cpu().numpy()
    abs_arr  = abs_sum[keep_idx].cpu().numpy()
    norm_arr = raw_arr / np.clip(abs_arr, 1e-15, None)
    sids     = (batch_idx[keep_idx] + sample_id_offset).cpu().numpy()
    return list(zip(sids.tolist(), raw_arr.tolist(),
                    norm_arr.tolist(), abs_arr.tolist())), keep_idx


@torch.no_grad()
def evaluate_model(exp_dir, device):
    val = PretrainCombinedLoader(str(VAL_PKL), batch_size=50, device=device,
                                 shuffle=False, drop_last=False, topology_filter='fan_smc')
    sample = next(iter(val))
    model, stats = load_model(exp_dir, device, sample)

    pred_records = []   # tuples (sid, raw_A, norm, abs_A)
    gt_records   = []
    sample_curr_logmae = []
    sample_id_offset = 0

    val = PretrainCombinedLoader(str(VAL_PKL), batch_size=50, device=device,
                                 shuffle=False, drop_last=False, topology_filter='fan_smc')
    n_seen = 0; n_batches = 0
    for batch in val:
        n_batches += 1
        attach_norm(batch, stats)
        out = model(batch)
        ic_norm = out['node_currents'].squeeze()
        log10I = ic_norm * stats['i_std'] + stats['i_mean']
        I_A = torch.pow(10.0, log10I).clamp_min(0.0)
        sign = batch.terminal_current_sign.float()
        I_pred_signed = I_A * sign

        # GT magnitudes from raw signed amps, then apply terminal_current_sign
        # (same convention as pred for clean apples-to-apples).
        I_gt_signed = batch.node_current_targets.abs().float() * sign

        kcl_inc = getattr(batch, 'kcl_include_mask', None)

        pred_recs, _ = per_net_residuals(I_pred_signed, batch, kcl_inc, sample_id_offset)
        gt_recs,   _ = per_net_residuals(I_gt_signed,   batch, kcl_inc, sample_id_offset)
        pred_records.extend(pred_recs)
        gt_records.extend(gt_recs)

        # Per-sample current log-MAE for correlation
        it = batch.node_current_targets
        im = batch.has_current_mask
        log_tgt = it.abs().clamp_min(1e-12).log10()
        ptr = batch.ptr
        for g in range(len(ptr) - 1):
            gs = ptr[g].item(); ge = ptr[g + 1].item()
            mg = im[gs:ge]
            if mg.any():
                sample_curr_logmae.append(
                    (log10I[gs:ge][mg] - log_tgt[gs:ge][mg]).abs().mean().item())
            else:
                sample_curr_logmae.append(float('nan'))

        sample_id_offset += int(batch.num_graphs)
        n_seen += int(batch.num_graphs)
    print(f'  evaluated {n_seen} samples / {n_batches} batches → '
          f'{len(pred_records)} pred residuals, {len(gt_records)} GT residuals')
    return pred_records, gt_records, sample_curr_logmae


def summarize(records):
    raw  = np.array([r[1] for r in records])
    norm = np.array([r[2] for r in records])
    return {
        'N': int(len(raw)),
        'raw_mean_A':   float(raw.mean()),
        'raw_median_A': float(np.median(raw)),
        'raw_p90_A':    float(np.percentile(raw, 90)),
        'raw_p95_A':    float(np.percentile(raw, 95)),
        'raw_p99_A':    float(np.percentile(raw, 99)),
        'norm_mean':    float(norm.mean()),
        'norm_median':  float(np.median(norm)),
        'norm_p90':     float(np.percentile(norm, 90)),
        'norm_p95':     float(np.percentile(norm, 95)),
        'norm_p99':     float(np.percentile(norm, 99)),
    }


def sample_correlation(records, sample_curr_logmae):
    by_sample = {}
    for sid, raw, norm, abs_ in records:
        by_sample.setdefault(sid, []).append(norm)
    sids = sorted(by_sample.keys())
    res = np.array([np.mean(by_sample[s]) for s in sids])
    err = np.array([sample_curr_logmae[s] for s in sids])
    ok = np.isfinite(res) & np.isfinite(err)
    return float(np.corrcoef(res[ok], err[ok])[0, 1]) if ok.sum() > 2 else float('nan')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[setup] device={device}')

    results = {}
    for label, exp_dir in [('KCL_ON_usingnow',   KCL_ON_DIR),
                           ('KCL_OFF_n5000_s88', KCL_OFF_DIR)]:
        print(f'\n[eval] {label}')
        pred_recs, gt_recs, sample_curr = evaluate_model(exp_dir, device)
        results[label] = {
            'pred': summarize(pred_recs),
            'gt':   summarize(gt_recs),
            'corr_pred_norm_vs_currMAE': sample_correlation(pred_recs, sample_curr),
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved → {OUT_JSON}')

    print('\n' + '=' * 92)
    print('§4.3.3 KCL RESIDUAL ON fan_smc VAL  (definitions in script header)')
    print('=' * 92)
    for label, r in results.items():
        print(f'\n{label}')
        print(f'  PRED residuals  (N={r["pred"]["N"]:,})')
        print(f'    raw  [µA]   mean={r["pred"]["raw_mean_A"]*1e6:9.3f}  med={r["pred"]["raw_median_A"]*1e6:9.3f}  '
              f'p90={r["pred"]["raw_p90_A"]*1e6:9.3f}  p95={r["pred"]["raw_p95_A"]*1e6:9.3f}  p99={r["pred"]["raw_p99_A"]*1e6:9.3f}')
        print(f'    norm [%]    mean={r["pred"]["norm_mean"]*100:9.4f}  med={r["pred"]["norm_median"]*100:9.4f}  '
              f'p90={r["pred"]["norm_p90"]*100:9.4f}  p95={r["pred"]["norm_p95"]*100:9.4f}  p99={r["pred"]["norm_p99"]*100:9.4f}')
        print(f'  GT residuals    (N={r["gt"]["N"]:,})  [floor]')
        print(f'    raw  [nA]   mean={r["gt"]["raw_mean_A"]*1e9:9.4f}  med={r["gt"]["raw_median_A"]*1e9:9.4f}  '
              f'p90={r["gt"]["raw_p90_A"]*1e9:9.4f}  p95={r["gt"]["raw_p95_A"]*1e9:9.4f}  p99={r["gt"]["raw_p99_A"]*1e9:9.4f}')
        print(f'    norm [%]    mean={r["gt"]["norm_mean"]*100:9.5f}  med={r["gt"]["norm_median"]*100:9.5f}  '
              f'p90={r["gt"]["norm_p90"]*100:9.5f}  p95={r["gt"]["norm_p95"]*100:9.5f}  p99={r["gt"]["norm_p99"]*100:9.5f}')
        print(f'  corr(mean norm pred residual, sample current logMAE) = {r["corr_pred_norm_vs_currMAE"]:.3f}')


if __name__ == '__main__':
    main()
