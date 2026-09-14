#!/usr/bin/env python
"""Verify KCL implementation per topology: compute KCL residual on GT currents
for each of the 5 topologies. If implementation is correct, residuals should be
~ngspice numerical noise (sub-nA, sub-0.001% normalized).

If any topology shows residuals orders of magnitude larger than fan_smc's
(~25 pA / 0.0002% median), the implementation has a topology-specific bug
(likely in the netlist parsing / sign convention for that topology).
"""
from __future__ import annotations
import pickle
import numpy as np
from pathlib import Path
import torch

VAL_PKL = Path(__file__).resolve().parents[2] / 'datasets/opamp_3stage_pretrain_combined_5topo/dataset_val.pkl'
TOPOS = ['fan_smc', 'sau_cfcc', 'peng_tcfc', 'leung_nmcf', 'leung_nmcnr']


def compute_residuals_for_sample(g):
    """Per-net KCL residual on a single graph's GT currents.
    Returns list of (raw_amps, abs_amps, n_terms) for each valid internal net.
    """
    # Required fields on the graph
    edge_index = g.edge_index                    # (2, E)
    nct = g.node_current_targets                 # raw signed amps per node
    sign = g.terminal_current_sign               # signed: +1/-1/0 per terminal
    kcl_inc = g.kcl_include_mask                 # bool: which terminals to include
    num_terminals = int(g.num_terminals)
    train_mask = g.train_mask                    # bool: internal nets (we predict V here)

    num_nodes = nct.shape[0]
    # |I| at each terminal (since target is already raw amps, take abs)
    I_abs = nct.abs().float()
    I_signed = I_abs * sign.float()              # signed amps via terminal sign

    src, dst = edge_index
    # Terminal index = local idx < num_terminals
    local_idx = torch.arange(num_nodes)
    terminal_mask = local_idx < num_terminals
    internal_net_mask = train_mask & ~terminal_mask

    valid = terminal_mask[src] & internal_net_mask[dst]
    if kcl_inc is not None:
        valid = valid & kcl_inc[src]

    vsrc = src[valid]
    vdst = dst[valid]
    if vsrc.numel() == 0:
        return []

    signed = I_signed[vsrc]
    absval = I_abs[vsrc]

    signed_sum = torch.zeros(num_nodes, dtype=torch.float64)
    abs_sum    = torch.zeros(num_nodes, dtype=torch.float64)
    n_terms    = torch.zeros(num_nodes, dtype=torch.long)
    pos_cnt    = torch.zeros(num_nodes, dtype=torch.long)
    neg_cnt    = torch.zeros(num_nodes, dtype=torch.long)

    signed_sum.scatter_add_(0, vdst, signed.double())
    abs_sum.scatter_add_(0, vdst, absval.double())
    n_terms.scatter_add_(0, vdst, torch.ones_like(vdst))
    pos_cnt.scatter_add_(0, vdst, (signed > 0).long())
    neg_cnt.scatter_add_(0, vdst, (signed < 0).long())

    # Same filter as compute_kcl_loss: ≥2 terms, both signs, total > 1 nA
    keep = internal_net_mask & (n_terms >= 2) & (pos_cnt > 0) & (neg_cnt > 0) & (abs_sum > 1e-9)
    idx = torch.nonzero(keep, as_tuple=False).flatten()
    raw  = signed_sum[idx].abs().cpu().numpy()
    abs_ = abs_sum[idx].cpu().numpy()
    return list(zip(raw.tolist(), abs_.tolist(), n_terms[idx].cpu().tolist()))


def main():
    print(f'Loading {VAL_PKL}...')
    with open(VAL_PKL, 'rb') as f:
        data = pickle.load(f)
    print(f'  {len(data)} total val samples\n')

    print('='*92)
    print(f"{'GT KCL residuals per topology (val set) — should ALL be ~ngspice noise':^92}")
    print('='*92)
    print(f"{'topology':<13} {'N samples':>10} {'N net-pts':>11}  | "
          f"{'raw (nA)':>27}  | {'norm (%)':>27}")
    print(f"{'':<13} {'':>10} {'':>11}  | "
          f"{'mean':>7} {'med':>7} {'p95':>7} {'p99':>4}  | "
          f"{'mean':>7} {'med':>7} {'p95':>7} {'p99':>4}")
    print('-'*92)

    for topo in TOPOS:
        samples = [s for s in data if s.get('topology') == topo]
        if not samples:
            print(f"{topo:<13}  no samples found")
            continue
        all_raws, all_abss, all_norms = [], [], []
        for s in samples:
            recs = compute_residuals_for_sample(s['graph'])
            for raw, abs_, _ in recs:
                all_raws.append(raw)
                all_abss.append(abs_)
                all_norms.append(raw / max(abs_, 1e-15))
        if not all_raws:
            print(f"{topo:<13}  no valid KCL nets")
            continue
        raws = np.array(all_raws); norms = np.array(all_norms)
        raw_nA = raws * 1e9
        norm_pct = norms * 100
        print(f"{topo:<13} {len(samples):>10} {len(raws):>11}  | "
              f"{raw_nA.mean():>7.4f} {np.median(raw_nA):>7.4f} "
              f"{np.percentile(raw_nA,95):>7.4f} {np.percentile(raw_nA,99):>4.2f}  | "
              f"{norm_pct.mean():>7.5f} {np.median(norm_pct):>7.5f} "
              f"{np.percentile(norm_pct,95):>7.5f} {np.percentile(norm_pct,99):>4.3f}")

    print()
    print('Interpretation:')
    print('  - "Good" topology: median residual ~ pA range, normalized median ~ 1e-4 % range')
    print('  - "Bad" topology (implementation issue): median residual >> nA, normalized median > 0.1 %')


if __name__ == '__main__':
    main()
