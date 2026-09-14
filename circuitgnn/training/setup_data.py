"""Dataset loading and in-place batch preprocessing for train_v3.

Contains the three dataset-loading branches (prebatched pickle, fixed
topology, dynamic PyG) and the ordered in-place preprocessing pipeline.

ORDER IS LOAD-BEARING (frozen surface #7 in the release refactor spec): the
prebatched pipeline must run vdc normalization, then add_mosfet_gt_vov
(BEFORE current normalization), then current normalization, then
attach_normalization_stats, then the SS/Vth/region target attachment and the
SS/gm-Id/Vov/VgsVds stat computations, in exactly this order. Do not reorder.
"""

from dataclasses import dataclass

import numpy as np
import torch

from circuitgnn.training.data_loading import (
    PrebatchedLoader,
    load_prebatched_variant,
    load_prebatched_metadata,
    compute_vdc_normalization,
    compute_current_normalization,
    normalize_batches_vdc,
    normalize_batches_current,
    attach_normalization_stats,
    add_ss_node_targets,
    add_vth_node_targets,
    add_mosfet_gt_vov,
    add_region_node_targets,
    compute_ss_normalization,
    normalize_batches_ss,
    compute_vov_normalization,
    normalize_batches_vov,
)


@dataclass
class DataBundle:
    """Everything train_v3's main() needs from the data setup stage."""
    train_loader: object
    val_loader: object
    sample_batch: object
    variant_config: dict
    train_batches: object   # prebatched branch only, else None
    train_ds: object        # fixed-topology branch only, else None
    vdc_mean: float
    vdc_std: float
    current_mean: float
    current_std: float
    has_ss: bool
    ss_gm_mean: float
    ss_gm_std: float
    ss_gds_mean: float
    ss_gds_std: float
    ss_region_stats: object
    gm_id_mean: float
    gm_id_std: float
    vov_mean: float
    vov_std: float
    vgsvds_mean: object
    vgsvds_std: object


def setup_training_data(args, dataset_path, use_prebatched, use_fixed_topology):
    """Load one of the three dataset formats and run the in-place preprocessing pipeline."""
    target_norm_type = getattr(args, 'target_norm_type', 'zscore')
    vdd = getattr(args, 'vdd', 1.8)
    current_mean, current_std = 0.0, 1.0
    ss_gm_mean, ss_gm_std = 0.0, 1.0
    ss_gds_mean, ss_gds_std = 0.0, 1.0
    ss_region_stats = None
    gm_id_mean, gm_id_std = 0.0, 1.0
    vov_mean, vov_std = 0.0, 1.0
    vgsvds_mean, vgsvds_std = None, None
    has_ss = False
    variant_config = {'enabled': False}
    train_batches = None
    train_ds = None

    if use_prebatched:
        train_dir = dataset_path / 'train'
        val_dir = dataset_path / 'val'

        if not train_dir.exists():
            raise FileNotFoundError(f"Pre-batched train directory not found: {train_dir}")

        train_metadata = load_prebatched_metadata(train_dir)
        num_train_variants = train_metadata.get('num_variants', 1)
        preload_device = args.device if getattr(args, 'preload_to_gpu', False) and args.device != 'cpu' else None
        num_to_preload = min(getattr(args, 'max_preload_variants', 1), num_train_variants)

        all_train_variants = []
        print(f"Pre-loading {num_to_preload}/{num_train_variants} train variants...")
        for vid in range(num_to_preload):
            variant_batches = load_prebatched_variant(train_dir, variant_id=vid, device=preload_device)
            all_train_variants.append(variant_batches)
            print(f"  Loaded variant {vid}: {len(variant_batches)} batches")

        train_batches = all_train_variants[0]
        val_batches = load_prebatched_variant(val_dir, variant_id=0, device=preload_device) if val_dir.exists() else None
        if val_batches:
            print(f"Loaded {len(val_batches)} val batches")

        # Few-shot: subsample train set to first N graphs across all variants.
        # Val set stays full so we evaluate on the same val distribution.
        max_n = getattr(args, 'max_train_samples', None)
        if max_n is not None and max_n > 0:
            from torch_geometric.data import Batch as PyGBatch
            from circuitgnn.training.data_loading import _sort_edge_index
            def _subsample_variant(batches, n, dev):
                examples, remaining = [], n
                for b in batches:
                    if remaining <= 0:
                        break
                    for i in range(b.num_graphs):
                        if remaining <= 0:
                            break
                        examples.append(b.get_example(i))
                        remaining -= 1
                nb = PyGBatch.from_data_list(examples)
                if dev is not None:
                    nb = nb.to(dev)
                return [_sort_edge_index(nb)]
            print(f"[few-shot] Subsampling train to first {max_n} graphs (across {len(all_train_variants)} variants)...")
            for vid in range(len(all_train_variants)):
                all_train_variants[vid] = _subsample_variant(
                    all_train_variants[vid], max_n, preload_device,
                )
            train_batches = all_train_variants[0]
            print(f"[few-shot] After subsample: 1 batch with {train_batches[0].num_graphs} graphs")

        vdc_mean, vdc_std = compute_vdc_normalization(dataset_path, train_batches, target_norm_type, vdd)
        print(f"vdc normalization: mean={vdc_mean:.4f}, std={vdc_std:.4f}")

        for variant_batches in all_train_variants:
            normalize_batches_vdc(variant_batches, vdc_mean, vdc_std)
        if val_batches:
            normalize_batches_vdc(val_batches, vdc_mean, vdc_std)

        # Pre-compute GT Vov per MOSFET (must be before current normalization)
        for variant_batches in all_train_variants:
            add_mosfet_gt_vov(variant_batches)
        if val_batches:
            add_mosfet_gt_vov(val_batches)

        has_currents = any(hasattr(b, 'node_current_targets') and b.node_current_targets is not None for b in train_batches[:3])
        if has_currents:
            current_mean, current_std = compute_current_normalization(all_train_variants)
            current_z_clip = getattr(args, 'current_z_clip', 0.0)
            print(f"Current normalization: log10 mean={current_mean:.2f}, std={current_std:.2f}")
            if current_z_clip > 0:
                print(f"  Soft z-clip enabled at ±{current_z_clip}σ")
            for variant_batches in all_train_variants:
                normalize_batches_current(variant_batches, current_mean, current_std, z_clip=current_z_clip)
            if val_batches:
                normalize_batches_current(val_batches, current_mean, current_std, z_clip=current_z_clip)

        # Attach normalization stats to batches for physics-based current prediction
        for variant_batches in all_train_variants:
            attach_normalization_stats(variant_batches, vdc_mean, vdc_std, current_mean, current_std)
        if val_batches:
            attach_normalization_stats(val_batches, vdc_mean, vdc_std, current_mean, current_std)

        # Add per-node SS targets for mask-based gm/gds prediction
        for variant_batches in all_train_variants:
            add_ss_node_targets(variant_batches)
        if val_batches:
            add_ss_node_targets(val_batches)

        # Add per-node Vth targets for gm physics loss
        for variant_batches in all_train_variants:
            add_vth_node_targets(variant_batches)
        if val_batches:
            add_vth_node_targets(val_batches)

        # Add per-node region labels for region classification head
        for variant_batches in all_train_variants:
            add_region_node_targets(variant_batches)
        if val_batches:
            add_region_node_targets(val_batches)

        # Normalize SS targets (log10 gm/gds) with z-score
        has_ss = any(hasattr(b, 'mosfet_drain_mask') and b.mosfet_drain_mask.any() for b in train_batches[:3])
        ss_region_stats = None
        if has_ss:
            ss_per_region = getattr(args, 'ss_per_region_norm', False)
            if ss_per_region:
                ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std, ss_region_stats = compute_ss_normalization(all_train_variants, per_region=True)
                print(f"SS normalization (per-region):")
                print(f"  global: gm mean={ss_gm_mean:.2f}, std={ss_gm_std:.2f} | gds mean={ss_gds_mean:.2f}, std={ss_gds_std:.2f}")
                for r, name in enumerate(['cutoff', 'triode', 'saturation']):
                    s = ss_region_stats[r]
                    print(f"  {name}: gm mean={s['gm_mean']:.2f} std={s['gm_std']:.2f} | gds mean={s['gds_mean']:.2f} std={s['gds_std']:.2f}")
            else:
                ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std = compute_ss_normalization(all_train_variants)
                print(f"SS normalization: gm mean={ss_gm_mean:.2f}, std={ss_gm_std:.2f} | gds mean={ss_gds_mean:.2f}, std={ss_gds_std:.2f}")
            for variant_batches in all_train_variants:
                normalize_batches_ss(variant_batches, ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std, region_stats=ss_region_stats)
            if val_batches:
                normalize_batches_ss(val_batches, ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std, region_stats=ss_region_stats)

        # Compute gm/Id normalization (log10 z-score)
        ss_predict_gm_id = getattr(args, 'ss_head_config', {}).get('predict_gm_id', False)
        if (getattr(args, 'gm_id_aux_weight', 0.0) > 0 or ss_predict_gm_id) and has_ss:
            gm_id_vals = []
            for b in train_batches:
                gm = b.mosfet_gm if hasattr(b, 'mosfet_gm') else None
                if gm is not None:
                    valid = gm > 1e-12
                    if valid.any():
                        mi = b.mosfet_info.long()
                        num_m = mi.shape[0]
                        num_g = b.ptr.shape[0] - 1
                        mp = getattr(b, 'mosfet_ptr', None)
                        dev = mi.device
                        if mp is not None:
                            mg = torch.bucketize(torch.arange(num_m, device=dev), mp[1:].to(dev), right=True)
                        else:
                            mg = torch.arange(num_m, device=dev) // (num_m // num_g)
                        offs = b.ptr.to(dev)[mg]
                        drain_idx = mi[:, 1] + offs
                        log_gm = torch.log10(gm[valid].clamp(min=1e-20))
                        log_id = b.node_current_targets[drain_idx[valid]] * current_std + current_mean
                        gm_id_vals.append((log_gm - log_id).cpu())
            if gm_id_vals:
                all_gm_id = torch.cat(gm_id_vals)
                gm_id_mean = all_gm_id.mean().item()
                gm_id_std = all_gm_id.std().item()
                print(f"gm/Id normalization: mean={gm_id_mean:.2f}, std={gm_id_std:.2f}")

        # Normalize Vov targets (log10 z-score)
        has_vov = any(hasattr(b, 'mosfet_gt_vov') and (b.mosfet_gt_vov > 0).any() for b in train_batches[:3])
        vov_mean, vov_std = 0.0, 1.0
        if has_vov:
            vov_mean, vov_std = compute_vov_normalization(all_train_variants)
            print(f"Vov normalization: log10 mean={vov_mean:.2f}, std={vov_std:.2f}")
            for variant_batches in all_train_variants:
                normalize_batches_vov(variant_batches, vov_mean, vov_std)
            if val_batches:
                normalize_batches_vov(val_batches, vov_mean, vov_std)

        # Compute Vgs/Vds normalization stats if vgsvds head is enabled
        if getattr(args, 'vgsvds_loss_weight', 0.0) > 0 or getattr(args, 'vgsvds_config', {}).get('enabled', False):
            all_vgs, all_vds = [], []
            for b in train_batches:
                mi = b.mosfet_info.long()
                num_m = mi.shape[0]
                num_g = b.ptr.shape[0] - 1
                m_ptr = getattr(b, 'mosfet_ptr', None)
                if m_ptr is not None:
                    mg = torch.bucketize(torch.arange(num_m, device=mi.device), m_ptr[1:].to(mi.device), right=True)
                else:
                    mg = torch.arange(num_m, device=mi.device) // (num_m // num_g)
                offs = b.ptr.to(mi.device)[mg]
                vt = b.node_voltage_targets
                all_vgs.append((vt[mi[:, 0] + offs] - vt[mi[:, 2] + offs]).cpu())
                all_vds.append((vt[mi[:, 1] + offs] - vt[mi[:, 2] + offs]).cpu())
            all_vgs = torch.cat(all_vgs)
            all_vds = torch.cat(all_vds)
            vgsvds_mean = torch.tensor([all_vgs.mean().item(), all_vds.mean().item()])
            vgsvds_std = torch.tensor([all_vgs.std().item(), all_vds.std().item()])
            print(f"Vgs/Vds normalization: Vgs mean={vgsvds_mean[0]:.4f}, std={vgsvds_std[0]:.4f} | Vds mean={vgsvds_mean[1]:.4f}, std={vgsvds_std[1]:.4f}")

        train_loader = PrebatchedLoader(train_batches, shuffle=True)
        val_loader = PrebatchedLoader(val_batches, shuffle=False) if val_batches else None
        variant_config = {'enabled': num_to_preload > 1, 'num_variants': num_to_preload, 'all_train_variants': all_train_variants}
        sample_batch = train_batches[0]

    elif use_fixed_topology:
        from circuitgnn.data.fixed_topology_loader import build_fixed_topology_dataset, FixedTopologyLoader

        print(f"\n=== LOADING FIXED-TOPOLOGY DATASET ===")
        gpu_device = args.device if args.device != 'cpu' else None
        load_device = args.device  # load directly onto training device

        train_ds = build_fixed_topology_dataset(dataset_path / 'dataset_train.pkl', device=load_device)
        val_ds = build_fixed_topology_dataset(dataset_path / 'dataset_val.pkl', device=load_device)

        # VDC normalization (compute from raw stacked tensor)
        raw_vdc = train_ds.all_vdc.flatten()
        if target_norm_type == 'minmax':
            vdc_mean, vdc_std = 0.0, vdd
        else:
            vdc_mean = raw_vdc.mean().item()
            vdc_std = raw_vdc.std().item()
            if vdc_std == 0:
                vdc_std = 1.0
        print(f"vdc normalization: mean={vdc_mean:.4f}, std={vdc_std:.4f}")
        train_ds.normalize_vdc(vdc_mean, vdc_std)
        val_ds.normalize_vdc(vdc_mean, vdc_std)

        # Current normalization (use per-sample masks — has_current_mask varies across samples)
        log_eps = 1e-12
        masked_values = train_ds.all_currents[train_ds.all_has_current_mask].abs()
        log_c = torch.log10(masked_values + log_eps)
        current_mean = log_c.mean().item()
        current_std = log_c.std().item()
        if current_std == 0:
            current_std = 1.0
        print(f"Current normalization: log10 mean={current_mean:.2f}, std={current_std:.2f}")
        train_ds.normalize_currents(current_mean, current_std)
        val_ds.normalize_currents(current_mean, current_std)

        # SS normalization (log10 gm/gds z-score)
        has_ss = train_ds.all_node_log_gm is not None
        if has_ss:
            drain_mask = train_ds.mosfet_drain_mask  # [N]
            gm_vals = train_ds.all_node_log_gm[:, drain_mask].flatten()
            gds_vals = train_ds.all_node_log_gds[:, drain_mask].flatten()
            # Filter out -inf/nan from log10(0)
            gm_valid = gm_vals[torch.isfinite(gm_vals)]
            gds_valid = gds_vals[torch.isfinite(gds_vals)]
            ss_gm_mean, ss_gm_std = gm_valid.mean().item(), gm_valid.std().item()
            ss_gds_mean, ss_gds_std = gds_valid.mean().item(), gds_valid.std().item()
            if ss_gm_std == 0: ss_gm_std = 1.0
            if ss_gds_std == 0: ss_gds_std = 1.0
            print(f"SS normalization: gm mean={ss_gm_mean:.2f}, std={ss_gm_std:.2f} | gds mean={ss_gds_mean:.2f}, std={ss_gds_std:.2f}")
            train_ds.normalize_ss(ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std)
            val_ds.normalize_ss(ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std)

            # gm/Id normalization for fixed topology
            if getattr(args, 'gm_id_aux_weight', 0.0) > 0:
                mosfet_gm_all = train_ds.all_mosfet_gm  # [N_samples, M]
                valid = mosfet_gm_all > 1e-12
                log_gm = torch.log10(mosfet_gm_all[valid].clamp(min=1e-20))
                # Get drain currents: mosfet_info[:,1] gives drain terminal indices
                mi = train_ds.mosfet_info_fixed.long()
                drain_idx = mi[:, 1]  # [M] per-graph drain indices
                # all_node_current_targets: [N_samples, N_nodes] z-scored
                drain_currents = train_ds.all_node_current_targets[:, drain_idx]  # [N_samples, M]
                log_id = drain_currents[valid] * current_std + current_mean
                gm_id_all = log_gm - log_id
                gm_id_finite = gm_id_all[torch.isfinite(gm_id_all)]
                gm_id_mean = gm_id_finite.mean().item()
                gm_id_std = gm_id_finite.std().item()
                if gm_id_std == 0: gm_id_std = 1.0
                print(f"gm/Id normalization: mean={gm_id_mean:.2f}, std={gm_id_std:.2f}")

        ft_batch_size = getattr(args, 'batch_size', 1024)
        train_loader = FixedTopologyLoader(train_ds, batch_size=ft_batch_size, shuffle=True)
        val_loader = FixedTopologyLoader(val_ds, batch_size=ft_batch_size, shuffle=False)
        variant_config = {'enabled': False}
        sample_batch = train_ds.get_batch(list(range(min(4, len(train_ds)))))

    else:
        from torch_geometric.loader import DataLoader as PyGDataLoader
        from circuitgnn.training.data_loading import CircuitGraphDataset

        train_file = dataset_path / 'dataset_train.pkl'
        val_file = dataset_path / 'dataset_val.pkl'

        if not train_file.exists():
            raise FileNotFoundError(f"Training dataset not found: {train_file}")

        train_dataset = CircuitGraphDataset(train_file)
        val_dataset = CircuitGraphDataset(val_file) if val_file.exists() else None
        print(f"Loaded {len(train_dataset)} train samples")
        if val_dataset:
            print(f"Loaded {len(val_dataset)} val samples")

        # Compute normalization from training graphs
        all_vdc = torch.cat([g.vdc for g in train_dataset.graphs])
        if target_norm_type == 'minmax':
            vdc_mean, vdc_std = 0.0, vdd
        else:
            vdc_mean, vdc_std = all_vdc.mean().item(), all_vdc.std().item()
            if vdc_std == 0:
                vdc_std = 1.0
        print(f"vdc normalization: mean={vdc_mean:.4f}, std={vdc_std:.4f}")

        # Normalize graphs in-place
        for g in train_dataset.graphs:
            g.vdc = (g.vdc - vdc_mean) / vdc_std
        if val_dataset:
            for g in val_dataset.graphs:
                g.vdc = (g.vdc - vdc_mean) / vdc_std

        # Current normalization
        has_currents = any(hasattr(g, 'node_current_targets') and g.node_current_targets is not None for g in train_dataset.graphs[:3])
        if has_currents:
            log_epsilon = 1e-12
            all_currents = []
            for g in train_dataset.graphs:
                if hasattr(g, 'node_current_targets') and hasattr(g, 'has_current_mask'):
                    if g.node_current_targets is not None and g.has_current_mask is not None:
                        mask = g.has_current_mask
                        if mask.any():
                            all_currents.extend(g.node_current_targets[mask].abs().cpu().tolist())
            if all_currents:
                log_currents = np.log10(np.array(all_currents) + log_epsilon)
                current_mean, current_std = float(log_currents.mean()), float(log_currents.std())
                if current_std == 0:
                    current_std = 1.0
                print(f"Current normalization: log10 mean={current_mean:.2f}, std={current_std:.2f}")

                for g in train_dataset.graphs:
                    if hasattr(g, 'node_current_targets') and g.node_current_targets is not None:
                        log_targets = torch.log10(g.node_current_targets.abs() + log_epsilon)
                        g.node_current_targets = (log_targets - current_mean) / current_std
                if val_dataset:
                    for g in val_dataset.graphs:
                        if hasattr(g, 'node_current_targets') and g.node_current_targets is not None:
                            log_targets = torch.log10(g.node_current_targets.abs() + log_epsilon)
                            g.node_current_targets = (log_targets - current_mean) / current_std

        # Attach normalization stats to graphs for physics-based current prediction
        for g in train_dataset.graphs:
            g.voltage_mean = vdc_mean
            g.voltage_std = vdc_std
            g.current_mean = current_mean
            g.current_std = current_std
        if val_dataset:
            for g in val_dataset.graphs:
                g.voltage_mean = vdc_mean
                g.voltage_std = vdc_std
                g.current_mean = current_mean
                g.current_std = current_std

        batch_size = getattr(args, 'batch_size', 128)
        train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        print(f"Using batch_size={batch_size}")
        sample_batch = next(iter(train_loader))

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        sample_batch=sample_batch,
        variant_config=variant_config,
        train_batches=train_batches,
        train_ds=train_ds,
        vdc_mean=vdc_mean,
        vdc_std=vdc_std,
        current_mean=current_mean,
        current_std=current_std,
        has_ss=has_ss,
        ss_gm_mean=ss_gm_mean,
        ss_gm_std=ss_gm_std,
        ss_gds_mean=ss_gds_mean,
        ss_gds_std=ss_gds_std,
        ss_region_stats=ss_region_stats,
        gm_id_mean=gm_id_mean,
        gm_id_std=gm_id_std,
        vov_mean=vov_mean,
        vov_std=vov_std,
        vgsvds_mean=vgsvds_mean,
        vgsvds_std=vgsvds_std,
    )
