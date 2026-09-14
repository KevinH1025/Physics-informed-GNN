#!/usr/bin/env python3
"""
Training script for GNN model.

Features:
- Voltage and current prediction
- Pre-batched dataset support
- GPU pre-loading for fast training

Thin entry point: CLI/config merging lives in circuitgnn.training.cli, dataset
loading and normalization in circuitgnn.training.setup_data, fine-tune and
phase-2 checkpoint surgery in circuitgnn.training.transfer, the loss warmup
ramps in circuitgnn.training.warmup, the periodic and end-of-training reports
in circuitgnn.training.diagnostics, and checkpoint saving, the summary and the
curve plots in circuitgnn.training.finalize.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set CUBLAS workspace config for deterministic algorithms (must be before torch import)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
from tqdm import tqdm
import numpy as np
import time
import random

from circuitgnn.training.data_loading import PrebatchedLoader
from circuitgnn.training.loops import train_epoch, validate
from circuitgnn.training.losses import UncertaintyWeights
from circuitgnn.training.current_constraints import build_mirror_pair_indices, OPAMP_3STAGE_DEVICE_NAMES
from circuitgnn.training.scheduler import create_scheduler, apply_warmup, step_scheduler
from circuitgnn.training.checkpoint import create_model_from_args
from circuitgnn.training.cli import parse_args_with_config
from circuitgnn.training.setup_data import setup_training_data
from circuitgnn.training.warmup import LossSchedule, PairedLossSchedule
from circuitgnn.training import transfer
from circuitgnn.training import diagnostics
from circuitgnn.training import finalize


def main():
    args = parse_args_with_config()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Performance vs reproducibility tradeoff
    if getattr(args, 'fast', False):
        # Fast mode: prioritize speed over exact reproducibility
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
        torch.set_float32_matmul_precision('high')  # Use TensorCores
        print("Fast mode: cudnn.benchmark=True, deterministic=False")
    elif getattr(args, 'deterministic', False):
        # Full deterministic mode: ALL ops deterministic (scatter, atomics, etc.)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        print("Full deterministic mode: use_deterministic_algorithms=True (~3x slower)")
    else:
        # Default: cudnn deterministic only (scatter ops remain non-deterministic)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load dataset
    dataset_path = Path(args.dataset)
    use_fixed_topology = getattr(args, 'use_fixed_topology', False)
    use_prebatched = getattr(args, 'use_prebatched', False)
    # Auto-detect prebatched if train/ directory has variant files (skip if fixed_topology)
    if not use_fixed_topology and not use_prebatched and (dataset_path / 'train' / 'variant_0.pkl').exists():
        use_prebatched = True
        print("Auto-detected prebatched dataset")

    # Output directory: experiments/<name>/ if --name provided, else dataset root
    if args.name:
        output_path = dataset_path / 'experiments' / args.name
    else:
        output_path = dataset_path

    # Copy original config into experiment folder for reproducibility
    output_path.mkdir(parents=True, exist_ok=True)
    if args.config:
        import shutil
        shutil.copy2(args.config, output_path / 'original_config.yaml')
        # Append actual CLI overrides so the saved config reflects what ran
        with open(output_path / 'original_config.yaml', 'a') as f:
            f.write(f"\n# CLI overrides: seed={args.seed}\n")

    # Tee stdout to a log file in the output directory (overwritten each run)
    log_file = open(output_path / 'training.log', 'w')
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()
    sys.stdout = Tee(sys.__stdout__, log_file)

    print(f"\n=== LOADING DATASET ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Mode: {'prebatched' if use_prebatched else 'dynamic batching'}")

    data = setup_training_data(args, dataset_path, use_prebatched, use_fixed_topology)
    train_loader = data.train_loader
    val_loader = data.val_loader
    sample_batch = data.sample_batch
    variant_config = data.variant_config
    train_batches = data.train_batches
    train_ds = data.train_ds
    vdc_mean, vdc_std = data.vdc_mean, data.vdc_std
    current_mean, current_std = data.current_mean, data.current_std
    has_ss = data.has_ss
    ss_gm_mean, ss_gm_std = data.ss_gm_mean, data.ss_gm_std
    ss_gds_mean, ss_gds_std = data.ss_gds_mean, data.ss_gds_std
    ss_region_stats = data.ss_region_stats
    gm_id_mean, gm_id_std = data.gm_id_mean, data.gm_id_std
    vgsvds_mean, vgsvds_std = data.vgsvds_mean, data.vgsvds_std

    # Feature dimensions
    x_dim = sample_batch.x.shape[1]
    type_dim = sample_batch.type_tens.shape[1]
    net_type_dim = sample_batch.net_type.shape[1] if hasattr(sample_batch, 'net_type') and sample_batch.net_type is not None else 0
    spe_dim = sample_batch.structural_pe.shape[1] if hasattr(sample_batch, 'structural_pe') and sample_batch.structural_pe is not None else 0
    total_input_dim = x_dim + type_dim + net_type_dim + spe_dim
    print(f"Input features: {total_input_dim} (x={x_dim}, type={type_dim}, net={net_type_dim}, spe={spe_dim})")

    # Fine-tune: override architecture args from checkpoint config so model matches exactly
    if getattr(args, 'finetune', False):
        transfer.inherit_architecture_from_checkpoint(args)

    # Create model
    predict_currents = getattr(args, 'predict_currents', False)
    voltage_head_config = getattr(args, 'voltage_head_config', {})
    current_head_config = getattr(args, 'current_head_config', {})

    # Voltage-derived currents config
    derive_currents_from_voltage = getattr(args, 'derive_currents_from_voltage', False)
    if not hasattr(args, 'mosfet_current_mlp_config'):
        args.mosfet_current_mlp_config = {
            'hidden_dim': getattr(args, 'mosfet_current_mlp_hidden', 64),
            'num_layers': getattr(args, 'mosfet_current_mlp_layers', 2),
            'use_wl_ratio': True,
        }
    if derive_currents_from_voltage:
        print(f"Using voltage-derived currents (MLP: {args.mosfet_current_mlp_config})")

    model, model_class = create_model_from_args(args, total_input_dim, args.device)
    print(f"Using {model_class}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Set normalization stats for autograd gm/gds or physics DC gain (needed for denormalization)
    _needs_norm_stats = getattr(model, '_needs_autograd', False) or \
        (getattr(model, 'predict_dc_gain', False) and getattr(model, 'dc_gain_mode', '') == 'physics')
    if _needs_norm_stats:
        model.set_normalization_stats(
            vdc_mean, vdc_std,
            ss_gm_mean, ss_gm_std,
            ss_gds_mean, ss_gds_std,
            current_mean=current_mean, current_std=current_std,
        )
        _mode = 'autograd_ss' if getattr(model, 'use_autograd_ss', False) else \
                'iv_model' if getattr(model, 'use_iv_model', False) else 'dc_gain_physics'
        print(f"{_mode}: set normalization stats ("
              f"gm: {ss_gm_mean:.2f}/{ss_gm_std:.2f}, gds: {ss_gds_mean:.2f}/{ss_gds_std:.2f})")

    # Set gm/Id normalization stats if needed
    if hasattr(model, 'ss_gm_id_mean_buf') and gm_id_mean != 0.0:
        model.ss_gm_id_mean_buf.fill_(gm_id_mean)
        model.ss_gm_id_std_buf.fill_(gm_id_std)
        print(f"gm/Id normalization set on model: mean={gm_id_mean:.2f}, std={gm_id_std:.2f}")

    # torch.compile() for PyTorch 2.0+ optimization
    if getattr(args, 'compile', False):
        if hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled successfully")
        else:
            print("Warning: torch.compile() not available (requires PyTorch 2.0+)")

    # Phase 2: Load checkpoint and freeze backbone/voltage head
    if getattr(args, 'phase2', False):
        transfer.apply_phase2(model, args)

    # Fine-tune: load checkpoint weights, optionally freeze backbone
    if getattr(args, 'finetune', False):
        transfer.load_finetune_weights(model, args)

    # Uncertainty weighting (Kendall et al. 2018)
    use_uncertainty_weighting = getattr(args, 'use_uncertainty_weighting', False)
    uncertainty_weights = None
    if use_uncertainty_weighting:
        task_names = ['voltage', 'current']
        if getattr(args, 'ss_gm_loss_weight', 0.0) > 0:
            task_names.append('ss_gm')
        if getattr(args, 'ss_gds_loss_weight', 0.0) > 0:
            task_names.append('ss_gds')
        if getattr(args, 'vov_loss_weight', 0.0) > 0:
            task_names.append('vov')
        uncertainty_weights = UncertaintyWeights(task_names).to(args.device)
        print(f"Uncertainty weighting enabled for tasks: {task_names}")

    # Optimizer and scheduler (use only trainable params for Phase 2 / freeze-backbone)
    weight_decay = getattr(args, 'weight_decay', 0.0)
    adam_eps = getattr(args, 'adam_eps', 1e-8)
    use_fused = torch.cuda.is_available()
    uw_params = list(uncertainty_weights.parameters()) if uncertainty_weights is not None else []

    freeze_warmup_epochs = int(getattr(args, 'freeze_backbone_epochs', 0))
    in_freeze_warmup = (getattr(args, 'finetune', False) and freeze_warmup_epochs > 0)
    if getattr(args, 'phase2', False) or (getattr(args, 'finetune', False) and getattr(args, 'freeze_backbone', False)) or in_freeze_warmup:
        # Only optimize trainable (unfrozen) parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad] + uw_params
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=weight_decay, eps=adam_eps, fused=use_fused)
    elif getattr(args, 'finetune', False) and not getattr(args, 'freeze_backbone', False):
        # Discriminative LR: backbone gets lower LR, heads get full LR
        backbone_params, head_params = transfer.split_backbone_head_params(model)
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': args.lr * args.backbone_lr_scale},
            {'params': head_params + uw_params, 'lr': args.lr},
        ], weight_decay=weight_decay, eps=adam_eps, fused=use_fused)
        print(f"Discriminative LR: backbone={args.lr * args.backbone_lr_scale:.2e}, heads={args.lr:.2e}")
    else:
        optimizer = torch.optim.Adam(list(model.parameters()) + uw_params, lr=args.lr, weight_decay=weight_decay, eps=adam_eps, fused=use_fused)

    scheduler = create_scheduler(
        optimizer, args.scheduler, args.epochs, args.warmup,
        min_lr=getattr(args, 'end_lr', 1e-6),
        plateau_factor=getattr(args, 'plateau_factor', 0.5),
        plateau_patience=getattr(args, 'plateau_patience', 10)
    )

    # AMP default: disabled (FP32). Use --amp to force enable.
    if getattr(args, 'amp', False):
        use_amp = True
        amp_dtype = torch.bfloat16
        scaler = None  # bfloat16 doesn't need GradScaler
        print("Using AMP (BF16) — forced via --amp flag")
    else:
        use_amp = False
        amp_dtype = None
        scaler = None
        print("Using FP32 (AMP disabled)")


    # Training state
    best_val_loss, best_epoch = float('inf'), 0
    best_composite_score = 0.0
    train_losses, val_losses, val_maes = [], [], []
    train_voltage_losses, val_voltage_losses = [], []
    train_current_losses, val_current_losses = [], []
    train_ss_gm_losses, val_ss_gm_losses = [], []
    train_ss_gds_losses, val_ss_gds_losses = [], []
    train_kcl_losses, val_kcl_losses = [], []
    train_ac_losses, val_ac_losses = [], []
    train_ac_component_losses = {}  # {comp: [epoch_losses]}
    val_ac_component_losses = {}    # {comp: [epoch_losses]}
    train_dc_gain_losses, val_dc_gain_losses = [], []
    train_region_losses, val_region_losses = [], []
    max_grad_norms, avg_grad_norms = [], []
    train_gm_physics_losses, val_gm_physics_losses = [], []
    train_triode_physics_losses, val_triode_physics_losses = [], []
    train_cutoff_physics_losses, val_cutoff_physics_losses = [], []
    train_maes, train_current_maes, val_current_maes, learning_rates = [], [], [], []
    train_hc_mirror_losses, val_hc_mirror_losses = [], []
    epochs_without_improvement = 0
    best_model_state = None
    best_metrics = {'val_mae_mv': float('inf'), 'val_current_mae_ua': 0.0, 'acc80': 0.0, 'acc50': 0.0, 'acc20': 0.0, 'acc10': 0.0, 'current_acc50': 0.0, 'current_acc20': 0.0, 'current_acc10': 0.0, 'current_acc5': 0.0, 'rel_metrics': None}

    early_stopping_patience = getattr(args, 'early_stopping_patience', 80)
    val_freq = getattr(args, 'val_freq', 1)
    loss_type = getattr(args, 'loss_type', 'mse')
    huber_delta = getattr(args, 'huber_delta', 1.0)
    current_weight_target = getattr(args, 'current_weight', 1.0)
    current_warmup_epochs = getattr(args, 'current_warmup_epochs', 0)
    kcl_weight_target = getattr(args, 'kcl_weight', 0.0)
    kcl_warmup_epochs = getattr(args, 'kcl_warmup_epochs', 0)
    kcl_start_epoch_cfg = getattr(args, 'kcl_start_epoch', 0)
    kcl_min_current = getattr(args, 'kcl_min_current', 1e-9)
    kcl_mode = getattr(args, 'kcl_mode', 'logsumexp')
    kcl_exclusive = getattr(args, 'kcl_exclusive', False)
    kcl_detach_backbone = getattr(args, 'kcl_detach_backbone', False)
    kcl_mask_unsupervised = getattr(args, 'kcl_mask_unsupervised', True)
    kcl_violation_threshold = getattr(args, 'kcl_violation_threshold', 0.0)
    kcl_huber_delta = getattr(args, 'kcl_huber_delta', 0.0)
    kcl_gt_filter = getattr(args, 'kcl_gt_filter', 0.1)
    kcl_conservation = getattr(args, 'kcl_conservation', False)
    kcl_skip_two_term = getattr(args, 'kcl_skip_two_term', False)
    kcl_only_two_term = getattr(args, 'kcl_only_two_term', False)
    kcl_intermediate_weight = getattr(args, 'kcl_intermediate_weight', 0.0)
    vov_loss_weight = getattr(args, 'vov_loss_weight', 0.0)
    vth_loss_weight = getattr(args, 'vth_loss_weight', 0.0)
    vdiff_loss_weight = getattr(args, 'vdiff_loss_weight', 0.0)
    vgsvds_loss_weight = getattr(args, 'vgsvds_loss_weight', 0.0)
    # Physics constraint config (diff pair, current mirrors)
    constraint_weight_target = getattr(args, 'constraint_weight', 0.0)
    constraint_warmup_epochs = getattr(args, 'constraint_warmup_epochs', 0)
    constraint_start_epoch_cfg = getattr(args, 'constraint_start_epoch', 0)

    # Build hardcoded mirror pair indices (once at startup)
    mirror_pair_indices, mirror_pair_ratios, mirror_pair_names = None, None, None
    if constraint_weight_target > 0:
        # Try to get device names from first batch, fallback to hardcoded 3-stage names
        first_batch = next(iter(train_loader))
        names = None
        if hasattr(first_batch, 'mosfet_device_names') and first_batch.mosfet_device_names:
            names = first_batch.mosfet_device_names
            if isinstance(names[0], list):
                names = names[0]
        if names is None:
            names = OPAMP_3STAGE_DEVICE_NAMES
            print(f"Using hardcoded 3-stage device names ({len(names)} devices)")
        mirror_pair_indices, mirror_pair_ratios, mirror_pair_names = build_mirror_pair_indices(names)
        if len(mirror_pair_indices) > 0:
            mirror_pair_indices = mirror_pair_indices.to(args.device)
            mirror_pair_ratios = mirror_pair_ratios.to(args.device)
            print(f"Mirror pairs: {len(mirror_pair_indices)} pairs — {', '.join(mirror_pair_names)}")
        else:
            print("Warning: No mirror pairs found in device names")
            mirror_pair_indices = None

    # gm self-consistency physics loss config
    gm_physics_loss_weight_target = getattr(args, 'gm_physics_loss_weight', 0.0)
    gm_physics_loss_warmup_epochs = getattr(args, 'gm_physics_loss_warmup_epochs', 0)
    gm_physics_loss_start_epoch_cfg = getattr(args, 'gm_physics_loss_start_epoch', 0)
    gm_physics_min_vov = getattr(args, 'gm_physics_min_vov', 0.0)
    gm_physics_use_clm = getattr(args, "gm_physics_use_clm", False)
    gm_physics_use_smaxt = getattr(args, "gm_physics_use_smaxt", False)
    gm_physics_use_gt_voltages = getattr(args, 'gm_physics_use_gt_voltages', True)

    # AC prediction loss config
    ac_loss_weight_target = getattr(args, 'ac_loss_weight', 0.0)
    ac_loss_warmup_epochs = getattr(args, 'ac_loss_warmup_epochs', 0)
    ac_loss_start_epoch_cfg = getattr(args, 'ac_loss_start_epoch', 0)

    # Supervised gm/gds loss config
    ss_gm_loss_weight_target = getattr(args, 'ss_gm_loss_weight', 0.0)
    ss_gds_loss_weight_target = getattr(args, 'ss_gds_loss_weight', 0.0)
    ss_loss_warmup_epochs = getattr(args, 'ss_loss_warmup_epochs', 0)
    ss_loss_start_epoch_cfg = getattr(args, 'ss_loss_start_epoch', 0)

    # Triode physics regularizer loss config
    triode_physics_loss_weight_target = getattr(args, 'triode_physics_loss_weight', 0.0)
    triode_physics_loss_warmup_epochs = getattr(args, 'triode_physics_loss_warmup_epochs', 0)
    triode_physics_loss_start_epoch_cfg = getattr(args, 'triode_physics_loss_start_epoch', 0)
    triode_physics_config = getattr(args, 'triode_physics_config', None)

    # Cutoff/subthreshold physics loss config
    cutoff_physics_loss_weight_target = getattr(args, 'cutoff_physics_loss_weight', 0.0)
    cutoff_physics_loss_warmup_epochs = getattr(args, 'cutoff_physics_loss_warmup_epochs', 0)
    cutoff_physics_loss_start_epoch_cfg = getattr(args, 'cutoff_physics_loss_start_epoch', 0)
    cutoff_physics_n_nmos = getattr(args, 'cutoff_physics_n_nmos', 1.5)
    cutoff_physics_n_pmos = getattr(args, 'cutoff_physics_n_pmos', 2.0)

    # Device consistency loss config
    device_consistency_weight = getattr(args, 'device_consistency_weight', 0.0)

    # DC gain prediction loss config
    dc_gain_loss_weight_target = getattr(args, 'dc_gain_loss_weight', 0.0)
    dc_gain_loss_weight = dc_gain_loss_weight_target
    dc_gain_warmup_epochs = getattr(args, 'dc_gain_warmup_epochs', 0)
    dc_gain_start_epoch = getattr(args, 'dc_gain_start_epoch', 0)

    # Region classification loss config
    region_loss_weight_target = getattr(args, 'region_loss_weight', 0.0)
    region_loss_start_epoch_cfg = getattr(args, 'region_loss_start_epoch', 0)

    # Build AC components list from config
    ac_head_cfg = getattr(args, 'ac_head_config', {})
    ac_components = []
    if ac_head_cfg.get('predict_ugbw', True):
        ac_components.append('ugbw')
    if ac_head_cfg.get('predict_pm', True):
        ac_components.append('pm')
    if ac_head_cfg.get('predict_am', False):
        ac_components.append('am')

    # Compute AC normalization stats from training data (if AC loss enabled)
    ac_mean = None
    ac_std = None
    if ac_loss_weight_target > 0 and ac_components:
        # Collect raw values for each enabled component
        ac_raw = {comp: [] for comp in ac_components}
        if use_fixed_topology:
            # For FixedTopologyLoader, scan the dataset tensors directly
            if train_ds.all_ac_valid is not None and train_ds.all_ac_ugbw is not None:
                valid = train_ds.all_ac_valid.bool()
                if valid.any():
                    if 'ugbw' in ac_raw:
                        ac_raw['ugbw'].extend(torch.log10(train_ds.all_ac_ugbw[valid].clamp(min=1.0)).cpu().tolist())
                    if 'pm' in ac_raw:
                        ac_raw['pm'].extend(train_ds.all_ac_pm[valid].cpu().tolist())
                    if 'am' in ac_raw and train_ds.all_ac_am is not None:
                        ac_raw['am'].extend(train_ds.all_ac_am[valid].cpu().tolist())
        else:
            scan_batches = train_batches if use_prebatched else []
            for b in scan_batches:
                if hasattr(b, 'ac_valid') and hasattr(b, 'ac_ugbw'):
                    valid = b.ac_valid
                    if valid.any():
                        if 'ugbw' in ac_raw:
                            ac_raw['ugbw'].extend(torch.log10(b.ac_ugbw[valid].clamp(min=1.0)).tolist())
                        if 'pm' in ac_raw:
                            ac_raw['pm'].extend(b.ac_pm[valid].tolist())
                        if 'am' in ac_raw:
                            ac_raw['am'].extend(b.ac_am[valid].tolist())
        n_valid = len(ac_raw[ac_components[0]])
        if n_valid > 0:
            ac_mean = torch.tensor([np.mean(ac_raw[c]) for c in ac_components], dtype=torch.float32)
            ac_std = torch.tensor([np.std(ac_raw[c]) for c in ac_components], dtype=torch.float32).clamp(min=1e-6)
            print(f"AC components: {ac_components}")
            print(f"AC normalization: mean={ac_mean.tolist()}, std={ac_std.tolist()}")
            print(f"  ({n_valid} valid AC samples in training set)")
        else:
            print("WARNING: No valid AC samples found. AC loss will be disabled.")
            ac_loss_weight_target = 0.0

    # Compute DC gain normalization stats from training data (all samples)
    dc_gain_mean = 0.0
    dc_gain_std = 1.0
    if dc_gain_loss_weight > 0:
        dc_gain_vals = []
        for batch in train_loader:
            if hasattr(batch, 'ac_dc_gain'):
                dc_gain_vals.extend(batch.ac_dc_gain.tolist())
        if dc_gain_vals:
            dc_gain_mean = float(np.mean(dc_gain_vals))
            dc_gain_std = max(float(np.std(dc_gain_vals)), 1e-6)
            print(f"DC gain normalization: mean={dc_gain_mean:.2f} dB, std={dc_gain_std:.2f} dB")
            print(f"  ({len(dc_gain_vals)} samples)")
        else:
            print("WARNING: No dc_gain data found. DC gain loss will be disabled.")
            dc_gain_loss_weight = 0.0

    # Stage 2 node weighting config
    stage2_weight = getattr(args, 'stage2_weight', 1.0)
    stage2_nodes = getattr(args, 'stage2_nodes', None)
    node_weights = getattr(args, 'node_weights', None)

    # Terminal voltage supervision
    use_terminal_voltage_loss = getattr(args, 'use_terminal_voltage_loss', False)
    if use_terminal_voltage_loss:
        print("Terminal voltage supervision enabled (loss on terminal nodes)")

    training_start_time = time.time()

    # All-variants-per-epoch mode: concatenate all variant batches into one loader
    all_variants_per_epoch = getattr(args, 'all_variants_per_epoch', False)
    if all_variants_per_epoch and variant_config['enabled']:
        all_batches = [b for var in variant_config['all_train_variants'] for b in var]
        train_loader = PrebatchedLoader(all_batches, shuffle=True)
        print(f"All-variants-per-epoch: {len(all_batches)} batches/epoch "
              f"(from {variant_config['num_variants']} variants)")

    # Per-loss warmup ramps. Most physics losses default to starting after the
    # current-loss warmup when no explicit start epoch is configured; the DC
    # gain and region losses default to epoch 0.
    current_sched = LossSchedule(current_weight_target, current_warmup_epochs, 0, 0)
    kcl_sched = LossSchedule(kcl_weight_target, kcl_warmup_epochs, kcl_start_epoch_cfg, current_warmup_epochs)
    constraint_sched = LossSchedule(constraint_weight_target, constraint_warmup_epochs, constraint_start_epoch_cfg, current_warmup_epochs)
    gm_physics_sched = LossSchedule(gm_physics_loss_weight_target, gm_physics_loss_warmup_epochs, gm_physics_loss_start_epoch_cfg, current_warmup_epochs)
    ac_sched = LossSchedule(ac_loss_weight_target, ac_loss_warmup_epochs, ac_loss_start_epoch_cfg, current_warmup_epochs)
    ss_sched = PairedLossSchedule(ss_gm_loss_weight_target, ss_gds_loss_weight_target, ss_loss_warmup_epochs, ss_loss_start_epoch_cfg, current_warmup_epochs)
    triode_physics_sched = LossSchedule(triode_physics_loss_weight_target, triode_physics_loss_warmup_epochs, triode_physics_loss_start_epoch_cfg, current_warmup_epochs)
    cutoff_physics_sched = LossSchedule(cutoff_physics_loss_weight_target, cutoff_physics_loss_warmup_epochs, cutoff_physics_loss_start_epoch_cfg, current_warmup_epochs)
    dc_gain_sched = LossSchedule(dc_gain_loss_weight_target, dc_gain_warmup_epochs, dc_gain_start_epoch, 0)
    region_sched = LossSchedule(region_loss_weight_target, 0, region_loss_start_epoch_cfg, 0)

    print(f"\nTraining for {args.epochs} epochs (early stopping: patience={early_stopping_patience})...")

    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        # Progressive unfreeze: at the start of `freeze_backbone_epochs`, the
        # backbone is frozen and the optimizer only contains heads. At that
        # epoch boundary, unfreeze backbone params and rebuild the optimizer
        # so backbone gradients start flowing.
        if (in_freeze_warmup and epoch == freeze_warmup_epochs
                and not getattr(args, 'freeze_backbone', False)):
            backbone_params, head_params = transfer.split_backbone_head_params(model, unfreeze=True)
            optimizer = torch.optim.Adam([
                {'params': backbone_params, 'lr': args.lr * args.backbone_lr_scale},
                {'params': head_params + uw_params, 'lr': args.lr},
            ], weight_decay=weight_decay, eps=adam_eps, fused=use_fused)
            scheduler = create_scheduler(
                optimizer, args.scheduler, args.epochs - epoch, args.warmup,
                min_lr=getattr(args, 'end_lr', 1e-6),
                plateau_factor=getattr(args, 'plateau_factor', 0.5),
                plateau_patience=getattr(args, 'plateau_patience', 10),
            )
            print(f"\n[epoch {epoch}] UNFROZE backbone — rebuilt optimizer with "
                  f"backbone_lr={args.lr * args.backbone_lr_scale:.2e}, "
                  f"head_lr={args.lr:.2e}, scheduler reset")

        # Variant rotation (skip if using all variants per epoch)
        if variant_config['enabled'] and epoch > 0 and not all_variants_per_epoch:
            variant_id = epoch % variant_config['num_variants']
            train_batches = variant_config['all_train_variants'][variant_id]
            train_loader = PrebatchedLoader(train_batches, shuffle=True)

        apply_warmup(optimizer, epoch, args.warmup, args.lr)

        # Loss weights for this epoch (linear warmup ramps)
        current_weight = current_sched.weight(epoch)
        kcl_weight = kcl_sched.weight(epoch)
        constraint_weight = constraint_sched.weight(epoch)
        gm_physics_loss_weight = gm_physics_sched.weight(epoch)
        ac_loss_weight = ac_sched.weight(epoch)
        ss_gm_loss_weight, ss_gds_loss_weight = ss_sched.weights(epoch)
        triode_physics_loss_weight = triode_physics_sched.weight(epoch)
        cutoff_physics_loss_weight = cutoff_physics_sched.weight(epoch)
        dc_gain_loss_weight = dc_gain_sched.weight(epoch)
        region_loss_weight = region_sched.weight(epoch)

        # Update current epoch for warmup-aware modules (e.g. loop attention)
        if hasattr(model, 'current_epoch'):
            model.current_epoch = epoch

        loss, mae_norm, voltage_loss, current_loss, current_mae_ua, kcl_loss, diff_pair_loss, mirror_loss, output_stage_loss, lambda_mirror_loss, gm_physics_loss, ac_loss, ss_gm_loss, ss_gds_loss, triode_physics_loss, triode_eq1_loss, triode_eq2_loss, triode_eq3_loss, cutoff_physics_loss, region_loss, train_vov_loss, train_vth_loss, train_ac_comp, train_dc_gain_loss, max_grad_norm, avg_grad_norm, train_gm_id_loss, train_gm_id_aux_loss, train_hc_mirror_loss, train_mirror_pair_detail = train_epoch(
            model, train_loader, optimizer, args.gradient_clip, args.device, scaler,
            predict_currents=predict_currents, current_weight=current_weight,
            voltage_weight=getattr(args, 'voltage_weight', 1.0),
            kcl_weight=kcl_weight,
            current_mean=current_mean, current_std=current_std, loss_type=loss_type, huber_delta=huber_delta,
            kcl_min_current=kcl_min_current, constraint_weight=constraint_weight, amp_dtype=amp_dtype,
            vdc_mean=vdc_mean, vdc_std=vdc_std,
            stage2_nodes=stage2_nodes, stage2_weight=stage2_weight, node_weights=node_weights,
            use_terminal_voltage_loss=use_terminal_voltage_loss,
            gm_physics_loss_weight=gm_physics_loss_weight, gm_physics_min_vov=gm_physics_min_vov, gm_physics_use_clm=gm_physics_use_clm, gm_physics_use_smaxt=gm_physics_use_smaxt,
            gm_physics_use_gt_voltages=gm_physics_use_gt_voltages,
            ss_gm_mean=ss_gm_mean if has_ss else 0.0, ss_gm_std=ss_gm_std if has_ss else 1.0,
            ss_gds_mean=ss_gds_mean if has_ss else 0.0, ss_gds_std=ss_gds_std if has_ss else 1.0,
            ac_loss_weight=ac_loss_weight, ac_mean=ac_mean, ac_std=ac_std, ac_components=ac_components,
            ss_gm_loss_weight=ss_gm_loss_weight, ss_gds_loss_weight=ss_gds_loss_weight,
            ss_huber_delta=getattr(args, 'ss_huber_delta', 0.0),
            ss_region_stats=ss_region_stats,
            triode_physics_loss_weight=triode_physics_loss_weight, triode_physics_config=triode_physics_config,
            cutoff_physics_loss_weight=cutoff_physics_loss_weight, cutoff_physics_n_nmos=cutoff_physics_n_nmos, cutoff_physics_n_pmos=cutoff_physics_n_pmos,
            region_loss_weight=region_loss_weight,
            kcl_mode=kcl_mode,
            kcl_exclusive=kcl_exclusive,
            kcl_detach_backbone=kcl_detach_backbone,
            kcl_mask_unsupervised=kcl_mask_unsupervised,
            kcl_violation_threshold=kcl_violation_threshold,
            kcl_huber_delta=kcl_huber_delta,
            kcl_gt_filter=kcl_gt_filter,
            kcl_conservation=kcl_conservation,
            kcl_skip_two_term=kcl_skip_two_term,
            kcl_only_two_term=kcl_only_two_term,
            device_consistency_weight=device_consistency_weight,
            intermediate_v_weight=getattr(args, 'intermediate_v_weight', 0.0),
            kcl_intermediate_weight=kcl_intermediate_weight,
            vov_loss_weight=vov_loss_weight,
            vth_loss_weight=vth_loss_weight,
            uncertainty_weights=uncertainty_weights,
            iv_id_loss_weight=getattr(args, 'iv_id_loss_weight', 0.0),
            dc_gain_loss_weight=dc_gain_loss_weight,
            dc_gain_mean=dc_gain_mean,
            dc_gain_std=dc_gain_std,
            gm_id_consistency_weight=getattr(args, 'gm_id_consistency_weight', 0.0),
            gm_id_aux_weight=getattr(args, 'gm_id_aux_weight', 0.0),
            gm_id_mean=gm_id_mean,
            gm_id_std=gm_id_std,
            mirror_pair_indices=mirror_pair_indices,
            mirror_pair_ratios=mirror_pair_ratios,
            mirror_pair_names=mirror_pair_names,
            vdiff_loss_weight=vdiff_loss_weight,
            vgsvds_loss_weight=vgsvds_loss_weight,
            vgsvds_mean=vgsvds_mean,
            vgsvds_std=vgsvds_std,
        )

        mae_mv = mae_norm * vdc_std * 1000
        train_losses.append(loss)
        train_voltage_losses.append(voltage_loss)
        train_current_losses.append(current_loss)
        train_ss_gm_losses.append(ss_gm_loss)
        train_ss_gds_losses.append(ss_gds_loss)
        train_kcl_losses.append(kcl_loss)
        train_ac_losses.append(ac_loss)
        for comp, comp_val in train_ac_comp.items():
            train_ac_component_losses.setdefault(comp, []).append(comp_val)
        train_dc_gain_losses.append(train_dc_gain_loss)
        train_region_losses.append(region_loss)
        train_gm_physics_losses.append(gm_physics_loss)
        max_grad_norms.append(max_grad_norm)
        avg_grad_norms.append(avg_grad_norm)
        train_triode_physics_losses.append(triode_physics_loss)
        train_cutoff_physics_losses.append(cutoff_physics_loss)
        train_maes.append(mae_mv)
        train_current_maes.append(current_mae_ua)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        train_hc_mirror_losses.append(train_hc_mirror_loss)

        # Validation
        if val_loader and epoch % val_freq == 0:
            eval_model = model
            val_loss, val_mae_mv, val_v_loss, val_c_loss, val_c_mae, acc80, acc50, acc20, acc10, current_acc50, current_acc20, current_acc10, current_acc5, val_kcl_loss, val_dp_loss, val_mirror_loss, val_os_loss, val_lm_loss, val_gm_physics_loss, val_ac_loss, val_ss_gm_loss, val_ss_gds_loss, val_triode_physics_loss, val_triode_eq1_loss, val_triode_eq2_loss, val_triode_eq3_loss, val_cutoff_physics_loss, val_region_loss, val_rel_metrics, val_vov_loss, val_vth_loss, val_ac_comp, val_dc_gain_loss, val_gm_id_loss, val_gm_id_aux_loss, val_hc_mirror_loss, val_mirror_pair_detail = validate(
                eval_model, val_loader, args.device, vdc_mean, vdc_std, current_mean, current_std,
                predict_currents=predict_currents, current_weight=current_weight,
                voltage_weight=getattr(args, 'voltage_weight', 1.0),
                kcl_weight=kcl_weight,
                loss_type=loss_type, huber_delta=huber_delta, kcl_min_current=kcl_min_current,
                constraint_weight=constraint_weight,
                stage2_nodes=stage2_nodes, stage2_weight=stage2_weight, node_weights=node_weights,
                use_terminal_voltage_loss=use_terminal_voltage_loss,
                gm_physics_loss_weight=gm_physics_loss_weight, gm_physics_min_vov=gm_physics_min_vov, gm_physics_use_clm=gm_physics_use_clm, gm_physics_use_smaxt=gm_physics_use_smaxt,
                gm_physics_use_gt_voltages=gm_physics_use_gt_voltages,
                ss_gm_mean=ss_gm_mean if has_ss else 0.0, ss_gm_std=ss_gm_std if has_ss else 1.0,
                ss_gds_mean=ss_gds_mean if has_ss else 0.0, ss_gds_std=ss_gds_std if has_ss else 1.0,
                ac_loss_weight=ac_loss_weight, ac_mean=ac_mean, ac_std=ac_std, ac_components=ac_components,
                ss_gm_loss_weight=ss_gm_loss_weight, ss_gds_loss_weight=ss_gds_loss_weight,
                ss_huber_delta=getattr(args, 'ss_huber_delta', 0.0),
                ss_region_stats=ss_region_stats,
                triode_physics_loss_weight=triode_physics_loss_weight, triode_physics_config=triode_physics_config,
                cutoff_physics_loss_weight=cutoff_physics_loss_weight, cutoff_physics_n_nmos=cutoff_physics_n_nmos, cutoff_physics_n_pmos=cutoff_physics_n_pmos,
                region_loss_weight=region_loss_weight,
                kcl_mode=kcl_mode,
                kcl_exclusive=kcl_exclusive,
                kcl_detach_backbone=kcl_detach_backbone,
            kcl_mask_unsupervised=kcl_mask_unsupervised,
                kcl_violation_threshold=kcl_violation_threshold,
                kcl_huber_delta=kcl_huber_delta,
                kcl_gt_filter=kcl_gt_filter,
                kcl_conservation=kcl_conservation,
                kcl_skip_two_term=kcl_skip_two_term,
                kcl_only_two_term=kcl_only_two_term,
                amp_dtype=amp_dtype,
                device_consistency_weight=device_consistency_weight,
                vov_loss_weight=vov_loss_weight,
                vth_loss_weight=vth_loss_weight,
                iv_id_loss_weight=getattr(args, 'iv_id_loss_weight', 0.0),
                dc_gain_loss_weight=dc_gain_loss_weight,
                dc_gain_mean=dc_gain_mean,
                dc_gain_std=dc_gain_std,
                gm_id_consistency_weight=getattr(args, 'gm_id_consistency_weight', 0.0),
                gm_id_aux_weight=getattr(args, 'gm_id_aux_weight', 0.0),
                gm_id_mean=gm_id_mean,
                gm_id_std=gm_id_std,
                mirror_pair_indices=mirror_pair_indices,
                mirror_pair_ratios=mirror_pair_ratios,
                mirror_pair_names=mirror_pair_names,
                ac_pred_filter=getattr(args, 'ac_pred_filter', False),
            )

            val_losses.append(val_loss)
            val_maes.append(val_mae_mv)
            val_voltage_losses.append(val_v_loss)
            val_current_losses.append(val_c_loss)
            val_ss_gm_losses.append(val_ss_gm_loss)
            val_ss_gds_losses.append(val_ss_gds_loss)
            val_kcl_losses.append(val_kcl_loss)
            val_ac_losses.append(val_ac_loss)
            for comp, comp_val in val_ac_comp.items():
                val_ac_component_losses.setdefault(comp, []).append(comp_val)
            val_dc_gain_losses.append(val_dc_gain_loss)
            val_region_losses.append(val_region_loss)
            val_gm_physics_losses.append(val_gm_physics_loss)
            val_triode_physics_losses.append(val_triode_physics_loss)
            val_cutoff_physics_losses.append(val_cutoff_physics_loss)
            val_current_maes.append(val_c_mae)
            val_hc_mirror_losses.append(val_hc_mirror_loss)

            # Schedule plateau on the SUPERVISED components only (V+I+SS+DC_gain),
            # excluding KCL and other physics regularizers. These can ramp in mid-
            # training (e.g. kcl_warmup) and would otherwise spuriously trigger LR
            # reductions just because the loss landscape changed, not because the
            # model regressed on what we actually care about.
            v_w = getattr(args, 'voltage_weight', 1.0)
            val_loss_sched = (
                v_w * val_v_loss
                + current_weight * val_c_loss
                + ss_gm_loss_weight * val_ss_gm_loss
                + ss_gds_loss_weight * val_ss_gds_loss
                + dc_gain_loss_weight * val_dc_gain_loss
            )
            if epoch >= args.warmup:
                step_scheduler(scheduler, args.scheduler, val_loss_sched)

            # Compute composite score: V@5% + I@5% + gm@10% + gds@10%
            composite_score = 0.0
            use_composite = False
            if val_rel_metrics:
                vr = val_rel_metrics.get('v_rel_acc', {})
                ir = val_rel_metrics.get('i_rel_acc', {})
                ss = val_rel_metrics.get('ss_metrics') or {}
                gm_acc = ss.get('gm_acc', {}).get(10, 0)
                gds_acc = ss.get('gds_acc', {}).get(10, 0)
                gm_id_m = val_rel_metrics.get('gm_id_metrics')
                gm_id_acc = gm_id_m['acc'][10] if gm_id_m else 0
                composite_score = vr.get(5, 0) + ir.get(5, 0) + gm_acc + gds_acc + gm_id_acc
                if dc_gain_loss_weight > 0:
                    composite_score += val_rel_metrics.get('dc_gain_acc_3dB', 0)
                if ac_loss_weight > 0:
                    ugbw_acc = val_rel_metrics.get('ugbw_acc', {})
                    composite_score += ugbw_acc.get(5, 0)
                use_composite = composite_score > 0

            if use_composite:
                improved = composite_score > best_composite_score + 0.1
            else:
                # Use supervised-only loss for best-tracking too, same reason as scheduler.
                improved = val_loss_sched < best_val_loss - 1e-4

            if improved:
                best_composite_score = max(best_composite_score, composite_score)
                best_val_loss, best_epoch = val_loss_sched, epoch
                best_model_state = model.state_dict()
                epochs_without_improvement = 0
                best_metrics.update(val_mae_mv=val_mae_mv, val_current_mae_ua=val_c_mae, acc80=acc80, acc50=acc50, acc20=acc20, acc10=acc10,
                                     current_acc50=current_acc50, current_acc20=current_acc20, current_acc10=current_acc10, current_acc5=current_acc5,
                                     rel_metrics=val_rel_metrics)
            else:
                epochs_without_improvement += 1

            lr = optimizer.param_groups[0]['lr']
            postfix = {'loss': f'{loss:.4f}', 'v_mae': f'{mae_mv:.1f}mV', 'lr': f'{lr:.2e}', 'score': f'{composite_score:.1f}'}
            if predict_currents:
                postfix['i_mae'] = f'{current_mae_ua:.1f}µA'
            if predict_currents:
                postfix['kcl'] = f'{kcl_loss:.2e}'
            if constraint_weight > 0:
                postfix['dp'] = f'{diff_pair_loss:.2e}'
                postfix['mir'] = f'{mirror_loss:.2e}'
                postfix['os'] = f'{output_stage_loss:.2e}'
                postfix['lm'] = f'{lambda_mirror_loss:.2e}'
            if gm_physics_loss_weight > 0:
                postfix['gm_phy'] = f'{gm_physics_loss:.2e}'
            if ac_loss_weight > 0:
                postfix['ac'] = f'{ac_loss:.2e}'
            if ss_gm_loss_weight > 0:
                postfix['gm'] = f'{ss_gm_loss:.2e}'
            if ss_gds_loss_weight > 0:
                postfix['gds'] = f'{ss_gds_loss:.2e}'
            if triode_physics_loss_weight > 0:
                postfix['tri_phy'] = f'{triode_physics_loss:.2e}'
            if cutoff_physics_loss_weight > 0:
                postfix['cut_phy'] = f'{cutoff_physics_loss:.2e}'
            pbar.set_postfix(postfix)

            # Detailed progress every 50 epochs
            detail_interval = 50
            if epoch > 0 and epoch % detail_interval == 0:
                diagnostics.print_detail_report(
                    model, train_loader, val_loader, args, epoch, lr,
                    cfg={
                        'predict_currents': predict_currents,
                        'current_weight': current_weight,
                        'kcl_weight': kcl_weight,
                        'loss_type': loss_type,
                        'huber_delta': huber_delta,
                        'constraint_weight': constraint_weight,
                        'stage2_nodes': stage2_nodes,
                        'stage2_weight': stage2_weight,
                        'node_weights': node_weights,
                        'use_terminal_voltage_loss': use_terminal_voltage_loss,
                        'gm_physics_loss_weight': gm_physics_loss_weight,
                        'gm_physics_min_vov': gm_physics_min_vov,
                        'gm_physics_use_clm': gm_physics_use_clm,
                        'gm_physics_use_smaxt': gm_physics_use_smaxt,
                        'has_ss': has_ss,
                        'ss_gm_mean': ss_gm_mean, 'ss_gm_std': ss_gm_std,
                        'ss_gds_mean': ss_gds_mean, 'ss_gds_std': ss_gds_std,
                        'ac_loss_weight': ac_loss_weight,
                        'ac_mean': ac_mean, 'ac_std': ac_std, 'ac_components': ac_components,
                        'ss_gm_loss_weight': ss_gm_loss_weight,
                        'ss_gds_loss_weight': ss_gds_loss_weight,
                        'ss_region_stats': ss_region_stats,
                        'triode_physics_loss_weight': triode_physics_loss_weight,
                        'triode_physics_config': triode_physics_config,
                        'cutoff_physics_loss_weight': cutoff_physics_loss_weight,
                        'cutoff_physics_n_nmos': cutoff_physics_n_nmos,
                        'cutoff_physics_n_pmos': cutoff_physics_n_pmos,
                        'region_loss_weight': region_loss_weight,
                        'amp_dtype': amp_dtype,
                        'device_consistency_weight': device_consistency_weight,
                        'vov_loss_weight': vov_loss_weight,
                        'vth_loss_weight': vth_loss_weight,
                        'dc_gain_loss_weight': dc_gain_loss_weight,
                        'dc_gain_mean': dc_gain_mean, 'dc_gain_std': dc_gain_std,
                        'gm_id_mean': gm_id_mean, 'gm_id_std': gm_id_std,
                        'mirror_pair_indices': mirror_pair_indices,
                        'mirror_pair_ratios': mirror_pair_ratios,
                        'mirror_pair_names': mirror_pair_names,
                        'vdc_mean': vdc_mean, 'vdc_std': vdc_std,
                        'current_mean': current_mean, 'current_std': current_std,
                        'kcl_mode': kcl_mode,
                        'kcl_mask_unsupervised': kcl_mask_unsupervised,
                    },
                    metrics={
                        'best_val_loss': best_val_loss,
                        'best_epoch': best_epoch,
                        'composite_score': composite_score,
                        'best_composite_score': best_composite_score,
                        'max_grad_norm': max_grad_norm,
                        'avg_grad_norm': avg_grad_norm,
                        'train_gm_physics_loss': gm_physics_loss,
                        'train_vth_loss': train_vth_loss,
                        'val_loss': val_loss,
                        'val_mae_mv': val_mae_mv,
                        'val_v_loss': val_v_loss,
                        'val_c_loss': val_c_loss,
                        'val_c_mae': val_c_mae,
                        'acc80': acc80, 'acc50': acc50, 'acc20': acc20, 'acc10': acc10,
                        'current_acc50': current_acc50, 'current_acc20': current_acc20,
                        'current_acc10': current_acc10, 'current_acc5': current_acc5,
                        'val_kcl_loss': val_kcl_loss,
                        'val_dp_loss': val_dp_loss,
                        'val_mirror_loss': val_mirror_loss,
                        'val_os_loss': val_os_loss,
                        'val_lm_loss': val_lm_loss,
                        'val_gm_physics_loss': val_gm_physics_loss,
                        'val_ac_loss': val_ac_loss,
                        'val_ss_gm_loss': val_ss_gm_loss,
                        'val_ss_gds_loss': val_ss_gds_loss,
                        'val_triode_physics_loss': val_triode_physics_loss,
                        'val_triode_eq1_loss': val_triode_eq1_loss,
                        'val_triode_eq2_loss': val_triode_eq2_loss,
                        'val_triode_eq3_loss': val_triode_eq3_loss,
                        'val_cutoff_physics_loss': val_cutoff_physics_loss,
                        'val_region_loss': val_region_loss,
                        'val_rel_metrics': val_rel_metrics,
                        'val_vth_loss': val_vth_loss,
                        'val_ac_comp': val_ac_comp,
                        'val_dc_gain_loss': val_dc_gain_loss,
                        'val_hc_mirror_loss': val_hc_mirror_loss,
                        'val_mirror_pair_detail': val_mirror_pair_detail,
                    },
                    uncertainty_weights=uncertainty_weights,
                )

            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Save best model with full config
    if best_model_state is not None:
        finalize.save_best_model(
            args, output_path, dataset_path, best_model_state, best_epoch,
            best_val_loss, total_input_dim, predict_currents,
            voltage_head_config, current_head_config, current_weight_target,
            loss_type, huber_delta, val_freq, early_stopping_patience,
            derive_currents_from_voltage,
            vdc_mean, vdc_std, current_mean, current_std,
            has_ss, ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std,
            dc_gain_mean, dc_gain_std, dc_gain_loss_weight_target,
            ac_mean, ac_std, ac_components, ac_loss_weight_target,
        )

    # Print summary
    finalize.print_training_summary(
        best_metrics, best_val_loss, best_epoch, training_start_time, predict_currents)

    # End-of-training AC and region evaluation on best model
    if best_model_state is not None and val_loader:
        diagnostics.print_final_evaluation(
            model, val_loader, args, best_model_state,
            ac_loss_weight_target, ac_mean, ac_std, ac_components,
            dc_gain_loss_weight_target, dc_gain_mean, dc_gain_std,
            region_loss_weight_target,
        )

    # Plot training curves
    finalize.plot_curves(
        output_path,
        {
            'train_losses': train_losses, 'val_losses': val_losses,
            'train_voltage_losses': train_voltage_losses, 'val_voltage_losses': val_voltage_losses,
            'train_maes': train_maes, 'val_maes': val_maes,
            'learning_rates': learning_rates,
            'train_current_losses': train_current_losses, 'val_current_losses': val_current_losses,
            'train_current_maes': train_current_maes, 'val_current_maes': val_current_maes,
            'train_ss_gm_losses': train_ss_gm_losses, 'val_ss_gm_losses': val_ss_gm_losses,
            'train_ss_gds_losses': train_ss_gds_losses, 'val_ss_gds_losses': val_ss_gds_losses,
            'train_kcl_losses': train_kcl_losses, 'val_kcl_losses': val_kcl_losses,
            'train_ac_losses': train_ac_losses, 'val_ac_losses': val_ac_losses,
            'train_ac_component_losses': train_ac_component_losses,
            'val_ac_component_losses': val_ac_component_losses,
            'train_region_losses': train_region_losses, 'val_region_losses': val_region_losses,
            'train_gm_physics_losses': train_gm_physics_losses, 'val_gm_physics_losses': val_gm_physics_losses,
            'train_triode_physics_losses': train_triode_physics_losses,
            'val_triode_physics_losses': val_triode_physics_losses,
            'train_cutoff_physics_losses': train_cutoff_physics_losses,
            'val_cutoff_physics_losses': val_cutoff_physics_losses,
            'train_dc_gain_losses': train_dc_gain_losses, 'val_dc_gain_losses': val_dc_gain_losses,
            'max_grad_norms': max_grad_norms, 'avg_grad_norms': avg_grad_norms,
            'train_hc_mirror_losses': train_hc_mirror_losses,
            'val_hc_mirror_losses': val_hc_mirror_losses,
        },
        best_metrics, best_val_loss, best_epoch,
        val_freq=val_freq, predict_currents=predict_currents,
        triode_physics_loss_weight_target=triode_physics_loss_weight_target,
        cutoff_physics_loss_weight_target=cutoff_physics_loss_weight_target,
        dc_gain_loss_weight=dc_gain_loss_weight,
        constraint_weight_target=constraint_weight_target,
        num_train_batches=len(train_loader),
        num_val_batches=len(val_loader) if val_loader else 0,
    )


if __name__ == '__main__':
    main()
