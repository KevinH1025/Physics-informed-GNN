"""Fine-tune / Phase-2 checkpoint surgery for train_v3.

Holds the architecture-inheritance, weight-loading, head-reset, BatchNorm
reset and backbone freeze/unfreeze logic, plus the two backbone prefix
tuples they key on.

IMPORTANT: the two prefix tuples below are intentionally different. They were
two divergent hardcoded copies in train_v3 and both are load-bearing for
existing checkpoints and SLURM workflows:

- RESET_HEADS_BACKBONE_PREFIXES (used only by --reset-heads) includes the
  TowerGENConv-era names 'backbone.', 'backbone_jk.' and 'loop_attn_layers.'.
- FINETUNE_BACKBONE_PREFIXES (used by --freeze-backbone, discriminative LR
  and the --freeze-backbone-epochs unfreeze) does NOT include those names,
  so on TowerGENConv --freeze-backbone does not freeze the conv stack or
  loop attention. This is a known bug; fixing it is a separate,
  behavior-changing commit. Do not unify the tuples here.
"""

import torch

# Backbone prefixes used when --reset-heads drops head weights before loading
# a fine-tune checkpoint. The backbone is everything before the heads (input
# proj, GNN layers, JK, virtual node, loop attention).
RESET_HEADS_BACKBONE_PREFIXES = (
    'input_linear.', 'input_proj.',
    'backbone.', 'layers.',
    'backbone_jk.', 'jk_linear.', 'jk_attn.',
    'virtual_node.', 'vn_',
    'loop_attn_layers.', 'norms.',
)

# Backbone parameter names: GNN layers, JK, virtual node, input projection.
# Used by --freeze-backbone, discriminative LR and progressive unfreeze.
FINETUNE_BACKBONE_PREFIXES = (
    'layers.', 'jk_linear.', 'jk_attn.', 'input_proj.', 'input_linear.',
    'vn_', 'virtual_node', 'norms.',
)


def inherit_architecture_from_checkpoint(args):
    """Fine-tune: override architecture args from checkpoint config so model matches exactly."""
    if not args.checkpoint:
        raise ValueError("--finetune requires --checkpoint path to pre-trained model")
    print(f"\n=== FINE-TUNING: Loading architecture from checkpoint ===")
    _ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    _src_cfg = _ckpt.get('config', {})
    # Architecture args to inherit from checkpoint
    _arch_map = {
        'hidden': ('hidden', 'hidden_dim'),
        'layers': ('layers', 'num_layers'),
        'dropout': ('dropout',),
        'genconv_num_layers': ('genconv_num_layers',),
        'num_mlp_layers': ('num_mlp_layers',),
        'jk_mode': ('jk_mode',),
        'jk_attention': ('jk_attention',),
        'jk_learn_temperature': ('jk_learn_temperature',),
        'norm_type': ('norm_type',),
        'skip_connection': ('skip_connection',),
        'virtual_node': ('virtual_node', 'use_virtual_node'),
        'use_attention_pooling': ('use_attention_pooling',),
        'vn_learn_temperature': ('vn_learn_temperature',),
    }
    # Dict configs to inherit as-is
    _dict_configs = [
        'voltage_head_config', 'current_head_config', 'ss_head_config',
        'ac_head_config', 'region_head_config', 'mosfet_current_mlp_config',
        'current_gnn_config', 'frozen_device_mlp_config', 'refinement_config',
        'vov_head_config',
    ]
    overridden = []
    for arg_name, cfg_keys in _arch_map.items():
        for ck in cfg_keys:
            if ck in _src_cfg:
                old_val = getattr(args, arg_name, None)
                setattr(args, arg_name, _src_cfg[ck])
                if old_val != _src_cfg[ck]:
                    overridden.append(f"  {arg_name}: {old_val} -> {_src_cfg[ck]}")
                break
    for dc in _dict_configs:
        if dc in _src_cfg:
            old_val = getattr(args, dc, {})
            setattr(args, dc, _src_cfg[dc])
            if old_val != _src_cfg[dc]:
                overridden.append(f"  {dc}: {old_val} -> {_src_cfg[dc]}")
    # Bool configs
    for bc in ['predict_currents', 'derive_currents_from_voltage', 'use_gnn_current_prediction',
                'use_frozen_device_mlp', 'use_refinement_pass', 'gradient_checkpointing']:
        if bc in _src_cfg:
            old_val = getattr(args, bc, False)
            setattr(args, bc, _src_cfg[bc])
            if old_val != _src_cfg[bc]:
                overridden.append(f"  {bc}: {old_val} -> {_src_cfg[bc]}")
    # Disable gradient checkpointing when backbone is frozen (no backbone grads needed)
    if getattr(args, 'freeze_backbone', False) and getattr(args, 'gradient_checkpointing', False):
        args.gradient_checkpointing = False
        overridden.append(f"  gradient_checkpointing: True -> False (frozen backbone, not needed)")
    if overridden:
        print(f"Overrode {len(overridden)} args from checkpoint:")
        for o in overridden:
            print(o)
    else:
        print("All architecture args already match checkpoint.")
    del _ckpt  # free memory, will reload later for weights


def apply_phase2(model, args):
    """Phase 2: load Phase 1 weights, freeze everything but the current MLP."""
    if not args.checkpoint:
        raise ValueError("--phase2 requires --checkpoint path to Phase 1 model")

    print(f"\n=== PHASE 2: Current MLP Fine-tuning ===")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    # Load only backbone/voltage head weights (skip MLP to use fresh config)
    state_dict = ckpt['model_state_dict']
    mlp_keys = [k for k in state_dict.keys() if 'mosfet_current_mlp' in k]
    for k in mlp_keys:
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded backbone from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
    print(f"Fresh MLP with config: {args.mosfet_current_mlp_config}")

    # Freeze everything except mosfet_current_mlp
    for name, param in model.named_parameters():
        if 'mosfet_current_mlp' not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Frozen: {total - trainable:,} params, Trainable: {trainable:,} params")

    # Set model to Phase 2 mode (don't detach voltages)
    model.phase2_mode = True

    # Ensure current training is enabled
    args.predict_currents = True
    if getattr(args, 'current_weight', 0.0) == 0.0:
        args.current_weight = 1.0
        print(f"Setting current_weight=1.0 for Phase 2")


def load_finetune_weights(model, args):
    """Fine-tune: load checkpoint weights, optionally reset heads/BN, optionally freeze backbone."""
    print(f"\n=== FINE-TUNING: Loading weights ===")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = ckpt['model_state_dict']
    source_config = ckpt.get('config', {})

    # Filter out keys with shape mismatches (e.g., different head MLP depths)
    model_sd = model.state_dict()
    skipped = []
    for k in list(state_dict.keys()):
        if k in model_sd and state_dict[k].shape != model_sd[k].shape:
            skipped.append(f"{k}: ckpt {list(state_dict[k].shape)} vs model {list(model_sd[k].shape)}")
            del state_dict[k]
    if skipped:
        print(f"  Skipped {len(skipped)} shape-mismatched keys (will use random init):")
        for s in skipped:
            print(f"    {s}")

    # If --reset-heads, drop all non-backbone keys before loading. The
    # backbone is everything before the heads (input proj, GNN layers, JK,
    # virtual node, loop attention). Anything else (voltage_head, current
    # head, gm/gds heads, dc gain head, etc.) gets to keep its fresh init.
    if getattr(args, 'reset_heads', False):
        kept_keys = [k for k in state_dict.keys() if any(k.startswith(p) for p in RESET_HEADS_BACKBONE_PREFIXES)]
        dropped_keys = [k for k in state_dict.keys() if k not in kept_keys]
        state_dict = {k: state_dict[k] for k in kept_keys}
        print(f"  --reset-heads: kept {len(kept_keys)} backbone keys, dropped {len(dropped_keys)} head keys")
        if dropped_keys[:3]:
            print(f"    dropped (first 3): {dropped_keys[:3]}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys (randomly initialized): {len(missing)} keys")
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected}")
    _val_loss_disp = ckpt.get('val_loss', None)
    if _val_loss_disp is None:
        _val_loss_disp = ckpt.get('val_v', '?')
    if isinstance(_val_loss_disp, float):
        _val_loss_disp = f'{_val_loss_disp:.4f}'
    print(f"Loaded pre-trained weights from {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, val_loss {_val_loss_disp})")

    # ── Reset BatchNorm running statistics. The pretrain corpus has a
    # different activation distribution (different graph topologies → different
    # aggregation magnitudes), so the stored running_mean/var are wrong for
    # the new dataset. Keep the learned weight/bias (gamma/beta) — those
    # adapt fast through gradients — but reset the buffers so BN re-learns
    # the right statistics from the new data.
    if not getattr(args, 'no_bn_reset', False):
        n_bn = 0
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                if m.running_mean is not None:
                    m.running_mean.zero_()
                if m.running_var is not None:
                    m.running_var.fill_(1.0)
                if m.num_batches_tracked is not None:
                    m.num_batches_tracked.zero_()
                n_bn += 1
        print(f"  Reset BatchNorm running statistics in {n_bn} BN layers (gamma/beta preserved)")

    # Determine if we should freeze the backbone now: either fully (--freeze-backbone)
    # or just for the head-warmup window (--freeze-backbone-epochs N).
    freeze_for_warmup = getattr(args, 'freeze_backbone_epochs', 0) > 0
    if getattr(args, 'freeze_backbone', False) or freeze_for_warmup:
        frozen_count = 0
        trainable_count = 0
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in FINETUNE_BACKBONE_PREFIXES):
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                trainable_count += param.numel()
        stage_label = ('FROZEN for entire run' if getattr(args, 'freeze_backbone', False)
                       else f'FROZEN for first {args.freeze_backbone_epochs} epochs (head warmup)')
        print(f"Backbone {stage_label}: {frozen_count:,} params")
        print(f"Heads trainable: {trainable_count:,} params")
    else:
        # Phase 2 of fine-tuning: all params trainable with discriminative LR
        total = sum(p.numel() for p in model.parameters())
        print(f"All {total:,} params trainable (backbone LR scale: {args.backbone_lr_scale}x)")

    # Store source checkpoint info for saving
    args._finetune_source = str(args.checkpoint)
    args._finetune_source_epoch = ckpt.get('epoch', -1)


def split_backbone_head_params(model, unfreeze=False):
    """Partition parameters into (backbone, heads) by FINETUNE_BACKBONE_PREFIXES.

    With ``unfreeze=True``, sets ``requires_grad = True`` on every parameter
    while partitioning (used by the --freeze-backbone-epochs progressive
    unfreeze).
    """
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if unfreeze:
            param.requires_grad = True
        if any(name.startswith(p) for p in FINETUNE_BACKBONE_PREFIXES):
            backbone_params.append(param)
        else:
            head_params.append(param)
    return backbone_params, head_params
