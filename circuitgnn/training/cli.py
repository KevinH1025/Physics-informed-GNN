"""CLI argument parsing and YAML config merging for train_v3.

The merge semantics are load-bearing: CLI flags that were explicitly typed on
the command line override YAML config values, and explicitness is detected by
scanning ``sys.argv`` for the option strings of each argparse action. This is
fragile (it does not recognize the ``--opt=value`` form or abbreviated flags)
but 550+ historical configs and the SLURM launchers depend on its exact
behavior, so it is preserved verbatim. Do not "fix" it here.
"""

import sys
import argparse

import torch

from circuitgnn.training.config import load_config, parse_training_config


def build_parser():
    """Build the train_v3 argument parser (flags and defaults unchanged)."""
    parser = argparse.ArgumentParser(description='Train GNN model')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--dataset', type=str, default='datasets/opamp_v3', help='Dataset path')
    parser.add_argument('--epochs', type=int, default=700, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--layers', type=int, default=15, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--jk-mode', type=str, default='cat', help='Jumping knowledge mode')
    parser.add_argument('--jk-attention', action='store_true', help='Use attention for JK')
    parser.add_argument('--num-mlp-layers', type=int, default=3, help='MLP layers in head')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'poly', 'none'])
    parser.add_argument('--plateau-patience', type=int, default=10, help='Epochs to wait before reducing LR (plateau scheduler)')
    parser.add_argument('--plateau-factor', type=float, default=0.5, help='Factor to multiply LR on plateau (e.g., 0.5 = halve)')
    parser.add_argument('--warmup', type=int, default=0, help='Warmup epochs')
    parser.add_argument('--gradient-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--virtual-node', action='store_true', help='Use virtual node')
    parser.add_argument('--model-type', type=str, default=None,
                        help='Model type from registry (e.g., deepgen, deepgen_vn). Overrides --virtual-node if set.')
    parser.add_argument('--predict-currents', action='store_true', help='Enable current prediction')
    parser.add_argument('--current-weight', type=float, default=1.0, help='Weight for current loss')
    parser.add_argument('--derive-currents-from-voltage', action='store_true',
                        help='Derive MOSFET currents from predicted voltages instead of independent prediction')
    parser.add_argument('--mosfet-current-mlp-hidden', type=int, default=64,
                        help='Hidden dimension for MOSFET current MLP')
    parser.add_argument('--mosfet-current-mlp-layers', type=int, default=2,
                        help='Number of layers in MOSFET current MLP')
    parser.add_argument('--phase2', action='store_true',
                        help='Phase 2 training: freeze backbone/voltage, train current MLP only')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for Phase 2 or fine-tuning')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune from checkpoint on new dataset (cross-topology transfer)')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze GNN backbone during fine-tuning (train heads only)')
    parser.add_argument('--freeze-backbone-epochs', type=int, default=0,
                        help='Freeze backbone for the first N epochs of fine-tuning, '
                             'then unfreeze and continue with all params trainable. '
                             '0 disables this stage (default).')
    parser.add_argument('--no-bn-reset', action='store_true',
                        help='At fine-tune, do NOT reset BatchNorm running stats. '
                             'Default behavior is to reset because pretrain BN stats '
                             'reflect a different data distribution and degrade transfer.')
    parser.add_argument('--reset-heads', action='store_true',
                        help='At fine-tune, load only the backbone weights from the '
                             'checkpoint and randomly re-initialize the heads. Useful '
                             'when pretrain heads were tuned for a different distribution '
                             'and may bias predictions on the new task.')
    parser.add_argument('--backbone-lr-scale', type=float, default=0.1,
                        help='LR scale factor for backbone params when not frozen (default: 0.1 = 10x lower)')
    parser.add_argument('--preload-to-gpu', action='store_true',
                        help='Pre-load all batches to GPU (faster training, uses more VRAM)')
    parser.add_argument('--max-train-samples', type=int, default=None,
                        help='Subsample train set to first N graphs (for few-shot scaling experiments). '
                             'Val set unchanged.')
    parser.add_argument('--ss-gm-loss-weight', type=float, default=0.0, dest='ss_gm_loss_weight',
                        help='Weight for supervised gm loss (0 = disabled)')
    parser.add_argument('--ss-gds-loss-weight', type=float, default=0.0, dest='ss_gds_loss_weight',
                        help='Weight for supervised gds loss (0 = disabled)')
    parser.add_argument('--ac-loss-weight', type=float, default=0.0, dest='ac_loss_weight',
                        help='Weight for AC loss (UGBW, PM, AM) (0 = disabled)')
    parser.add_argument('--kcl-weight', type=float, default=0.0,
                        help='Weight for KCL physics loss (0 = disabled)')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: disable deterministic algorithms, enable cudnn.benchmark')
    parser.add_argument('--deterministic', action='store_true',
                        help='Full deterministic mode: forces all ops (scatter, atomics) to be deterministic. ~3x slower.')
    parser.add_argument('--no-amp', action='store_true', dest='no_amp',
                        help='Disable AMP (use FP32 instead of BF16)')
    parser.add_argument('--amp', action='store_true',
                        help='Force enable AMP (BF16)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for model optimization (PyTorch 2.0+)')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (saves outputs to experiments/<name>/)')
    parser.add_argument('--all-variants-per-epoch', action='store_true', dest='all_variants_per_epoch',
                        help='Train on all variants per epoch instead of rotating (10x more steps/epoch)')
    return parser


def parse_args_with_config():
    """Parse CLI args and merge YAML config, CLI-explicit flags winning.

    Reads ``sys.argv`` (both via ``parse_args()`` and via the explicitness
    scan), exactly like the original inline code in train_v3's main().
    """
    parser = build_parser()
    args = parser.parse_args()

    # Track which CLI args were explicitly provided (for overriding config)
    cli_explicit = {action.dest for action in parser._actions
                    if action.dest in vars(args) and
                    any(opt in sys.argv for opt in action.option_strings)}

    # Convert model-type to model_type for consistency
    args.model_type = getattr(args, 'model_type', None)

    # Load config from YAML if provided
    if args.config:
        config = load_config(args.config)
        parsed = parse_training_config(config)

        for key, value in parsed.items():
            if value is not None:
                setattr(args, key, value)

        # CLI-explicit args override config values
        cli_args = parser.parse_args()
        for key in cli_explicit:
            setattr(args, key, getattr(cli_args, key))

        print(f"Loaded config from {args.config}")

    return args
