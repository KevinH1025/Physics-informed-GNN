"""
Training utilities for GNN model.

This module provides utilities for:
- Data loading and batch iteration
- Loss computation
- Metrics calculation
- Learning rate scheduling
"""

from circuitgnn.training.data_loading import (
    PrebatchedLoader,
    load_prebatched_variant,
    load_prebatched_metadata,
    compute_vdc_normalization,
    compute_current_normalization,
    normalize_batches_vdc,
    normalize_batches_current,
    attach_normalization_stats,
)
from circuitgnn.training.losses import (
    compute_loss,
    compute_combined_loss,
)
from circuitgnn.training.metrics import (
    compute_voltage_accuracy,
    compute_current_accuracy,
    denormalize_voltage,
    denormalize_current,
)
from circuitgnn.training.scheduler import (
    create_scheduler,
    apply_warmup,
    step_scheduler,
)
from circuitgnn.training.loops import (
    train_epoch,
    validate,
    validate_simple,
)
from circuitgnn.training.plotting import plot_training_curves
from circuitgnn.training.checkpoint import (
    create_model,
    create_model_from_args,
    save_checkpoint,
    load_checkpoint,
    build_full_config,
    build_config_yaml,
)

__all__ = [
    # Data loading
    'PrebatchedLoader',
    'load_prebatched_variant',
    'load_prebatched_metadata',
    'compute_vdc_normalization',
    'compute_current_normalization',
    'normalize_batches_vdc',
    'normalize_batches_current',
    'attach_normalization_stats',
    # Losses
    'compute_loss',
    'compute_combined_loss',
    # Metrics
    'compute_voltage_accuracy',
    'compute_current_accuracy',
    'denormalize_voltage',
    'denormalize_current',
    # Scheduler
    'create_scheduler',
    'apply_warmup',
    'step_scheduler',
    # Loops
    'train_epoch',
    'validate',
    'validate_simple',
    # Plotting
    'plot_training_curves',
    # Checkpoint
    'create_model',
    'create_model_from_args',
    'save_checkpoint',
    'load_checkpoint',
    'build_full_config',
    'build_config_yaml',
]
