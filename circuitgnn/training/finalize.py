"""End-of-run steps for train_v3: best-checkpoint saving, summary, plots.

The printed summary is a frozen surface (the figure scripts regex-parse
'Best Val MAE: ...' and the accuracy tables out of training.log), so nothing
here may be reworded, respaced or rounded differently.

NOTE: the best-checkpoint aliasing bug is deliberately preserved. train_v3
snapshots ``best_model_state = model.state_dict()`` without a deep copy, so the
tensors alias live parameters and the checkpoint written here holds final-epoch
weights. Fixing that is a separate, behavior-changing commit; this module just
writes whatever state dict it is handed.
"""

import time

from circuitgnn.training.checkpoint import save_checkpoint, build_full_config
from circuitgnn.training.plotting import plot_training_curves


def save_best_model(args, output_path, dataset_path, best_model_state, best_epoch,
                    best_val_loss, total_input_dim, predict_currents,
                    voltage_head_config, current_head_config, current_weight_target,
                    loss_type, huber_delta, val_freq, early_stopping_patience,
                    derive_currents_from_voltage,
                    vdc_mean, vdc_std, current_mean, current_std,
                    has_ss, ss_gm_mean, ss_gm_std, ss_gds_mean, ss_gds_std,
                    dc_gain_mean, dc_gain_std, dc_gain_loss_weight_target,
                    ac_mean, ac_std, ac_components, ac_loss_weight_target):
    """Save best_model.pt with the full config and the normalization stats."""
    save_path = output_path / 'best_model.pt'

    full_config = build_full_config(
        args, total_input_dim, predict_currents, voltage_head_config,
        current_head_config, current_weight_target, loss_type, huber_delta,
        val_freq, early_stopping_patience, dataset_path,
        derive_currents_from_voltage, args.mosfet_current_mlp_config,
        getattr(args, 'use_gnn_current_prediction', False),
        getattr(args, 'current_gnn_config', {}),
        getattr(args, 'use_frozen_device_mlp', False),
        getattr(args, 'frozen_device_mlp_config', {}),
        getattr(args, 'use_refinement_pass', False),
        getattr(args, 'refinement_config', {}),
    )
    stats = {
        'vdc': {'mean': vdc_mean, 'std': vdc_std},
        'current': {'mean': current_mean, 'std': current_std},
        'ss_gm': {'mean': ss_gm_mean, 'std': ss_gm_std} if has_ss else None,
        'ss_gds': {'mean': ss_gds_mean, 'std': ss_gds_std} if has_ss else None,
        'dc_gain': {'mean': dc_gain_mean, 'std': dc_gain_std} if dc_gain_loss_weight_target > 0 else None,
        'ac': {'mean': ac_mean.tolist() if ac_mean is not None else None,
               'std': ac_std.tolist() if ac_std is not None else None,
               'components': ac_components} if ac_loss_weight_target > 0 else None,
    }

    # Add finetune provenance info
    if hasattr(args, '_finetune_source'):
        full_config['finetune_source'] = args._finetune_source
        full_config['finetune_source_epoch'] = args._finetune_source_epoch
        full_config['freeze_backbone'] = getattr(args, 'freeze_backbone', False)
        full_config['backbone_lr_scale'] = getattr(args, 'backbone_lr_scale', 0.1)

    save_checkpoint(save_path, best_model_state, best_epoch, best_val_loss, full_config, stats)
    print(f"\nSaved best model (epoch {best_epoch}) to {save_path}")
    print(f"Saved config to {output_path / 'config.yaml'}")


def print_training_summary(best_metrics, best_val_loss, best_epoch,
                           training_start_time, predict_currents):
    """Print the TRAINING COMPLETE block, accuracy tables and SS/gm-Id summaries."""
    training_time = time.time() - training_start_time
    rm = best_metrics.get('rel_metrics') or {}
    ia = rm.get('i_abs_acc', {})
    vr = rm.get('v_rel_acc', {})
    ir = rm.get('i_rel_acc', {})
    ss_m = rm.get('ss_metrics')

    print(f"\n{'='*50}\n=== TRAINING COMPLETE ===\n{'='*50}")
    print(f"Training time: {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")
    if rm:
        print(f"Best Val MAE: {best_metrics['val_mae_mv']:.2f}mV (median rel: {rm['v_rel_median']:.2f}%)")
    else:
        print(f"Best Val MAE: {best_metrics['val_mae_mv']:.2f}mV")
    if predict_currents:
        if rm:
            print(f"Best Val Current MAE: {best_metrics['val_current_mae_ua']:.1f}µA (median rel: {rm['i_rel_median']:.2f}%)")
        else:
            print(f"Best Val Current MAE: {best_metrics['val_current_mae_ua']:.1f}µA")

    if predict_currents and ia:
        print(f"Voltage Abs Acc @80mV: {best_metrics['acc80']:5.2f}% | Current Abs Acc @50uA: {ia[50]:5.2f}%")
        print(f"Voltage Abs Acc @50mV: {best_metrics['acc50']:5.2f}% | Current Abs Acc @20uA: {ia[20]:5.2f}%")
        print(f"Voltage Abs Acc @20mV: {best_metrics['acc20']:5.2f}% | Current Abs Acc  @5uA: {ia[5]:5.2f}%")
        print(f"Voltage Abs Acc @10mV: {best_metrics['acc10']:5.2f}% | Current Abs Acc  @2uA: {ia[2]:5.2f}%")
    else:
        print(f"Voltage Abs Acc @80mV: {best_metrics['acc80']:.2f}%")
        print(f"Voltage Abs Acc @50mV: {best_metrics['acc50']:.2f}%")
        print(f"Voltage Abs Acc @20mV: {best_metrics['acc20']:.2f}%")
        print(f"Voltage Abs Acc @10mV: {best_metrics['acc10']:.2f}%")

    if vr:
        if predict_currents:
            print(f"Voltage Rel Acc  @1%: {vr[1]:5.2f}% | Current Rel Acc  @1%: {ir[1]:5.2f}%")
            print(f"Voltage Rel Acc  @5%: {vr[5]:5.2f}% | Current Rel Acc  @5%: {ir[5]:5.2f}%")
            print(f"Voltage Rel Acc @10%: {vr[10]:5.2f}% | Current Rel Acc @10%: {ir[10]:5.2f}%")
            print(f"Voltage Rel Acc @20%: {vr[20]:5.2f}% | Current Rel Acc @20%: {ir[20]:5.2f}%")
        else:
            print(f"Voltage Rel Acc  @1%: {vr[1]:5.2f}%")
            print(f"Voltage Rel Acc  @5%: {vr[5]:5.2f}%")
            print(f"Voltage Rel Acc @10%: {vr[10]:5.2f}%")
            print(f"Voltage Rel Acc @20%: {vr[20]:5.2f}%")

    if ss_m is not None:
        print(f"\n--- SS Evaluation ---")
        print(f"gm  MAE: {ss_m['gm_log_mae']:.3f} log10  (median {ss_m['gm_log_median']:.3f}, median rel: {ss_m['gm_median']:.1f}%)")
        print(f"gds MAE: {ss_m['gds_log_mae']:.3f} log10  (median {ss_m['gds_log_median']:.3f}, median rel: {ss_m['gds_median']:.1f}%)")
        print(f"gm  Acc @10%: {ss_m['gm_acc'][10]:5.1f}% | gds Acc @10%: {ss_m['gds_acc'][10]:5.1f}%")
        print(f"gm  Acc @20%: {ss_m['gm_acc'][20]:5.1f}% | gds Acc @20%: {ss_m['gds_acc'][20]:5.1f}%")
        print(f"gm  Acc @50%: {ss_m['gm_acc'][50]:5.1f}% | gds Acc @50%: {ss_m['gds_acc'][50]:5.1f}%")

    gm_id_m = rm.get('gm_id_metrics') if rm else None
    if gm_id_m is not None:
        print(f"\n--- gm/Id Evaluation ---")
        print(f"gm/Id MAE: {gm_id_m['log_mae']:.3f} log10  (median {gm_id_m['log_median']:.3f}, median rel: {gm_id_m['median_rel']:.1f}%)")
        print(f"gm/Id Acc @10%: {gm_id_m['acc'][10]:5.1f}%")
        print(f"gm/Id Acc @20%: {gm_id_m['acc'][20]:5.1f}%")
        print(f"gm/Id Acc @50%: {gm_id_m['acc'][50]:5.1f}%")

    if ss_m is not None:
        # MoE diagnostics
        if 'moe_region_acc' in ss_m:
            print(f"\n--- MoE Diagnostics ---")
            print(f"Region head accuracy: {ss_m['moe_region_acc']:.1%}")
            if ss_m.get('moe_avg_probs') is not None:
                p = ss_m['moe_avg_probs']
                print(f"Avg routing probs: cut={p[0]:.3f}  tri={p[1]:.3f}  sat={p[2]:.3f}")
            for name in ['cutoff', 'triode', 'saturation']:
                gm_k = f'moe_gm_mae_{name}'
                gds_k = f'moe_gds_mae_{name}'
                if gm_k in ss_m:
                    print(f"  {name:12s}  gm MAE={ss_m[gm_k]:.4f}  gds MAE={ss_m[gds_k]:.4f}")


def plot_curves(output_path, history, best_metrics, best_val_loss, best_epoch,
                val_freq, predict_currents, triode_physics_loss_weight_target,
                cutoff_physics_loss_weight_target, dc_gain_loss_weight,
                constraint_weight_target, num_train_batches, num_val_batches):
    """Write training_curve.png from the collected per-epoch history."""
    h = history
    plot_training_curves(
        h['train_losses'], h['val_losses'], h['train_voltage_losses'], h['val_voltage_losses'],
        h['train_maes'], h['val_maes'], h['learning_rates'], best_metrics, best_val_loss, best_epoch,
        output_path / 'training_curve.png', val_freq=val_freq, predict_currents=predict_currents,
        train_current_losses=h['train_current_losses'], val_current_losses=h['val_current_losses'],
        train_current_maes=h['train_current_maes'], val_current_maes=h['val_current_maes'],
        train_ss_gm_losses=h['train_ss_gm_losses'], val_ss_gm_losses=h['val_ss_gm_losses'],
        train_ss_gds_losses=h['train_ss_gds_losses'], val_ss_gds_losses=h['val_ss_gds_losses'],
        train_kcl_losses=h['train_kcl_losses'], val_kcl_losses=h['val_kcl_losses'],
        train_ac_losses=h['train_ac_losses'], val_ac_losses=h['val_ac_losses'],
        train_ac_component_losses=h['train_ac_component_losses'], val_ac_component_losses=h['val_ac_component_losses'],
        train_region_losses=h['train_region_losses'], val_region_losses=h['val_region_losses'],
        train_gm_physics_losses=h['train_gm_physics_losses'], val_gm_physics_losses=h['val_gm_physics_losses'],
        train_triode_physics_losses=h['train_triode_physics_losses'] if triode_physics_loss_weight_target > 0 else None,
        val_triode_physics_losses=h['val_triode_physics_losses'] if triode_physics_loss_weight_target > 0 else None,
        train_cutoff_physics_losses=h['train_cutoff_physics_losses'] if cutoff_physics_loss_weight_target > 0 else None,
        val_cutoff_physics_losses=h['val_cutoff_physics_losses'] if cutoff_physics_loss_weight_target > 0 else None,
        train_dc_gain_losses=h['train_dc_gain_losses'] if dc_gain_loss_weight > 0 else None,
        val_dc_gain_losses=h['val_dc_gain_losses'] if dc_gain_loss_weight > 0 else None,
        max_grad_norms=h['max_grad_norms'], avg_grad_norms=h['avg_grad_norms'],
        num_train_batches=num_train_batches, num_val_batches=num_val_batches,
        train_mirror_losses=h['train_hc_mirror_losses'] if constraint_weight_target > 0 else None,
        val_mirror_losses=h['val_hc_mirror_losses'] if constraint_weight_target > 0 else None,
    )
