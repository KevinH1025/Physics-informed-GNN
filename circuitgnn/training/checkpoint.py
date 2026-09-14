"""
Checkpoint save/load utilities for GNN models.

Handles saving complete configs and loading models with architecture auto-detection.

The full model-parameter set is defined ONCE in the MODEL_KWARGS table below.
create_model, create_model_from_args, load_checkpoint and build_full_config all
derive their kwarg enumerations from that table, so the four cannot drift apart.
The historical gaps between the four enumerations (kwargs that were never saved
by build_full_config or never read back by load_checkpoint) are preserved as-is
and encoded explicitly per entry via ``ckpt=None`` / ``save=None``.
"""

import inspect
from pathlib import Path
import torch
import yaml

from circuitgnn.gnn import get_model, list_models


def _from_config(keys, default):
    """
    Checkpoint-config resolver: chained flat-config lookup.

    _from_config(('a', 'b'), d) resolves exactly like the original
    ``config.get('a', config.get('b', d))``. Dict defaults are copied per call
    so resolved configs never share one mutable empty dict.
    """
    def resolve(config, state_dict):
        value = dict(default) if isinstance(default, dict) else default
        for key in reversed(keys):
            value = config.get(key, value)
        return value
    return resolve


def _kw(name, default, is_config=False, args_attr=None, args_required=False,
        ckpt=None, save=None):
    """
    Build one MODEL_KWARGS entry.

    Args:
        name: create_model kwarg name (also the model kwarg passed to get_model)
        default: create_model signature default
        is_config: config-dict kwarg; create_model applies ``value or {}`` and
            create_model_from_args / build_full_config fall back to ``{}``
        args_attr: attribute read off the flat args namespace (default: name).
            Differs only for hidden_dim -> hidden, num_layers -> layers,
            use_virtual_node -> virtual_node.
        args_required: read via plain attribute access (AttributeError if the
            args namespace lacks it), matching the original direct ``args.x``
        ckpt: how load_checkpoint resolves this kwarg from a saved checkpoint:
            None       -> never read back; the table default is used (this is
                          the historical load_checkpoint lossiness, kept as-is)
            True       -> config.get(name, default) with the standard default
            tuple      -> chained config.get over the given flat-key aliases
            callable   -> resolver(config, state_dict), used for the
                          state-dict-substring auto-detection quirks
        save: how build_full_config persists this kwarg into the flat config:
            None             -> never persisted (historical gap, kept as-is)
            'args'           -> read off args (same access as
                                create_model_from_args) under key args_attr
            'param'          -> taken verbatim from build_full_config's own
                                parameter of the same name
            'param_or_empty' -> same, with ``or {}`` applied
    """
    if ckpt is True:
        ckpt = _from_config((name,), {} if is_config else default)
    elif isinstance(ckpt, tuple):
        ckpt = _from_config(ckpt, {} if is_config else default)
    return {
        'name': name,
        'default': default,
        'is_config': is_config,
        'args_attr': args_attr if args_attr is not None else name,
        'args_required': args_required,
        'ckpt': ckpt,
        'save': save,
    }


# Single source of truth for the model-parameter set. Order matters: it is the
# create_model signature order, which is also the order of the kwargs dict
# passed to get_model (kept identical to the original hand-written code).
MODEL_KWARGS = [
    _kw('hidden_dim', 128, args_attr='hidden', args_required=True,
        ckpt=('hidden', 'hidden_dim'), save='args'),
    _kw('num_layers', 15, args_attr='layers', args_required=True,
        ckpt=('layers', 'num_layers'), save='args'),
    _kw('dropout', 0.0, args_required=True, ckpt=True, save='args'),
    _kw('genconv_num_layers', 2, ckpt=True, save='args'),
    # conv_type/conv_num_heads/mlp_expansion/mlp_depth are neither saved nor
    # loaded: a GATv2/GIN checkpoint is only reconstructible via the
    # YAML-reparse path (parse_training_config + create_model_from_args).
    _kw('conv_type', 'genconv'),
    _kw('conv_num_heads', 4),
    _kw('mlp_expansion', 2),
    _kw('mlp_depth', 2),
    _kw('num_mlp_layers', 3, ckpt=True, save='args'),
    _kw('jk_mode', 'cat', args_required=True, ckpt=True, save='args'),
    _kw('jk_attention', False, args_required=True, ckpt=True, save='args'),
    _kw('jk_learn_temperature', False, ckpt=True, save='args'),
    _kw('norm_type', 'layer', ckpt=True, save='args'),
    _kw('act_type', 'relu',
        # Auto-detect GELU from state_dict keys (quirk kept as-is: nn.GELU is
        # parameter-free so no 'gelu' key ever exists and this cannot fire)
        ckpt=lambda config, state_dict: config.get(
            'act_type',
            'gelu' if any('gelu' in k.lower() for k in state_dict.keys()) else 'relu'),
        save='args'),
    _kw('skip_connection', True, ckpt=True, save='args'),
    _kw('predict_currents', False, ckpt=True, save='param'),
    _kw('voltage_head_config', None, is_config=True,
        ckpt=('voltage_head_config', 'voltage_head'), save='param'),
    _kw('current_head_config', None, is_config=True,
        ckpt=('current_head_config', 'current_head'), save='param'),
    # Virtual node options (model handles these internally)
    _kw('use_virtual_node', False, args_attr='virtual_node', args_required=True,
        # Detect virtual node from state_dict keys (backward compatibility with
        # old checkpoints that used the separate deepgen_vn model)
        ckpt=lambda config, state_dict: config.get(
            'virtual_node', config.get(
                'use_virtual_node',
                any('vn_' in k or 'virtual_node' in k for k in state_dict.keys()))),
        save='args'),
    _kw('use_attention_pooling', True, ckpt=True, save='args'),
    _kw('vn_learn_temperature', False, ckpt=True, save='args'),
    # vn_gate_broadcast/vn_mode/vn_num_heads/vn_head_dim are never persisted as
    # flat keys; load_checkpoint reads them from a nested 'vn_config' that the
    # training entry points never populate, so on load they resolve via the
    # state-dict sniffs below (num_heads/head_dim: always the defaults).
    _kw('vn_gate_broadcast', False,
        ckpt=lambda config, state_dict: config.get('vn_config', {}).get(
            'gate_broadcast',
            any('virtual_node.gate_projs' in k for k in state_dict.keys()))),
    _kw('vn_mode', 'default',
        ckpt=lambda config, state_dict: config.get('vn_config', {}).get(
            'mode',
            'mha' if any('virtual_node.W_q' in k for k in state_dict.keys()) else 'default')),
    _kw('vn_num_heads', 4,
        ckpt=lambda config, state_dict: config.get('vn_config', {}).get('num_heads', 4)),
    _kw('vn_head_dim', 32,
        ckpt=lambda config, state_dict: config.get('vn_config', {}).get('head_dim', 32)),
    _kw('vn_apply_to', 'backbone'),
    _kw('gradient_checkpointing', False,
        ckpt=lambda config, state_dict: False,  # Not needed for inference
        save='args'),
    # Voltage-derived current options
    _kw('derive_currents_from_voltage', False, ckpt=True, save='param'),
    _kw('mosfet_current_mlp_config', None, is_config=True, ckpt=True,
        save='param_or_empty'),
    # GNN-based current prediction options
    _kw('use_gnn_current_prediction', False, ckpt=True, save='param'),
    _kw('current_gnn_config', None, is_config=True, ckpt=True,
        save='param_or_empty'),
    # Frozen Device MLP options
    _kw('use_frozen_device_mlp', False, ckpt=True, save='param'),
    _kw('frozen_device_mlp_config', None, is_config=True, ckpt=True,
        save='param_or_empty'),
    # Device-level pooling current head
    _kw('use_device_pooling_current', False,
        ckpt=lambda config, state_dict: config.get(
            'use_device_pooling_current',
            any('device_current_head.' in k for k in state_dict.keys())),
        save='args'),
    # Device aggregation layer (auto-detected on load, never persisted)
    _kw('device_aggregation_config', None, is_config=True,
        ckpt=lambda config, state_dict: config.get(
            'device_aggregation_config',
            {'enabled': True} if any('device_agg.' in k for k in state_dict.keys()) else {})),
    # Intermediate voltage prediction
    _kw('intermediate_voltage_config', None, is_config=True),
    # Edge features
    _kw('use_edge_features', False),
    _kw('edge_feature_dim', 6),
    _kw('edge_feature_indices', None),
    _kw('input_dropout', 0.0),
    # Refinement pass options
    _kw('use_refinement_pass', False, ckpt=True, save='param'),
    _kw('refinement_config', None, is_config=True, ckpt=True,
        save='param_or_empty'),
    # AC readout head
    _kw('ac_head_config', None, is_config=True, ckpt=True, save='args'),
    # gm/gds prediction head
    _kw('ss_head_config', None, is_config=True, ckpt=True, save='args'),
    # Region classification head
    _kw('region_head_config', None, is_config=True, ckpt=True, save='args'),
    # Z-space KCL projection
    _kw('kcl_zspace_projection', False),
    _kw('kcl_blend_alpha', 0.0),
    # Tower architecture options
    _kw('backbone_layers', 6, ckpt=True, save='args'),
    _kw('state_tower_layers', 2, ckpt=True, save='args'),
    _kw('sensitivity_tower_layers', 2, ckpt=True, save='args'),
    _kw('backbone_jk_config', None, is_config=True, ckpt=True, save='args'),
    _kw('state_tower_jk_config', None, is_config=True, ckpt=True, save='args'),
    _kw('sensitivity_tower_jk_config', None, is_config=True, ckpt=True,
        save='args'),
    # Vov prediction head (read on load but never persisted -> always {})
    _kw('vov_head_config', None, is_config=True, ckpt=True),
    # Vth prediction head (read on load but never persisted -> always {})
    _kw('vth_head_config', None, is_config=True, ckpt=True),
    # Loop attention config (auto-detected fallback hardcodes 4 heads / dim 32
    # / backbone / gate fusion, as before)
    _kw('loop_attention_config', None, is_config=True,
        ckpt=lambda config, state_dict: config.get(
            'loop_attention_config',
            {'enabled': True, 'num_heads': 4, 'head_dim': 32,
             'apply_to': 'backbone', 'fusion': 'gate'}
            if any('loop_attn' in k for k in state_dict.keys()) else {}),
        save='args'),
    # DC gain prediction head
    _kw('dc_gain_config', None, is_config=True, ckpt=True, save='args'),
    # gm/Id auxiliary head
    _kw('gm_id_head_config', None, is_config=True,
        ckpt=lambda config, state_dict: config.get(
            'gm_id_head_config',
            {'enabled': True} if any('gm_id_head.' in k for k in state_dict.keys()) else {}),
        save='args'),
    # Subcircuit DAG
    _kw('subcircuit_dag_config', None, is_config=True),
    # Vgs/Vds prediction head
    _kw('vgsvds_config', None, is_config=True),
    # Pretrained IV-surface autoencoder embedding as MOSFET input feature
    _kw('iv_embedder_config', None, is_config=True),
    # Physics-exact currents from LUT lookup on predicted V
    _kw('lut_current_config', None, is_config=True),
    # Iterative-refinement per-node LUT op-point features
    _kw('lut_op_features_config', None, is_config=True),
    # Stacking: forward baseline predictions as extra inputs
    _kw('stack_features_config', None, is_config=True),
    # End-to-end LUT-residual (heads predict deltas, LUT is physics layer)
    _kw('lut_residual_config', None, is_config=True),
    # 5-dim per-MOSFET physical descriptor (LUT lookups at canonical biases)
    _kw('mosfet_descriptor_config', None, is_config=True),
    # 7-dim per-MOSFET functional role one-hot
    _kw('mosfet_role_config', None, is_config=True),
    # 5-dim per-net role one-hot (VDD/GND/SIG_IN/SIG_OUT/INTERNAL)
    _kw('net_role_config', None, is_config=True),
]


def _args_value(args, spec):
    """
    Read one model kwarg off the flat args namespace, exactly as the original
    create_model_from_args / build_full_config did: required entries via plain
    attribute access, the rest via getattr with the standard default ({} for
    config-dict entries, the signature default otherwise).
    """
    if spec['args_required']:
        return getattr(args, spec['args_attr'])
    return getattr(args, spec['args_attr'], {} if spec['is_config'] else spec['default'])


def create_model(
    node_feature_dim,
    hidden_dim=128,
    num_layers=15,
    dropout=0.0,
    genconv_num_layers=2,
    conv_type='genconv',
    conv_num_heads=4,
    mlp_expansion=2,
    mlp_depth=2,
    num_mlp_layers=3,
    jk_mode='cat',
    jk_attention=False,
    jk_learn_temperature=False,
    norm_type='layer',
    act_type='relu',
    skip_connection=True,
    predict_currents=False,
    voltage_head_config=None,
    current_head_config=None,
    use_virtual_node=False,
    use_attention_pooling=True,
    vn_learn_temperature=False,
    vn_gate_broadcast=False,
    vn_mode='default',
    vn_num_heads=4,
    vn_head_dim=32,
    vn_apply_to='backbone',
    gradient_checkpointing=False,
    device='cuda',
    model_type=None,
    derive_currents_from_voltage=False,
    mosfet_current_mlp_config=None,
    use_gnn_current_prediction=False,
    current_gnn_config=None,
    use_frozen_device_mlp=False,
    frozen_device_mlp_config=None,
    use_device_pooling_current=False,
    device_aggregation_config=None,
    intermediate_voltage_config=None,
    use_edge_features=False,
    edge_feature_dim=6,
    edge_feature_indices=None,
    input_dropout=0.0,
    use_refinement_pass=False,
    refinement_config=None,
    ac_head_config=None,
    ss_head_config=None,
    region_head_config=None,
    kcl_zspace_projection=False,
    kcl_blend_alpha=0.0,
    # Tower architecture options
    backbone_layers=6,
    state_tower_layers=2,
    sensitivity_tower_layers=2,
    backbone_jk_config=None,
    state_tower_jk_config=None,
    sensitivity_tower_jk_config=None,
    vov_head_config=None,
    vth_head_config=None,
    loop_attention_config=None,
    dc_gain_config=None,
    gm_id_head_config=None,
    subcircuit_dag_config=None,
    vgsvds_config=None,
    iv_embedder_config=None,
    lut_current_config=None,
    lut_op_features_config=None,
    stack_features_config=None,
    lut_residual_config=None,
    mosfet_descriptor_config=None,
    mosfet_role_config=None,
    net_role_config=None,
):
    """
    Create a GNN model with the specified configuration.

    The signature is kept explicit for API compatibility; its parameter names,
    order and defaults are verified against MODEL_KWARGS at import time (see
    _assert_table_matches_signature), so the two cannot drift apart.

    Args:
        node_feature_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        dropout: Dropout rate
        genconv_num_layers: Layers in each GENConv
        num_mlp_layers: MLP layers in head
        jk_mode: Jumping knowledge mode ('cat', 'last', 'max', 'sum')
        jk_attention: Use attention for JK aggregation
        jk_learn_temperature: Learn JK attention temperature
        norm_type: Normalization type ('layer' or 'batch')
        skip_connection: Concatenate input to head
        predict_currents: Enable current prediction head
        voltage_head_config: Config dict for voltage head
        current_head_config: Config dict for current head
        use_virtual_node: Enable virtual node (config option, not separate model)
        use_attention_pooling: Use attention pooling for virtual node
        vn_learn_temperature: Learn virtual node attention temperature
        gradient_checkpointing: Enable gradient checkpointing
        device: Device to place model on
        model_type: Model type from registry (default: 'deepgen')
        derive_currents_from_voltage: Derive MOSFET currents from predicted voltages
        mosfet_current_mlp_config: Config dict for MOSFET current MLP
        use_gnn_current_prediction: Use GNN layers for current prediction
        current_gnn_config: Config dict for current GNN backbone
        use_frozen_device_mlp: Use pre-trained frozen Device MLP for current prediction
        frozen_device_mlp_config: Config dict for frozen Device MLP
        use_refinement_pass: Enable two-pass refinement architecture
        refinement_config: Config dict for refinement GNN
        ss_head_config: Config dict for gm/gds prediction head

    Returns:
        model: Created model on specified device
        model_class: String name of model class used
    """
    # Default to 'deepgen' - VN is now a config option, not a separate model
    if model_type is None:
        model_type = 'deepgen'

    # All kwargs passed to model - VN is handled internally. Derived from the
    # MODEL_KWARGS table in table order (= original hand-written dict order);
    # config-dict entries get the original ``value or {}`` normalization.
    values = locals()
    model_kwargs = {'node_feature_dim': node_feature_dim}
    for spec in MODEL_KWARGS:
        value = values[spec['name']]
        model_kwargs[spec['name']] = (value or {}) if spec['is_config'] else value

    # Create model using registry
    model = get_model(model_type, **model_kwargs)
    model_class = type(model).__name__

    model = model.to(device)
    return model, model_class


def _assert_table_matches_signature():
    """Fail loudly at import time if MODEL_KWARGS and create_model drift."""
    non_table = ('node_feature_dim', 'device', 'model_type')
    signature_items = [
        (name, param.default)
        for name, param in inspect.signature(create_model).parameters.items()
        if name not in non_table
    ]
    table_items = [(spec['name'], spec['default']) for spec in MODEL_KWARGS]
    if signature_items != table_items:
        raise RuntimeError(
            'MODEL_KWARGS is out of sync with the create_model signature; '
            'update both together. signature=%r table=%r'
            % (signature_items, table_items))


_assert_table_matches_signature()


def create_model_from_args(args, input_dim, device='cuda'):
    """
    Create model from argparse args (convenience wrapper for training scripts).

    Args:
        args: Argument namespace with model config
        input_dim: Input feature dimension
        device: Device to place model on

    Returns:
        model: Created model
        model_class: String name of model class
    """
    # Get model_type from args (default: 'deepgen')
    model_type = getattr(args, 'model_type', 'deepgen')

    kwargs = {spec['name']: _args_value(args, spec) for spec in MODEL_KWARGS}
    return create_model(
        node_feature_dim=input_dim,
        device=device,
        model_type=model_type,
        **kwargs,
    )


def _infer_input_dim(config, state_dict):
    """Infer input feature dim from state_dict, falling back to the config."""
    if 'input_proj.weight' in state_dict:
        return state_dict['input_proj.weight'].shape[1]
    elif 'input_linear.weight' in state_dict:
        return state_dict['input_linear.weight'].shape[1]
    return config.get('input_dim', 20)


def resolve_model_kwargs(config, state_dict):
    """
    Resolve every create_model kwarg for a saved checkpoint from its flat
    config dict plus state-dict-key auto-detection, exactly as load_checkpoint
    has always done. Entries that were never persisted (ckpt=None in
    MODEL_KWARGS) resolve to the table default - the historical lossiness of
    the saved config, preserved as-is.

    Args:
        config: Flat config dict stored in the checkpoint
        state_dict: Model state dict (used for substring auto-detection)

    Returns:
        Dict mapping every MODEL_KWARGS name to its resolved value
    """
    kwargs = {}
    for spec in MODEL_KWARGS:
        if spec['ckpt'] is None:
            kwargs[spec['name']] = spec['default']
        else:
            kwargs[spec['name']] = spec['ckpt'](config, state_dict)
    return kwargs


def save_checkpoint(
    save_path,
    model_state_dict,
    epoch,
    val_loss,
    config,
    stats,
    save_yaml=True,
):
    """
    Save model checkpoint with full config.

    Args:
        save_path: Path to save checkpoint (.pt file)
        model_state_dict: Model state dict
        epoch: Best epoch number
        val_loss: Best validation loss
        config: Dict with all model and training config
        stats: Dict with normalization stats {'vdc': {...}, 'curr': {...}}
        save_yaml: Also save config.yaml alongside checkpoint

    The checkpoint additionally stores the resolved create_model kwargs under
    the top-level key 'model_kwargs' (resolved from config + state_dict with
    the same rules load_checkpoint applies). Purely additive: loading does not
    require it and old checkpoints keep loading through the existing path.
    """
    save_path = Path(save_path)

    resolved_model_kwargs = {
        'model_type': config.get('model_type', 'deepgen'),
        'node_feature_dim': _infer_input_dim(config, model_state_dict),
    }
    resolved_model_kwargs.update(resolve_model_kwargs(config, model_state_dict))

    torch.save({
        'model_state_dict': model_state_dict,
        'epoch': epoch,
        'val_loss': val_loss,
        'config': config,
        'stats': stats,
        'model_kwargs': resolved_model_kwargs,
    }, save_path)

    if save_yaml:
        config_yaml = build_config_yaml(config, stats)
        config_path = save_path.parent / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_yaml, f, default_flow_style=False, sort_keys=False)
        return save_path, config_path

    return save_path, None


def build_config_yaml(config, stats):
    """Build YAML-friendly config dict from flat config."""
    return {
        'model': {
            'hidden_dim': config.get('hidden', config.get('hidden_dim', 128)),
            'num_layers': config.get('layers', config.get('num_layers', 15)),
            'dropout': config.get('dropout', 0.0),
            'genconv_num_layers': config.get('genconv_num_layers', 2),
            'use_virtual_node': config.get('virtual_node', config.get('use_virtual_node', False)),
            'predict_currents': config.get('predict_currents', False),
            'jk_mode': config.get('jk_mode', 'cat'),
            'jk_attention': config.get('jk_attention', False),
            'jk_learn_temperature': config.get('jk_learn_temperature', False),
            'norm_type': config.get('norm_type', 'layer'),
            'skip_connection': config.get('skip_connection', True),
            'use_attention_pooling': config.get('use_attention_pooling', True),
            'vn_learn_temperature': config.get('vn_learn_temperature', False),
            'voltage_head': config.get('voltage_head_config', config.get('voltage_head', {})),
            'current_head': config.get('current_head_config', config.get('current_head', {})),
            'derive_currents_from_voltage': config.get('derive_currents_from_voltage', False),
            'mosfet_current_mlp_config': config.get('mosfet_current_mlp_config', {}),
        },
        'training': {
            'dataset_path': config.get('dataset', ''),
            'learning_rate': config.get('learning_rate', config.get('lr', 0.003)),
            'weight_decay': config.get('weight_decay', 0.0),
            'gradient_clip': config.get('gradient_clip', 1.0),
            'scheduler': config.get('scheduler', 'plateau'),
            'warmup_epochs': config.get('warmup_epochs', config.get('warmup', 0)),
            'current_weight': config.get('current_weight', 1.0),
            'early_stopping_patience': config.get('early_stopping_patience', 80),
            'val_freq': config.get('val_freq', 1),
            'seed': config.get('seed', 42),
        },
        'normalization': {
            'vdc_mean': stats.get('vdc', {}).get('mean', 0.0),
            'vdc_std': stats.get('vdc', {}).get('std', 1.0),
            'current_mean': stats.get('current', stats.get('curr', {})).get('mean', 0.0),
            'current_std': stats.get('current', stats.get('curr', {})).get('std', 1.0),
        },
        'loss': {
            'loss_type': config.get('loss_type', 'mse'),
            'huber_delta': config.get('huber_delta', 1.0),
        },
    }


def load_checkpoint(checkpoint_path, device='cuda'):
    """
    Load model from checkpoint with architecture auto-detection.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        model: Loaded model in eval mode
        config: Config dict
        stats: Normalization stats dict
    """
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    config = ckpt.get('config', {})
    stats = ckpt.get('stats', {})
    state_dict = ckpt['model_state_dict']

    # Get model_type from config (default: 'deepgen')
    model_type = config.get('model_type', 'deepgen')

    # Infer input dim from state_dict
    input_dim = _infer_input_dim(config, state_dict)

    # Resolve every model kwarg from the saved flat config, with the
    # state-dict-substring auto-detection fallbacks encoded in MODEL_KWARGS
    model_kwargs = resolve_model_kwargs(config, state_dict)

    # Create model - VN is now a config option on 'deepgen'
    model, model_class = create_model(
        node_feature_dim=input_dim,
        device='cpu',  # Load to CPU first, then transfer after loading weights
        model_type=model_type,  # Use detected model_type from config
        **model_kwargs,
    )

    # Load weights and move to device
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Add detection info to config for reference
    config['_detected'] = {
        'input_dim': input_dim,
        'has_virtual_node': model_kwargs['use_virtual_node'],
        'model_class': model_class,
        'model_type': model_type,
    }

    return model, config, stats


def build_full_config(args, total_input_dim, predict_currents, voltage_head_config,
                      current_head_config, current_weight, loss_type, huber_delta,
                      val_freq, early_stopping_patience, dataset_path,
                      derive_currents_from_voltage=False, mosfet_current_mlp_config=None,
                      use_gnn_current_prediction=False, current_gnn_config=None,
                      use_frozen_device_mlp=False, frozen_device_mlp_config=None,
                      use_refinement_pass=False, refinement_config=None):
    """
    Build complete config dict from training args for saving.

    Args:
        args: Argument namespace from argparse
        total_input_dim: Total input feature dimension
        predict_currents: Whether model predicts currents
        voltage_head_config: Voltage head config dict
        current_head_config: Current head config dict
        current_weight: Weight for current loss
        loss_type: Loss type string
        huber_delta: Huber delta value
        val_freq: Validation frequency
        early_stopping_patience: Early stopping patience
        dataset_path: Path to dataset
        derive_currents_from_voltage: Derive MOSFET currents from predicted voltages
        mosfet_current_mlp_config: Config dict for MOSFET current MLP
        use_gnn_current_prediction: Use GNN layers for current prediction
        current_gnn_config: Config dict for current GNN backbone
        use_frozen_device_mlp: Use pre-trained frozen Device MLP for current prediction
        frozen_device_mlp_config: Config dict for frozen Device MLP
        use_refinement_pass: Enable two-pass refinement architecture
        refinement_config: Config dict for refinement GNN

    Returns:
        Dict with all config values
    """
    model_type = getattr(args, 'model_type', 'deepgen')

    # Values arriving as explicit parameters of this function rather than
    # through args (train_v3.py computes these separately)
    param_values = {
        'predict_currents': predict_currents,
        'voltage_head_config': voltage_head_config,
        'current_head_config': current_head_config,
        'derive_currents_from_voltage': derive_currents_from_voltage,
        'mosfet_current_mlp_config': mosfet_current_mlp_config,
        'use_gnn_current_prediction': use_gnn_current_prediction,
        'current_gnn_config': current_gnn_config,
        'use_frozen_device_mlp': use_frozen_device_mlp,
        'frozen_device_mlp_config': frozen_device_mlp_config,
        'use_refinement_pass': use_refinement_pass,
        'refinement_config': refinement_config,
    }

    config = {
        # Model architecture
        'model_type': model_type,
    }
    for spec in MODEL_KWARGS:
        source = spec['save']
        if source is None:
            # Never persisted (historical gap in the saved config, kept as-is)
            continue
        if source == 'param':
            value = param_values[spec['name']]
        elif source == 'param_or_empty':
            value = param_values[spec['name']] or {}
        else:  # 'args'
            value = _args_value(args, spec)
        config[spec['args_attr']] = value

    config.update({
        # Virtual node config (args.vn_config is never set by the training
        # entry points, so this stays {} and vn_* settings are not persisted)
        'vn_config': getattr(args, 'vn_config', {}),
        # Training params
        'learning_rate': args.lr,
        'weight_decay': getattr(args, 'weight_decay', 0.0),
        'gradient_clip': args.gradient_clip,
        'scheduler': args.scheduler,
        'warmup_epochs': args.warmup,
        'current_weight': current_weight,
        'loss_type': loss_type,
        'huber_delta': huber_delta,
        'val_freq': val_freq,
        'early_stopping_patience': early_stopping_patience,
        'seed': args.seed,
        # Data info
        'dataset': str(dataset_path),
        'input_dim': total_input_dim,
    })
    return config
