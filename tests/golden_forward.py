"""Golden equivalence harness: prove a refactor changed no numbers.

Builds a set of representative configs with a fixed seed, runs one forward
pass on a real prebatched batch and records the state dict key set, tensor
shapes and every output tensor. Run once on the reference commit to capture,
then again on the changed tree with --compare.

    python tests/golden_forward.py --dataset datasets/<name> --out goldens.pt
    python tests/golden_forward.py --dataset datasets/<name> --out goldens.pt --compare

Comparison is bitwise, so capture and compare must run on the same hardware
with the same library versions. The two things it guards are the ones that
break saved models silently: the set of state_dict keys, because checkpoints
are loaded by name, and the forward output values, because the physics losses
read specific entries out of the output dict.
"""
import argparse
import pickle
import sys
import traceback
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Chosen to span the architecture space: the default tower, a GIN backbone with
# no tower, a GATv2 backbone, the subcircuit DAG with the analytic UGBW head and
# the unified scalar jumping-knowledge variant.
DEFAULT_CONFIGS = [
    'configs/gnn/tower/w512_dc_h128_gelu_cascode_loopwarm_v9_5k.yaml',
    'configs/gnn/tower/w512_ginbn_3layer_bb8_notower_vn_kcl_w10_v9_5k.yaml',
    'configs/gnn/tower/w512_dc_h128_gelu_cascode_loopwarm_jklast_gatv2_v9_5k.yaml',
    'configs/gnn/tower/w512_dc_h128_gelu_cascode_loopwarm_dagnn_ugbw_physics_v9_5k.yaml',
    'configs/gnn/tower/w512_dc_h128_gelu_cascode_loopwarm_unifiedjk_scalar_v9_5k.yaml',
]

# Placeholder normalization stats. The model only needs these to be present and
# stable across capture and compare, it does not matter that they are not the
# dataset's real statistics.
PLACEHOLDER_STATS = [
    ('vdc_mean', 0.9), ('vdc_std', 0.5),
    ('current_mean', -5.55), ('current_std', 1.42),
    ('ss_gm_mean', -4.43), ('ss_gm_std', 1.44),
    ('ss_gds_mean', -5.89), ('ss_gds_std', 1.89),
    ('voltage_mean', 0.9), ('voltage_std', 0.5),
]

SEED = 1234


def run_forward(model, batch):
    """Forward under no_grad unless the model differentiates internally."""
    if getattr(model, '_needs_autograd', False):
        return model(batch)
    with torch.no_grad():
        return model(batch)


def make_small_batch(batch, n_graphs):
    """Re-collate the first n graphs so the harness fits in modest memory."""
    from torch_geometric.data import Batch

    from circuitgnn.data.batching import _add_device_ptr_tensors

    for attr in ['mosfet_ptr', 'resistor_ptr', 'isource_ptr', 'capacitor_ptr']:
        if hasattr(batch, attr):
            delattr(batch, attr)
    graphs = batch.to_data_list()[:n_graphs]
    small = Batch.from_data_list(graphs)
    _add_device_ptr_tensors(small, graphs)
    return small


def attach_norm(batch):
    for name, value in PLACEHOLDER_STATS:
        setattr(batch, name, torch.tensor(float(value)))


def load_batch(dataset_dir, n_graphs):
    variant = Path(dataset_dir) / 'train' / 'variant_0.pkl'
    with open(variant, 'rb') as handle:
        batches = pickle.load(handle)
    batch = make_small_batch(batches[0], n_graphs)
    attach_norm(batch)
    return batch


def input_dim(batch):
    dims = batch.x.shape[1] + batch.type_tens.shape[1]
    for optional in ['net_type', 'structural_pe']:
        tensor = getattr(batch, optional, None)
        if tensor is not None:
            dims += tensor.shape[1]
    return dims


def build_model(config_path, in_dim):
    from circuitgnn.training.checkpoint import create_model_from_args
    from circuitgnn.training.config import load_config, parse_training_config

    parsed = parse_training_config(load_config(str(REPO_ROOT / config_path)))
    namespace = argparse.Namespace()
    for key, value in parsed.items():
        setattr(namespace, key, value)
    torch.manual_seed(SEED)
    model, _ = create_model_from_args(namespace, in_dim, 'cpu')
    model.eval()
    # Past any warmup so warmup-gated modules are active during the comparison.
    model.current_epoch = 10 ** 6
    return model


def load_checkpoint_model(checkpoint_path):
    from circuitgnn.training.checkpoint import load_checkpoint

    loaded = load_checkpoint(str(checkpoint_path))
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    model.eval()
    model.current_epoch = 10 ** 6
    return model


def capture(args, batch):
    in_dim = input_dim(batch)
    print(f'batch: {batch.num_nodes} nodes, in_dim={in_dim}')
    results = {'in_dim': in_dim}

    for config_path in args.configs:
        entry = {}
        try:
            model = build_model(config_path, in_dim)
            state = model.state_dict()
            entry['sd_keys'] = sorted(state)
            entry['sd_shapes'] = {k: tuple(v.shape) for k, v in state.items()}
            entry['param_count'] = sum(p.numel() for p in model.parameters())
            out = run_forward(model, batch)
            entry['out'] = {k: v.detach().clone() for k, v in out.items() if torch.is_tensor(v)}
        except Exception:
            entry['error'] = traceback.format_exc()
            print(f'{config_path}: ERROR {entry["error"].splitlines()[-1]}')
        else:
            print(f'{config_path}: ok ({len(entry["out"])} output tensors)')
        results[config_path] = entry

    if args.checkpoint:
        entry = {}
        try:
            model = load_checkpoint_model(args.checkpoint)
            entry['sd_keys'] = sorted(model.state_dict())
            out = run_forward(model, batch)
            entry['out'] = {k: v.detach().clone() for k, v in out.items() if torch.is_tensor(v)}
        except Exception:
            entry['error'] = traceback.format_exc()
            print(f'checkpoint: ERROR {entry["error"].splitlines()[-1]}')
        else:
            print('checkpoint: ok')
        results['checkpoint'] = entry

    torch.save(results, args.out)
    print(f'saved {args.out}')
    return 0


def compare(args, batch):
    reference = torch.load(args.out, weights_only=False)
    in_dim = reference['in_dim']
    failures = []

    for config_path in args.configs:
        expected = reference.get(config_path)
        if expected is None:
            failures.append(f'{config_path}: absent from the reference capture')
            continue
        try:
            model = build_model(config_path, in_dim)
            keys = sorted(model.state_dict())
            if keys != expected.get('sd_keys'):
                before, after = set(expected.get('sd_keys') or []), set(keys)
                failures.append(
                    f'{config_path}: state_dict keys differ '
                    f'(missing {sorted(before - after)[:5]}, added {sorted(after - before)[:5]})')
                continue
            out = run_forward(model, batch)
            for name, want in expected['out'].items():
                got = out[name].detach()
                if got.shape != want.shape:
                    failures.append(f'{config_path}:{name} shape {tuple(want.shape)} -> {tuple(got.shape)}')
                elif not torch.equal(got, want):
                    failures.append(
                        f'{config_path}:{name} values differ '
                        f'(max abs diff {(got - want).abs().max().item():.3e})')
        except Exception:
            if 'error' not in expected:
                failures.append(f'{config_path}: now raises {traceback.format_exc().splitlines()[-1]}')

    expected = reference.get('checkpoint')
    if args.checkpoint and expected:
        try:
            model = load_checkpoint_model(args.checkpoint)
            out = run_forward(model, batch)
            for name, want in expected.get('out', {}).items():
                if not torch.equal(out[name].detach(), want):
                    failures.append(f'checkpoint:{name} values differ')
        except Exception:
            if 'error' not in expected:
                failures.append(f'checkpoint: now raises {traceback.format_exc().splitlines()[-1]}')

    if failures:
        print('GOLDEN COMPARISON FAILED:')
        for failure in failures:
            print('  -', failure)
        return 1
    print('GOLDEN COMPARISON PASSED (bitwise identical)')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', required=True,
                        help='dataset directory containing train/variant_0.pkl')
    parser.add_argument('--out', default='goldens.pt', help='capture file to write or compare against')
    parser.add_argument('--compare', action='store_true', help='compare against --out instead of capturing')
    parser.add_argument('--checkpoint', default=None, help='optional .pt to exercise checkpoint loading')
    parser.add_argument('--configs', nargs='*', default=DEFAULT_CONFIGS, help='configs to build')
    parser.add_argument('--graphs', type=int, default=24, help='graphs to keep from the batch')
    parser.add_argument('--threads', type=int, default=8, help='torch CPU threads')
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    batch = load_batch(args.dataset, args.graphs)
    return compare(args, batch) if args.compare else capture(args, batch)


if __name__ == '__main__':
    sys.exit(main())
