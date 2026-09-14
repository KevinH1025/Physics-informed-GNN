"""Smoke tests: the package imports cleanly and its public surface is intact.

These catch the most common refactor regression, a moved module whose old
import path was not kept working. They need torch and PyG but build no models
and touch no datasets, so they run in a few seconds on any machine.
"""
import importlib

import pytest


PUBLIC_MODULES = [
    'circuitgnn',
    'circuitgnn.circuits.parser',
    'circuitgnn.data.batching',
    'circuitgnn.data.graph_builder',
    'circuitgnn.data.loop_info',
    'circuitgnn.data.net_roles',
    'circuitgnn.data.schema',
    'circuitgnn.gnn',
    'circuitgnn.gnn.registry',
    'circuitgnn.gnn.architectures.base',
    'circuitgnn.gnn.architectures.deepgen',
    'circuitgnn.gnn.architectures.tower',
    'circuitgnn.gnn.architectures.pretrain_backbone',
    'circuitgnn.gnn.components',
    'circuitgnn.gnn.components.layers',
    'circuitgnn.gnn.components.virtual_node',
    'circuitgnn.training.checkpoint',
    'circuitgnn.training.config',
    'circuitgnn.training.data_loading',
    'circuitgnn.training.loops',
    'circuitgnn.training.losses',
]


@pytest.mark.parametrize('name', PUBLIC_MODULES)
def test_module_imports(name):
    importlib.import_module(name)


# Modules that moved during the public-release refactor. Their old import paths
# stay valid so existing scripts and checkpoint tooling keep working.
LEGACY_PATHS = {
    'circuitgnn.gnn.architectures.tower_genconv': ['TowerGENConv', 'JKAggregation', 'log10_add'],
    'circuitgnn.gnn.components.current_from_voltage': ['MOSFETCurrentMLP'],
    'circuitgnn.models.device_mlp': ['DeviceMLP'],
    'circuitgnn.training.losses': ['compute_combined_loss', 'compute_kcl_loss', 'get_device_graph_idx'],
}


@pytest.mark.parametrize('module_name,symbols', sorted(LEGACY_PATHS.items()))
def test_legacy_import_paths_still_work(module_name, symbols):
    module = importlib.import_module(module_name)
    for symbol in symbols:
        assert hasattr(module, symbol), f'{module_name} no longer exports {symbol}'


def test_tower_shim_is_the_same_class():
    from circuitgnn.gnn.architectures.tower import TowerGENConv as canonical
    from circuitgnn.gnn.architectures.tower_genconv import TowerGENConv as shimmed

    assert canonical is shimmed


def test_registry_contains_both_architectures():
    import circuitgnn.gnn.architectures  # noqa: F401  (import registers the models)
    from circuitgnn.gnn.registry import MODEL_REGISTRY

    assert set(MODEL_REGISTRY) == {'deepgen', 'tower_genconv'}
