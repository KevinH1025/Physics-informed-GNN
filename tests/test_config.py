"""Every shipped experiment config must still parse.

A renamed or mistyped YAML key does not raise in this codebase, it silently
falls back to a default, so a config that stops parsing is one of the few
failures that surface loudly. These tests walk the whole configs/ tree.
"""
from pathlib import Path

import pytest

from circuitgnn.training.config import load_config, parse_training_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / 'configs'

ALL_CONFIGS = sorted(CONFIG_ROOT.rglob('*.yaml'))
MODEL_CONFIGS = [p for p in ALL_CONFIGS if 'opamp_dataset' not in p.parts]


def _config_id(path):
    return str(path.relative_to(CONFIG_ROOT))


@pytest.mark.skipif(not ALL_CONFIGS, reason='configs/ tree not present')
@pytest.mark.parametrize('path', ALL_CONFIGS, ids=_config_id)
def test_config_is_valid_yaml(path):
    assert isinstance(load_config(str(path)), dict)


@pytest.mark.skipif(not MODEL_CONFIGS, reason='configs/ tree not present')
@pytest.mark.parametrize('path', MODEL_CONFIGS, ids=_config_id)
def test_training_config_parses(path):
    args = parse_training_config(load_config(str(path)))
    assert 'hidden' in args
    assert 'layers' in args


@pytest.mark.skipif(not MODEL_CONFIGS, reason='configs/ tree not present')
def test_model_types_are_registered():
    import circuitgnn.gnn.architectures  # noqa: F401  (import registers the models)
    from circuitgnn.gnn.registry import MODEL_REGISTRY

    for path in MODEL_CONFIGS:
        model_type = (load_config(str(path)).get('model') or {}).get('type')
        if model_type is not None:
            assert model_type in MODEL_REGISTRY, f'{path.name} wants unknown model {model_type}'
