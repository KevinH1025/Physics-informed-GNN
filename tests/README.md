# Tests

Two tiers, split by what they need.

## Fast tests (no data required)

```bash
pip install -e ".[dev]"
pytest tests/
```

`test_imports.py` checks that the package imports, that the module paths which moved during the refactor still resolve and that both architectures register. `test_config.py` parses every YAML under `configs/` and checks each names a registered model. Config keys in this codebase fall back to defaults rather than raising, so a config that fails to parse is one of the few loud failures available, which is why the whole tree is walked.

## Golden equivalence harness (needs a dataset and a checkpoint)

`golden_forward.py` is the gate used while refactoring: it builds several representative configs with a fixed seed, runs one forward pass on a real prebatched batch and records state dict keys, shapes and output tensors. Run it once on the reference commit, then again on the changed tree to compare.

```bash
# On the reference commit
python tests/golden_forward.py --dataset <dataset dir> --out goldens.pt

# After refactoring, on the same machine
python tests/golden_forward.py --dataset <dataset dir> --out goldens.pt --compare
```

Comparison is bitwise, so run both halves on the same hardware and library versions. Floating point reductions differ across CPU architectures and between CPU and GPU, so a mismatch caused by moving machines is not a real regression.

The harness is deliberately strict about two things that silently break saved models. The first is the set of `state_dict` keys, because checkpoints from earlier runs are loaded by name. The second is the forward output values, because most of the physics losses read specific output entries.
