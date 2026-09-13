# Public release refactor design

Date: 2026-09-13
Branch: `public-release` (from `autograd-gm-gds` at ea429f8)

## Goal

Turn the thesis working tree into a clean public GitHub repository without changing any numerical behavior: same models, same training results, same checkpoint compatibility. Oversized files get split, duplicated logic gets a single home, one-off scripts get organized and the repo gets an installable package layout plus a README.

## Frozen surfaces (must not change)

These are load-bearing external contracts. Every implementation step must preserve them byte for byte:

1. **state_dict key names**: every nn.Module attribute and registered buffer of `TowerGENConv`, `DeepGENConv`, `PretrainBackbone` and all components. Existing `best_model.pt` files from 66+ ablation runs must keep loading. Module creation order in `__init__` also stays fixed so seeded weight init is bitwise identical.
2. **Registry names** `tower_genconv` and `deepgen` and the checkpoint dict schema `{model_state_dict, epoch, val_loss, config, stats}` including all flat config key aliases.
3. **YAML config keys**: all keys read by `parse_training_config` and the `*_config` dicts. 550+ configs in history must keep meaning exactly what they meant.
4. **Dataset attribute names**: every attribute read from pickled PyG Data/Batch objects (`mosfet_info`, `mosfet_ptr`, `node_log_gm`, `kcl_include_mask`, `edge_attr` in its 6, 10 and 11 dim generations and the rest). The bipartite terminal-net node ordering (terminals first in drain, gate, source, bulk order, then alphabetically sorted nets) is an on-disk contract.
5. **training.log format strings**: the printed metric lines are regex-parsed by the figure scripts. No wording, spacing or precision changes.
6. **Entry point paths and CLI flags**: `scripts/train_v3.py`, `scripts/generate_dataset.py`, `scripts/evaluate.py` and the pretrain scripts keep their paths and argparse surfaces. SLURM launchers call them by exact path.
7. **Loss semantics**: validate() zeroing physics weights, the KCL raw-current pathway, lazy LUT init after warmup, the `model.current_epoch` protocol and the in-place order-sensitive batch normalization pipeline.

## Verification harness

- Forward goldens: five representative tower configs built with a fixed seed, forwarded on a real re-collated 24-graph batch. state_dict keys, shapes and output tensors must match bitwise before and after every wave (SLURM CPU job).
- End-to-end goldens: two deterministic 2-epoch `train_v3.py` runs (base tower config and physics mirror config) from a pre-refactor worktree snapshot. The refactored tree must reproduce the same logged losses (SLURM GPU job).
- `python -m py_compile` and import smoke checks after each wave.

## Target layout

```
circuitgnn/            renamed from src/, installable flat package
  circuits/            netlist parser, PySpice simulator
  data/                graph building, batching, loaders, schema notes
  physics/             MOSFET LUT and physics constants
  gnn/
    registry.py
    architectures/
      base.py
      tower/           split of the 3722 line tower_genconv.py
      deepgen.py       legacy baseline (kept, dead branches removed)
      pretrain_backbone.py
    components/        live components plus lut/ subpackage
    legacy/            deepgen-only components and device_mlp
  training/
    config.py
    losses/            split of the 2286 line losses.py
    loops.py
    checkpoint.py
    ...
  evaluation/          shared eval helpers promoted from scripts
scripts/               thin entry points
  train_v3.py, generate_dataset.py, evaluate.py, pretrain_*.py
  analysis/            dataset and error analysis one-offs
  figures/             one script per thesis figure
  migrations/          the applied patch_dataset_* scripts, documented
  slurm/               consolidated parameterized launchers
configs/               tracked, active families at top, archive/ for legacy
netlists/              SPICE templates, sky130 include path parameterized
docs/                  this spec plus architecture notes
tests/                 smoke tests and the golden equivalence harness
```

Old import paths keep working during and after the transition through re-export shims (`src/` is gone but every moved module re-exports from its old dotted path within `circuitgnn`).

## Waves

- **Wave 0, package rename**: `git mv src circuitgnn`, rewrite `from src.` imports across all Python files, add `pyproject.toml` and fixed `requirements.txt`. Pure text transformation.
- **Wave 1, parallel splits** (disjoint file ownership, old paths re-exported):
  - `tower_genconv.py` into `gnn/architectures/tower/` (constants, physics formulas as pure functions, JK, SS heads, subcircuit DAG, input features, LUT paths, model). Deduplicates the three copies of the DC gain formula and three copies of the UGBW estimate.
  - `losses.py` into `training/losses/` (data terms, KCL, device physics, supervised heads, aggregate). The shared MOSFET index helper moves to a neutral module, breaking the model-imports-losses cycle.
  - components liveness split: legacy deepgen-only components to `gnn/legacy/`, SKY130 LUT loading deduplicated into `components/lut/`, dead `aggregation.py` removed.
  - data layer: loop info and net role deduplication, dead module removal, schema documentation.
- **Wave 2, coupled cluster**: `train_v3.py` main() dismantled into modules, `loops.py` tuple returns converted to named results, `evaluate.py` and `eval_checkpoint.py` unified on the same helpers, checkpoint model-kwargs single source of truth. One agent owns all files in this cluster.
- **Wave 3, surface**: scripts triage into subfolders with SLURM launchers updated in the same commit, SLURM clone collapse, pretrain script dedup, README and docs, config archive split.

## Known bugs fixed on the way (each its own commit, behavior change documented)

- `best_model_state` aliasing in train_v3: the saved best checkpoint was actually final-epoch weights. Fix with a detached deep copy.
- `vgsvds_loss` computed but never added to the static weighted sum (and `vth_loss` missing from the uncertainty branch). Both sums now generated from one mapping.
- `generate_dataset.py` NameError in the non-LHS branch.
- `eval_checkpoint.py` split detection globbing `*.pt` instead of variant files.
- Empty-tensor return in `compute_isource_currents` missing the device argument.

## Out of scope

- No pruning of experimental config branches: every mode referenced by a tracked config keeps working.
- No dataset regeneration and no changes to stored dataset schemas.
- No renaming of metrics, thresholds or evaluation semantics.
