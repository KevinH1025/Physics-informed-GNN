# Architecture

Written for someone who wants to read, extend or reproduce this code. It covers how a circuit becomes a prediction, where each concern lives and which contracts you must not break.

## The pipeline

```
netlist template + sampled device sizes
        |  circuitgnn/data/sampling.py
        v
SPICE simulation (ngspice via PySpice)
        |  circuitgnn/circuits/simulator.py
        v
operating point, currents, gm/gds, AC response
        |  circuitgnn/data/graph_builder.py
        v
bipartite PyG graph + targets + masks
        |  circuitgnn/data/batching.py
        v
prebatched shuffle variants on disk
        |  circuitgnn/training/
        v
trained model -> predictions
```

Dataset generation is expensive (SPICE dominates) and happens once. Training reads prebatched pickles, which is why the on-disk graph layout is treated as a frozen schema rather than an implementation detail.

## Graph representation

Circuits become **bipartite graphs**: device terminals and circuit nets are both nodes. An edge joins a terminal to the net it connects to. Topology is therefore given to the network rather than inferred from features.

Node ordering is a contract, not a convenience: terminals come first in netlist order, with each MOSFET's four terminals consecutive as drain, gate, source, bulk, then net nodes sorted alphabetically. Index arithmetic all over the codebase depends on this, for example reaching a MOSFET's bulk as source + 1. `circuitgnn/data/schema.py` is the authoritative description of every attribute, shape and column, including the ones added later by the dataset migrations.

Why bipartite rather than one node per device: currents are a per-terminal quantity and Kirchhoff's law is a per-net statement, so both losses have a natural home. A device-only graph would force one of them into edge features.

## Model

`TowerGENConv` (`circuitgnn/gnn/architectures/tower/`) implements the architecture. The class name is historical: the reported model runs with **both towers disabled**, because the ablation found a single shared backbone beats splitting the last layers into per-output towers. The tower depths stay configurable so that ablation is still reproducible.

The reference configuration is a shared backbone of eight layers at hidden width 128. Each layer applies three gated additive updates in sequence: global self-attention for circuit-wide context, GINE message passing over the device-net edges and loop attention over the circuit's fundamental cycles. Outputs of all eight layers are merged by attention-weighted jumping knowledge and concatenated with a skip connection carrying the raw device and type features.

Four heads read that representation: a linear voltage head, a device-pooled current head, a small-signal head that gathers a transistor's four terminals plus operating-point context and a DC-gain head that refines an analytical estimate.

Loop attention is the component that matters most. Removing it more than doubles the voltage error, more than any other single ablation.

The class is large and stays whole on purpose. Its `state_dict` key names are the interface to every checkpoint saved during the study, so splitting it into subclasses would rename keys and orphan the saved models. Internally it is decomposed: `__init__` calls focused `_build_*` methods and `forward` calls `_forward_*` stage methods, in a fixed order.

The file worth knowing about is `tower/physics.py`. The three-stage DC gain formula lives there as a pure function in log10 space, single-sourced rather than the three copy-pasted inline versions it had before. The Miller UGBW estimate sits beside it, used only by the bandwidth experiments that the thesis reports as unsuccessful.

## Losses

`circuitgnn/training/losses/` splits by kind:

| Module | Contents |
|--------|----------|
| `data_terms.py` | supervised voltage and current losses |
| `kcl.py` | Kirchhoff current law residual at internal nets |
| `device_physics.py` | gm, triode, cutoff and saturation device equations |
| `supervised_heads.py` | gm/gds, AC, DC gain, Vov, Vth and related heads |
| `aggregate.py` | `compute_combined_loss`, which weights and sums everything |

**Only the KCL term is active in the reported model.** The device-physics module is kept because the thesis needs it: its central experiment is showing that every approximate device relation degrades accuracy against a BSIM4 ground truth. Those losses exist to be switched on for that ablation, not to be used. If you are building on this, leave them at weight zero unless you are reproducing that study.

KCL is warmup-scheduled: it stays at zero until epoch 200 and then ramps in over 100 epochs, because a current-conservation residual computed on untrained current predictions is noise. Weights, start epochs and warmup lengths are all config-driven.

## Configuration

One YAML per experiment, covering dataset, architecture, heads, loss weights and schedules, optimizer and seed. The parser accepts both nested and flat legacy spellings, because configs written across the project's life are all still expected to reproduce.

One sharp edge worth knowing: unknown keys do not raise. A mistyped loss weight silently reads as its default, meaning the term is quietly disabled rather than loudly wrong. If an ablation shows no effect, check the key spelling before concluding the term does nothing. The config tests in `tests/test_config.py` exist because a parse failure is one of the few errors this system reports loudly.

## Contracts you must not break

These are external interfaces even though nothing in the language enforces them:

1. **`state_dict` key names.** Module attribute and buffer names are how saved checkpoints are matched. Renaming `self.virtual_node` orphans every checkpoint. Checkpoint loading also infers architecture options by matching substrings of key names, so nesting modules under a new parent changes behavior as well as names.
2. **Dataset attribute names.** Loaders read attributes off unpickled graphs by exact string, guarded with `getattr(..., None)` fallbacks. A renamed attribute does not raise, it silently feeds zeros to the model. Assert loudly rather than relying on the guards.
3. **Config key names.** See above: silent defaults, not errors.
4. **Log line format.** The figure scripts regex-parse `training.log`. Changing a printed metric's wording or precision breaks figure regeneration rather than training.
5. **Module construction order.** Seeded weight initialization consumes RNG in construction order, so reordering module creation changes trained weights even when the architecture is identical.

`tests/golden_forward.py` is the gate for the first and last of these: it compares `state_dict` keys and forward outputs bitwise across a change.

## Extending

**A new circuit topology.** Add a SPICE template pair (DC and AC) under `netlists/`, add a dataset config under `configs/opamp_dataset/` and generate. The graph builder parses the netlist itself, so no model change is needed for a new topology of the same device family. What does need attention is anything holding a hardcoded device table. The mirror-constraint pairs and the signal-path MOSFET indices used by the gain formula are specific to the three-stage topology they were written for. On a topology whose device names do not match they fail quietly.

**A new loss term.** Add the function to the module matching its kind, add its weight to `compute_combined_loss` and to the shared `statically_weighted` list in `aggregate.py` so both weighting branches pick it up. The list exists because those two sums previously drifted and a term went unsummed for an entire ablation.

**A new prediction head.** Add the module in a `_build_*` method and the corresponding `_forward_*` stage, gated on a config flag that defaults to off, so existing checkpoints keep loading.

## Known rough edges

Honest notes for anyone building on this:

- Thermal voltage appears as both 0.026 and 0.02585 in different device physics losses. Preserved deliberately during the refactor so past results stay reproducible, but it should be unified with a documented rerun.
- The uniform devices-per-graph fallback used when pointer tensors are missing is only correct for single-topology batches. Mixed-topology batches need the pointer path.
- Dataset generation reproduces the graph builder's current output, which is not identical to the older datasets that the migration scripts patched in place. `scripts/migrations/` records what was added when.
