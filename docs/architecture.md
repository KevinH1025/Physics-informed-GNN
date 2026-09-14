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

`TowerGENConv` (`circuitgnn/gnn/architectures/tower/`) is the current architecture. A shared message-passing backbone feeds two specialized towers:

- **State tower** predicts quantities that describe the operating point: node voltages and terminal currents.
- **Sensitivity tower** predicts small-signal quantities: gm and gds per MOSFET, which then feed the analytic gain formula.

Around the backbone sit a virtual node for global context, optional loop attention over circuit cycles found by cycle-basis decomposition and jumping-knowledge aggregation across layers. Every one of these is a config switch, because the thesis ablations turn them on and off individually.

The class is large and stays whole on purpose. Its `state_dict` key names are the interface to every checkpoint saved during the study, so splitting it into subclasses would rename keys and orphan the saved models. Internally it is decomposed: `__init__` calls focused `_build_*` methods and `forward` calls `_forward_*` stage methods, in a fixed order.

The single file it is worth knowing about is `tower/physics.py`. The three-stage DC gain formula and the Miller UGBW estimate live there as pure functions in log10 space. These formulas are the physics contribution, so they get one tested home rather than the three copy-pasted inlined versions they had before.

## Losses

`circuitgnn/training/losses/` splits by kind:

| Module | Contents |
|--------|----------|
| `data_terms.py` | supervised voltage and current losses |
| `kcl.py` | Kirchhoff current law residual at internal nets |
| `device_physics.py` | gm, triode, cutoff and saturation device equations |
| `supervised_heads.py` | gm/gds, AC, DC gain, Vov, Vth and related heads |
| `aggregate.py` | `compute_combined_loss`, which weights and sums everything |

Physics terms are typically warmup-scheduled: they switch on after the data terms have found a reasonable operating point, because a physics residual computed on noise is not informative. Weights, start epochs and warmup lengths are all config-driven.

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
