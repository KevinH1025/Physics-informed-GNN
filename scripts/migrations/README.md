# One-shot dataset migrations

Every script in this folder is a **one-shot migration that has already been
applied** to the stored datasets under `datasets/`. They are kept for
provenance only: they document how each field in the pickled graphs came to
exist, and they let a dataset regenerated from scratch be brought up to the
same schema. None of them is part of the training or evaluation path, and
re-running one against a dataset that already carries the field is either a
no-op or a corruption risk. Most are idempotent by construction but not all of
them state so, so read the script before you run it.

The logic these scripts encode has since been folded into
`circuitgnn/data/graph_builder.py`, so freshly generated datasets already carry
the fields below.

## Node feature lineage (`graph.x` and per-node tensors)

| Step | Script | What it added |
|------|--------|---------------|
| 1 | `patch_dataset_add_m.py` | M (multiplier) as a 4th per-device property, taking `x.shape[1]` from 8 to 9 (4 device props W, L, W/L, M plus 5 global features). Effective per-device M is recomputed from group params and the netlist template multipliers, since raw netlists are not stored in the dataset. |
| 2 | `patch_dataset_log_norm.py` | Re-normalized `x[:, 0]` (W) and `x[:, 1]` (L) to log scale. The original linear min-max normalization crammed 50% of log-uniformly sampled values into 8% of the feature range. |
| 3 | `patch_dataset_net_role.py` | `batch.net_role` `[N, 5]` one-hot over VDD, GND, SIG_IN, SIG_OUT and INTERNAL, classified from the node name. Terminal nodes stay all-zero because they carry a terminal role instead. |
| 4 | `patch_dataset_lut_op_features.py` | `graph.node_lut_features` `[N, 3]`: a frozen pass-1 model predicts V0, the per-MOSFET bias (Vgs, Vds, Vbs) is derived from it, the SKY130 LUT is queried for physics-exact (id, gm, gds), and the log10 z-scored triple is scattered to the four terminal nodes of each MOSFET. |
| 5 | `patch_dataset_stack_features.py` | `graph.node_stack_features` `[N, 4]` = (V, I, gm, gds) in z-score space. Same pass-1 forward-stacking idea as step 4 but using the baseline's own predictions instead of a LUT query. |

## Edge attribute lineage (`graph.edge_attr`)

The builder emits a 6-dim one-hot of the source-side terminal type
(gate, drain, source, bulk, p, n). Two migrations extend it:

| Dims | Script | Meaning |
|------|--------|---------|
| 0-5 | (graph builder) | terminal type one-hot |
| 6 | `patch_dataset_edge_features.py` | `loop_flag`: destination edge is in `loop_edge_index` |
| 7 | `patch_dataset_edge_features.py` | `bound_flag`: destination node has a known voltage (VDD/GND/V-source) |
| 8 | `patch_dataset_edge_features.py` | `out_flag`: destination node is an output node |
| 9 | `patch_dataset_edge_features.py` | `cap_flag`: destination node is a capacitor terminal (AC-coupled) |
| 10 | `patch_dataset_edge_current_sign.py` | `current_sign`: +1 drain, -1 source, 0 for gate/bulk/cap and for non-physical terminal-to-terminal or net-to-net edges |

Final shape is `[E, 11]`. `patch_dataset_edge_features.py` is idempotent: it
only appends when `edge_attr` is exactly 6-dim, and it must run after the
dataset has been prebatched.

## Per-MOSFET tensors

| Script | What it added |
|--------|---------------|
| `patch_dataset_mosfet_wl.py` | `graph.mosfet_wl_um` `[M, 2]` in micrometres and `graph.mosfet_terminal_idx` `[M, 4]` (G, D, S, B), both aligned with `mosfet_info`. Without these the LUT and the IV-embedder cannot be driven. Also re-runs `create_prebatched_dataset` for every split so the `variant_*.pkl` files stay in lockstep. |
| `patch_dataset_mosfet_role.py` | `mosfet_role` `[M, 7]` functional one-hot for the fan_smc / openloop topology: BIASCM_P, BIASCM_N, GM1, GM2, GMF2, LOAD2 and GM3. |
| `patch_dataset_mosfet_descriptor.py` | `mosfet_descriptor` `[M, 5]`: log10(Id), log10(gm), log10(gds) at the canonical strong-inversion bias (Vgs=0.7, Vds=0.9, Vbs=0), Vth at Vbs=0 and a weak-inversion log10(gm) at (Vgs=0.5, Vds=0.5, Vbs=0). |
| `patch_dataset_physics.py` | `mosfet_vth` and `node_mosfet_vth`, read from `specs['_mosfet_regions']` in the raw pkl files and matched to prebatched graphs by x-feature fingerprint. No SPICE re-run. |
| `patch_2stage_dataset.py` | The 2-stage equivalent: `mosfet_vth`, `node_mosfet_vth`, `mosfet_drain_mask`, `node_log_gm` and `node_log_gds`. |
| `patch_terminal_currents.py` | Replaced the single shared current magnitude on every terminal with the actual per-device \|Id\| from SPICE `_all_device_currents`. The original scheme broke KCL validation. |

## Corpus assembly and re-batching

| Script | Purpose |
|--------|---------|
| `build_5topo_corpus.py` | Merges `opamp_3stage_pretrain_combined` (sau_cfcc, peng_tcfc, leung_nmcf, leung_nmcnr) with `opamp_3stage_fan_smc_openloop_5k` (fan_smc) into `datasets/opamp_3stage_pretrain_combined_5topo`, adding the `topology` field. |
| `patch_pretrain_combined.py` | Backfills `mosfet_wl_um`, `mosfet_m` and `mosfet_terminal_idx` onto the 4-topology combined pretrain set by parsing each topology's `.sp` template for the device-to-group mapping and per-device M prefix. |
| `rebatch_dataset.py` | Adds `terminal_train_mask` and `terminal_vdc` to raw samples and regenerates the prebatched variants, avoiding a SPICE re-run. |
| `build_mosfet_lut.py` | Not a dataset patch. Builds the SKY130 5-D MOSFET DC operating-point LUT keyed by (W, L, Vbs, Vds, Vgs) per polarity, which steps 4 and 5 above and `mosfet_descriptor` all query. Output is an HDF5 file under `datasets/lut/`. |
