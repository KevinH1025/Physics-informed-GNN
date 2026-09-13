"""
SKY130 MOSFET LUT components.

Shared pipeline for the HDF5 device tables:
- io: array transforms (axis reorientation, log10 flooring, Vbs magnitude
  sorting, polarity stacking)
- interp: differentiable axis bracketing and quintlinear interpolation
- iv_embedder: frozen IV-surface autoencoder embedder (IVEmbedder)
- lut_id_query: differentiable |id| lookup (LUTIdQuery)
- lut_op_query: differentiable (id, gm, gds) lookup (LUTOpQuery)

The class submodules import h5py at module scope, so import them directly
(e.g. circuitgnn.gnn.components.lut.lut_id_query) as needed rather than
through eager package imports.
"""
