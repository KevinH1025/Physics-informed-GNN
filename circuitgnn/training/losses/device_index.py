"""Batched device-to-graph index helpers shared by loss functions and model code."""

import torch


def get_device_graph_idx(num_devices: int, num_graphs: int,
                         device_ptr: torch.Tensor = None,
                         device: torch.device = None) -> torch.Tensor:
    """Get graph assignment index for each device, supporting variable counts per graph.

    Args:
        num_devices: Total number of devices across all graphs in the batch.
        num_graphs: Number of graphs in the batch.
        device_ptr: Optional [num_graphs + 1] cumulative size tensor.
            If provided, uses bucketize for variable-size support.
            If None, falls back to uniform integer division (legacy behavior).
        device: Torch device for output tensor.

    Returns:
        Tensor [num_devices] mapping each device to its graph index.
    """
    if device_ptr is not None:
        return torch.bucketize(
            torch.arange(num_devices, device=device),
            device_ptr[1:].to(device),
            right=True,
        )
    # Fallback: uniform (backward compat for single-topology batches)
    devices_per_graph = num_devices // num_graphs
    return torch.arange(num_devices, device=device) // devices_per_graph


def _batch_offsets(num_devices, device_ptr, ptr, device):
    """Compute per-device node offsets for batched graphs."""
    if device_ptr is not None:
        graph_idx = torch.bucketize(
            torch.arange(num_devices, device=device),
            device_ptr[1:].to(device), right=True)
    else:
        num_graphs = len(ptr) - 1
        devices_per_graph = num_devices // num_graphs
        graph_idx = torch.arange(num_devices, device=device) // devices_per_graph
    return ptr[graph_idx]
