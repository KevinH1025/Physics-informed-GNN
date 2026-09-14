"""Repository path resolution for evaluation helpers.

Everything is derived from the location of this file, so the package works
from a checkout at any prefix instead of the author's original absolute path.
"""
from __future__ import annotations

from pathlib import Path

# circuitgnn/evaluation/paths.py -> circuitgnn/evaluation -> circuitgnn -> <repo root>
REPO = Path(__file__).resolve().parents[2]

DATASET = REPO / 'datasets/opamp_3stage_pretrain_combined_5topo'
EXP = DATASET / 'experiments'

__all__ = ['REPO', 'DATASET', 'EXP']
