"""Evaluation helpers shared by the scripts under ``scripts/``.

Previously these lived inside one-off scripts and were pulled in by bare
sibling imports (``from eval_all_checkpoints import attach_norm``). They are
importable as a normal package now, so the scripts no longer have to place
their own directory on ``sys.path``.
"""
from .checkpoints import (
    TOPOS,
    _drain_idx_batched,
    attach_norm,
    eval_one,
)
from .paths import DATASET, EXP, REPO

__all__ = [
    'REPO', 'DATASET', 'EXP',
    'TOPOS', 'attach_norm', '_drain_idx_batched', 'eval_one',
]
