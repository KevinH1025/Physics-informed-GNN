#!/usr/bin/env python
"""Plot pretraining curves from a training.log file.

Parses lines of the form:
    epoch <N>: train_loss=<X> [lr=<X>] | val_loss=<X> val_mae=<X> [BEST]

Saves training_curves.png next to the log.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


_EPOCH_RE = re.compile(
    r'epoch\s+(\d+):\s*train_loss=([\d.eE+\-]+)'
    r'(?:\s+lr=([\d.eE+\-]+))?'
    r'(?:\s*\|\s*val_loss=([\d.eE+\-]+)\s+val_mae=([\d.eE+\-]+))?'
    r'(?:\s+\[BEST\])?'
)


def parse(log_path: Path):
    epochs, train, val_loss, val_mae, lrs, bests = [], [], [], [], [], []
    with open(log_path) as f:
        for line in f:
            m = _EPOCH_RE.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            tr = float(m.group(2))
            lr = float(m.group(3)) if m.group(3) else None
            vl = float(m.group(4)) if m.group(4) else None
            vmae = float(m.group(5)) if m.group(5) else None
            epochs.append(ep)
            train.append(tr)
            lrs.append(lr)
            val_loss.append(vl)
            val_mae.append(vmae)
            bests.append('[BEST]' in line)
    return epochs, train, val_loss, val_mae, lrs, bests


def plot(log_path: Path, out_path: Path):
    epochs, train, val_loss, val_mae, lrs, bests = parse(log_path)
    if not epochs:
        raise RuntimeError(f'No parseable lines in {log_path}')

    has_lr = any(l is not None for l in lrs)
    has_val = any(v is not None for v in val_loss)

    n_panels = 1 + (1 if has_val else 0) + (1 if has_lr else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    ax_iter = iter(axes)

    # Loss panel
    ax = next(ax_iter)
    ax.plot(epochs, train, label='train_loss', color='C0', alpha=0.8)
    if has_val:
        v_ep = [e for e, v in zip(epochs, val_loss) if v is not None]
        v_va = [v for v in val_loss if v is not None]
        ax.plot(v_ep, v_va, label='val_loss', color='C1', alpha=0.8)
        # Mark best epochs
        best_ep = [e for e, b, v in zip(epochs, bests, val_loss) if b and v is not None]
        best_v = [v for b, v in zip(bests, val_loss) if b and v is not None]
        if best_ep:
            ax.scatter(best_ep, best_v, marker='*', color='red', s=40, zorder=5,
                       label=f'[BEST] (n={len(best_ep)})')
    ax.set_ylabel('MSE loss')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title(f'Pretraining curves — {log_path.parent.name}')

    # Val MAE panel
    if has_val:
        ax = next(ax_iter)
        v_ep = [e for e, v in zip(epochs, val_mae) if v is not None]
        v_va = [v for v in val_mae if v is not None]
        ax.plot(v_ep, v_va, color='C2')
        ax.set_ylabel('val_mae')
        ax.grid(alpha=0.3)

    # LR panel
    if has_lr:
        ax = next(ax_iter)
        lr_ep = [e for e, l in zip(epochs, lrs) if l is not None]
        lr_v = [l for l in lrs if l is not None]
        ax.plot(lr_ep, lr_v, color='C3')
        ax.set_ylabel('learning rate')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('epoch')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f'wrote {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', type=Path, help='Path to training.log')
    ap.add_argument('--out', type=Path, default=None,
                    help='Output PNG path (default: training_curves.png next to log)')
    args = ap.parse_args()
    out = args.out or args.log.parent / 'training_curves.png'
    plot(args.log, out)


if __name__ == '__main__':
    main()
