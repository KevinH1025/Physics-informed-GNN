#!/usr/bin/env python
"""Plot multiple openloop training logs on one figure for comparison.

Parses lines like:
    Epoch  50 | LR=3.00e-03 | BestLoss=0.4826 | Score=66.1 (best=89.8, ep=44) | ...

Tracks BestLoss and Score-best per run, plots both panels.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


_LINE_RE = re.compile(
    r'Epoch\s+(\d+)\s*\|\s*LR=([\d.eE+\-]+)\s*\|\s*BestLoss=([\d.eE+\-]+)\s*\|'
    r'\s*Score=([\d.\-eE+]+)\s+\(best=([\d.\-eE+]+),\s*ep=(\d+)\)'
)


def parse(log_path: Path):
    eps, best_loss, best_score, lrs = [], [], [], []
    with open(log_path) as f:
        for line in f:
            m = _LINE_RE.search(line)
            if not m:
                continue
            eps.append(int(m.group(1)))
            lrs.append(float(m.group(2)))
            best_loss.append(float(m.group(3)))
            best_score.append(float(m.group(5)))
    return eps, best_loss, best_score, lrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='paths to training.log files')
    ap.add_argument('--out', default='openloop_compare.png')
    ap.add_argument('--labels', nargs='+', default=None,
                    help='labels per run (default: parent dir name)')
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        raise ValueError('labels must match number of runs')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i, run in enumerate(args.runs):
        run = Path(run)
        eps, bl, bs, lrs = parse(run)
        if not eps:
            print(f'no parseable lines in {run}')
            continue
        label = args.labels[i] if args.labels else run.parent.name
        # Shorten very long names
        if len(label) > 50:
            label = '...' + label[-47:]
        ax1.plot(eps, bl, label=label, linewidth=1.5)
        ax2.plot(eps, bs, label=label, linewidth=1.5)

    ax1.set_ylabel('Best Val Loss')
    ax1.set_yscale('log')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title('Openloop training comparison')

    ax2.set_ylabel('Best Val Score')
    ax2.set_xlabel('Epoch')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='lower right', fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
