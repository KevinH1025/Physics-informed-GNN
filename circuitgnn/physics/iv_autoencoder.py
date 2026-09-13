"""Compact convolutional autoencoder for MOSFET IV surfaces.

Straight Conv→BN→ReLU encoder / ConvTranspose→BN→ReLU decoder, no residuals.
Compresses a log10|I_D|(Vgs, Vds) surface (181 × 361) to a bottleneck-dim
embedding.  This is the single-conv-per-stage architecture that was best in
the bottleneck / data sweeps (197 K params at b=64, 7 conv layers).

Surface is padded symmetrically to 192 × 384, then 6 stride-2 convs reach
64 × 3 × 6, and a final asymmetric Conv(3,6) collapses to bottleneck × 1 × 1.
Decoder mirrors with a learned ConvTranspose(3,6) expansion.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────── Target padded shape ──────────────────────────
INPUT_H, INPUT_W = 181, 361
PADDED_H, PADDED_W = 192, 384
PAD_H = PADDED_H - INPUT_H    # 11
PAD_W = PADDED_W - INPUT_W    # 23
PAD_T = PAD_H // 2            # 5
PAD_B = PAD_H - PAD_T         # 6
PAD_L = PAD_W // 2            # 11
PAD_R = PAD_W - PAD_L         # 12


def pad_to_padded(x: torch.Tensor) -> torch.Tensor:
    return F.pad(x, (PAD_L, PAD_R, PAD_T, PAD_B), mode='replicate')


def crop_center(x: torch.Tensor) -> torch.Tensor:
    return x[..., PAD_T:PAD_T + INPUT_H, PAD_L:PAD_L + INPUT_W]


def _conv_bn_relu(in_ch, out_ch, k, s, p):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _convT_bn_relu(in_ch, out_ch, k, s, p, opad):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s,
                           padding=p, output_padding=opad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ─────────────────────────────── Encoder ────────────────────────────────
class Encoder(nn.Module):
    """(B, 1, 181, 361) → (B, bottleneck).

    Channels double every stride-2 stage. Last stage matches `bottleneck`:
      64-dim:  1 → 2 → 4 → 8 → 16 → 32 → 64
      128-dim: 1 → 4 → 8 → 16 → 32 → 64 → 128
    """

    def __init__(self, bottleneck: int = 64):
        super().__init__()
        self.bottleneck = bottleneck
        assert bottleneck in (64, 128), f'Supported: 64, 128; got {bottleneck}'
        base = bottleneck // 32   # 2 for 64-dim, 4 for 128-dim
        c = [base * (2 ** i) for i in range(6)]

        self.net = nn.Sequential(
            _conv_bn_relu(1,    c[0], k=5, s=2, p=2),
            _conv_bn_relu(c[0], c[1], k=3, s=2, p=1),
            _conv_bn_relu(c[1], c[2], k=3, s=2, p=1),
            _conv_bn_relu(c[2], c[3], k=3, s=2, p=1),
            _conv_bn_relu(c[3], c[4], k=3, s=2, p=1),
            _conv_bn_relu(c[4], c[5], k=3, s=2, p=1),
            nn.Conv2d(c[5], bottleneck, kernel_size=(3, 6), stride=(3, 6),
                      padding=0, bias=False),
            nn.BatchNorm2d(bottleneck),
        )

    def forward(self, x):
        x = pad_to_padded(x)
        x = self.net(x)                  # (B, bottleneck, 1, 1)
        return x.flatten(1)              # (B, bottleneck)


# ─────────────────────────────── Decoder ────────────────────────────────
class Decoder(nn.Module):
    """(B, bottleneck) → (B, 1, 181, 361). Mirrors Encoder's channel schedule."""

    def __init__(self, bottleneck: int = 64):
        super().__init__()
        self.bottleneck = bottleneck
        assert bottleneck in (64, 128), f'Supported: 64, 128; got {bottleneck}'
        base = bottleneck // 32
        c = [base * (2 ** i) for i in range(6)]   # ascending; we consume descending

        self.net = nn.Sequential(
            _convT_bn_relu(bottleneck, c[5], k=(3, 6), s=(3, 6), p=0, opad=0),
            _convT_bn_relu(c[5], c[4], k=3, s=2, p=1, opad=1),
            _convT_bn_relu(c[4], c[3], k=3, s=2, p=1, opad=1),
            _convT_bn_relu(c[3], c[2], k=3, s=2, p=1, opad=1),
            _convT_bn_relu(c[2], c[1], k=3, s=2, p=1, opad=1),
            _convT_bn_relu(c[1], c[0], k=3, s=2, p=1, opad=1),
            nn.ConvTranspose2d(c[0], 1, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=True),
        )

    def forward(self, z):
        x = z.view(-1, self.bottleneck, 1, 1)
        x = self.net(x)
        return crop_center(x)


# ─────────────────────────── Autoencoder wrapper ────────────────────────
class IVAutoencoder(nn.Module):
    def __init__(self, bottleneck: int = 64):
        super().__init__()
        self.encoder = Encoder(bottleneck)
        self.decoder = Decoder(bottleneck)

    def forward(self, x):
        return self.decoder(self.encoder(x))
