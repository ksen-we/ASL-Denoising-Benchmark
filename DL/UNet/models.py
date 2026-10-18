"""U-Net with same padding, GroupNorm, and linear regression output."""
from math import prod

import torch
from torch import nn
from torch.nn import functional as F
from common.output import configure_output, image_output


def group_norm(channels):
    """Use at least four channels per group for small-batch training."""
    groups = min(8, max(1, channels // 4))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DoubleConv(nn.Module):
    """Apply two 3x3 convolutions, each followed by GroupNorm and ReLU."""
    def __init__(self, ndim, in_channels, out_channels):
        super().__init__()
        conv = nn.Conv3d if ndim == 3 else nn.Conv2d
        self.layers = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            group_norm(out_channels), nn.ReLU(inplace=True),
            conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            group_norm(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNetDenoiser(nn.Module):
    """Double-convolution encoder-decoder with pooling and transposed-convolution upsampling."""
    def __init__(self, ndim=3, in_channels=1, base_channels=32, levels=4,
                 downsample_factors=None, residual=False,
                 global_residual=None, zero_init_output=None):
        super().__init__()
        if ndim not in (2, 3) or base_channels < 4 or levels < 2:
            raise ValueError("UNet requires ndim=2/3, base_channels>=4, and levels>=2")
        if downsample_factors is None:
            downsample_factors = ([(1, 2, 2)] + [(2, 2, 2)] * (levels - 2)
                                  if ndim == 3 else [(2, 2)] * (levels - 1))
        factors = tuple(tuple(f) for f in downsample_factors)
        if len(factors) != levels - 1 or any(len(f) != ndim or any(v not in (1, 2) for v in f) for f in factors):
            raise ValueError("downsample_factors requires levels-1 entries of ndim factors, each 1 or 2")
        self.ndim = ndim
        self.in_channels = in_channels
        self.multiple = tuple(prod(f[axis] for f in factors) for axis in range(ndim))
        conv = nn.Conv3d if ndim == 3 else nn.Conv2d
        pool = nn.MaxPool3d if ndim == 3 else nn.MaxPool2d
        trans = nn.ConvTranspose3d if ndim == 3 else nn.ConvTranspose2d
        widths = [base_channels * 2 ** i for i in range(levels)]
        self.encoder = nn.ModuleList([DoubleConv(ndim, in_channels if i == 0 else widths[i - 1], c)
                                      for i, c in enumerate(widths)])
        self.pool = nn.ModuleList([pool(kernel_size=f, stride=f) for f in factors])
        self.up = nn.ModuleList([trans(widths[i + 1], widths[i], kernel_size=factors[i], stride=factors[i])
                                for i in reversed(range(levels - 1))])
        self.decoder = nn.ModuleList([DoubleConv(ndim, 2 * widths[i], widths[i])
                                     for i in reversed(range(levels - 1))])

        self.output = conv(widths[0], 1, kernel_size=1)
        configure_output(self, self.output, ndim, in_channels, global_residual,
                         zero_init_output, legacy_residual=residual, legacy_zero_init=residual)
        self.residual = self.global_residual

    def forward(self, x):
        if x.ndim != self.ndim + 2:
            raise ValueError(f"UNet expects {self.ndim + 2} dimensions, got {x.ndim}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"UNet expects {self.in_channels} input channels, got {x.shape[1]}")
        source = x
        shape = x.shape[2:]

        padding = tuple(v for n, m in reversed(list(zip(shape, self.multiple))) for v in (0, (-n) % m))
        x = F.pad(x, padding)
        skips = []
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)
            if i < len(self.pool):
                skips.append(x)
                x = self.pool[i](x)
        for up, decoder, skip in zip(self.up, self.decoder, reversed(skips)):
            x = decoder(torch.cat([up(x), skip], dim=1))
        output = self.output(x)
        output = output[(slice(None), slice(None)) + tuple(slice(0, n) for n in shape)]
        return image_output(self, output, source)
