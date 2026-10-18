"""HUST-inspired Swin encoder-decoder with patch merging and U-shaped skips."""
import torch
from torch import nn
from common.swin import DualUpsample, PatchMerging, SwinStage, conv_nd
from common.output import configure_output, image_output


class SwinUNetDenoiser(nn.Module):
    def __init__(self, ndim=3, in_channels=1, embed_dim=48, depths=(2, 2, 2),
                 heads=(3, 6, 12), window=(2, 8, 8), mlp_ratio=4.0,
                 use_checkpoint=True, downsample_factors=None,
                 global_residual=None, zero_init_output=None):
        super().__init__()
        levels = len(depths)
        if levels < 2 or len(heads) != levels:
            raise ValueError("SwinUNet depths/heads must have equal lengths and at least two levels")
        if downsample_factors is None:
            downsample_factors = [(1, 2, 2)] + [(2, 2, 2)] * (levels - 2) if ndim == 3 else [(2, 2)] * (levels - 1)
        if len(downsample_factors) != levels - 1:
            raise ValueError("downsample_factors must have one fewer entry than the number of levels")
        conv = conv_nd(ndim)
        widths = [embed_dim * 2 ** i for i in range(levels)]
        self.stem = conv(in_channels, embed_dim, 3, padding=1)
        self.encoder = nn.ModuleList([SwinStage(c, d, window, h, mlp_ratio, use_checkpoint) for c, d, h in zip(widths, depths, heads)])
        self.down = nn.ModuleList([PatchMerging(widths[i], downsample_factors[i]) for i in range(levels - 1)])
        self.up = nn.ModuleList([DualUpsample(ndim, widths[i + 1], widths[i], downsample_factors[i]) for i in reversed(range(levels - 1))])
        self.fuse = nn.ModuleList([conv(2 * widths[i], widths[i], 1) for i in reversed(range(levels - 1))])
        self.decoder = nn.ModuleList([SwinStage(widths[i], depths[i], window, heads[i], mlp_ratio, use_checkpoint) for i in reversed(range(levels - 1))])
        self.output = conv(embed_dim, 1, 3, padding=1)
        configure_output(self, self.output, ndim, in_channels, global_residual, zero_init_output)

    def forward(self, x):
        features, skips = self.stem(x), []
        for i, stage in enumerate(self.encoder):
            features = stage(features)
            if i < len(self.down):
                skips.append(features)
                features = self.down[i](features)
        for up, fuse, stage, skip in zip(self.up, self.fuse, self.decoder, reversed(skips)):
            features = stage(fuse(torch.cat([up(features, skip.shape[2:]), skip], dim=1)))
        return image_output(self, self.output(features), x)
