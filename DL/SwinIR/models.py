"""Shallow convolution, residual Swin groups, and image reconstruction."""
from torch import nn
from common.swin import SwinStage, conv_nd
from common.output import configure_output, image_output


class ResidualSwinGroup(nn.Module):
    def __init__(self, ndim, dim, depth, window, heads, mlp_ratio, use_checkpoint):
        super().__init__()
        self.stage = SwinStage(dim, depth, window, heads, mlp_ratio, use_checkpoint)
        self.conv = conv_nd(ndim)(dim, dim, 3, padding=1)

    def forward(self, x):
        return x + self.conv(self.stage(x))


class SwinIRDenoiser(nn.Module):
    def __init__(self, ndim=3, in_channels=1, embed_dim=60, depths=(6, 6, 6, 6, 6, 6),
                 heads=6, window=(2, 6, 6), mlp_ratio=2.0, use_checkpoint=True,
                 global_residual=None, zero_init_output=None):
        super().__init__()
        conv = conv_nd(ndim)
        self.shallow = conv(in_channels, embed_dim, 3, padding=1)
        self.groups = nn.Sequential(*[ResidualSwinGroup(ndim, embed_dim, d, window, heads, mlp_ratio, use_checkpoint) for d in depths])
        self.deep = conv(embed_dim, embed_dim, 3, padding=1)
        self.output = conv(embed_dim, 1, 3, padding=1)

        configure_output(self, self.output, ndim, in_channels, global_residual,
                         zero_init_output, legacy_residual=True, legacy_zero_init=True)

    def forward(self, x):
        shallow = self.shallow(x)
        output = self.output(shallow + self.deep(self.groups(shallow)))
        return image_output(self, output, x)
