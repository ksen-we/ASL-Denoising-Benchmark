"""Pix2Pix adapted to small ASL volumes with GroupNorm and linear output."""
import torch
from torch import nn
from torch.nn import functional as F
from common.swin import conv_nd, crop_spatial, pad_spatial
from common.output import configure_output, image_output


def norm(channels):

    groups = min(8, max(1, channels // 4))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class UNetGenerator(nn.Module):
    def __init__(self, ndim=3, in_channels=1, base_channels=32, dropout=0.5,
                 global_residual=None, zero_init_output=None):
        super().__init__()
        conv = conv_nd(ndim)
        trans = nn.ConvTranspose3d if ndim == 3 else nn.ConvTranspose2d
        strides = [(1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)] if ndim == 3 else [(2, 2)] * 4
        self.multiple = (8, 16, 16) if ndim == 3 else (16, 16)
        widths = [base_channels * k for k in (1, 2, 4, 8)]
        self.down = nn.ModuleList()
        prev = in_channels
        for i, (width, stride) in enumerate(zip(widths, strides)):
            kernel = tuple(4 if s == 2 else 3 for s in stride)
            layers = [conv(prev, width, kernel, stride=stride, padding=1)]
            if i > 0:
                layers.append(norm(width))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.down.append(nn.Sequential(*layers))
            prev = width
        self.up = nn.ModuleList()
        for i in (3, 2, 1):
            stride = strides[i]
            layers = [nn.ReLU(), trans(prev, widths[i - 1], 4, stride=stride, padding=1), norm(widths[i - 1])]
            if i >= 2 and dropout:
                layers.append(nn.Dropout(dropout))
            self.up.append(nn.Sequential(*layers))
            prev = widths[i - 1] * 2
        stride = strides[0]
        kernel = tuple(4 if s == 2 else 3 for s in stride)
        self.output = nn.Sequential(nn.ReLU(), trans(prev, 1, kernel, stride=stride, padding=1))
        configure_output(self, self.output[-1], ndim, in_channels, global_residual, zero_init_output)

    def forward(self, x):
        source = x
        x, shape = pad_spatial(x, self.multiple)
        skips = []
        for layer in self.down:
            x = layer(x)
            skips.append(x)
        for layer, skip in zip(self.up, reversed(skips[:-1])):
            x = torch.cat([layer(x), skip], dim=1)
        return image_output(self, crop_spatial(self.output(x), shape), source)


class PatchDiscriminator(nn.Module):
    """Map concatenated source and candidate images to unsquashed patch logits."""
    def __init__(self, ndim=3, in_channels=1, base_channels=32):
        super().__init__()
        conv = conv_nd(ndim)
        layers, prev = [], in_channels + 1
        strides = [(1, 2, 2), (2, 2, 2), (1, 1, 1)] if ndim == 3 else [(2, 2), (2, 2), (1, 1)]
        for i, stride in enumerate(strides):
            width = base_channels * 2 ** i
            kernel = tuple(4 if s == 2 else 3 for s in stride)
            layers.append(conv(prev, width, kernel, stride=stride, padding=1))
            if i:
                layers.append(norm(width))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev = width
        layers.append(conv(prev, 1, 3, padding=1))
        self.network = nn.Sequential(*layers)


        receptive_field, jump, padding = [1] * ndim, [1] * ndim, [0] * ndim
        for layer in self.network:
            if isinstance(layer, (nn.Conv2d, nn.Conv3d)):
                if any(d != 1 for d in layer.dilation) or layer.padding_mode != "zeros":
                    raise ValueError("Foreground mapping requires the current nondilated, zero-padded convolution structure")
                for axis in range(ndim):
                    receptive_field[axis] += (layer.kernel_size[axis] - 1) * jump[axis]
                    padding[axis] += layer.padding[axis] * jump[axis]
                    jump[axis] *= layer.stride[axis]
        self.receptive_field = tuple(receptive_field)
        self._mask_stride, self._mask_padding = tuple(jump), tuple(padding)
        self.register_buffer("_mask_kernel", torch.ones(1, 1, *receptive_field), persistent=False)

    @torch.no_grad()
    def foreground_weights(self, mask):
        """Compute geometric convolution-support foreground fractions, excluding GroupNorm coupling."""
        ndim = len(self.receptive_field)
        if mask.ndim != ndim + 2 or mask.shape[1] != 1:
            raise ValueError("PatchGAN masks must be [B,1,H,W] or [B,1,D,H,W], matching discriminator dimensions")
        with torch.autocast(device_type=mask.device.type, enabled=False):
            m = mask.detach().float()
            if not torch.isfinite(m).all() or (m < 0).any() or (m > 1).any():
                raise ValueError("PatchGAN masks must be finite and within [0,1]")
            conv = F.conv3d if ndim == 3 else F.conv2d
            kernel = self._mask_kernel.float()
            foreground = conv(m, kernel, stride=self._mask_stride, padding=self._mask_padding)

            coverage = conv(torch.ones_like(m[:1]), kernel, stride=self._mask_stride, padding=self._mask_padding)
            return foreground / coverage.clamp_min(1)

    def forward(self, condition, image):
        return self.network(torch.cat([condition, image], dim=1))
