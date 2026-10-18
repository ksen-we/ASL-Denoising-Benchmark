"""Windowed and shifted-window attention for 2D and 3D tensors."""
from functools import lru_cache
from itertools import product
from math import prod

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def conv_nd(ndim):
    return nn.Conv3d if ndim == 3 else nn.Conv2d


def pad_spatial(x, multiples):
    """Pad the right side of spatial axes and return the original shape."""
    shape = x.shape[2:]
    padding = tuple(v for n, w in reversed(list(zip(shape, multiples))) for v in (0, (-n) % w))
    return F.pad(x, padding), shape


def crop_spatial(x, shape):
    return x[(slice(None), slice(None)) + tuple(slice(0, n) for n in shape)]


def partition_windows(x, window):
    """B,*spatial,C -> B*nWindows,nTokens,C。"""
    ndim = len(window)
    grid = [n // w for n, w in zip(x.shape[1:-1], window)]
    shape = [x.shape[0]] + [v for pair in zip(grid, window) for v in pair] + [x.shape[-1]]
    order = [0] + list(range(1, 2 * ndim, 2)) + list(range(2, 2 * ndim + 1, 2)) + [2 * ndim + 1]
    return x.reshape(shape).permute(order).reshape(-1, prod(window), x.shape[-1])


def reverse_windows(windows, window, spatial, batch):
    ndim = len(window)
    grid = [n // w for n, w in zip(spatial, window)]
    x = windows.reshape(batch, *grid, *window, windows.shape[-1])
    order = [0] + [v for pair in zip(range(1, ndim + 1), range(ndim + 1, 2 * ndim + 1)) for v in pair] + [2 * ndim + 1]
    return x.permute(order).reshape(batch, *spatial, windows.shape[-1])


@lru_cache(maxsize=16)
def attention_mask(original, padded, window, shift, device):
    """Exclude padded tokens and artificial cyclic-shift neighbors."""
    region = torch.zeros((1, *padded, 1), device=device)
    slices = [(slice(0, -w), slice(-w, -s), slice(-s, None)) if s else (slice(None),)
              for w, s in zip(window, shift)]
    for index, section in enumerate(product(*slices)):
        region[(slice(None),) + section + (slice(None),)] = index
    ids = partition_windows(region, window).squeeze(-1)
    blocked = ids.unsqueeze(1) != ids.unsqueeze(2)
    valid = torch.zeros_like(region, dtype=torch.bool)
    valid[(slice(None),) + tuple(slice(0, n) for n in original) + (slice(None),)] = True
    if any(shift):
        valid = torch.roll(valid, tuple(-s for s in shift), dims=tuple(range(1, len(window) + 1)))
    keys = partition_windows(valid, window).squeeze(-1)
    return blocked | ~keys.unsqueeze(1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window, heads):
        super().__init__()
        if dim % heads:
            raise ValueError("Channel count must be divisible by the number of attention heads")
        self.heads, self.window = heads, tuple(window)
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(torch.zeros(prod(2 * w - 1 for w in window), heads))
        coords = torch.stack(torch.meshgrid(*(torch.arange(w) for w in window), indexing="ij")).flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        index = torch.zeros(relative.shape[1:], dtype=torch.long)
        for axis, size in enumerate(window):
            index = index * (2 * size - 1) + relative[axis] + size - 1
        self.register_buffer("relative_index", index, persistent=False)
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, x, mask):
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, tokens, 3, self.heads, channels // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_bias[self.relative_index.reshape(-1)].reshape(tokens, tokens, self.heads).permute(2, 0, 1)
        scores = scores + bias.to(scores.dtype)[None]
        nwin = mask.shape[0]
        scores = scores.reshape(-1, nwin, self.heads, tokens, tokens)


        scores = scores.masked_fill(mask[None, :, None], -10000.0).reshape(batch_windows, self.heads, tokens, tokens)
        attn = scores.float().softmax(dim=-1).to(v.dtype)
        return self.proj((attn @ v).transpose(1, 2).reshape(batch_windows, tokens, channels))


class SwinBlock(nn.Module):
    def __init__(self, dim, window, heads, shifted=False, mlp_ratio=4.0):
        super().__init__()
        self.window, self.shifted = tuple(window), shifted
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = WindowAttention(dim, window, heads)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        shortcut = x
        normalized = self.norm1(x.movedim(1, -1)).movedim(-1, 1)
        normalized, original = pad_spatial(normalized, self.window)
        spatial = tuple(normalized.shape[2:])
        shift = tuple(w // 2 if self.shifted and n > w else 0 for n, w in zip(original, self.window))
        dims = tuple(range(1, len(self.window) + 1))
        normalized = normalized.movedim(1, -1)
        if any(shift):
            normalized = torch.roll(normalized, tuple(-s for s in shift), dims)
        mask = attention_mask(tuple(original), spatial, self.window, shift, str(x.device))
        windows = self.attention(partition_windows(normalized, self.window), mask)
        out = reverse_windows(windows, self.window, spatial, x.shape[0])
        if any(shift):
            out = torch.roll(out, shift, dims)
        out = shortcut + crop_spatial(out.movedim(-1, 1), original)
        return out + self.mlp(self.norm2(out.movedim(1, -1))).movedim(-1, 1)


class SwinStage(nn.Module):
    def __init__(self, dim, depth, window, heads, mlp_ratio=4.0, use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([SwinBlock(dim, window, heads, bool(i % 2), mlp_ratio) for i in range(depth)])
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.use_checkpoint and self.training else block(x)
        return x


class PatchMerging(nn.Module):
    """Concatenate neighboring features, normalize, and linearly project."""
    def __init__(self, dim, factors):
        super().__init__()
        self.factors = tuple(factors)
        self.norm = nn.LayerNorm(dim * prod(factors))
        self.reduction = nn.Linear(dim * prod(factors), 2 * dim, bias=False)

    def forward(self, x):
        x, _ = pad_spatial(x, self.factors)
        parts = [x[(slice(None), slice(None)) + tuple(slice(o, None, f) for o, f in zip(offset, self.factors))]
                 for offset in product(*(range(f) for f in self.factors))]
        return self.reduction(self.norm(torch.cat(parts, dim=1).movedim(1, -1))).movedim(-1, 1)


class DualUpsample(nn.Module):
    """Fuse transposed-convolution and interpolation-convolution branches."""
    def __init__(self, ndim, in_channels, out_channels, factors):
        super().__init__()
        conv = conv_nd(ndim)
        trans = nn.ConvTranspose3d if ndim == 3 else nn.ConvTranspose2d
        self.learned = trans(in_channels, out_channels, kernel_size=factors, stride=factors)
        self.smooth = conv(in_channels, out_channels, 3, padding=1)
        self.fuse = conv(2 * out_channels, out_channels, 1)
        self.mode = "trilinear" if ndim == 3 else "bilinear"

    def forward(self, x, target_shape):
        learned = crop_spatial(self.learned(x), target_shape)
        smooth = self.smooth(F.interpolate(x, size=target_shape, mode=self.mode, align_corners=False))
        return self.fuse(torch.cat([learned, smooth], dim=1))
