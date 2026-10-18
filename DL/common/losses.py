"""Masked L1/SSIM reconstruction and weighted adversarial BCE."""
import torch
from torch import nn
from torch.nn import functional as F


def weighted_bce_with_logits(logits, targets, weights=None):
    """Normalize BCE across the batch and score grid using foreground weights."""
    if weights is None:
        return F.binary_cross_entropy_with_logits(logits, targets)
    if logits.shape != targets.shape or weights.shape != logits.shape:
        raise ValueError("BCE logits, labels, and foreground weights must have identical shapes")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        w = weights.detach().float()
        if not torch.isfinite(w).all() or (w < 0).any():
            raise ValueError("BCE weights must be finite and nonnegative")
        total = w.sum()
        if total.item() <= 0:
            raise ValueError("No valid foreground in the BCE weighting region")
        loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
        return (loss * w).sum() / total


def as_slices(x):
    if x.ndim == 5:
        return x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], *x.shape[-2:])
    if x.ndim != 4:
        raise ValueError("Input must have BCHW or BCDHW shape")
    return x


class ReconstructionLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.l1_weight, self.ssim_weight = cfg.l1_weight, cfg.ssim_weight
        self.data_range = cfg.ssim_data_range
        self.c1, self.c2 = (0.01 * cfg.ssim_data_range) ** 2, (0.03 * cfg.ssim_data_range) ** 2
        coords = torch.arange(cfg.ssim_window, dtype=torch.float32) - cfg.ssim_window // 2
        g = torch.exp(-coords.square() / (2 * cfg.ssim_sigma ** 2))
        g /= g.sum()
        self.register_buffer("kernel", (g[:, None] * g[None, :])[None, None])

    def forward(self, prediction, target, mask=None):

        with torch.autocast(device_type=prediction.device.type, enabled=False):
            x, y = as_slices(prediction.float()), as_slices(target.float())
            m = torch.ones_like(x) if mask is None else as_slices(mask.float())
            if x.shape != y.shape or m.shape != x.shape or x.shape[1] != 1:
                raise ValueError("Loss requires matching single-channel prediction, target, and mask tensors")
            count = m.sum()
            if count.item() <= 0:
                raise ValueError("No valid voxels in the loss region")
            kernel = self.kernel.to(device=x.device, dtype=x.dtype)
            def smooth(v):
                return F.conv2d(v, kernel, padding=kernel.shape[-1] // 2)

            mass = smooth(m).clamp_min(1e-8)
            ux, uy = smooth(x * m) / mass, smooth(y * m) / mass
            vx = (smooth(x.square() * m) / mass - ux.square()).clamp_min(0)
            vy = (smooth(y.square() * m) / mass - uy.square()).clamp_min(0)
            cov = smooth(x * y * m) / mass - ux * uy
            ssim_map = ((2 * ux * uy + self.c1) * (2 * cov + self.c2)) / (
                (ux.square() + uy.square() + self.c1) * (vx + vy + self.c2))
            score = (ssim_map.clamp(-1, 1) * m).sum() / count
            l1 = ((x - y).abs() * m).sum() / count
            loss = self.l1_weight * l1 + self.ssim_weight * (1 - score)
        return loss, {"l1": l1, "ssim": score}
