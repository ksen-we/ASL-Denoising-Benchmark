"""Hierarchical Swin U-Net model defaults."""
from common.config import DEFAULT_GLOBAL_RESIDUAL, ExperimentConfig


def make_config(mode="3d", global_residual=None):
    cfg = ExperimentConfig(method="SwinUNet", mode=mode)
    cfg.batch_size = 1 if mode == "3d" else 16
    cfg.patch_size = (16, 48, 48) if mode == "3d" else (48, 48)
    cfg.model = dict(embed_dim=48, depths=(2, 2, 2), heads=(3, 6, 12),
                     window=(2, 8, 8) if mode == "3d" else (8, 8),
                     mlp_ratio=4.0, use_checkpoint=True)

    cfg.zero_init_output = True
    cfg.set_global_residual(DEFAULT_GLOBAL_RESIDUAL if global_residual is None else global_residual)
    return cfg
