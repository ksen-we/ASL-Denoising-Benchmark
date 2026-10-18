"""Pix2Pix defaults; the release entry selects the study training policies."""
from common.config import DEFAULT_GLOBAL_RESIDUAL, ExperimentConfig


def make_config(mode="3d", global_residual=None):
    cfg = ExperimentConfig(method="Pix2Pix", mode=mode)
    cfg.batch_size = 1 if mode == "3d" else 16
    cfg.patch_size = (16, 48, 48) if mode == "3d" else (48, 48)
    cfg.model = dict(base_channels=32, dropout=0.5)
    cfg.use_gan = True
    cfg.adversarial_weight = 0.01
    cfg.adversarial_mask_mode = "none"
    cfg.lr_policy = "plateau"
    cfg.early_stopping_patience = 30
    cfg.early_stopping_start = "immediate"

    cfg.zero_init_output = True
    cfg.set_global_residual(DEFAULT_GLOBAL_RESIDUAL if global_residual is None else global_residual)
    return cfg
