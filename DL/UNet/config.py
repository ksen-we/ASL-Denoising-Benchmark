"""U-Net model defaults."""
from common.config import DEFAULT_GLOBAL_RESIDUAL, ExperimentConfig


def make_config(mode="3d", global_residual=None):
    cfg = ExperimentConfig(method="UNet", mode=mode)
    cfg.batch_size = 1 if mode == "3d" else 16
    cfg.patch_size = (16, 48, 48) if mode == "3d" else (48, 48)
    cfg.model = dict(base_channels=32, levels=4)

    cfg.zero_init_output = True
    cfg.set_global_residual(DEFAULT_GLOBAL_RESIDUAL if global_residual is None else global_residual)
    return cfg
