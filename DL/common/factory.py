"""Construct models from the complete saved configuration."""
def build_model(cfg):
    ndim = 3 if cfg.mode == "3d" else 2
    inputs = 1 if ndim == 3 else 3
    if cfg.method == "SwinIR":
        from SwinIR.models import SwinIRDenoiser
        cls = SwinIRDenoiser
    elif cfg.method == "Pix2Pix":
        from Pix2Pix.models import UNetGenerator
        cls = UNetGenerator
    elif cfg.method == "SwinUNet":
        from SwinUNet.models import SwinUNetDenoiser
        cls = SwinUNetDenoiser
    elif cfg.method == "UNet":
        from UNet.models import UNetDenoiser
        cls = UNetDenoiser
    else:
        raise ValueError(cfg.method)
    if (cfg.global_residual is not None and "residual" in cfg.model
            and cfg.model["residual"] != cfg.global_residual):
        raise ValueError("Legacy model.residual conflicts with global_residual; remove the legacy field")
    return cls(ndim=ndim, in_channels=inputs, global_residual=cfg.global_residual,
               zero_init_output=cfg.zero_init_output, **cfg.model)


def build_discriminator(cfg):
    if not cfg.use_gan:
        return None
    from Pix2Pix.models import PatchDiscriminator
    return PatchDiscriminator(3 if cfg.mode == "3d" else 2, 1 if cfg.mode == "3d" else 3,
                              cfg.discriminator_channels)
