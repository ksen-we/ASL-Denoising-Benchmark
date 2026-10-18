"""Image-level residual output and output-head initialization."""
from torch import nn


def configure_output(model, layer, ndim, in_channels, global_residual,
                     zero_init_output, *, legacy_residual=False, legacy_zero_init=False):
    """Configure residual addition and head initialization without adding parameters."""
    for name, value in (("global_residual", global_residual), ("zero_init_output", zero_init_output)):
        if value is not None and type(value) is not bool:
            raise ValueError(f"{name} must be bool or None for legacy compatibility")
    model.global_residual = legacy_residual if global_residual is None else global_residual
    model.zero_init_output = legacy_zero_init if zero_init_output is None else zero_init_output
    if model.global_residual and in_channels != (1 if ndim == 3 else 3):
        raise ValueError("Global residual requires one 3D channel or three 2.5D channels (center slice only)")
    model.center = in_channels // 2
    if model.zero_init_output:
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def image_output(model, correction, source):
    """Apply image residual addition after restoring the original spatial shape."""
    if model.global_residual:
        return source[:, model.center:model.center + 1] + correction
    return correction
