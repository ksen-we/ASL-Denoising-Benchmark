"""Validate public CUDA parameters without requiring a GPU."""
import math
from numbers import Integral, Real


def validate_parameters(shape, sigma, bs_ht=4, step_ht=3, max_stack_ht=16,
                        search_ht=7, tau_ht=2.9527, lambda_thr=3.0,
                        bs_wiener=5, step_wiener=3, max_stack_wiener=32,
                        search_wiener=7, tau_wiener=0.7689):
    integers = dict(bs_ht=bs_ht, bs_wiener=bs_wiener, step_ht=step_ht,
                    step_wiener=step_wiener, max_stack_ht=max_stack_ht,
                    max_stack_wiener=max_stack_wiener, search_ht=search_ht,
                    search_wiener=search_wiener)
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in integers.values()):
        raise ValueError("Block sizes, steps, stack sizes and search radii must be integers")
    if bs_ht != 4:
        raise ValueError("HT uses a fixed 4-point transform: bs_ht must equal 4")
    if not 1 <= bs_wiener <= 5:
        raise ValueError("Wiener kernels support block sizes 1..5 only")
    for stage, bs, step, stack, search in (
        ("HT", bs_ht, step_ht, max_stack_ht, search_ht),
        ("Wiener", bs_wiener, step_wiener, max_stack_wiener, search_wiener),
    ):
        if not 1 <= step <= bs:
            raise ValueError(f"{stage}: require 1 <= step <= block size to avoid uncovered voxels")
        if stack < 1 or stack > 32 or stack & (stack - 1):
            raise ValueError(f"{stage}: max_stack must be a power of two in 1..32")
        if search < 0 or search > 2**30:
            raise ValueError(f"{stage}: invalid search radius")
    if len(shape) != 3 or min(shape) < max(bs_ht, bs_wiener):
        raise ValueError("Input must be 3D with every axis at least as large as both blocks")
    if math.prod(shape) >= 2**31:
        raise ValueError("Volume exceeds the CUDA kernels' signed int32 indexing limit")
    for name, value in dict(sigma=sigma, tau_ht=tau_ht, tau_wiener=tau_wiener,
                            lambda_thr=lambda_thr).items():
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real scalar")
        if value < 0 or (name.startswith("tau") and value == 0):
            raise ValueError(f"{name} is outside its valid range")
