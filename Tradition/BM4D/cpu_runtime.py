"""Optional official CPU backend; no automatic fallback from CUDA."""
from importlib import metadata
from numbers import Integral

import numpy as np


def validate_cpu_input(shape, sigma, num_threads):
    if isinstance(num_threads, bool) or not isinstance(num_threads, Integral) or num_threads < 1:
        raise ValueError("CPU_THREADS must be a positive integer.")
    if len(shape) != 3 or min(shape) < 5:
        raise ValueError("The CPU np profile requires a 3-D volume with at least 5 voxels per axis.")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative.")


def preflight(num_threads=4):
    """Run both stages on a synthetic volume and return the denoising callable."""
    validate_cpu_input((8, 9, 10), 0.1, num_threads)
    try:
        from bm4d import bm4d, BM4DProfile, BM4DStages
    except ImportError as exc:
        raise RuntimeError("The CPU backend requires bm4d==4.2.5; install requirements-cpu.txt.") from exc

    def denoise(volume, sigma, verbose=False):
        x = np.asarray(volume)
        validate_cpu_input(x.shape, sigma, num_threads)
        if x.dtype.kind not in "fiu" or not np.isfinite(x).all():
            raise ValueError("CPU BM4D input must contain finite real values.")
        x = np.ascontiguousarray(x, dtype=np.float32)
        if sigma == 0:
            return x.copy()
        # Create a fresh profile because the package mutates noise parameters.
        profile = BM4DProfile()  # Official normal (np) profile.
        profile.num_threads = int(num_threads)
        profile.print_info = bool(verbose)
        result = np.asarray(bm4d(x, float(sigma), profile=profile,
                                stage_arg=BM4DStages.ALL_STAGES), dtype=np.float32)
        if result.shape != x.shape or not np.isfinite(result).all():
            raise FloatingPointError("CPU BM4D returned an invalid shape or nonfinite values.")
        return result

    sample = np.random.default_rng(0).normal(0.5, 0.1, (8, 9, 10)).astype(np.float32)
    denoise(sample, 0.1)
    profile = BM4DProfile()
    profile.num_threads = int(num_threads)
    settings = {name: getattr(profile, name) for name in dir(profile)
                if not name.startswith("_") and not callable(getattr(profile, name))}
    return denoise, {"status": "ok", "backend": "official_cpu", "device": "CPU",
                     "packages": {"bm4d": metadata.version("bm4d")},
                     "profile": "np", "profile_settings": settings}
