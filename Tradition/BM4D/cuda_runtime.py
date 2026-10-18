"""Validate CUDA dependencies and both BM4D stages before processing cases."""
import ctypes
from importlib import metadata
from pathlib import Path
import sysconfig

_LIBRARIES = []


def load_cuda_libraries():
    """Load matching CUDA shared libraries from installed NVIDIA wheels."""
    names = []
    for name in ("cupy", "cupy-cuda11x", "cupy-cuda12x", "cupy-cuda13x"):
        try:
            names.append((name, metadata.version(name)))
        except metadata.PackageNotFoundError:
            pass
    if len(names) != 1:
        raise RuntimeError(f"Install exactly one CuPy package; found {names}. See requirements-cuda12.txt.")
    package = names[0][0]
    major = next((v for v in (11, 12, 13) if f"cuda{v}x" in package), None)
    root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if major is not None:
        # NVIDIA component wheels coexist with PyTorch, but their lib folders
        # need not be in LD_LIBRARY_PATH when an IDE launches Python.
        patterns = (("cuda_runtime", f"libcudart.so.{major}*"),
                    ("cuda_nvrtc", f"libnvrtc-builtins.so.{major}*"),
                    ("cuda_nvrtc", f"libnvrtc.so.{major}*"))
        for component, pattern in patterns:
            for path in sorted((root / component / "lib").glob(pattern)):
                _LIBRARIES.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
    return dict(names)


def preflight(gpu_id=0):
    packages = load_cuda_libraries()
    import cupy as cp
    import numpy as np
    try:
        nvrtc_version = cp.cuda.nvrtc.getVersion()
        count = cp.cuda.runtime.getDeviceCount()
        if not isinstance(gpu_id, int) or not 0 <= gpu_id < count:
            raise ValueError(f"GPU_ID={gpu_id} is invalid for {count} visible GPUs (zero-based).")
        cp.cuda.Device(gpu_id).use()
        from bm4d_cuda import bm4d
        # This launches both stages, not just an import/version check. A zero
        # basic group also exercises the Wiener zero-denominator protection.
        probe = np.zeros((6, 7, 8), dtype=np.float32, order="F")
        output = bm4d(probe, 0.1)
        if output.shape != probe.shape or not np.isfinite(output).all() or np.max(np.abs(output)) > 1e-6:
            raise RuntimeError("BM4D zero-volume kernel check failed.")
        props = cp.cuda.runtime.getDeviceProperties(gpu_id)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()
        return bm4d, {"status": "passed", "packages": packages, "nvrtc": list(nvrtc_version),
                      "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                      "cuda_driver": cp.cuda.runtime.driverGetVersion(), "gpu_id": gpu_id,
                      "gpu_name": name, "zero_volume_kernel_check": True}
    except Exception as exc:
        raise RuntimeError(
            "BM4D CUDA preflight failed before processing any cases. "
            "Check the visible GPU index and matching CuPy/NVRTC major versions. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
