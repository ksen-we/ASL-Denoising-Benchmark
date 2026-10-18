# BM4D baseline

The final experiments use `BACKEND="cuda"` in `Run_BM4D.py`.
The implementation is adapted from Matin Mahzad's
[SeismicFlow/bm4d-cuda](https://github.com/SeismicFlow/bm4d-cuda).
Its upstream README identifies the license as MIT. Source attribution and
the scope of local changes are recorded in
[../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

```bash
pip install -r Tradition/BM4D/requirements-cuda12.txt
python Tradition/BM4D/Run_BM4D.py
```

Set `GPU_ID` to an index among the GPUs visible to the process. The default
is `0`. A compatible NVIDIA driver is required. Install only one CuPy
distribution. `CHECK_ONLY=True` runs a synthetic backend check;
`DRY_RUN=True` only estimates input noise.

Noise is estimated from 3D Haar high-frequency coefficients of the source.
The high-SNR target is used only for evaluation. Parameters are stored with
file and implementation fingerprints; incompatible existing outputs are
rejected rather than silently reused.

`requirements-cpu.txt` and `BACKEND="cpu"` provide an optional official
`bm4d==4.2.5` backend. It is not the backend used for the final comparison,
and switching backends requires a separate output directory.
