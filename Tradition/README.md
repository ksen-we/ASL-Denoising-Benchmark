# Classical ASL denoising

BM4D and DT-CWT-NLM denoise the same low-SNR PWI volumes used by the deep
learning models. Both methods process the full 3D field of view. M0 is used
to define the evaluation mask, not as an additional denoising input.

## Run

1. Arrange the data as described in the project README.
2. For BM4D, install `BM4D/requirements-cuda12.txt`, select the visible GPU
   in `BM4D/Run_BM4D.py`, then run that file with Python.
3. For DT-CWT-NLM, install the external ONLM MEX dependency described in
   [DTCWT_NLM/README.md](DTCWT_NLM/README.md), then run
   `DTCWT_NLM/MyRun_DTCWT_NLM.m` in MATLAB.
4. Run `python Tradition/evaluate_results.py` from the project root.

Dataset paths, split selection and output names are configured at the top
of each entry script. Outputs are stored under
`Data/Dataset001/TraditionResult/<method>/<split>/<case>/`.
Existing predictions are preserved by default.

## Evaluation

The default `minmax` normalization and `legacy` evaluation match the final
`wtm0mask_ori` deep learning runs. The evaluation label is
`wtm0mask_seed42`, and reports are saved in
`evaluation/val/wtM0mask/` under each method's result directory.

Evaluation uses the same source-derived scale, M0 foreground mask and
metric functions as `DL/common`. Do not normalize predictions separately.
Only complete validation results enter `DL/compare_results.py`.
Traditional 3D results are grouped with the 3D deep learning results.

The BM4D results use the local CUDA implementation. The optional official
CPU backend is a different implementation and is not interchangeable with
the reported CUDA results. Denoising noise estimation and internal scaling
are method-specific; the final comparison uses the shared evaluation above.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution
and external dependencies.
