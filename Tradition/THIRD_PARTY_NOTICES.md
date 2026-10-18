# Third-party sources

This file records the terms and attribution found with the upstream
components. It does not assign a new license to them.

## BM4D CUDA

- Source: [SeismicFlow/bm4d-cuda](https://github.com/SeismicFlow/bm4d-cuda),
  Matin Mahzad.
- Upstream README license statement: **MIT**. A separate license file was
  not present in the checked upstream repository (2026-10-18).
- `BM4D/bm4d_cuda.py` preserves the upstream source documentation and
  references, including the acknowledged
  [Logrus/BM4D-GPU](https://github.com/Logrus/BM4D-GPU) transform architecture.
- Local modifications validate parameters and array layouts, protect
  zero-denominator aggregation and nonfinite results, and integrate
  input-derived noise estimation and NIfTI I/O. The optional official
  CPU `bm4d` package is installed separately and retains its own terms.

## DT-CWT-NLM and wavelet functions

- Source: [DT-CWT-NLM on NITRC](https://www.nitrc.org/projects/dt-cwt-nlm),
  Xiaoyun Liang, Alan Connelly and Fernando Calamante.
- Original notice: Copyright 2015 Florey Institute of Neuroscience and
  Mental Health, Melbourne, Australia. Written by Xiaoyun Liang.
- The official project page identifies **GNU General Public License
  (GPL)** without specifying a version. See the
  [GNU GPL](https://www.gnu.org/licenses/gpl.html) and the upstream release
  for applicable terms. No GPL version has been invented for this copy.
- The wavelet functions retain their original attribution to Ivan
  Selesnick and coworkers at Polytechnic University, Brooklyn, NY.
- Local changes to `DTCWT.m` make wavelet level and fusion slope explicit
  and estimate noise at the current decomposition level. Original source
  comments and attribution are retained.
- ONLM is the optimized blockwise NLM implementation attributed upstream
  to Pierrick Coupe and the 2008 IEEE TMI paper by Coupe et al.
  Precompiled ONLM binaries are not redistributed here; obtain them from
  the official DT-CWT-NLM release and retain the accompanying terms.

## NIfTI toolbox

`DTCWT_NLM/vendor/NIfTI_20140122` contains Jimmy Shen's pure MATLAB NIfTI
I/O subset. Its original 2014 copyright notice and redistribution terms
are preserved in `license.txt`. Vendor `.m` files are unchanged.
