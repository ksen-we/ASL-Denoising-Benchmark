# DT-CWT-NLM baseline

This MATLAB baseline adapts the toolbox accompanying Liang, Connelly and
Calamante, *Voxel-wise functional connectomics using arterial spin labeling
fMRI: the role of denoising*, Brain Connectivity (2015),
[doi:10.1089/brain.2014.0290](https://doi.org/10.1089/brain.2014.0290).

## Dependencies

- MATLAB with Signal Processing Toolbox (`upfirdn`). The local pipeline was
  developed with MATLAB R2018b.
- A platform-compatible ONLM MEX binary, obtained separately from the
  [official DT-CWT-NLM release](https://www.nitrc.org/projects/dt-cwt-nlm).
  Download [DT-CWT-NLM1.01.zip](https://www.nitrc.org/frs/download.php/9402/DT-CWT-NLM1.01.zip), find
  `onlm.mexa64` (Linux x86-64) or `onlm.mexw64` (Windows x86-64), and place
  the matching file beside `MyRun_DTCWT_NLM.m`. The MEX files and their
  source are not bundled in this release copy. Follow the upstream terms.
- Pure MATLAB NIfTI I/O is included under `vendor/NIfTI_20140122` with its
  original license. SPM and REST are not needed by this entry script.

In MATLAB, use `mexext` and `which onlm -all` to check the installed binary.
`external_dependencies.json` records the archive and binary hashes. Both
locally used binaries match the corresponding files in the official 1.01
archive by SHA256.
Do not add the upstream package ahead of this directory on the MATLAB path:
the included `DTCWT.m` contains the task adaptation used here.

## Run

Edit the configuration at the top of `MyRun_DTCWT_NLM.m`, then run it.
Input and output filenames default to `rpwi_ave_4.nii` and
`rpwi_dtcwt_nlm.nii`. Existing outputs are preserved when `overwrite=false`.
`dryRun=true` estimates noise without invoking ONLM.

The entry script estimates noise from the source, shifts signed PWI only
for ONLM, reverses that shift before wavelet fusion, and saves the original
NIfTI geometry. The selected settings use a search radius of 3, patch
radii of 1 and 2, three wavelet levels and an ONLM noise divisor of 4.
`DTCWT.m` uses a separate noise estimate for each decomposition level and
accepts the configured fusion slope. These are the local task settings,
not a claim of an unchanged implementation of the paper.

Run `Tradition/evaluate_results.py` with Python after denoising. MATLAB
denoising itself does not require Python.
