# DESI + NewEra KAN-Payne Preprocessing Notes

This note records the working choices for using NewEra V3 LowRes spectra to train
KAN-Payne and infer DESI DR1 MWS stellar parameters.

## Data Format

NewEra V3 LowRes files use a GAIA-library style text format: one header line
followed by one flux line per model. The header contains the wavelength range in
nm, the wavelength sampling, and stellar labels including `Teff`, `logg`, iron
abundance, and `alphaFe`. The conversion script derives `M_H` as
`feAbund - 7.50` and uses the four labels `Teff, logg, M_H, alpha_M`.

DESI DR1 coadd files contain one row per target in a healpix/program file. The
spectra are stored separately for the `B`, `R`, and `Z` cameras as wavelength,
flux, inverse variance, mask, and resolution-matrix HDUs. The MWS VAC SP tables
provide `SNR_MED`, `RV_ADOP`, `TEFF`, `LOGG`, `FEH`, and `ALPHAFE`; the first
two are used for sample quality and rest-frame correction, while the atmospheric
parameters are retained only for comparison.

## Current Processing Choices

The training grid is built on a fixed rest-frame wavelength grid, currently
`3600-9800 A` with `1 A` spacing. NewEra spectra are continuum-normalized with a
running median and convolved to a conservative approximate resolving power,
default `R=2000`.

Observed DESI spectra are corrected to the stellar rest frame with
`wave_rest = wave_observed / (1 + RV_ADOP / c)`. The `B`, `R`, and `Z` arms are
continuum-normalized separately before inverse-variance weighted merging onto the
same wavelength grid. This reduces arm-to-arm flux calibration offsets and broad
continuum-shape mismatch between PHOENIX/NewEra and DESI coadds.

The selected production threshold is `SPTAB.SNR_MED >= 30`. The pilot sample is
constructed by taking the densest program-healpix groups, so a 10k target pilot
requires only 70 coadd/redrock healpix groups rather than the full 25k groups
covered by all S/N-selected targets.

## Main Systematic Risks

Continuum placement is the largest first-order mismatch. DESI flux calibration,
extinction, binary contamination, imperfect sky subtraction, and broad-band model
differences can all bias the low-frequency spectral shape. The current solution
is pseudo-continuum normalization per camera arm. For publication-quality DESI
work, the inference stage should also test low-order multiplicative polynomial
nuisance terms or masked continuum anchor pixels.

Resolution mismatch is the second major risk. DESI resolution changes with
wavelength, camera, fiber, and exposure history; the coadd files include
resolution matrices. The current code uses a conservative common-resolution
approximation. A more exact production version should convolve each synthetic
spectrum with the DESI resolution matrix, or downgrade both theory and data to a
common resolution lower than the worst DESI resolving power in the fitted range.

Radial velocities must be applied before interpolation. The current pipeline
uses MWS `RV_ADOP`. For low-quality or problematic RVs, fit residuals near narrow
features will dominate and can bias labels. The production catalog should cut or
flag large `RV_ERR`, failed SP/RVS fits, and high chi-square cases.

## Relevant Code

- `scripts/desi_prepare_newera_grid.py`: parses NewEra V3 LowRes/GAIA-format
  archives or HDF5 files and writes a `PayneGrid` NPZ.
- `scripts/run_desi_newera_kan_payne_pipeline.sh`: runs grid preparation,
  KAN-Payne training, validation, DESI pilot preprocessing, and label fitting.
- `scripts/desi_prepare_observed_spectra.py`: reads DESI coadd FITS files,
  applies RV correction, per-arm normalization, and writes observed NPZ.
- `scripts/payne_fit_observed_spectra.py`: generic observed-spectrum label
  optimizer for 4-label NewEra or 25-label APOGEE checkpoints.
