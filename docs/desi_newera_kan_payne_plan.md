# DESI NewEra KAN-Payne Implementation Plan

This document records the DESI stellar-parameter project built on the NewEra
PHOENIX/1D LTE model grid and the existing KAN-Payne training code.

## Input Grid

The uploaded PDF is the NewEra model-grid paper, not the grid data itself. The
paper states that the grid contains 37,438 LTE PHOENIX/1D models covering
2300-12000 K, log g from 0.0 to 6.0, [M/H] from -4.0 to +0.5, and additional
alpha variations for part of the metallicity range. The data products are
distributed as HDF5 files and as low-resolution archive products. For a first
DESI experiment, the low-resolution spectra from 2500 Angstrom to 2.5 micron
with 0.1 Angstrom sampling are the practical starting point because DESI covers
roughly the optical 3600-9800 Angstrom range at R of a few thousand.

Required downloads from the NewEra repository:

- `list_of_available_NewEra_models.txt`
- `get_NewEra_from_FDR.py`
- either a selected set of HDF5 spectra or the low-resolution archive
  `PHOENIX-NewEra-LowRes-SPECTRA.tar.gz`

The first pilot should not download every product blindly. Start with a
restricted FGK/red-giant and metal-poor grid region, then expand after the
conversion and training pipeline is verified.

## Synthetic Grid Conversion

The conversion entry point is:

```bash
python scripts/desi_prepare_newera_grid.py \
  --input-root data/external/newera/lowres_extracted \
  --glob "**/*" \
  --output data/processed/desi_newera_grid.npz \
  --wave-min 3600 \
  --wave-max 9800 \
  --wave-step 1.0 \
  --target-resolution 3000 \
  --continuum-window 301 \
  --teff-range 3500,8000 \
  --logg-range 0,5 \
  --mh-range -3.0,0.5 \
  --alpha-range -0.2,0.8
```

The output file uses the same `PayneGrid` NPZ schema as the APOGEE experiments,
with labels:

```text
Teff, logg, M_H, alpha_M
```

The script performs a first-pass DESI-like preprocessing:

- select the 3600-9800 Angstrom wavelength range,
- interpolate onto a uniform optical grid,
- apply approximate Gaussian smoothing to a target resolving power,
- divide a running-median pseudo-continuum,
- split the synthetic grid into train, validation, and test subsets.

This is a pilot-level approximation. A production DESI catalog should replace
the single-resolution smoothing with DESI camera-dependent line-spread
functions and should use the same continuum normalization for synthetic and
observed spectra.

## Emulator Training

The existing training entry point can be reused:

```bash
python scripts/payne_train_emulator.py \
  --grid data/processed/desi_newera_grid.npz \
  --model kan_payne \
  --output-dir data/processed/desi_newera_emulators \
  --epochs 10000 \
  --batch-size 64 \
  --lr 1e-4 \
  --hidden-sizes 64,256 \
  --activation gelu \
  --device cuda
```

Recommended pilot baselines:

- `payne_mlp`, with `--hidden-sizes 300,300`
- `kan_payne`, with `--hidden-sizes 64,256`
- optional `transformer_payne`, only after the vector-output models are stable

The first DESI method paper should focus on KAN-Payne versus Payne-MLP. The
TransformerPayne comparison can be included if compute time allows, but full
DESI catalog inference should use the fastest validated model.

## DESI Observed Spectra

Observed DESI spectra should be handled after the synthetic emulator passes
validation. The planned data product for the pilot is DR1 coadded stellar
spectra, not all single exposures.

Observed preprocessing should:

- read DESI coadd spectra by target/healpix,
- combine or separately handle B/R/Z arms,
- apply rest-frame correction using DESI radial velocity or a fitted RV,
- use inverse variance and mask arrays to remove bad pixels and sky residuals,
- normalize each camera arm with the same pseudo-continuum logic used for the
  synthetic spectra,
- resample to the same grid as `desi_newera_grid.npz`.

The first inference target should be a crossmatched validation subset:

- DESI stellar VAC stars with RVS/SP labels,
- APOGEE/GALAH/LAMOST crossmatches for external validation,
- Gaia HR-diagram sanity checks.

## Parameter Fitting

The APOGEE fitter is currently single-star and step-loop based. For DESI this
must become batched fitting before any large catalog is attempted. The target
design is:

```text
labels: [batch, 4]
flux:   [batch, n_pix]
ivar:   [batch, n_pix]
mask:   [batch, n_pix]
loss = mean(mask * ivar * (model(labels) - flux)^2)
```

Required output columns:

- fitted `Teff`, `logg`, `[M/H]`, `[alpha/M]`,
- fitted scaled labels,
- success flag,
- boundary flag,
- chi-square and flux MAE,
- number of valid pixels,
- S/N summaries by camera arm,
- model and grid version metadata.

## Paper Positioning

The DESI paper should not be framed as simply filling a missing catalog because
DESI DR1 already has stellar-parameter VAC products. The stronger framing is an
independent synthetic-emulator forward-model catalog:

```text
KAN-Payne for DESI DR1 Stellar Atmospheric Parameters:
an independent NewEra-based forward-model catalog with quality flags and
cross-survey validation.
```

The current APOGEE paper should be completed first. It establishes the
KAN-Payne method and validates the architecture against Payne-MLP and
TransformerPayne. The DESI work can then become the natural large-survey
application paper.

