# APOGEE DR17 Data Workflow

This workflow prepares APOGEE DR17 spectra for the KAN-Payne, Payne-MLP, and
TransformerPayne comparison.

## Data Choice

Use `allStar-dr17-synspec_rev1.fits` as the official ASPCAP comparison catalog
and use `aspcapStar` files as the main observed spectra. The `synspec_rev1`
tree is preferred because SDSS recommends it after correcting a subset of LCO
line-spread-function assignments.

The main spectrum product is `aspcapStar` because it contains the
pseudo-continuum-normalized, radial-velocity-corrected, combined APOGEE
spectrum on the standard vacuum wavelength grid. A smaller `apStar/asStar`
subset should be kept for a robustness test with an independent continuum
normalization.

## Recommended First Run

1. Download the official allStar catalog:

```bash
python scripts/apogee_dr17_make_manifest.py \
  --download-allstar \
  --max-stars 10000 \
  --output data/manifests/apogee_dr17_clean_10k.csv
```

2. Download the selected `aspcapStar` spectra:

```bash
python scripts/apogee_dr17_download.py \
  --manifest data/manifests/apogee_dr17_clean_10k.csv \
  --product aspcapstar \
  --workers 8 \
  --progress-every 100
```

3. Convert the downloaded spectra into an analysis-ready NPZ file:

```bash
python scripts/apogee_dr17_preprocess.py \
  --manifest data/manifests/apogee_dr17_clean_10k.csv \
  --output data/processed/apogee_dr17_clean_10k.npz
```

If the Payne training wavelength grid is available, add:

```bash
  --wavelength-npz path/to/apogee_wavelength.npz
```

If a local `allStarLite-dr17-synspec_rev1.fits` file is already available, it
can be used for the first 10k run:

```bash
python scripts/apogee_dr17_make_manifest.py \
  --allstar path/to/allStarLite-dr17-synspec_rev1.fits \
  --max-stars 10000 \
  --output data/manifests/apogee_dr17_clean_10k.csv \
  --reader simple
```

`allStarLite` does not include every allStar column, but it contains the
columns needed for the default clean-sample selection and the primary ASPCAP
parameter comparison.

## Default Clean Sample

The default manifest selection is intentionally conservative:

- `TELESCOPE in (apo25m, lco25m)`
- `SNREV > 100`
- `NVISITS >= 2`
- red-giant-focused parameter range:
  `3500 < TEFF < 5500`, `0 < LOGG < 3.8`, `-2.0 < FE_H < 0.5`
- excludes verbose ASPCAP flags:
  `STAR_BAD`, `CHI2_BAD`, `NO_ASPCAP_RESULT`, `TEFF_BAD`, `LOGG_BAD`, `M_H_BAD`
- excludes verbose STAR flags:
  `BAD_PIXELS`, `VERY_BRIGHT_NEIGHBOR`, `LOW_SNR`,
  `BAD_RV_COMBINATION`, `RV_FAIL`
- excludes `EXTRATARG` duplicate observations.

The first paper should use this clean sample for the main comparison, then add
a wider validation sample after the fitting pipeline is stable.

## Comparison Catalog Columns

Use the calibrated named ASPCAP columns as the primary comparison:

- `TEFF`, `LOGG`, `M_H`, `FE_H`, `ALPHA_M`
- selected abundances such as `C_FE`, `N_FE`, `O_FE`, `MG_FE`, `SI_FE`,
  `CA_FE`, `TI_FE`, `NI_FE`
- corresponding uncertainty columns where present.

For an appendix or systematic test, compare against spectroscopic/raw columns
such as `TEFF_SPEC`, `LOGG_SPEC`, and the `FPARAM` array.

## Scaling Up

After the 10k run is verified, create a larger manifest:

```bash
python scripts/apogee_dr17_make_manifest.py \
  --allstar data/raw/apogee_dr17/allStar-dr17-synspec_rev1.fits \
  --max-stars 100000 \
  --sample-mode random \
  --output data/manifests/apogee_dr17_clean_100k.csv
```

For full-scale download, prefer `rsync` or Globus. The full `aspcapStar` set is
over 40 GB, while all `apStar/asStar` files are much larger.
