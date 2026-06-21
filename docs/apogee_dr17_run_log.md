# APOGEE DR17 Run Log

## 2026-06-20

Created the first APOGEE DR17 clean sample for the Payne-MLP, KAN-Payne, and
TransformerPayne comparison.

Inputs:

- Catalog: local `allStarLite-dr17-synspec_rev1.fits`
- Manifest: `data/manifests/apogee_dr17_clean_10k.csv`
- Spectrum product: `aspcapStar` from the SDSS DR17 `synspec_rev1` tree

Selection summary:

- Total allStarLite rows: 733,901
- Clean pool after default cuts: 220,350
- Manifest rows: 10,000
- Telescopes: `apo25m`, `lco25m`
- Default cuts: `SNREV > 100`, `NVISITS >= 2`,
  `3500 < TEFF < 5500`, `0 < LOGG < 3.8`, `-2.0 < FE_H < 0.5`,
  with bad ASPCAP/STAR verbose flags excluded.

Download summary:

- Downloaded `aspcapStar` files: 10,000
- Raw `aspcapStar` directory size: about 2.3 GB
- One interrupted full `allStar` download was preserved as
  `data/raw/apogee_dr17/allStar-dr17-synspec_rev1.fits.partial`.

Preprocessed output:

- File: `data/processed/apogee_dr17_clean_10k.npz`
- Size: about 400 MB
- Wavelength grid: 8,575 APOGEE pixels, 15100.8015-16999.8074 Angstrom
- Flux array: `(10000, 8575)`, `float32`
- Error array: `(10000, 8575)`, `float32`
- Mask array: `(10000, 8575)`, masked fraction about 0.1237

ASPCAP label QA:

| Label | NaN count | Min | Median | Max |
|---|---:|---:|---:|---:|
| `TEFF` | 0 | 3500.24 | 4657.94 | 5494.19 |
| `LOGG` | 0 | 0.0105 | 2.3941 | 3.7974 |
| `FE_H` | 0 | -1.9918 | -0.2337 | 0.4956 |
| `M_H` | 0 | -2.0278 | -0.2286 | 0.5091 |
| `ALPHA_M` | 0 | -0.6556 | 0.0657 | 0.4975 |
| `MG_FE` | 7 | -0.4629 | 0.0911 | 0.6890 |
| `C_FE` | 4 | -1.2114 | -0.0058 | 0.9360 |
| `N_FE` | 29 | -0.5560 | 0.2009 | 1.6785 |
| `O_FE` | 3 | -0.5798 | 0.1033 | 0.8070 |
