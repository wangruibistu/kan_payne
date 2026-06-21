# KAN-Payne

KAN-Payne is a stellar-spectrum emulator project that replaces the MLP
emulator in The Payne-style spectral modeling with Kolmogorov-Arnold Networks
(KANs). The working research goal is to compare Payne-MLP, KAN-Payne, and
TransformerPayne on theoretical APOGEE-like spectra and on APOGEE DR17 stellar
parameter inference.

## Manuscript Data

The compact data package for the Universe manuscript is available on Zenodo:

https://doi.org/10.5281/zenodo.20782050

It contains the continuum mask, training histories, validation metrics, APOGEE
fit summaries, trained checkpoints, and revision diagnostic tables used by the
paper. The original APOGEE DR17 spectra and The Payne synthetic spectra should
be downloaded from their public upstream archives.

## Current Workflow

This repository now has two reproducible data paths:

1. prepare APOGEE DR17 spectra and the official ASPCAP comparison labels;
2. prepare The Payne GitHub Kurucz/APOGEE synthetic grid and train or reproduce
   Payne-MLP, KAN-Payne, and TransformerPayne emulators.

### Payne Synthetic Grid and Emulator Baselines

Prepare the unified theoretical training set:

```bash
python scripts/payne_prepare_training_grid.py --download-missing
```

Reproduce the GitHub pretrained Payne-MLP baseline with pure NumPy:

```bash
python scripts/payne_evaluate_pretrained_mlp.py \
  --grid data/processed/payne_apogee_synthetic_grid.npz
```

Dry-run the three emulator training configurations without importing PyTorch:

```bash
python scripts/payne_train_emulator.py --model payne_mlp --dry-run
python scripts/payne_train_emulator.py --model kan_payne --dry-run --hidden-sizes 64,256
python scripts/payne_train_emulator.py --model transformer_payne --dry-run
```

Start actual training after confirming the local PyTorch environment:

```bash
python scripts/payne_train_emulator.py --model payne_mlp --device mps --epochs 200
python scripts/payne_train_emulator.py --model kan_payne --device mps --hidden-sizes 64,256 --activation gelu --epochs 200
python scripts/payne_train_emulator.py --model transformer_payne --device mps --epochs 200 --batch-size 32
```

See [docs/payne_emulator_workflow.md](docs/payne_emulator_workflow.md) for the
dataset format, split convention, model settings, and comparison metrics. See
[docs/payne_emulator_run_log.md](docs/payne_emulator_run_log.md) for the current
data-product QA and local MPS environment checks.

### APOGEE DR17 Data Preparation

Install the lightweight data-preparation dependencies:

```bash
pip install -r requirements-apogee.txt
```

Create a clean `aspcapStar` manifest from the official DR17 `allStar` table:

```bash
python scripts/apogee_dr17_make_manifest.py \
  --download-allstar \
  --max-stars 10000 \
  --output data/manifests/apogee_dr17_clean_10k.csv
```

Download the selected pseudo-continuum-normalized APOGEE spectra:

```bash
python scripts/apogee_dr17_download.py \
  --manifest data/manifests/apogee_dr17_clean_10k.csv \
  --product aspcapstar \
  --workers 8 \
  --progress-every 100
```

Preprocess the downloaded FITS spectra into one analysis-ready NPZ file:

```bash
python scripts/apogee_dr17_preprocess.py \
  --manifest data/manifests/apogee_dr17_clean_10k.csv \
  --output data/processed/apogee_dr17_clean_10k.npz
```

If using the original Payne wavelength grid, pass it explicitly:

```bash
python scripts/apogee_dr17_preprocess.py \
  --manifest data/manifests/apogee_dr17_clean_10k.csv \
  --wavelength-npz path/to/apogee_wavelength.npz \
  --output data/processed/apogee_dr17_clean_10k_payne_grid.npz
```

See [docs/apogee_dr17_workflow.md](docs/apogee_dr17_workflow.md) for sample
cuts, data-product choices, and the scaling plan.

## Smoke Test

The repository includes a one-star APOGEE DR17 manifest fixture. This validates
the download and preprocessing path without downloading the 4 GB `allStar`
catalog:

```bash
python scripts/apogee_dr17_download.py \
  --manifest tests/fixtures/apogee_dr17_smoke_manifest.csv \
  --data-root data/raw/apogee_dr17_smoke \
  --max-files 1

python scripts/apogee_dr17_preprocess.py \
  --manifest tests/fixtures/apogee_dr17_smoke_manifest.csv \
  --data-root data/raw/apogee_dr17_smoke \
  --output data/processed/apogee_dr17_smoke.npz \
  --max-stars 1
```

Expected preprocessed shape for the fixture is one spectrum with 8575 APOGEE
pixels.
