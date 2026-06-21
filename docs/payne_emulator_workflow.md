# Payne Emulator Workflow

This workflow prepares the theoretical APOGEE-like Kurucz training set from
The Payne repository and trains or reproduces three common emulators:
Payne-MLP, KAN-Payne, and TransformerPayne.

## 1. Prepare The Payne Synthetic Grid

The expected external files are:

- `data/external/the_payne/other_data/kurucz_training_spectra.npz`
- `data/external/the_payne/other_data/apogee_wavelength.npz`
- `data/external/the_payne/other_data/apogee_mask.npz`
- `data/external/the_payne/neural_nets/NN_normalized_spectra.npz`

If they are not present, the preparation script can download the small GitHub
files:

```bash
python scripts/payne_prepare_training_grid.py --download-missing
```

The default output is:

```text
data/processed/payne_apogee_synthetic_grid.npz
```

It contains:

- `wavelength`: Payne APOGEE grid, 7214 pixels
- `mask`: Payne APOGEE pixel mask; `True` means omit/bad
- `labels`: shape `(N, 25)`
- `label_scaled`: Payne convention `(label - train_min) / span - 0.5`
- `spectra`: shape `(N, 7214)`
- `train_idx`, `valid_idx`, `test_idx`
- `label_min`, `label_max`, `label_names`

By default the split follows The Payne repository: first 800 spectra for
training and the remaining 200 for validation.

## 2. Reproduce the GitHub Payne-MLP Baseline

Evaluate the pretrained MLP coefficients from The Payne repository without
importing PyTorch:

```bash
python scripts/payne_evaluate_pretrained_mlp.py \
  --grid data/processed/payne_apogee_synthetic_grid.npz \
  --pretrained-mlp data/external/the_payne/neural_nets/NN_normalized_spectra.npz \
  --output data/processed/payne_pretrained_mlp_validation_metrics.json
```

This reports MAE and RMSE in `flux residual * 1e4` units for all pixels and for
unmasked Payne pixels.

## 3. Train Payne-MLP

The default Payne-MLP follows the original two hidden-layer architecture:
25 labels -> 300 -> 300 -> 7214 flux pixels, with LeakyReLU activations and raw
flux targets.

```bash
python scripts/payne_train_emulator.py \
  --model payne_mlp \
  --grid data/processed/payne_apogee_synthetic_grid.npz \
  --device mps \
  --epochs 200 \
  --batch-size 64 \
  --lr 1e-4
```

Use `--dry-run` first on a new machine to validate the grid and resolved
configuration without importing PyTorch.

On Apple Silicon, use a native `arm64` Python/PyTorch environment before
training:

```bash
python -c "import platform, torch; print(platform.machine()); print(torch.backends.mps.is_available())"
```

The expected output is `arm64` and `True`. If Python reports `x86_64`, rebuild
the environment with a native arm64 Python before using `--device mps`.

## 4. Train KAN-Payne

KAN-Payne uses the same grid and label scaling, but replaces the vector-output
MLP with `efficient_kan.KAN`.

The `efficient_kan` import used by the original project comes from the
`Blealtan/efficient-kan` GitHub implementation, not from a published PyPI
package. Install it into the active PyTorch environment with:

```bash
pip install git+https://github.com/Blealtan/efficient-kan.git
```

```bash
python scripts/payne_train_emulator.py \
  --model kan_payne \
  --grid data/processed/payne_apogee_synthetic_grid.npz \
  --device mps \
  --hidden-sizes 64,256 \
  --activation gelu \
  --epochs 200 \
  --batch-size 64 \
  --lr 1e-4
```

For direct emulator comparisons, keep `--spectra-normalization none`. If KAN
training is unstable, run a secondary experiment with
`--spectra-normalization minmax` and report it as an optimization variant.

## 5. Train TransformerPayne

The TransformerPayne entrypoint predicts scalar flux values conditioned on both
labels and wavelength. During training it samples wavelength pixels per
spectrum; during validation it scans all Payne pixels in chunks.

```bash
python scripts/payne_train_emulator.py \
  --model transformer_payne \
  --grid data/processed/payne_apogee_synthetic_grid.npz \
  --device mps \
  --epochs 200 \
  --batch-size 32 \
  --lr 1e-4 \
  --d-model 128 \
  --n-label-tokens 16 \
  --n-heads 4 \
  --n-layers 4 \
  --pixels-per-spectrum 256
```

The settings above are deliberately smaller than the full paper-scale model and
are appropriate for the 1000-spectrum GitHub training subset. A paper-scale run
should sweep `d_model`, layer count, and sampled pixels after the baseline is
stable.

## 6. Put APOGEE DR17 on the Same Grid

After preparing APOGEE DR17 spectra, resample the observed spectra to the Payne
grid:

```bash
python scripts/payne_export_apogee_observed_on_payne_grid.py \
  --input data/processed/apogee_dr17_clean_10k.npz \
  --payne-grid data/processed/payne_apogee_synthetic_grid.npz \
  --output data/processed/apogee_dr17_clean_10k_payne_grid.npz
```

The output keeps the official ASPCAP comparison columns from the original NPZ
as `label_*` arrays.

## Recommended Experiment Table

Use the same train/validation split and report at least:

- validation MAE/RMSE over all pixels
- validation MAE/RMSE over unmasked Payne pixels
- model parameter count
- training wall time and device
- APOGEE DR17 parameter residuals versus ASPCAP after the fitting pipeline is added

The emulator-only table should be separated from the APOGEE parameter-inference
table in the paper.
