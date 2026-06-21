# Payne Emulator Run Log

## 2026-06-20

### Data Products

Prepared the unified The Payne APOGEE/Kurucz synthetic grid:

```text
data/processed/payne_apogee_synthetic_grid.npz
```

Summary:

- spectra: `(1000, 7214)`
- labels: `(1000, 25)`
- wavelength range: `15168.1285` to `16936.7470` Angstrom
- Payne mask fraction: `0.1212919`
- split: first 800 training spectra, last 200 validation spectra

Reproduced the GitHub pretrained Payne-MLP baseline:

```text
data/processed/payne_pretrained_mlp_validation_metrics.json
```

Validation metrics in `flux residual * 1e4` units:

- all pixels: MAE `18.2938`, RMSE `70.8920`
- unmasked Payne pixels: MAE `17.4419`, RMSE `68.3113`

Prepared a small APOGEE DR17 smoke product on the Payne grid:

```text
data/processed/apogee_dr17_clean_100_payne_grid.npz
```

Shape:

- flux: `(100, 7214)`
- err: `(100, 7214)`
- mask: `(100, 7214)`

### Python/Torch Environment Checks

The available Conda environments under `/Users/wangr/opt/anaconda3` were tested
for `torch` import and MPS availability. They are x86_64 environments and
timed out during `import torch`:

- `/Users/wangr/opt/anaconda3/bin/python`
- `/Users/wangr/opt/anaconda3/envs/open_manus/bin/python`
- `/Users/wangr/opt/anaconda3/envs/gpt/bin/python`
- `/Users/wangr/opt/anaconda3/envs/py311/bin/python`
- `/Users/wangr/opt/anaconda3/envs/code/bin/python`
- `/Users/wangr/opt/anaconda3/envs/ai_scientist/bin/python`

A project-local ARM64 venv was created at:

```text
.venv/
```

It uses Apple CommandLineTools Python 3.9.6 ARM64. Two PyTorch variants were
tested:

- `torch 2.8.0` with `numpy 2.0.2`
- `torch 2.0.1` with `numpy 1.26.4`

Both stalled during `import torch` / early Python import when launched from this
Codex terminal. Process sampling showed ARM64 execution and Metal/MPS framework
loading for the torch import attempt. Therefore emulator training was not run
in this session.

Recommended next environment step:

```bash
python -c "import platform, torch; print(platform.machine()); print(torch.__version__); print(torch.backends.mps.is_available())"
```

Proceed with training only when this returns `arm64` and `True`.

### A100 Remote Training Smoke Test

The Linux server `lily` is reachable by SSH and has two NVIDIA A100-SXM4-40GB
GPUs. The selected training environment is:

```text
/home/wangrui/miniconda3/envs/py311/bin/python
Python 3.11.14
torch 2.2.1+cu121
CUDA available: True
GPU count: 2
```

The project was synced to:

```text
~/kan_payne_a100
```

The `efficient_kan` dependency was installed from a locally cloned copy of
`Blealtan/efficient-kan` with `pip install --no-deps -e`, so the working
`torch 2.2.1+cu121` environment was not upgraded.

The remote Payne-MLP pretrained reproduction matched the local baseline:

- all pixels: MAE `18.2938`, RMSE `70.8921`
- unmasked Payne pixels: MAE `17.4419`, RMSE `68.3113`

One-epoch CUDA smoke training completed for all three emulator families:

| model | config | valid good MAE x1e4 | valid good RMSE x1e4 |
| --- | --- | ---: | ---: |
| Payne-MLP | hidden `300,300`, batch 128 | `9725.7406` | `9748.1096` |
| KAN-Payne | hidden `32,64`, batch 64 | `9745.4822` | `9756.8822` |
| TransformerPayne | `d_model=64`, `n_layers=2`, `pixels_per_spectrum=64` | `722.4659` | `940.1631` |

Smoke checkpoints and histories were copied back locally under:

```text
data/processed/remote_smoke_emulators/
```

### 10000-Epoch A100 Run and Pending APOGEE Inference

A longer run was launched after the 200-epoch baseline showed continued
validation improvement:

```text
run id: formal_a100_10k_20260620_161437
remote path: ~/kan_payne_a100/data/processed/formal_a100_10k_20260620_161437
```

Payne-MLP, KAN-Payne, and TransformerPayne use the same configurations as the
formal 200-epoch run, but with `--epochs 10000` and unbuffered logs. Payne-MLP
completed first; KAN-Payne and TransformerPayne continued running in parallel
on the two A100 GPUs. A postprocess watcher is running remotely and will
evaluate the best checkpoint for all three models once the run writes `DONE`.

APOGEE DR17 clean 10k spectra were resampled to the Payne wavelength grid:

```text
data/processed/apogee_dr17_clean_10k_payne_grid.npz
```

The file was synced to `lily` for downstream inference. The APOGEE fitting
script passed a three-star smoke test with the 200-epoch KAN checkpoint:

```text
data/processed/formal_a100_20260620_160739/apogee_fit_kan_smoke3.npz
```

New analysis scripts added during the long run:

- `scripts/payne_compare_emulator_runs.py`
- `scripts/payne_fit_apogee_spectra.py`
- `scripts/payne_compare_apogee_fits.py`
- `scripts/payne_make_paper_figures.py`

The manuscript draft was started at:

```text
paper/universe_kan_payne_mdpi.tex
paper/references.bib
```

The draft is currently positioned as a KAN-Payne methods paper. Payne-MLP and
TransformerPayne are treated as comparison baselines, not as co-primary methods.
A figure and comparison plan was added at:

```text
docs/paper_figure_analysis_plan.md
```
