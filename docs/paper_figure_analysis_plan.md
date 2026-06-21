# KAN-Payne Paper Figure and Comparison Plan

This paper is positioned as a methods paper introducing KAN-Payne. Payne-MLP
and TransformerPayne are comparison baselines. The figures should therefore make
KAN-Payne's behavior visible, not just list three numbers.

## Literature Patterns to Follow

- ASPCAP/FERRE pipeline papers emphasize external validation, residual trends,
  calibration behavior, and parameter-space coverage.
- The Cannon and DD-Payne style papers emphasize label-transfer residuals,
  one-to-one comparisons with reference labels, and systematic trends across
  label space.
- StarNet and astroNN style papers often show predicted-versus-reference labels,
  residual distributions, and performance as a function of S/N or reference
  label value.
- The Payne and TransformerPayne style emulator papers emphasize synthetic
  validation residuals, wavelength-dependent residual structure, and downstream
  label recovery.

## Required Figures

1. **Training history**
   - File target: `validation_mae_history.pdf`, `training_l1_history.pdf`
   - Purpose: show whether KAN-Payne converges faster/slower than Payne-MLP and
     whether it reaches a lower validation floor.
   - Text emphasis: KAN-Payne as the proposed model; MLP and Transformer as
     reference curves.

2. **Synthetic validation residual distribution**
   - File target: `synthetic_residual_histogram.pdf`
   - Purpose: compare residual width and tails, not only MAE/RMSE.
   - Text emphasis: a lower median residual is not enough if the tails are
     worse.

3. **Wavelength-resolved residuals**
   - File target: `pixel_residual_summary.pdf`
   - Purpose: identify whether KAN-Payne improves broad spectral regions or
     only selected pixels.
   - Text emphasis: locate difficult APOGEE wavelength intervals and discuss
     whether KAN edge functions help around line-rich regions.

4. **APOGEE fitted label versus ASPCAP**
   - File target: `apogee_aspcap_comparison.pdf`
   - Purpose: match the common survey-pipeline presentation of predicted versus
     reference labels.
   - Text emphasis: Teff, logg, and Fe/H should be shown first. Abundances can
     be added after the pipeline is stable.

5. **Residual trends versus reference labels**
   - File target: `apogee_residual_trends.pdf`
   - Purpose: expose calibration-like trends and edge effects.
   - Text emphasis: show whether KAN-Payne has smaller or flatter residual
     trends than Payne-MLP.

6. **Kiel diagram residual map**
   - File target: `apogee_kiel_residual_map.pdf`
   - Purpose: test whether residuals cluster by evolutionary state.
   - Text emphasis: this is especially important for red giants because
     Teff-logg correlations can hide in one-dimensional residual plots.

## Scripts

Use:

```bash
python scripts/payne_make_paper_figures.py \
  --run-dir data/processed/formal_a100_10k_20260620_161437 \
  --output-dir paper/figures \
  --apogee-fit \
    data/processed/formal_a100_10k_20260620_161437/apogee_fit_payne_mlp.npz \
    data/processed/formal_a100_10k_20260620_161437/apogee_fit_kan_payne.npz \
    data/processed/formal_a100_10k_20260620_161437/apogee_fit_transformer_payne.npz
```

The same script also works before APOGEE fitting if `--apogee-fit` is omitted;
in that case it generates only the synthetic emulator figures.
