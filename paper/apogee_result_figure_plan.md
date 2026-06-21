# APOGEE DR17 Result Figure Plan

This note records the analysis figures to generate after the three APOGEE
inference files are available. The plotting entry point is:

```bash
python scripts/payne_apogee_result_diagnostics.py \
  --fit \
    data/processed/formal_a100_10k_20260620_161437/apogee_inference_10k_payne_norm/apogee_fit_payne_mlp_10k_payne_norm.npz \
    data/processed/formal_a100_10k_20260620_161437/apogee_inference_10k_payne_norm/apogee_fit_kan_payne_10k_payne_norm.npz \
    data/processed/formal_a100_10k_20260620_161437/apogee_inference_10k_payne_norm/apogee_fit_transformer_payne_10k_payne_norm.npz \
  --output-dir paper/figures/apogee_diagnostics
```

The script writes PDF and PNG figures, plus `apogee_quality_summary.csv`,
`apogee_residual_summary.csv`, and `apogee_figure_descriptions.md`.

## Figure Set

`apogee_aspcap_hexbin_grid.pdf` is the main label-comparison figure. It follows
the style used in APOGEE validation and Payne-style analyses: fitted labels are
shown against ASPCAP labels, the one-to-one relation is overplotted, and each
panel reports the median residual and robust scatter.

`apogee_residual_trend_grid.pdf` is the main systematics figure. It plots
fit-minus-ASPCAP residuals as a function of ASPCAP label and overlays a running
median with a 16th--84th percentile band. This reveals edge effects and
calibration trends that are hidden in a global scatter value.

`apogee_residual_distribution.pdf` compares residual histograms for
Payne-MLP, KAN-Payne, and TransformerPayne. This figure should be used to state
whether KAN-Payne improves the residual core, the tails, or both.

`apogee_quality_diagnostics.pdf` summarizes the quality-control selection. It
reports the number of successful and quality-selected spectra and shows the
spectral-fit chi-square and flux-MAE distributions.

`apogee_kiel_systematics.pdf` places KAN-Payne results on a Kiel diagram and
colors the points by residuals in [Fe/H], Teff, and log g. This checks whether
systematics depend on evolutionary state.

`apogee_model_to_model_residuals.pdf` compares labels between emulators. It is
useful for separating disagreement with ASPCAP from disagreement among the
models.

## Suggested Quality Cut

The default cut is conservative and reproducible:

```text
success == True
npix >= 5000
max(abs(fitted_label_scaled)) <= 0.49
chi2 <= model-specific 99th percentile after the above cuts
mae_x1e4 <= model-specific 99th percentile after the above cuts
```

The exact retained sample size and thresholds are written to
`apogee_quality_summary.csv`.

## Manuscript Text Template

After the final run, replace bracketed values with the CSV outputs:

```text
After applying the quality-control cuts described above, the retained sample
contains [N_KAN] KAN-Payne fits, [N_MLP] Payne-MLP fits, and [N_TR] 
TransformerPayne fits. Relative to ASPCAP, KAN-Payne gives median residuals of
[dTeff] K, [dlogg] dex, and [dFeH] dex for Teff, log g, and [Fe/H], with robust
scatters of [sTeff] K, [slogg] dex, and [sFeH] dex. The residual-trend figure
shows [describe trend: flat / cool-end drift / low-metallicity drift]. The Kiel
diagram indicates [describe whether residuals are concentrated along RGB/RC or
distributed broadly].
```

