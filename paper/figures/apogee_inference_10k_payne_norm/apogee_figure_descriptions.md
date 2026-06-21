# APOGEE DR17 Inference Figure Descriptions

Use these descriptions after replacing bracketed values with the numbers in
`apogee_residual_summary.csv` and `apogee_quality_summary.csv`.

Figure `apogee_aspcap_hexbin_grid.pdf` compares fitted labels with the official
ASPCAP reference values for the quality-controlled 10000-star sample. Each
column corresponds to one emulator and each row to one atmospheric label. The
one-to-one line is shown in red, and each panel reports the median residual and
robust scatter. This figure follows the comparison style used in APOGEE
pipeline validation and Payne-style label analyses.

Figure `apogee_residual_trend_grid.pdf` shows fit-minus-ASPCAP residuals as a
function of the ASPCAP label. The running median and 16th--84th percentile band
make calibration trends visible. This is the main figure for judging whether a
model has edge effects at cool temperatures, low gravities, or low metallicity.

Figure `apogee_residual_distribution.pdf` compares the residual distributions
of Payne-MLP, KAN-Payne, and TransformerPayne. This compact view is useful for
reporting whether KAN-Payne narrows the residual core or mainly changes the
outlier tails.

Figure `apogee_quality_diagnostics.pdf` summarizes the quality-control cut and
the spectral-fit quality distributions. Use it to state how many spectra pass
success, boundary, valid-pixel, chi-square, and MAE cuts.

Figure `apogee_kiel_systematics.pdf` places the proposed KAN-Payne labels in
the Kiel diagram and colors the stars by residuals in [Fe/H], Teff, and log g.
This checks whether the method has evolutionary-state-dependent systematics.

Figure `apogee_model_to_model_residuals.pdf` compares the fitted labels between
emulators. This separates disagreement with ASPCAP from disagreement among the
emulators themselves.
