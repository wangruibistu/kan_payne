#!/usr/bin/env python
"""Make APOGEE DR17 inference diagnostics from fitted Payne-label NPZ files.

The script is intentionally result-file driven: once the remote inference job
has produced the three `apogee_fit_*_payne_norm.npz` files, run this script to
generate publication-ready PDF figures, summary CSV tables, quality-control
cuts, and short text snippets for the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODEL_LABELS = {
    "payne_mlp": "Payne-MLP",
    "kan_payne": "KAN-Payne",
    "transformer_payne": "TransformerPayne",
}

MODEL_COLORS = {
    "payne_mlp": "#4c78a8",
    "kan_payne": "#f58518",
    "transformer_payne": "#54a24b",
}

LABEL_PAIRS = [
    ("Teff", "TEFF", r"$T_{\rm eff}$", "K"),
    ("logg", "LOGG", r"$\log g$", "dex"),
    ("Fe_H", "FE_H", r"[Fe/H]", "dex"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", nargs="+", required=True, help="APOGEE fit NPZ files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-npix", type=int, default=5000)
    parser.add_argument("--max-boundary-distance", type=float, default=0.49)
    parser.add_argument("--chi2-quantile", type=float, default=0.99)
    parser.add_argument("--mae-quantile", type=float, default=0.99)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _model_name(path: Path, data: dict) -> str:
    if "model" in data:
        value = data["model"]
        return str(value.tolist() if hasattr(value, "tolist") else value)
    stem = path.stem
    for model in MODEL_LABELS:
        if model in stem:
            return model
    return stem


def _read_fit(path_text: str):
    import numpy as np

    path = Path(path_text)
    with np.load(path, allow_pickle=True) as payload:
        data = {key: payload[key] for key in payload.files}
    model = _model_name(path, data)
    return model, path, data


def _label_index(label_names, label: str) -> int | None:
    for index, name in enumerate(label_names):
        if str(name) == label:
            return index
    return None


def _finite(values):
    import numpy as np

    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _robust_stats(values):
    import numpy as np

    values = _finite(values)
    if values.size == 0:
        return {
            "n": 0,
            "median": float("nan"),
            "mad_sigma": float("nan"),
            "p16": float("nan"),
            "p84": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    median = float(np.median(values))
    return {
        "n": int(values.size),
        "median": median,
        "mad_sigma": float(1.4826 * np.median(np.abs(values - median))),
        "p16": float(np.percentile(values, 16)),
        "p84": float(np.percentile(values, 84)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _running_median(x, y, bins=24, min_count=20):
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < min_count:
        return None
    x = x[valid]
    y = y[valid]
    edges = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(bins, np.nan)
    lo = np.full(bins, np.nan)
    hi = np.full(bins, np.nan)
    for i in range(bins):
        take = (x >= edges[i]) & (x < edges[i + 1])
        if np.count_nonzero(take) < min_count:
            continue
        vals = y[take]
        med[i] = np.nanmedian(vals)
        lo[i] = np.nanpercentile(vals, 16)
        hi[i] = np.nanpercentile(vals, 84)
    good = np.isfinite(med)
    if not np.any(good):
        return None
    return centers[good], med[good], lo[good], hi[good]


def _format_axes(ax):
    ax.grid(alpha=0.18, linewidth=0.55)
    ax.tick_params(direction="in", top=True, right=True)


def _save(fig, output_dir: Path, name: str, dpi: int):
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png", dpi=dpi)


def _quality_mask(data: dict, *, min_npix: int, max_boundary_distance: float, chi2_q: float, mae_q: float):
    import numpy as np

    n = len(data["success"])
    success = np.asarray(data["success"], dtype=bool)
    npix = np.asarray(data.get("npix", np.zeros(n)), dtype=float)
    chi2 = np.asarray(data.get("chi2", np.full(n, np.nan)), dtype=float)
    mae = np.asarray(data.get("mae_x1e4", np.full(n, np.nan)), dtype=float)
    scaled = np.asarray(data.get("fitted_label_scaled", np.full((n, 1), np.nan)), dtype=float)
    finite_labels = np.all(np.isfinite(scaled), axis=1)
    boundary_ok = np.nanmax(np.abs(scaled), axis=1) <= max_boundary_distance

    base = success & finite_labels & boundary_ok & (npix >= min_npix) & np.isfinite(chi2) & np.isfinite(mae)
    if np.count_nonzero(base) == 0:
        return base, {"chi2_max": float("nan"), "mae_max": float("nan")}
    chi2_max = float(np.nanquantile(chi2[base], chi2_q))
    mae_max = float(np.nanquantile(mae[base], mae_q))
    quality = base & (chi2 <= chi2_max) & (mae <= mae_max)
    return quality, {"chi2_max": chi2_max, "mae_max": mae_max}


def _residual_arrays(data: dict, fit_label: str, aspcap_label: str):
    import numpy as np

    label_names = [str(item) for item in data["label_names"]]
    idx = _label_index(label_names, fit_label)
    key = f"aspcap_{aspcap_label}"
    if idx is None or key not in data:
        return None
    fitted = np.asarray(data["fitted_labels"], dtype=float)[:, idx]
    reference = np.asarray(data[key], dtype=float)
    residual = fitted - reference
    return reference, fitted, residual


def write_summary_tables(fits, quality_masks, thresholds, output_dir: Path):
    import numpy as np

    residual_rows = []
    quality_rows = []
    for model, path, data in fits:
        quality = quality_masks[model]
        success = np.asarray(data["success"], dtype=bool)
        quality_rows.append(
            {
                "model": model,
                "label": MODEL_LABELS.get(model, model),
                "fit_file": str(path),
                "n_total": int(success.size),
                "n_success": int(np.count_nonzero(success)),
                "n_quality": int(np.count_nonzero(quality)),
                "success_fraction": float(np.count_nonzero(success) / success.size),
                "quality_fraction": float(np.count_nonzero(quality) / success.size),
                "chi2_max": thresholds[model]["chi2_max"],
                "mae_x1e4_max": thresholds[model]["mae_max"],
                **{f"chi2_{k}": v for k, v in _robust_stats(np.asarray(data["chi2"], dtype=float)[quality]).items()},
                **{f"mae_x1e4_{k}": v for k, v in _robust_stats(np.asarray(data["mae_x1e4"], dtype=float)[quality]).items()},
            }
        )
        for fit_label, asp_label, axis_label, unit in LABEL_PAIRS:
            arrays = _residual_arrays(data, fit_label, asp_label)
            if arrays is None:
                continue
            reference, fitted, residual = arrays
            valid = quality & np.isfinite(reference) & np.isfinite(fitted) & np.isfinite(residual)
            stats = _robust_stats(residual[valid])
            residual_rows.append(
                {
                    "model": model,
                    "label": fit_label,
                    "axis_label": axis_label,
                    "unit": unit,
                    "aspcap_label": asp_label,
                    **stats,
                }
            )

    for name, rows in [
        ("apogee_quality_summary.csv", quality_rows),
        ("apogee_residual_summary.csv", residual_rows),
    ]:
        if not rows:
            continue
        with (output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "apogee_diagnostics_summary.json").open("w") as handle:
        json.dump({"quality": quality_rows, "residuals": residual_rows}, handle, indent=2, sort_keys=True)


def plot_reference_comparison(fits, quality_masks, output_dir: Path, dpi: int):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(LABEL_PAIRS), len(fits), figsize=(4.05 * len(fits), 9.4))
    if len(fits) == 1:
        axes = np.asarray(axes)[:, None]
    for col, (model, _, data) in enumerate(fits):
        quality = quality_masks[model]
        for row, (fit_label, asp_label, axis_label, unit) in enumerate(LABEL_PAIRS):
            ax = axes[row, col]
            arrays = _residual_arrays(data, fit_label, asp_label)
            if arrays is None:
                ax.set_axis_off()
                continue
            reference, fitted, residual = arrays
            valid = quality & np.isfinite(reference) & np.isfinite(fitted)
            hb = ax.hexbin(reference[valid], fitted[valid], gridsize=55, bins="log", mincnt=1, cmap="viridis")
            both = np.concatenate([reference[valid], fitted[valid]])
            low, high = np.nanpercentile(both, [1, 99])
            ax.plot([low, high], [low, high], color="crimson", linewidth=0.9)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            stats = _robust_stats(residual[valid])
            ax.text(
                0.04,
                0.96,
                f"N={stats['n']}\nmed={stats['median']:.2g}\nMAD={stats['mad_sigma']:.2g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
            )
            if row == 0:
                ax.set_title(MODEL_LABELS.get(model, model))
            ax.set_xlabel(f"ASPCAP {axis_label} ({unit})")
            ax.set_ylabel(f"Fitted {axis_label} ({unit})")
            _format_axes(ax)
    fig.colorbar(hb, ax=axes.ravel().tolist(), shrink=0.72, label="log density")
    _save(fig, output_dir, "apogee_aspcap_hexbin_grid", dpi)
    plt.close(fig)


def plot_residual_trends(fits, quality_masks, output_dir: Path, dpi: int):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(LABEL_PAIRS), len(fits), figsize=(4.05 * len(fits), 9.4))
    if len(fits) == 1:
        axes = np.asarray(axes)[:, None]
    for col, (model, _, data) in enumerate(fits):
        quality = quality_masks[model]
        for row, (fit_label, asp_label, axis_label, unit) in enumerate(LABEL_PAIRS):
            ax = axes[row, col]
            arrays = _residual_arrays(data, fit_label, asp_label)
            if arrays is None:
                ax.set_axis_off()
                continue
            reference, _, residual = arrays
            valid = quality & np.isfinite(reference) & np.isfinite(residual)
            ax.axhline(0.0, color="0.25", linestyle="--", linewidth=0.8)
            ax.hexbin(reference[valid], residual[valid], gridsize=55, bins="log", mincnt=1, cmap="magma")
            trend = _running_median(reference[valid], residual[valid])
            if trend is not None:
                x, med, lo, hi = trend
                ax.plot(x, med, color="#00bcd4", linewidth=1.6, label="running median")
                ax.fill_between(x, lo, hi, color="#00bcd4", alpha=0.18, linewidth=0)
            ylo, yhi = np.nanpercentile(residual[valid], [1, 99])
            pad = 0.05 * (yhi - ylo) if np.isfinite(yhi - ylo) else 1.0
            ax.set_ylim(ylo - pad, yhi + pad)
            if row == 0:
                ax.set_title(MODEL_LABELS.get(model, model))
            ax.set_xlabel(f"ASPCAP {axis_label} ({unit})")
            ax.set_ylabel(f"Fit - ASPCAP {axis_label} ({unit})")
            _format_axes(ax)
    _save(fig, output_dir, "apogee_residual_trend_grid", dpi)
    plt.close(fig)


def plot_residual_distributions(fits, quality_masks, output_dir: Path, dpi: int):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(LABEL_PAIRS), figsize=(4.0 * len(LABEL_PAIRS), 3.6))
    for ax, (fit_label, asp_label, axis_label, unit) in zip(axes, LABEL_PAIRS):
        for model, _, data in fits:
            arrays = _residual_arrays(data, fit_label, asp_label)
            if arrays is None:
                continue
            _, _, residual = arrays
            valid = quality_masks[model] & np.isfinite(residual)
            vals = residual[valid]
            if vals.size == 0:
                continue
            lo, hi = np.nanpercentile(vals, [0.5, 99.5])
            bins = np.linspace(lo, hi, 90)
            ax.hist(
                vals,
                bins=bins,
                histtype="step",
                density=True,
                linewidth=1.3,
                color=MODEL_COLORS.get(model),
                label=MODEL_LABELS.get(model, model),
            )
        ax.axvline(0, color="0.25", linestyle="--", linewidth=0.8)
        ax.set_xlabel(f"Fit - ASPCAP {axis_label} ({unit})")
        ax.set_ylabel("Density")
        _format_axes(ax)
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, output_dir, "apogee_residual_distribution", dpi)
    plt.close(fig)


def plot_quality_diagnostics(fits, quality_masks, output_dir: Path, dpi: int):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    positions = np.arange(len(fits))
    labels = [MODEL_LABELS.get(model, model) for model, _, _ in fits]
    n_success = []
    n_quality = []
    for model, _, data in fits:
        n_success.append(np.count_nonzero(np.asarray(data["success"], dtype=bool)))
        n_quality.append(np.count_nonzero(quality_masks[model]))
    axes[0].bar(positions - 0.18, n_success, width=0.36, color="0.65", label="success")
    axes[0].bar(positions + 0.18, n_quality, width=0.36, color="#4c78a8", label="quality")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Number of spectra")
    axes[0].legend(frameon=False, fontsize=8)
    _format_axes(axes[0])

    for model, _, data in fits:
        quality = quality_masks[model]
        chi2 = np.asarray(data["chi2"], dtype=float)
        mae = np.asarray(data["mae_x1e4"], dtype=float)
        for ax, values, xlabel in [
            (axes[1], chi2[quality], r"Reduced $\chi^2$ proxy"),
            (axes[2], mae[quality], r"Spectral MAE ($10^{-4}$ flux)"),
        ]:
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            lo, hi = np.nanpercentile(values, [0.5, 99.5])
            bins = np.linspace(lo, hi, 80)
            ax.hist(values, bins=bins, histtype="step", density=True, linewidth=1.3, color=MODEL_COLORS.get(model), label=MODEL_LABELS.get(model, model))
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Density")
            _format_axes(ax)
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, output_dir, "apogee_quality_diagnostics", dpi)
    plt.close(fig)


def plot_kiel_and_systematics(fits, quality_masks, output_dir: Path, dpi: int):
    import numpy as np
    import matplotlib.pyplot as plt

    chosen = None
    for fit in fits:
        if fit[0] == "kan_payne":
            chosen = fit
            break
    if chosen is None:
        chosen = fits[0]
    model, _, data = chosen
    quality = quality_masks[model]

    label_names = [str(item) for item in data["label_names"]]
    teff_idx = _label_index(label_names, "Teff")
    logg_idx = _label_index(label_names, "logg")
    feh_idx = _label_index(label_names, "Fe_H")
    if teff_idx is None or logg_idx is None or feh_idx is None or "aspcap_FE_H" not in data:
        return
    fitted = np.asarray(data["fitted_labels"], dtype=float)
    teff = fitted[:, teff_idx]
    logg = fitted[:, logg_idx]
    feh_resid = fitted[:, feh_idx] - np.asarray(data["aspcap_FE_H"], dtype=float)
    teff_resid = teff - np.asarray(data["aspcap_TEFF"], dtype=float)
    logg_resid = logg - np.asarray(data["aspcap_LOGG"], dtype=float)
    valid = quality & np.isfinite(teff) & np.isfinite(logg) & np.isfinite(feh_resid)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True, sharey=True)
    panels = [
        (feh_resid, r"Fit [Fe/H] $-$ ASPCAP [Fe/H]", -0.4, 0.4),
        (teff_resid, r"Fit $T_{\rm eff}$ $-$ ASPCAP $T_{\rm eff}$", -250, 250),
        (logg_resid, r"Fit $\log g$ $-$ ASPCAP $\log g$", -0.5, 0.5),
    ]
    for ax, (color_value, cbar_label, vmin, vmax) in zip(axes, panels):
        c = np.asarray(color_value, dtype=float)
        take = valid & np.isfinite(c)
        sc = ax.scatter(teff[take], logg[take], c=c[take], s=5, cmap="coolwarm", vmin=vmin, vmax=vmax, linewidths=0)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel(r"Fitted $T_{\rm eff}$ (K)")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(cbar_label)
        _format_axes(ax)
    axes[0].set_ylabel(r"Fitted $\log g$")
    axes[1].set_title(f"{MODEL_LABELS.get(model, model)} quality sample")
    _save(fig, output_dir, "apogee_kiel_systematics", dpi)
    plt.close(fig)


def plot_model_to_model(fits, quality_masks, output_dir: Path, dpi: int):
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt

    if len(fits) < 2:
        return
    pairs = list(itertools.combinations(fits, 2))
    fig, axes = plt.subplots(len(LABEL_PAIRS), len(pairs), figsize=(4.1 * len(pairs), 9.2))
    if len(pairs) == 1:
        axes = np.asarray(axes)[:, None]
    for col, ((m1, _, d1), (m2, _, d2)) in enumerate(pairs):
        q = quality_masks[m1] & quality_masks[m2]
        names1 = [str(item) for item in d1["label_names"]]
        names2 = [str(item) for item in d2["label_names"]]
        for row, (fit_label, _, axis_label, unit) in enumerate(LABEL_PAIRS):
            ax = axes[row, col]
            i1 = _label_index(names1, fit_label)
            i2 = _label_index(names2, fit_label)
            if i1 is None or i2 is None:
                ax.set_axis_off()
                continue
            y1 = np.asarray(d1["fitted_labels"], dtype=float)[:, i1]
            y2 = np.asarray(d2["fitted_labels"], dtype=float)[:, i2]
            residual = y1 - y2
            valid = q & np.isfinite(y1) & np.isfinite(y2)
            ax.axhline(0, color="0.25", linestyle="--", linewidth=0.8)
            ax.hexbin(y2[valid], residual[valid], gridsize=55, bins="log", mincnt=1, cmap="cividis")
            stats = _robust_stats(residual[valid])
            ax.text(0.04, 0.96, f"MAD={stats['mad_sigma']:.2g}", transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2})
            if row == 0:
                ax.set_title(f"{MODEL_LABELS.get(m1, m1)} - {MODEL_LABELS.get(m2, m2)}")
            ax.set_xlabel(f"{MODEL_LABELS.get(m2, m2)} {axis_label} ({unit})")
            ax.set_ylabel(f"Difference {axis_label} ({unit})")
            _format_axes(ax)
    _save(fig, output_dir, "apogee_model_to_model_residuals", dpi)
    plt.close(fig)


def write_paper_snippets(output_dir: Path):
    text = """# APOGEE DR17 Inference Figure Descriptions

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
"""
    (output_dir / "apogee_figure_descriptions.md").write_text(text)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    fits = [_read_fit(path) for path in args.fit]
    order = {name: index for index, name in enumerate(MODEL_LABELS)}
    fits.sort(key=lambda item: order.get(item[0], 99))

    quality_masks = {}
    thresholds = {}
    for model, _, data in fits:
        mask, threshold = _quality_mask(
            data,
            min_npix=args.min_npix,
            max_boundary_distance=args.max_boundary_distance,
            chi2_q=args.chi2_quantile,
            mae_q=args.mae_quantile,
        )
        quality_masks[model] = mask
        thresholds[model] = threshold

    write_summary_tables(fits, quality_masks, thresholds, output_dir)
    plot_reference_comparison(fits, quality_masks, output_dir, args.dpi)
    plot_residual_trends(fits, quality_masks, output_dir, args.dpi)
    plot_residual_distributions(fits, quality_masks, output_dir, args.dpi)
    plot_quality_diagnostics(fits, quality_masks, output_dir, args.dpi)
    plot_kiel_and_systematics(fits, quality_masks, output_dir, args.dpi)
    plot_model_to_model(fits, quality_masks, output_dir, args.dpi)
    write_paper_snippets(output_dir)
    print(f"Wrote APOGEE diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
