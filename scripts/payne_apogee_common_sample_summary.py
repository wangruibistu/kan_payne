#!/usr/bin/env python
"""Compute APOGEE DR17 atmospheric-label metrics on a common three-model sample."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LABELS = [
    ("Teff", "TEFF", r"$T_{\rm eff}$", "K"),
    ("logg", "LOGG", r"$\log g$", "dex"),
    ("Fe_H", "FE_H", "[Fe/H]", "dex"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-npix", type=int, default=5000)
    parser.add_argument("--boundary", type=float, default=0.49)
    parser.add_argument("--chi2-quantile", type=float, default=0.99)
    parser.add_argument("--mae-quantile", type=float, default=0.99)
    return parser.parse_args()


def _model_name(path: Path) -> str:
    name = path.name
    if "kan_payne" in name:
        return "KAN-Payne"
    if "transformer_payne" in name:
        return "TransformerPayne"
    if "payne_mlp" in name:
        return "Payne-MLP"
    return path.stem


def _mad_sigma(values):
    import numpy as np

    med = np.nanmedian(values)
    return 1.4826 * np.nanmedian(np.abs(values - med))


def _per_model_quality(data, label_index, args):
    import numpy as np

    success = data["success"].astype(bool)
    base = success & (data["npix"] >= args.min_npix)
    for label, _, _, _ in LABELS:
        base &= np.abs(data["fitted_label_scaled"][:, label_index[label]]) < args.boundary
    fit_base = success & (data["npix"] >= args.min_npix)
    chi2_cut = np.nanquantile(data["chi2"][fit_base], args.chi2_quantile)
    mae_cut = np.nanquantile(data["mae_x1e4"][fit_base], args.mae_quantile)
    return base & (data["chi2"] <= chi2_cut) & (data["mae_x1e4"] <= mae_cut), chi2_cut, mae_cut


def main():
    args = parse_args()

    import numpy as np

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = []
    common = None
    for fit in args.fit:
        path = Path(fit)
        data = np.load(path, allow_pickle=True)
        names = [str(item) for item in data["label_names"]]
        label_index = {name: i for i, name in enumerate(names)}
        quality, chi2_cut, mae_cut = _per_model_quality(data, label_index, args)
        if common is None:
            common = quality.copy()
        else:
            common &= quality
        models.append(
            {
                "name": _model_name(path),
                "path": str(path),
                "data": data,
                "label_index": label_index,
                "quality": quality,
                "chi2_cut": float(chi2_cut),
                "mae_cut": float(mae_cut),
            }
        )

    if common is None:
        raise SystemExit("No fit files supplied")

    rows = []
    quality_rows = []
    for model in models:
        data = model["data"]
        label_index = model["label_index"]
        quality_rows.append(
            {
                "model": model["name"],
                "n_total": int(common.size),
                "n_model_quality": int(model["quality"].sum()),
                "n_common_quality": int(common.sum()),
                "model_quality_fraction": float(model["quality"].mean()),
                "common_quality_fraction": float(common.mean()),
                "chi2_cut": model["chi2_cut"],
                "mae_x1e4_cut": model["mae_cut"],
                "chi2_median_common": float(np.nanmedian(data["chi2"][common])),
                "mae_x1e4_median_common": float(np.nanmedian(data["mae_x1e4"][common])),
            }
        )
        for label, aspcap, _, unit in LABELS:
            pred = data["fitted_labels"][:, label_index[label]]
            ref = data[f"aspcap_{aspcap}"]
            mask = common & np.isfinite(pred) & np.isfinite(ref)
            residual = pred[mask] - ref[mask]
            rows.append(
                {
                    "model": model["name"],
                    "label": label,
                    "unit": unit,
                    "n": int(mask.sum()),
                    "median": float(np.nanmedian(residual)),
                    "mad_sigma": float(_mad_sigma(residual)),
                    "p16": float(np.nanpercentile(residual, 16)),
                    "p84": float(np.nanpercentile(residual, 84)),
                    "mean": float(np.nanmean(residual)),
                    "std": float(np.nanstd(residual)),
                }
            )

    with (out / "apogee_common_quality_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quality_rows[0].keys()))
        writer.writeheader()
        writer.writerows(quality_rows)
    with (out / "apogee_common_residual_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out / "apogee_common_summary.json").write_text(
        json.dumps({"quality": quality_rows, "residuals": rows}, indent=2, sort_keys=True) + "\n"
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(LABELS), len(models), figsize=(11, 8.5), constrained_layout=True)
    for col, model in enumerate(models):
        data = model["data"]
        label_index = model["label_index"]
        for row_idx, (label, aspcap, axis_label, _) in enumerate(LABELS):
            ax = axes[row_idx, col]
            pred = data["fitted_labels"][:, label_index[label]]
            ref = data[f"aspcap_{aspcap}"]
            mask = common & np.isfinite(pred) & np.isfinite(ref)
            hb = ax.hexbin(ref[mask], pred[mask], gridsize=45, bins="log", mincnt=1, cmap="viridis")
            del hb
            lo = float(np.nanpercentile(np.concatenate([ref[mask], pred[mask]]), 1))
            hi = float(np.nanpercentile(np.concatenate([ref[mask], pred[mask]]), 99))
            ax.plot([lo, hi], [lo, hi], color="crimson", lw=1)
            residual = pred[mask] - ref[mask]
            ax.text(
                0.04,
                0.96,
                f"N={mask.sum()}\nmed={np.nanmedian(residual):.3g}\nscatter={_mad_sigma(residual):.3g}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
            if row_idx == 0:
                ax.set_title(model["name"])
            if col == 0:
                ax.set_ylabel(f"Fitted {axis_label}")
            if row_idx == len(LABELS) - 1:
                ax.set_xlabel(f"ASPCAP {axis_label}")
    fig.savefig(out / "apogee_common_aspcap_hexbin_grid.pdf")
    fig.savefig(out / "apogee_common_aspcap_hexbin_grid.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(LABELS), figsize=(11, 3.5), constrained_layout=True)
    colors = {"Payne-MLP": "tab:blue", "KAN-Payne": "tab:green", "TransformerPayne": "tab:orange"}
    for ax, (label, aspcap, axis_label, unit) in zip(axes, LABELS):
        for model in models:
            data = model["data"]
            label_index = model["label_index"]
            pred = data["fitted_labels"][:, label_index[label]]
            ref = data[f"aspcap_{aspcap}"]
            mask = common & np.isfinite(pred) & np.isfinite(ref)
            residual = pred[mask] - ref[mask]
            ax.hist(
                residual,
                bins=70,
                density=True,
                histtype="step",
                lw=1.5,
                color=colors.get(model["name"]),
                label=model["name"],
            )
        ax.axvline(0, color="0.25", lw=1, ls="--")
        ax.set_xlabel(f"Fitted - ASPCAP {axis_label} ({unit})")
        ax.set_ylabel("Density")
    axes[0].legend(fontsize=8)
    fig.savefig(out / "apogee_common_residual_distribution.pdf")
    fig.savefig(out / "apogee_common_residual_distribution.png", dpi=220)
    plt.close(fig)

    print(json.dumps({"quality": quality_rows, "residuals": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
