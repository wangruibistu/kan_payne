#!/usr/bin/env python
"""Make APOGEE atmospheric-label comparison figures with atmospheric-label QC."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_LABELS = {
    "payne_mlp": "Payne-MLP",
    "kan_payne": "KAN-Payne",
    "transformer_payne": "TransformerPayne",
}
LABELS = [
    ("Teff", "TEFF", r"$T_{\rm eff}$", "K"),
    ("logg", "LOGG", r"$\log g$", "dex"),
    ("Fe_H", "FE_H", "[Fe/H]", "dex"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _infer_model(path: Path) -> str:
    name = path.name
    for key in MODEL_LABELS:
        if key in name:
            return key
    return path.stem


def _mad_sigma(values):
    import numpy as np

    med = np.nanmedian(values)
    return 1.4826 * np.nanmedian(np.abs(values - med))


def _quality(data, name_idx):
    import numpy as np

    success = data["success"].astype(bool)
    good = success & (data["npix"] >= 5000)
    for label, _, _, _ in LABELS:
        good &= np.abs(data["fitted_label_scaled"][:, name_idx[label]]) < 0.49
    chi2_cut = np.nanquantile(data["chi2"][success & (data["npix"] >= 5000)], 0.99)
    mae_cut = np.nanquantile(data["mae_x1e4"][success & (data["npix"] >= 5000)], 0.99)
    good &= (data["chi2"] <= chi2_cut) & (data["mae_x1e4"] <= mae_cut)
    return good, float(chi2_cut), float(mae_cut)


def main():
    args = parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    loaded = []
    quality_rows = []
    residual_rows = []
    for fit in args.fit:
        path = Path(fit)
        data = np.load(path, allow_pickle=True)
        model = _infer_model(path)
        names = [str(item) for item in data["label_names"]]
        name_idx = {name: i for i, name in enumerate(names)}
        quality, chi2_cut, mae_cut = _quality(data, name_idx)
        loaded.append((model, MODEL_LABELS.get(model, model), data, name_idx, quality))
        quality_rows.append(
            {
                "model": model,
                "label": MODEL_LABELS.get(model, model),
                "n_total": int(quality.size),
                "n_success": int(data["success"].sum()),
                "n_atmospheric_quality": int(quality.sum()),
                "atmospheric_quality_fraction": float(quality.mean()),
                "chi2_cut": chi2_cut,
                "mae_x1e4_cut": mae_cut,
                "chi2_median": float(np.nanmedian(data["chi2"][quality])),
                "mae_x1e4_median": float(np.nanmedian(data["mae_x1e4"][quality])),
            }
        )
        for label, aspcap, _, unit in LABELS:
            pred = data["fitted_labels"][:, name_idx[label]]
            ref = data[f"aspcap_{aspcap}"]
            m = quality & np.isfinite(pred) & np.isfinite(ref)
            res = pred[m] - ref[m]
            residual_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "label": label,
                    "unit": unit,
                    "n": int(m.sum()),
                    "median": float(np.nanmedian(res)),
                    "mad_sigma": float(_mad_sigma(res)),
                    "p16": float(np.nanpercentile(res, 16)),
                    "p84": float(np.nanpercentile(res, 84)),
                    "mean": float(np.nanmean(res)),
                    "std": float(np.nanstd(res)),
                }
            )

    with (out / "apogee_atmospheric_quality_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quality_rows[0].keys()))
        writer.writeheader()
        writer.writerows(quality_rows)
    with (out / "apogee_atmospheric_residual_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(residual_rows[0].keys()))
        writer.writeheader()
        writer.writerows(residual_rows)
    (out / "apogee_atmospheric_summary.json").write_text(
        json.dumps({"quality": quality_rows, "residuals": residual_rows}, indent=2, sort_keys=True)
        + "\n"
    )

    fig, axes = plt.subplots(len(LABELS), len(loaded), figsize=(11, 8.5), constrained_layout=True)
    for col, (model, model_label, data, name_idx, quality) in enumerate(loaded):
        for row, (label, aspcap, axis_label, unit) in enumerate(LABELS):
            ax = axes[row, col]
            pred = data["fitted_labels"][:, name_idx[label]]
            ref = data[f"aspcap_{aspcap}"]
            m = quality & np.isfinite(pred) & np.isfinite(ref)
            ax.hexbin(ref[m], pred[m], gridsize=45, bins="log", mincnt=1, cmap="viridis")
            lo = float(np.nanpercentile(np.concatenate([ref[m], pred[m]]), 1))
            hi = float(np.nanpercentile(np.concatenate([ref[m], pred[m]]), 99))
            ax.plot([lo, hi], [lo, hi], color="crimson", lw=1)
            res = pred[m] - ref[m]
            ax.text(
                0.04,
                0.96,
                f"N={m.sum()}\nmed={np.nanmedian(res):.3g}\nscatter={_mad_sigma(res):.3g}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
            if row == 0:
                ax.set_title(model_label)
            if col == 0:
                ax.set_ylabel(f"Fitted {axis_label}")
            if row == len(LABELS) - 1:
                ax.set_xlabel(f"ASPCAP {axis_label}")
    fig.savefig(out / "apogee_atmospheric_aspcap_hexbin_grid.pdf")
    fig.savefig(out / "apogee_atmospheric_aspcap_hexbin_grid.png", dpi=args.dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(LABELS), figsize=(11, 3.5), constrained_layout=True)
    colors = {"payne_mlp": "tab:blue", "kan_payne": "tab:green", "transformer_payne": "tab:orange"}
    for ax, (label, aspcap, axis_label, unit) in zip(axes, LABELS):
        for model, model_label, data, name_idx, quality in loaded:
            pred = data["fitted_labels"][:, name_idx[label]]
            ref = data[f"aspcap_{aspcap}"]
            m = quality & np.isfinite(pred) & np.isfinite(ref)
            res = pred[m] - ref[m]
            ax.hist(
                res,
                bins=70,
                density=True,
                histtype="step",
                lw=1.5,
                color=colors.get(model),
                label=model_label,
            )
        ax.axvline(0, color="0.25", lw=1, ls="--")
        ax.set_xlabel(f"Fitted - ASPCAP {axis_label} ({unit})")
        ax.set_ylabel("Density")
    axes[0].legend(fontsize=8)
    fig.savefig(out / "apogee_atmospheric_residual_distribution.pdf")
    fig.savefig(out / "apogee_atmospheric_residual_distribution.png", dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote atmospheric APOGEE diagnostics to {out}")


if __name__ == "__main__":
    main()
