#!/usr/bin/env python
"""Create paper-style figures for KAN-Payne emulator and APOGEE analyses."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="paper/figures")
    parser.add_argument(
        "--apogee-fit",
        nargs="*",
        default=[],
        help="Optional APOGEE fit NPZ files from payne_fit_apogee_spectra.py.",
    )
    parser.add_argument(
        "--apogee-observed",
        default=None,
        help="Optional APOGEE NPZ on the Payne grid; used for a sample observed-spectrum figure.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def _history(run_dir: Path, model: str):
    path = run_dir / "emulators" / model / f"{model}_history.json"
    if not path.exists():
        return None
    return _load_json(path)["history"]


def _residual_npz(run_dir: Path, model: str) -> Path:
    return run_dir / "evaluations" / f"{model}_valid_residuals.npz"


def _format_axes(ax):
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.tick_params(direction="in", top=True, right=True)


def _save(fig, output_dir: Path, name: str, dpi: int):
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png", dpi=dpi)


def plot_training_histories(run_dir: Path, output_dir: Path, dpi: int):
    import matplotlib.pyplot as plt

    histories = {model: _history(run_dir, model) for model in MODEL_LABELS}
    histories = {model: history for model, history in histories.items() if history}
    if not histories:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for model, history in histories.items():
        ax.plot(
            [row["epoch"] for row in history],
            [row["valid_good_mae_x1e4"] for row in history],
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            linewidth=1.4,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Validation MAE, unmasked pixels ($10^{-4}$ flux)")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    _format_axes(ax)
    _save(fig, output_dir, "validation_mae_history", dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for model, history in histories.items():
        ax.plot(
            [row["epoch"] for row in history],
            [row["train_l1_x1e4"] for row in history],
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            linewidth=1.4,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Training L1 ($10^{-4}$ flux)")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    _format_axes(ax)
    _save(fig, output_dir, "training_l1_history", dpi)
    plt.close(fig)


def plot_synthetic_residuals(run_dir: Path, output_dir: Path, dpi: int):
    import numpy as np
    import matplotlib.pyplot as plt

    residuals = {}
    wavelength = None
    good = None
    for model in MODEL_LABELS:
        path = _residual_npz(run_dir, model)
        if not path.exists():
            continue
        with np.load(path) as payload:
            residuals[model] = np.asarray(payload["residual"], dtype=float)
            wavelength = np.asarray(payload["wavelength"], dtype=float)
            good = ~np.asarray(payload["mask"], dtype=bool)
    if not residuals:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bins = np.linspace(-1000, 1000, 160)
    for model, residual in residuals.items():
        data = (residual[:, good].ravel() * 1.0e4) if good is not None else residual.ravel() * 1.0e4
        data = data[np.isfinite(data)]
        ax.hist(
            data,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=1.4,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    ax.set_xlabel(r"Validation residual ($10^{-4}$ flux)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    _format_axes(ax)
    _save(fig, output_dir, "synthetic_residual_histogram", dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for model, residual in residuals.items():
        pixel_mae = np.nanmean(np.abs(residual), axis=0) * 1.0e4
        ax.plot(
            wavelength,
            pixel_mae,
            linewidth=0.8,
            alpha=0.8,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    if good is not None:
        masked_wave = wavelength[~good]
        if masked_wave.size:
            ax.scatter(
                masked_wave,
                np.full(masked_wave.shape, ax.get_ylim()[0]),
                s=1,
                color="0.75",
                marker="|",
                label="Payne mask",
            )
    ax.set_xlabel(r"Wavelength ($\AA$)")
    ax.set_ylabel(r"Per-pixel MAE ($10^{-4}$ flux)")
    ax.legend(frameon=False, ncol=2)
    _format_axes(ax)
    _save(fig, output_dir, "pixel_residual_summary", dpi)
    plt.close(fig)


def _label_index(label_names, label):
    for index, name in enumerate(label_names):
        if str(name) == label:
            return index
    return None


def _read_fit(path: Path):
    import numpy as np

    with np.load(path, allow_pickle=True) as payload:
        data = {key: payload[key] for key in payload.files}
    model = str(data.get("model", path.stem))
    if hasattr(data.get("model"), "tolist"):
        model = str(data["model"].tolist())
    return model, data


def plot_apogee_comparisons(fit_paths, output_dir: Path, dpi: int):
    import numpy as np
    import matplotlib.pyplot as plt

    if not fit_paths:
        return
    fits = []
    for path_text in fit_paths:
        path = Path(path_text)
        if path.exists():
            fits.append(_read_fit(path))
    if not fits:
        return

    labels = [
        ("Teff", "TEFF", r"$T_{\rm eff}$", "K"),
        ("logg", "LOGG", r"$\log g$", "dex"),
        ("Fe_H", "FE_H", r"[Fe/H]", "dex"),
    ]

    fig, axes = plt.subplots(len(labels), len(fits), figsize=(4.0 * len(fits), 10.0))
    if len(fits) == 1:
        axes = np.asarray(axes)[:, None]
    for col, (model, data) in enumerate(fits):
        label_names = [str(item) for item in data["label_names"]]
        fitted = np.asarray(data["fitted_labels"], dtype=float)
        success = np.asarray(data["success"], dtype=bool)
        for row, (fit_label, asp_label, axis_label, unit) in enumerate(labels):
            ax = axes[row, col]
            idx = _label_index(label_names, fit_label)
            key = f"aspcap_{asp_label}"
            if idx is None or key not in data:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.set_axis_off()
                continue
            x = np.asarray(data[key], dtype=float)
            y = fitted[:, idx]
            valid = success & np.isfinite(x) & np.isfinite(y)
            ax.hexbin(x[valid], y[valid], gridsize=45, bins="log", mincnt=1, cmap="viridis")
            low = np.nanpercentile(np.concatenate([x[valid], y[valid]]), 1)
            high = np.nanpercentile(np.concatenate([x[valid], y[valid]]), 99)
            ax.plot([low, high], [low, high], color="crimson", linewidth=1.0)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_xlabel(f"ASPCAP {axis_label} ({unit})")
            ax.set_ylabel(f"Fitted {axis_label} ({unit})")
            if row == 0:
                ax.set_title(MODEL_LABELS.get(model, model))
            _format_axes(ax)
    _save(fig, output_dir, "apogee_aspcap_comparison", dpi)
    plt.close(fig)

    fig, axes = plt.subplots(len(labels), len(fits), figsize=(4.0 * len(fits), 10.0), sharex=False)
    if len(fits) == 1:
        axes = np.asarray(axes)[:, None]
    for col, (model, data) in enumerate(fits):
        label_names = [str(item) for item in data["label_names"]]
        fitted = np.asarray(data["fitted_labels"], dtype=float)
        success = np.asarray(data["success"], dtype=bool)
        for row, (fit_label, asp_label, axis_label, unit) in enumerate(labels):
            ax = axes[row, col]
            idx = _label_index(label_names, fit_label)
            key = f"aspcap_{asp_label}"
            if idx is None or key not in data:
                ax.set_axis_off()
                continue
            x = np.asarray(data[key], dtype=float)
            residual = fitted[:, idx] - x
            valid = success & np.isfinite(x) & np.isfinite(residual)
            ax.axhline(0, color="crimson", linewidth=1.0)
            ax.hexbin(x[valid], residual[valid], gridsize=45, bins="log", mincnt=1, cmap="magma")
            ax.set_xlabel(f"ASPCAP {axis_label} ({unit})")
            ax.set_ylabel(f"Fit - ASPCAP {axis_label} ({unit})")
            if row == 0:
                ax.set_title(MODEL_LABELS.get(model, model))
            _format_axes(ax)
    _save(fig, output_dir, "apogee_residual_trends", dpi)
    plt.close(fig)

    # KAN-Payne centered Kiel diagram, because this is the proposed method.
    chosen = None
    for model, data in fits:
        if model == "kan_payne":
            chosen = (model, data)
            break
    if chosen is None:
        chosen = fits[0]
    model, data = chosen
    label_names = [str(item) for item in data["label_names"]]
    fitted = np.asarray(data["fitted_labels"], dtype=float)
    success = np.asarray(data["success"], dtype=bool)
    teff_idx = _label_index(label_names, "Teff")
    logg_idx = _label_index(label_names, "logg")
    feh_idx = _label_index(label_names, "Fe_H")
    if teff_idx is not None and logg_idx is not None and feh_idx is not None and "aspcap_FE_H" in data:
        teff = fitted[:, teff_idx]
        logg = fitted[:, logg_idx]
        feh_resid = fitted[:, feh_idx] - np.asarray(data["aspcap_FE_H"], dtype=float)
        valid = success & np.isfinite(teff) & np.isfinite(logg) & np.isfinite(feh_resid)
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        sc = ax.scatter(teff[valid], logg[valid], c=feh_resid[valid], s=5, cmap="coolwarm", vmin=-0.4, vmax=0.4)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel(r"Fitted $T_{\rm eff}$ (K)")
        ax.set_ylabel(r"Fitted $\log g$")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(r"Fitted [Fe/H] $-$ ASPCAP [Fe/H]")
        ax.set_title(MODEL_LABELS.get(model, model))
        _format_axes(ax)
        _save(fig, output_dir, "apogee_kiel_residual_map", dpi)
        plt.close(fig)


def plot_apogee_sample_spectrum(observed_path: str | None, output_dir: Path, dpi: int):
    if not observed_path:
        return

    import json
    import numpy as np
    import matplotlib.pyplot as plt

    path = Path(observed_path)
    if not path.exists():
        return

    with np.load(path, allow_pickle=True) as payload:
        data = {key: payload[key] for key in payload.files}
    wave = np.asarray(data["wave"], dtype=float)
    flux = np.asarray(data["flux"], dtype=float)
    err = np.asarray(data["err"], dtype=float)
    mask = np.asarray(data["mask"], dtype=bool)
    payne_mask = np.asarray(data.get("payne_mask", np.zeros(wave.size, dtype=bool)), dtype=bool)
    continuum_pixels = np.asarray(
        data.get("continuum_pixels", np.zeros(wave.size, dtype=bool)),
        dtype=bool,
    )

    good = np.isfinite(flux) & np.isfinite(err) & (err > 0) & ~mask & ~payne_mask[None, :]
    snr = np.nanmedian(np.where(good, flux / err, np.nan), axis=1)
    flux_for_stats = np.where(good, flux, np.nan)
    median_flux = np.nanmedian(flux_for_stats, axis=1)
    p05_flux = np.nanpercentile(flux_for_stats, 5, axis=1)
    p95_flux = np.nanpercentile(flux_for_stats, 95, axis=1)
    coverage = np.mean(good, axis=1)
    typical = (
        np.isfinite(snr)
        & (snr > 80.0)
        & (snr < 150.0)
        & (median_flux > 0.965)
        & (median_flux < 1.01)
        & (p95_flux < 1.04)
        & (coverage > 0.85)
    )
    if "label_FE_H" in data:
        feh = np.asarray(data["label_FE_H"], dtype=float)
        typical &= np.isfinite(feh) & (feh > -1.0) & (feh < 0.3)
    if "label_LOGG" in data:
        logg = np.asarray(data["label_LOGG"], dtype=float)
        typical &= np.isfinite(logg) & (logg > 1.0) & (logg < 3.8)
    if np.any(typical):
        row = int(np.nanargmax(np.where(typical, p05_flux, -np.inf)))
    else:
        score = (
            np.abs(snr - 120.0) / 120.0
            + np.abs(median_flux - 0.99) * 5.0
            + np.maximum(0.0, 0.82 - p05_flux) * 4.0
            + np.maximum(0.0, p95_flux - 1.04) * 4.0
            - coverage
        )
        finite_score = np.isfinite(score)
        row = int(np.nanargmin(np.where(finite_score, score, np.inf))) if np.any(finite_score) else 0

    metadata = {}
    if "metadata_json" in data:
        try:
            metadata = json.loads(str(data["metadata_json"][row]))
        except Exception:
            metadata = {}

    chips = [
        (0, 2920, "blue detector"),
        (2920, 5320, "green detector"),
        (5320, wave.size, "red detector"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 6.0), sharey=True)
    y_min, y_max = 0.70, 1.075
    for ax, (lo, hi, chip_name) in zip(axes, chips):
        local = slice(lo, hi)
        local_good = good[row, local]
        local_flux = flux[row, local].copy()
        local_flux[~local_good] = np.nan
        ax.plot(
            wave[local],
            local_flux,
            color="#1f4e79",
            linewidth=0.55,
            solid_capstyle="butt",
        )
        cont = continuum_pixels[local] & local_good
        if np.any(cont):
            ax.scatter(
                wave[local][cont],
                local_flux[cont],
                s=3,
                color="#d95f02",
                alpha=0.45,
                linewidths=0,
            )
        ax.axhline(1.0, color="0.25", linewidth=0.7, linestyle="--", zorder=0)
        ax.text(
            0.012,
            0.12,
            chip_name,
            transform=ax.transAxes,
            fontsize=8.5,
            color="0.25",
            ha="left",
            va="bottom",
        )
        ax.set_xlim(wave[lo], wave[hi - 1])
        ax.set_ylim(y_min, y_max)
        _format_axes(ax)
    axes[1].set_ylabel("Continuum-normalized flux")
    axes[-1].set_xlabel(r"Wavelength ($\AA$)")
    title_bits = []
    if metadata.get("APOGEE_ID"):
        title_bits.append(str(metadata["APOGEE_ID"]))
    label_bits = []
    label_map = [
        ("label_TEFF", r"$T_{\rm eff}$", "K", ".0f"),
        ("label_LOGG", r"$\log g$", "", ".2f"),
        ("label_FE_H", r"[Fe/H]", "", ".2f"),
    ]
    for key, label, unit, fmt in label_map:
        if key in data:
            value = float(np.asarray(data[key])[row])
            if np.isfinite(value):
                suffix = f" {unit}" if unit else ""
                label_bits.append(f"{label}={value:{fmt}}{suffix}")
    if np.isfinite(snr[row]):
        title_bits.append(f"median S/N={snr[row]:.0f}")
    if label_bits:
        title_bits.append(", ".join(label_bits))
    if title_bits:
        axes[0].set_title(title_bits[0], fontsize=9.5, pad=6)
    _save(fig, output_dir, "apogee_sample_spectrum", dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    run_dir = Path(args.run_dir)
    plot_training_histories(run_dir, output_dir, args.dpi)
    plot_synthetic_residuals(run_dir, output_dir, args.dpi)
    plot_apogee_comparisons(args.apogee_fit, output_dir, args.dpi)
    plot_apogee_sample_spectrum(args.apogee_observed, output_dir, args.dpi)
    print(f"Wrote paper figures to {output_dir}")


if __name__ == "__main__":
    main()
