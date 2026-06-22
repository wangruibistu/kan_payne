#!/usr/bin/env python
"""Evaluate a trained Payne emulator checkpoint on a unified Payne grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="data/processed/payne_apogee_synthetic_grid.npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "valid", "all"), default="valid")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--wave-chunk", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _split_indices(grid, split: str):
    import numpy as np

    if split == "train":
        return grid.train_idx
    if split == "valid":
        return grid.valid_idx
    return np.arange(grid.labels.shape[0], dtype=np.int64)


def _choose_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _inverse_spectra(values, norm):
    if not norm or norm.get("mode") != "minmax":
        return values
    return values * (norm["y_max"] - norm["y_min"]) + norm["y_min"]


def _wave_scaled(wavelength):
    import numpy as np

    wave = np.asarray(wavelength, dtype=np.float32)
    return (2.0 * (wave - np.nanmin(wave)) / (np.nanmax(wave) - np.nanmin(wave)) - 1.0).astype(
        np.float32
    )


def _predict_vector(model, labels_scaled, *, batch_size: int, device):
    import numpy as np
    import torch

    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, labels_scaled.shape[0], batch_size):
            batch = torch.from_numpy(labels_scaled[start : start + batch_size]).to(device)
            rows.append(model(batch).detach().cpu().numpy().astype(np.float32))
    return np.vstack(rows)


def _predict_transformer(
    model,
    labels_scaled,
    wavelength,
    *,
    batch_size: int,
    wave_chunk: int,
    device,
):
    import numpy as np
    import torch

    wave = _wave_scaled(wavelength)
    predicted = np.empty((labels_scaled.shape[0], wave.shape[0]), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for row_start in range(0, labels_scaled.shape[0], batch_size):
            row_slice = slice(row_start, row_start + batch_size)
            labels = torch.from_numpy(labels_scaled[row_slice]).to(device)
            for pix_start in range(0, wave.shape[0], wave_chunk):
                pix_slice = slice(pix_start, pix_start + wave_chunk)
                wave_batch = torch.from_numpy(wave[pix_slice]).to(device)
                predicted[row_slice, pix_slice] = (
                    model(labels, wave_batch).detach().cpu().numpy().astype(np.float32)
                )
    return predicted


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch

    from kan_payne.payne_data import load_payne_grid, regression_metrics
    from kan_payne.payne_emulators import build_emulator

    grid = load_payne_grid(args.grid)
    idx = _split_indices(grid, args.split)
    device = _choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = dict(checkpoint["model_config"])
    model_name = config["model"]
    model, _ = build_emulator(
        model_name,
        n_labels=int(config["n_labels"]),
        n_pixels=int(config["n_pixels"]),
        hidden_sizes=tuple(config.get("hidden_sizes", (300, 300))),
        activation=config.get("activation", "leaky_relu"),
        d_model=int(config.get("d_model", 128)),
        n_label_tokens=int(config.get("n_label_tokens", 16)),
        n_heads=int(config.get("n_heads", 4)),
        n_layers=int(config.get("n_layers", 4)),
        dim_feedforward=int(config.get("dim_feedforward", 256)),
        wave_frequencies=int(config.get("wave_frequencies", 16)),
        dropout=float(config.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    labels_scaled = grid.label_scaled[idx].astype(np.float32)
    if model_name == "transformer_payne":
        predicted = _predict_transformer(
            model,
            labels_scaled,
            grid.wavelength,
            batch_size=args.batch_size,
            wave_chunk=args.wave_chunk,
            device=device,
        )
    else:
        predicted = _predict_vector(
            model,
            labels_scaled,
            batch_size=args.batch_size,
            device=device,
        )

    norm = checkpoint.get("spectra_normalization", {"mode": "none"})
    predicted = _inverse_spectra(predicted, norm).astype(np.float32)
    target = grid.spectra[idx].astype(np.float32)
    residual = (predicted - target).astype(np.float32)
    good = grid.good_pixel_mask

    metrics_all = regression_metrics(predicted, target)
    metrics_good = regression_metrics(predicted, target, good_pixel_mask=good)
    pixel_residual = residual[:, good]
    star_mae = np.mean(np.abs(pixel_residual), axis=1) * 1.0e4
    pixel_mae = np.mean(np.abs(residual), axis=0) * 1.0e4
    result = {
        "grid": args.grid,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "model": model_name,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "n_spectra": int(idx.size),
        "n_pixels": int(grid.spectra.shape[1]),
        "metrics_all_pixels": metrics_all,
        "metrics_good_pixels": metrics_good,
        "star_mae_x1e4": {
            "median": float(np.median(star_mae)),
            "p16": float(np.percentile(star_mae, 16)),
            "p84": float(np.percentile(star_mae, 84)),
            "max": float(np.max(star_mae)),
        },
        "pixel_mae_x1e4": {
            "median": float(np.median(pixel_mae)),
            "p16": float(np.percentile(pixel_mae, 16)),
            "p84": float(np.percentile(pixel_mae, 84)),
            "max": float(np.max(pixel_mae)),
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.output_npz:
        output_npz = Path(args.output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz,
            indices=idx,
            predicted=predicted,
            target=target,
            residual=residual,
            wavelength=grid.wavelength,
            mask=grid.mask,
            star_mae_x1e4=star_mae.astype(np.float32),
            pixel_mae_x1e4=pixel_mae.astype(np.float32),
        )
        print(f"Wrote residual arrays to {output_npz}")


if __name__ == "__main__":
    main()
