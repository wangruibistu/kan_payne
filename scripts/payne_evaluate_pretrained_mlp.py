#!/usr/bin/env python
"""Evaluate The Payne GitHub pretrained MLP on the unified synthetic grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        default="data/processed/payne_apogee_synthetic_grid.npz",
        help="Unified Payne grid produced by payne_prepare_training_grid.py.",
    )
    parser.add_argument(
        "--pretrained-mlp",
        default="data/external/the_payne/neural_nets/NN_normalized_spectra.npz",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "all"),
        default="valid",
        help="Grid split to evaluate.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/payne_pretrained_mlp_validation_metrics.json",
        help="JSON metrics output.",
    )
    parser.add_argument(
        "--save-residuals",
        default=None,
        help="Optional NPZ path for predicted spectra and residuals.",
    )
    return parser.parse_args()


def _split_indices(grid, split: str):
    import numpy as np

    if split == "train":
        return grid.train_idx
    if split == "valid":
        return grid.valid_idx
    return np.arange(grid.labels.shape[0], dtype=np.int64)


def main() -> None:
    args = parse_args()

    import numpy as np

    from kan_payne.payne_data import (
        load_payne_grid,
        load_pretrained_payne_mlp,
        predict_pretrained_payne_mlp,
        regression_metrics,
    )

    grid = load_payne_grid(args.grid)
    coeffs = load_pretrained_payne_mlp(args.pretrained_mlp)
    idx = _split_indices(grid, args.split)
    predicted = predict_pretrained_payne_mlp(grid.labels[idx], coeffs)
    target = grid.spectra[idx]

    result = {
        "grid": args.grid,
        "pretrained_mlp": args.pretrained_mlp,
        "split": args.split,
        "n_spectra": int(idx.size),
        "n_pixels": int(grid.spectra.shape[1]),
        "metrics_all_pixels": regression_metrics(predicted, target),
        "metrics_good_pixels": regression_metrics(
            predicted,
            target,
            good_pixel_mask=grid.good_pixel_mask,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(f"Wrote metrics to {output}")
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.save_residuals:
        residual_output = Path(args.save_residuals)
        residual_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            residual_output,
            indices=idx,
            predicted=predicted.astype(np.float32),
            target=target.astype(np.float32),
            residual=(predicted - target).astype(np.float32),
            wavelength=grid.wavelength,
            mask=grid.mask,
        )
        print(f"Wrote residuals to {residual_output}")


if __name__ == "__main__":
    main()
