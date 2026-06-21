#!/usr/bin/env python
"""Train Payne-MLP, KAN-Payne, or TransformerPayne on the unified grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="data/processed/payne_apogee_synthetic_grid.npz")
    parser.add_argument(
        "--model",
        choices=("payne_mlp", "kan_payne", "transformer_payne"),
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/payne_emulators",
        help="Directory for checkpoints and history JSON.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--spectra-normalization",
        choices=("none", "minmax"),
        default="none",
        help="Use none for The Payne-like raw flux targets.",
    )
    parser.add_argument(
        "--use-mask",
        action="store_true",
        help="Train/evaluate only unmasked Payne APOGEE pixels where applicable.",
    )
    parser.add_argument(
        "--hidden-sizes",
        default="300,300",
        help="Comma-separated hidden sizes for Payne-MLP/KAN-Payne.",
    )
    parser.add_argument(
        "--activation",
        choices=("leaky_relu", "gelu", "relu"),
        default="leaky_relu",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-label-tokens", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--wave-frequencies", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pixels-per-spectrum", type=int, default=256)
    parser.add_argument("--eval-wave-chunk", type=int, default=512)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the grid and print the resolved training config without importing torch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from kan_payne.payne_data import grid_summary, load_payne_grid
    from kan_payne.payne_training import parse_hidden_sizes

    grid = load_payne_grid(args.grid)
    model_output_dir = Path(args.output_dir) / args.model
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    resolved = {
        "grid": args.grid,
        "model": args.model,
        "output_dir": str(model_output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "seed": args.seed,
        "spectra_normalization": args.spectra_normalization,
        "use_mask": args.use_mask,
        "hidden_sizes": hidden_sizes,
        "activation": args.activation,
        "d_model": args.d_model,
        "n_label_tokens": args.n_label_tokens,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dim_feedforward": args.dim_feedforward,
        "wave_frequencies": args.wave_frequencies,
        "dropout": args.dropout,
        "pixels_per_spectrum": args.pixels_per_spectrum,
        "eval_wave_chunk": args.eval_wave_chunk,
        "grid_summary": grid_summary(grid),
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return

    from kan_payne.payne_training import train_emulator

    common = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "spectra_normalization": args.spectra_normalization,
        "use_mask": args.use_mask,
        "device_name": args.device,
        "seed": args.seed,
    }
    if args.model == "transformer_payne":
        kwargs = {
            **common,
            "d_model": args.d_model,
            "n_label_tokens": args.n_label_tokens,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dim_feedforward": args.dim_feedforward,
            "wave_frequencies": args.wave_frequencies,
            "dropout": args.dropout,
            "pixels_per_spectrum": args.pixels_per_spectrum,
            "eval_wave_chunk": args.eval_wave_chunk,
        }
    else:
        kwargs = {
            **common,
            "hidden_sizes": hidden_sizes,
            "activation": args.activation,
        }

    summary = train_emulator(
        grid,
        model_name=args.model,
        output_dir=model_output_dir,
        **kwargs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
