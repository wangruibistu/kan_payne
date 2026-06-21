#!/usr/bin/env python
"""Prepare a unified Payne APOGEE/Kurucz synthetic training grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--the-payne-root", default="data/external/the_payne")
    parser.add_argument(
        "--output",
        default="data/processed/payne_apogee_synthetic_grid.npz",
        help="Output unified grid NPZ.",
    )
    parser.add_argument(
        "--split",
        choices=("official", "random"),
        default="official",
        help="Use The Payne first-800 split or a deterministic random split.",
    )
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing The Payne GitHub files into --the-payne-root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from kan_payne.payne_data import (
        build_unified_payne_grid,
        ensure_the_payne_files,
        grid_summary,
        save_payne_grid,
    )

    paths = ensure_the_payne_files(
        args.the_payne_root,
        keys=("training", "wavelength", "mask"),
        download_missing=args.download_missing,
    )
    grid = build_unified_payne_grid(
        training_npz=paths["training"],
        wavelength_npz=paths["wavelength"],
        mask_npz=paths["mask"],
        split=args.split,
        train_size=args.train_size,
        train_fraction=args.train_fraction,
        valid_fraction=args.valid_fraction,
        random_seed=args.random_seed,
    )
    output = save_payne_grid(grid, args.output)
    print(f"Wrote unified Payne grid to {output}")
    print(json.dumps(grid_summary(grid), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
