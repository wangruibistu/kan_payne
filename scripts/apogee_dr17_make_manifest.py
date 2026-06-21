#!/usr/bin/env python
"""Create a clean APOGEE DR17 aspcapStar download manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_range(text: str | None) -> tuple[float, float] | None:
    if text is None or text.lower() == "none":
        return None
    left, right = text.split(",", 1)
    return float(left), float(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allstar",
        default="data/raw/apogee_dr17/allStar-dr17-synspec_rev1.fits",
        help="Path to allStar-dr17-synspec_rev1.fits.",
    )
    parser.add_argument(
        "--download-allstar",
        action="store_true",
        help="Download allStar if --allstar does not exist. The file is about 4 GB.",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/apogee_dr17_clean.csv",
        help="Output CSV manifest.",
    )
    parser.add_argument("--max-stars", type=int, default=10000)
    parser.add_argument("--snrev-min", type=float, default=100.0)
    parser.add_argument("--snr-min", type=float, default=None)
    parser.add_argument("--nvisits-min", type=int, default=2)
    parser.add_argument("--teff-range", default="3500,5500")
    parser.add_argument("--logg-range", default="0,3.8")
    parser.add_argument("--feh-range", default="-2.0,0.5")
    parser.add_argument("--telescopes", default="apo25m,lco25m")
    parser.add_argument(
        "--reader",
        choices=("simple", "astropy"),
        default="simple",
        help="FITS table reader for allStar. The simple reader avoids astropy.io.fits.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("first", "random", "snrev"),
        default="random",
        help="How to choose rows after filtering.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np

    from kan_payne.apogee_dr17 import (
        ALLSTAR_URL,
        DEFAULT_ALLSTAR_COLUMNS,
        download_file,
        read_fits_bintable_columns,
        select_clean_allstar,
        write_manifest,
    )

    allstar_path = Path(args.allstar)
    if not allstar_path.exists():
        if not args.download_allstar:
            raise SystemExit(
                f"Missing {allstar_path}. Re-run with --download-allstar or place the "
                "official allStar file there."
            )
        print(f"Downloading {ALLSTAR_URL} -> {allstar_path}")
        download_file(ALLSTAR_URL, allstar_path)

    print(f"Reading {allstar_path}")
    if args.reader == "simple":
        allstar = read_fits_bintable_columns(allstar_path, DEFAULT_ALLSTAR_COLUMNS)
    else:
        from astropy.table import Table

        allstar = Table.read(allstar_path, hdu=1, memmap=True)
    telescopes = tuple(item.strip() for item in args.telescopes.split(",") if item.strip())
    clean_mask = select_clean_allstar(
        allstar,
        telescopes=telescopes,
        snrev_min=args.snrev_min,
        snr_min=args.snr_min,
        nvisits_min=args.nvisits_min,
        teff_range=parse_range(args.teff_range),
        logg_range=parse_range(args.logg_range),
        feh_range=parse_range(args.feh_range),
    )
    selected_indices = np.flatnonzero(clean_mask)
    print(f"Clean sample: {len(selected_indices)} / {len(allstar)}")

    if args.sample_mode == "random":
        rng = np.random.default_rng(args.seed)
        if args.max_stars and len(selected_indices) > args.max_stars:
            selected_indices = rng.choice(selected_indices, size=args.max_stars, replace=False)
            selected_indices.sort()
    elif args.sample_mode == "snrev" and args.max_stars:
        snrev = np.asarray(allstar["SNREV"], dtype=float)[selected_indices]
        order = np.argsort(snrev)[::-1]
        selected_indices = selected_indices[order[: args.max_stars]]
    elif args.sample_mode == "first" and args.max_stars:
        selected_indices = selected_indices[: args.max_stars]

    rows = (allstar[index] for index in selected_indices)
    output = write_manifest(rows, args.output)
    print(f"Wrote {len(selected_indices)} rows to {output}")


if __name__ == "__main__":
    main()
