#!/usr/bin/env python
"""Preprocess downloaded APOGEE DR17 aspcapStar spectra into one NPZ file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default="data/raw/apogee_dr17")
    parser.add_argument("--output", default="data/processed/apogee_dr17_clean.npz")
    parser.add_argument(
        "--wavelength-npz",
        default=None,
        help="Optional The Payne apogee_wavelength.npz. If omitted, keep aspcapStar grid.",
    )
    parser.add_argument("--max-stars", type=int, default=None)
    parser.add_argument("--error-floor", type=float, default=0.005)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip rows whose spectrum file is not downloaded.",
    )
    return parser.parse_args()


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    args = parse_args()
    import numpy as np

    from kan_payne.apogee_dr17 import (
        DEFAULT_REFERENCE_COLUMNS,
        load_payne_wavelength,
        read_aspcapstar,
        read_manifest,
        save_preprocessed_npz,
    )

    rows = read_manifest(args.manifest)
    if args.max_stars is not None:
        rows = rows[: args.max_stars]

    target_wave = None
    if args.wavelength_npz:
        target_wave = load_payne_wavelength(args.wavelength_npz)

    data_root = Path(args.data_root)
    flux_rows = []
    err_rows = []
    mask_rows = []
    metadata = []
    labels = {column: [] for column in DEFAULT_REFERENCE_COLUMNS}
    wave = target_wave

    for index, row in enumerate(rows, start=1):
        spectrum_path = data_root / row["aspcapstar_path"]
        if not spectrum_path.exists():
            message = f"Missing {spectrum_path}"
            if args.skip_missing:
                print(message)
                continue
            raise SystemExit(message)

        spectrum = read_aspcapstar(
            spectrum_path,
            target_wavelength=target_wave,
            error_floor=args.error_floor,
        )
        if wave is None:
            wave = np.asarray(spectrum["wave"], dtype=float)
        flux_rows.append(np.asarray(spectrum["flux"], dtype=np.float32))
        err_rows.append(np.asarray(spectrum["err"], dtype=np.float32))
        mask_rows.append(np.asarray(spectrum["mask"], dtype=bool))
        metadata.append(
            {
                "APOGEE_ID": row.get("APOGEE_ID", ""),
                "TELESCOPE": row.get("TELESCOPE", ""),
                "FIELD": row.get("FIELD", ""),
                "aspcapstar_path": row.get("aspcapstar_path", ""),
            }
        )
        for column in DEFAULT_REFERENCE_COLUMNS:
            labels[column].append(_to_float(row.get(column, "")))
        if index % 100 == 0:
            print(f"Preprocessed {index} manifest rows")

    if not flux_rows:
        raise SystemExit("No spectra were preprocessed")

    label_arrays = {key: np.asarray(value, dtype=np.float32) for key, value in labels.items()}
    output = save_preprocessed_npz(
        args.output,
        wave=np.asarray(wave, dtype=float),
        flux=np.stack(flux_rows),
        err=np.stack(err_rows),
        mask=np.stack(mask_rows),
        reference_labels=label_arrays,
        metadata=metadata,
    )
    print(f"Wrote {len(flux_rows)} spectra to {output}")


if __name__ == "__main__":
    main()
