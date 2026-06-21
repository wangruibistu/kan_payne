#!/usr/bin/env python
"""Resample a preprocessed APOGEE DR17 NPZ onto the unified Payne wavelength grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/processed/apogee_dr17_clean_10k.npz",
        help="Preprocessed APOGEE NPZ from apogee_dr17_preprocess.py.",
    )
    parser.add_argument(
        "--payne-grid",
        default="data/processed/payne_apogee_synthetic_grid.npz",
        help="Unified Payne grid NPZ containing wavelength.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/apogee_dr17_clean_10k_payne_grid.npz",
    )
    parser.add_argument(
        "--continuum-pixels",
        default=None,
        help=(
            "Optional NPZ containing pixels_cannon. When provided, spectra are "
            "renormalized on the Payne grid with The Payne-style per-chip "
            "fourth-order Chebyshev continua."
        ),
    )
    parser.add_argument("--max-stars", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def _fit_payne_chip_continuum(wave_scaled, flux, err, cont_pixels, *, degree: int = 4):
    import numpy as np

    usable = (
        np.asarray(cont_pixels, dtype=bool)
        & np.isfinite(flux)
        & np.isfinite(err)
        & (err > 0)
        & (flux > 0)
    )
    if np.count_nonzero(usable) <= degree + 1:
        return np.ones_like(flux, dtype=np.float32)
    weights = 1.0 / np.maximum(err[usable], 1.0e-4)
    try:
        poly = np.polynomial.Chebyshev.fit(
            wave_scaled[usable],
            flux[usable],
            degree,
            w=weights,
        )
        continuum = np.asarray(poly(wave_scaled), dtype=np.float32)
    except Exception:
        continuum = np.ones_like(flux, dtype=np.float32)
    bad_cont = ~np.isfinite(continuum) | (continuum <= 0.05)
    continuum[bad_cont] = 1.0
    return continuum


def _payne_style_continuum_normalize(flux_rows, err_rows, mask_rows, cont_pixels):
    import numpy as np

    cont_pixels = np.asarray(cont_pixels, dtype=bool)
    if cont_pixels.shape[0] != flux_rows.shape[1]:
        raise ValueError(
            f"continuum pixel mask has {cont_pixels.shape[0]} pixels, "
            f"but spectra have {flux_rows.shape[1]}"
        )
    chips = ((0, 2920), (2920, 5320), (5320, flux_rows.shape[1]))
    continuum_rows = np.empty_like(flux_rows, dtype=np.float32)
    normalized_flux = np.empty_like(flux_rows, dtype=np.float32)
    normalized_err = np.empty_like(err_rows, dtype=np.float32)
    for row in range(flux_rows.shape[0]):
        continuum = np.ones(flux_rows.shape[1], dtype=np.float32)
        for lo, hi in chips:
            x = np.linspace(-1.0, 1.0, hi - lo, dtype=np.float32)
            chip_cont = _fit_payne_chip_continuum(
                x,
                flux_rows[row, lo:hi],
                err_rows[row, lo:hi],
                cont_pixels[lo:hi] & ~mask_rows[row, lo:hi],
            )
            continuum[lo:hi] = chip_cont
        continuum_rows[row] = continuum
        normalized_flux[row] = flux_rows[row] / continuum
        normalized_err[row] = err_rows[row] / continuum
    return normalized_flux, normalized_err, continuum_rows


def main() -> None:
    args = parse_args()

    import numpy as np

    from kan_payne.payne_data import load_payne_grid

    grid = load_payne_grid(args.payne_grid)
    target_wave = np.asarray(grid.wavelength, dtype=np.float64)

    with np.load(args.input) as payload:
        source_wave = np.asarray(payload["wave"], dtype=np.float64)
        source_flux = np.asarray(payload["flux"], dtype=np.float32)
        source_err = np.asarray(payload["err"], dtype=np.float32)
        source_mask = np.asarray(payload["mask"], dtype=bool)
        n_rows = source_flux.shape[0]
        if args.max_stars is not None:
            n_rows = min(n_rows, args.max_stars)

        output_flux = np.empty((n_rows, target_wave.size), dtype=np.float32)
        output_err = np.empty((n_rows, target_wave.size), dtype=np.float32)
        output_mask = np.empty((n_rows, target_wave.size), dtype=bool)

        same_grid = source_wave.shape == target_wave.shape and np.allclose(
            source_wave,
            target_wave,
            rtol=0.0,
            atol=1.0e-8,
        )
        if same_grid:
            output_flux[:] = source_flux[:n_rows]
            output_err[:] = source_err[:n_rows]
            output_mask[:] = source_mask[:n_rows]
        else:
            for row in range(n_rows):
                good = (
                    np.isfinite(source_flux[row])
                    & np.isfinite(source_err[row])
                    & (source_err[row] > 0.0)
                    & ~source_mask[row]
                )
                if np.count_nonzero(good) < 2:
                    output_flux[row] = np.nan
                    output_err[row] = np.nan
                    output_mask[row] = True
                    continue

                output_flux[row] = np.interp(
                    target_wave,
                    source_wave[good],
                    source_flux[row, good],
                    left=np.nan,
                    right=np.nan,
                ).astype(np.float32)
                output_err[row] = np.interp(
                    target_wave,
                    source_wave[good],
                    source_err[row, good],
                    left=np.nan,
                    right=np.nan,
                ).astype(np.float32)
                output_mask[row] = (
                    ~np.isfinite(output_flux[row])
                    | ~np.isfinite(output_err[row])
                    | (output_err[row] <= 0.0)
                    | grid.mask
                )
                if args.progress_every and (row + 1) % args.progress_every == 0:
                    print(f"Resampled {row + 1} spectra")

        continuum_rows = None
        continuum_pixels = None
        if args.continuum_pixels:
            with np.load(args.continuum_pixels) as cont_payload:
                if "pixels_cannon" not in cont_payload:
                    raise KeyError(
                        f"{args.continuum_pixels} does not contain pixels_cannon"
                    )
                continuum_pixels = np.asarray(cont_payload["pixels_cannon"], dtype=bool)
            output_flux, output_err, continuum_rows = _payne_style_continuum_normalize(
                output_flux,
                output_err,
                output_mask,
                continuum_pixels,
            )
            output_mask |= (
                ~np.isfinite(output_flux)
                | ~np.isfinite(output_err)
                | (output_err <= 0.0)
            )

        output_payload = {
            "wave": target_wave,
            "flux": output_flux,
            "err": output_err,
            "mask": output_mask,
            "source_wave": source_wave,
            "payne_mask": grid.mask,
        }
        if continuum_rows is not None:
            output_payload["continuum"] = continuum_rows
            output_payload["continuum_pixels"] = continuum_pixels
            output_payload["normalization"] = np.asarray("payne_chebyshev_cannon_pixels")
        for key in payload.files:
            if key.startswith("label_") or key == "metadata_json":
                output_payload[key] = payload[key][:n_rows]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **output_payload)
    print(
        f"Wrote {n_rows} APOGEE spectra on Payne grid to {output} "
        f"with shape {output_flux.shape}"
    )


if __name__ == "__main__":
    main()
