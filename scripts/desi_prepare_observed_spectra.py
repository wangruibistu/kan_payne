#!/usr/bin/env python
"""Prepare DESI DR1 stellar coadd spectra for NewEra/KAN-Payne inference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

C_KMS = 299792.458


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-targets", required=True, help="CSV from the S/N-selected manifest.")
    parser.add_argument("--coadd-root", required=True, help="Directory containing coadd-*.fits files.")
    parser.add_argument(
        "--sp-catalog",
        action="append",
        required=True,
        help="MWS SP catalog FITS file with SPTAB extension. May repeat.",
    )
    parser.add_argument("--grid", required=True, help="NewEra PayneGrid NPZ; supplies target wavelength.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-stars", type=int, default=None)
    parser.add_argument("--continuum-window", type=int, default=301)
    parser.add_argument(
        "--normalize-per-arm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize DESI B/R/Z arms separately before merging.",
    )
    parser.add_argument(
        "--smooth-observed-sigma-pix",
        type=float,
        default=0.0,
        help="Optional extra Gaussian smoothing on the merged normalized spectrum.",
    )
    parser.add_argument("--min-coverage", type=float, default=0.70)
    parser.add_argument("--max-mask-fraction", type=float, default=0.35)
    parser.add_argument("--snr-min", type=float, default=30.0)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def _running_median(values, window: int):
    import numpy as np

    if window <= 1:
        return np.ones_like(values)
    if window % 2 == 0:
        window += 1
    try:
        from scipy.ndimage import median_filter

        cont = median_filter(values, size=window, mode="nearest")
    except Exception:
        radius = window // 2
        padded = np.pad(values, radius, mode="edge")
        cont = np.empty_like(values)
        for i in range(values.size):
            cont[i] = np.nanmedian(padded[i : i + window])
    finite = np.isfinite(values) & (values > 0)
    fallback = float(np.nanmedian(values[finite])) if np.any(finite) else 1.0
    bad = ~np.isfinite(cont) | (cont <= 0)
    cont[bad] = fallback
    return cont


def _gaussian_smooth(values, sigma_pix: float):
    if sigma_pix <= 0:
        return values
    try:
        from scipy.ndimage import gaussian_filter1d

        return gaussian_filter1d(values, sigma_pix, mode="nearest")
    except Exception:
        import numpy as np

        radius = max(2, int(round(4.0 * sigma_pix)))
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma_pix) ** 2)
        kernel /= np.sum(kernel)
        return np.convolve(values, kernel, mode="same")


def _load_selected_targets(path: Path, max_stars: int | None):
    rows = []
    seen = set()
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (str(row["survey"]), str(row["program"]), int(row["healpix"]), int(row["targetid"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "survey": key[0],
                    "program": key[1],
                    "healpix": key[2],
                    "targetid": key[3],
                }
            )
            if max_stars is not None and len(rows) >= max_stars:
                break
    return rows


def _load_sp_catalogs(paths):
    import numpy as np
    from astropy.io import fits

    out = {}
    wanted = [
        "TARGETID",
        "TEFF",
        "LOGG",
        "FEH",
        "ALPHAFE",
        "SNR_MED",
        "RV_ADOP",
        "RV_ERR",
        "CHISQ_TOT",
        "SUCCESS",
        "HEALPIX",
    ]
    for path in paths:
        with fits.open(path, memmap=True) as hdul:
            table = hdul["SPTAB"].data
            cols = set(hdul["SPTAB"].columns.names)
            for i, targetid in enumerate(np.asarray(table["TARGETID"], dtype=np.int64)):
                record = {}
                for col in wanted:
                    if col in cols:
                        record[col.lower()] = table[col][i]
                out[int(targetid)] = record
    return out


def _coadd_path(root: Path, survey: str, program: str, healpix: int) -> Path:
    flat = root / f"coadd-{survey}-{program}-{healpix}.fits"
    if flat.exists():
        return flat
    nested = root / survey / program / str(healpix // 100) / str(healpix) / flat.name
    if nested.exists():
        return nested
    matches = list(root.rglob(flat.name))
    if matches:
        return matches[0]
    return flat


def _read_arms_from_coadd(path: Path, targetid: int):
    import numpy as np
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        fibermap = hdul["FIBERMAP"].data
        targetids = np.asarray(fibermap["TARGETID"], dtype=np.int64)
        matches = np.flatnonzero(targetids == int(targetid))
        if matches.size == 0:
            raise KeyError(f"TARGETID {targetid} not found in {path}")
        row = int(matches[0])
        arms = []
        for band in ("B", "R", "Z"):
            wave = np.asarray(hdul[f"{band}_WAVELENGTH"].data, dtype=float)
            flux = np.asarray(hdul[f"{band}_FLUX"].data[row], dtype=float)
            ivar = np.asarray(hdul[f"{band}_IVAR"].data[row], dtype=float)
            mask = np.asarray(hdul[f"{band}_MASK"].data[row])
            good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0) & (mask == 0)
            arms.append({"band": band, "wave": wave[good], "flux": flux[good], "ivar": ivar[good]})
    return arms


def _interp_one_arm(wave_obs, flux_obs, ivar_obs, target_wave, rv_kms: float, continuum_window: int):
    import numpy as np

    wave_rest = np.asarray(wave_obs, dtype=float) / (1.0 + float(rv_kms) / C_KMS)
    order = np.argsort(wave_rest)
    wave_rest = wave_rest[order]
    flux_obs = np.asarray(flux_obs, dtype=float)[order]
    ivar_obs = np.asarray(ivar_obs, dtype=float)[order]
    finite = np.isfinite(wave_rest) & np.isfinite(flux_obs) & np.isfinite(ivar_obs) & (ivar_obs > 0)
    wave_rest = wave_rest[finite]
    flux_obs = flux_obs[finite]
    ivar_obs = ivar_obs[finite]
    if wave_rest.size < 10:
        raise ValueError("too few usable pixels")

    flux = np.interp(target_wave, wave_rest, flux_obs, left=np.nan, right=np.nan)
    ivar = np.interp(target_wave, wave_rest, ivar_obs, left=0.0, right=0.0)
    valid = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
    fallback = np.nanmedian(flux[valid]) if np.any(valid) else 1.0
    flux[~valid] = fallback
    continuum = _running_median(flux, continuum_window)
    norm_flux = flux / continuum
    err = np.full_like(norm_flux, np.inf, dtype=float)
    err[valid] = 1.0 / np.sqrt(ivar[valid]) / continuum[valid]
    mask = ~valid | ~np.isfinite(norm_flux) | ~np.isfinite(err) | (err <= 0)
    return norm_flux.astype("float32"), err.astype("float32"), mask.astype(bool)


def _interp_observed_arms(
    arms,
    target_wave,
    rv_kms: float,
    continuum_window: int,
    *,
    normalize_per_arm: bool,
    smooth_observed_sigma_pix: float,
):
    import numpy as np

    if normalize_per_arm:
        numerator = np.zeros(target_wave.size, dtype=float)
        denominator = np.zeros(target_wave.size, dtype=float)
        for arm in arms:
            arm_flux, arm_err, arm_mask = _interp_one_arm(
                arm["wave"],
                arm["flux"],
                arm["ivar"],
                target_wave,
                rv_kms,
                continuum_window,
            )
            good = ~arm_mask
            weight = np.zeros_like(arm_flux, dtype=float)
            weight[good] = 1.0 / np.square(arm_err[good])
            numerator[good] += arm_flux[good] * weight[good]
            denominator[good] += weight[good]
        mask = denominator <= 0
        flux = np.ones(target_wave.size, dtype=float)
        flux[~mask] = numerator[~mask] / denominator[~mask]
        err = np.full(target_wave.size, np.inf, dtype=float)
        err[~mask] = 1.0 / np.sqrt(denominator[~mask])
    else:
        wave = np.concatenate([arm["wave"] for arm in arms])
        raw_flux = np.concatenate([arm["flux"] for arm in arms])
        ivar = np.concatenate([arm["ivar"] for arm in arms])
        flux, err, mask = _interp_one_arm(wave, raw_flux, ivar, target_wave, rv_kms, continuum_window)

    if smooth_observed_sigma_pix > 0:
        good = ~mask
        smoothed = _gaussian_smooth(flux, smooth_observed_sigma_pix)
        flux[good] = smoothed[good]
    return flux.astype("float32"), err.astype("float32"), mask.astype(bool)


def main() -> None:
    args = parse_args()

    import numpy as np

    from kan_payne.payne_data import load_payne_grid

    grid = load_payne_grid(args.grid)
    target_wave = np.asarray(grid.wavelength, dtype=np.float64)
    rows = _load_selected_targets(Path(args.selected_targets), args.max_stars)
    sp_lookup = _load_sp_catalogs([Path(path) for path in args.sp_catalog])
    coadd_root = Path(args.coadd_root)

    flux_rows = []
    err_rows = []
    mask_rows = []
    metadata_rows = []
    skipped = []
    for i, row in enumerate(rows, start=1):
        targetid = int(row["targetid"])
        sp = sp_lookup.get(targetid)
        if sp is None:
            skipped.append((targetid, "missing SP record"))
            continue
        snr = float(sp.get("snr_med", np.nan))
        rv = float(sp.get("rv_adop", np.nan))
        if not np.isfinite(snr) or snr < args.snr_min:
            skipped.append((targetid, "below SNR cut"))
            continue
        if not np.isfinite(rv):
            skipped.append((targetid, "missing RV_ADOP"))
            continue
        path = _coadd_path(coadd_root, row["survey"], row["program"], int(row["healpix"]))
        try:
            arms = _read_arms_from_coadd(path, targetid)
            flux, err, mask = _interp_observed_arms(
                arms,
                target_wave,
                rv,
                args.continuum_window,
                normalize_per_arm=args.normalize_per_arm,
                smooth_observed_sigma_pix=args.smooth_observed_sigma_pix,
            )
            coverage = 1.0 - float(np.mean(mask))
            if coverage < args.min_coverage or float(np.mean(mask)) > args.max_mask_fraction:
                skipped.append((targetid, f"insufficient coverage {coverage:.3f}"))
                continue
        except Exception as exc:
            skipped.append((targetid, str(exc)))
            continue
        flux_rows.append(flux)
        err_rows.append(err)
        mask_rows.append(mask)
        metadata_rows.append({**row, **sp})
        if args.progress_every and len(flux_rows) % args.progress_every == 0:
            print(f"Prepared {len(flux_rows)} spectra; skipped {len(skipped)}", flush=True)

    if not flux_rows:
        raise SystemExit(f"No spectra prepared; first skipped rows: {skipped[:10]}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "wave": target_wave.astype("float32"),
        "flux": np.asarray(flux_rows, dtype="float32"),
        "err": np.asarray(err_rows, dtype="float32"),
        "mask": np.asarray(mask_rows, dtype=bool),
        "targetid": np.asarray([row["targetid"] for row in metadata_rows], dtype=np.int64),
        "survey": np.asarray([row["survey"] for row in metadata_rows]),
        "program": np.asarray([row["program"] for row in metadata_rows]),
        "healpix": np.asarray([row["healpix"] for row in metadata_rows], dtype=np.int64),
        "rv_adop": np.asarray([row.get("rv_adop", np.nan) for row in metadata_rows], dtype="float32"),
        "rv_err": np.asarray([row.get("rv_err", np.nan) for row in metadata_rows], dtype="float32"),
        "snr_med": np.asarray([row.get("snr_med", np.nan) for row in metadata_rows], dtype="float32"),
        "sp_success": np.asarray([row.get("success", False) for row in metadata_rows]).astype(bool),
        "label_TEFF": np.asarray([row.get("teff", np.nan) for row in metadata_rows], dtype="float32"),
        "label_LOGG": np.asarray([row.get("logg", np.nan) for row in metadata_rows], dtype="float32"),
        "label_M_H": np.asarray([row.get("feh", np.nan) for row in metadata_rows], dtype="float32"),
        "label_alpha_M": np.asarray([row.get("alphafe", np.nan) for row in metadata_rows], dtype="float32"),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "selected_targets": args.selected_targets,
                    "coadd_root": args.coadd_root,
                    "sp_catalog": args.sp_catalog,
                    "grid": args.grid,
                    "continuum_window": args.continuum_window,
                    "normalize_per_arm": args.normalize_per_arm,
                    "smooth_observed_sigma_pix": args.smooth_observed_sigma_pix,
                    "snr_min": args.snr_min,
                    "n_input_rows": len(rows),
                    "n_prepared": len(flux_rows),
                    "n_skipped": len(skipped),
                    "first_skipped": skipped[:20],
                    "rv_correction": "wave_rest = wave_observed / (1 + RV_ADOP / c)",
                    "normalization": "running-median pseudo-continuum",
                },
                sort_keys=True,
            )
        ),
    }
    np.savez_compressed(output, **payload)
    print(f"Wrote {len(flux_rows)} DESI spectra to {output}; skipped {len(skipped)}")


if __name__ == "__main__":
    main()
