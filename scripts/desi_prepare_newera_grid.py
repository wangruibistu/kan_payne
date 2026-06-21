#!/usr/bin/env python
"""Convert NewEra synthetic spectra into a DESI-like PayneGrid NPZ.

The NewEra paper distributes spectra as HDF5 files and low-resolution archive
products. This script keeps the input reader deliberately tolerant: it can read
NewEra HDF5 spectra or simple two-column text spectra after the archive has
been extracted. The output uses the same NPZ schema as the APOGEE Payne grid,
so the existing `payne_train_emulator.py` entry point can train KAN-Payne,
Payne-MLP, or TransformerPayne without modification.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_LABEL_NAMES = ("Teff", "logg", "M_H", "alpha_M")
SOLAR_FE_ABUNDANCE = 7.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        help="Directory containing extracted NewEra spectra or HDF5 files.",
    )
    parser.add_argument(
        "--glob",
        default="**/*",
        help="File glob under --input-root. Unsupported files are skipped.",
    )
    parser.add_argument("--output", default="data/processed/desi_newera_grid.npz")
    parser.add_argument("--wave-min", type=float, default=3600.0, help="Angstrom.")
    parser.add_argument("--wave-max", type=float, default=9800.0, help="Angstrom.")
    parser.add_argument("--wave-step", type=float, default=1.0, help="Output Angstrom step.")
    parser.add_argument(
        "--target-resolution",
        type=float,
        default=3000.0,
        help="Approximate Gaussian resolving power for first-pass DESI-like smoothing.",
    )
    parser.add_argument(
        "--continuum-window",
        type=int,
        default=301,
        help="Odd-pixel running-median window for pseudo-continuum normalization; 0 disables.",
    )
    parser.add_argument("--teff-range", default="3500,8000")
    parser.add_argument("--logg-range", default="0,5")
    parser.add_argument("--mh-range", default="-3.0,0.5")
    parser.add_argument("--alpha-range", default="-0.2,0.8")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files and print the first parsed labels without writing output.",
    )
    return parser.parse_args()


def _parse_range(text: str) -> tuple[float, float]:
    left, right = text.split(",", 1)
    return float(left), float(right)


def _in_range(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def _signed_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1).replace("m", "-").replace("p", "+"))


def parse_newera_labels(path: Path) -> dict[str, float] | None:
    """Extract Teff/logg/[M/H]/alpha from common NewEra/PHOENIX names."""

    text = path.name
    lower = text.lower()

    teff = _signed_float(r"(?:teff|lte)[_\-+]?([0-9]{4,5}(?:\.[0-9]+)?)", lower)
    if teff is None:
        # Common PHOENIX-style names begin with lte05000-2.50-0.5.
        match = re.search(r"lte([0-9]{5}|[0-9]{4})", lower)
        if match:
            teff = float(match.group(1))

    logg = _signed_float(r"(?:logg|log_g|g)[_=:\-]?([+-]?[0-9]+(?:\.[0-9]+)?)", lower)
    mh = _signed_float(r"(?:m_h|mh|feh|metal|z)[_=:\-]?([+-]?[0-9]+(?:\.[0-9]+)?)", lower)
    alpha = _signed_float(r"(?:alpha|afe|a_fe|alpha_m)[_=:\-]?([+-]?[0-9]+(?:\.[0-9]+)?)", lower)

    if logg is None or mh is None:
        match = re.search(
            r"lte[0-9]{4,5}([+-][0-9]+(?:\.[0-9]+)?)([+-][0-9]+(?:\.[0-9]+)?)",
            lower,
        )
        if match:
            logg = logg if logg is not None else float(match.group(1))
            mh = mh if mh is not None else float(match.group(2))

    if teff is None or logg is None or mh is None:
        return None
    if alpha is None:
        alpha = 0.0
    return {"Teff": float(teff), "logg": float(logg), "M_H": float(mh), "alpha_M": float(alpha)}


def _read_hdf5(path: Path):
    import numpy as np

    try:
        import h5py
    except Exception as exc:
        raise RuntimeError("Reading NewEra HDF5 files requires h5py") from exc

    with h5py.File(path, "r") as handle:
        candidates = [
            ("/PHOENIX_SPECTRUM_LSR/wl", "/PHOENIX_SPECTRUM_LSR/fl"),
            ("/PHOENIX_SPECTRUM/wl", "/PHOENIX_SPECTRUM/flux"),
        ]
        for wave_key, flux_key in candidates:
            if wave_key in handle and flux_key in handle:
                wave = np.asarray(handle[wave_key], dtype=float)
                flux = np.asarray(handle[flux_key], dtype=float)
                # NewEra HDF5 stores log10 flux for these spectrum groups.
                if np.nanmedian(flux) < 0.0:
                    flux = 10.0 ** flux
                return wave, flux
    raise KeyError(f"{path} does not contain a recognized NewEra spectrum group")


def _read_text_spectrum(path: Path):
    import numpy as np

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="ignore") as handle:
        rows = []
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric two-column spectrum found in {path}")
    arr = np.asarray(rows, dtype=float)
    wave, flux = arr[:, 0], arr[:, 1]
    # GAIA-format NewEra low-resolution files use nm and W/m2/nm.
    if np.nanmax(wave) < 3000.0:
        wave = wave * 10.0
    return wave, flux


def _looks_like_gaia_header(line: str) -> bool:
    parts = line.split()
    if len(parts) < 30:
        return False
    try:
        int(float(parts[8]))
        float(parts[9])
        float(parts[10])
        float(parts[11])
        float(parts[12])
        float(parts[13])
    except ValueError:
        return False
    return True


def _labels_from_gaia_header(parts: list[str]) -> dict[str, float]:
    teff = float(parts[12])
    logg = float(parts[13])
    alpha = float(parts[19])
    fe_abund = float(parts[26])
    return {
        "Teff": teff,
        "logg": logg,
        "M_H": fe_abund - SOLAR_FE_ABUNDANCE,
        "alpha_M": alpha,
    }


def _read_gaia_flux_line(line: str, n_wave: int):
    import numpy as np

    flux = np.loadtxt(io.StringIO(line), dtype=float)
    if flux.size != n_wave:
        raise ValueError(f"GAIA/NewEra flux line has {flux.size} samples, expected {n_wave}")
    return flux


def iter_newera_spectra(path: Path):
    """Yield (source_id, labels, wave_A, flux) from HDF5, text, or GAIA-format archives."""

    import numpy as np

    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        labels = parse_newera_labels(path)
        if labels is None:
            return
        wave, flux = _read_hdf5(path)
        yield str(path), labels, wave, flux
        return

    if suffix not in {".txt", ".dat", ".csv", ".gz"}:
        return

    opener = gzip.open if suffix == ".gz" else open
    with opener(path, "rt", errors="ignore") as handle:
        first = ""
        while not first:
            first = handle.readline()
            if first == "":
                return
            first = first.strip()
            if first.startswith("#"):
                first = ""

        if _looks_like_gaia_header(first):
            header = first
            index = 0
            while header:
                parts = header.split()
                try:
                    n_wave = int(float(parts[8]))
                    wave_start_nm = float(parts[9])
                    wave_end_nm = float(parts[10])
                    resolution_nm = float(parts[7])
                    labels = _labels_from_gaia_header(parts)
                    flux_line = handle.readline()
                    if not flux_line:
                        break
                    wave_nm = np.linspace(
                        wave_start_nm,
                        wave_end_nm,
                        num=int((wave_end_nm - wave_start_nm) / resolution_nm) + 1,
                    )
                    if wave_nm.size != n_wave:
                        wave_nm = np.linspace(wave_start_nm, wave_end_nm, num=n_wave)
                    flux = _read_gaia_flux_line(flux_line, n_wave) * 1.0e10
                    yield f"{path}#{index}", labels, wave_nm * 10.0, flux
                except Exception as exc:
                    raise ValueError(f"Could not parse GAIA/NewEra spectrum {path}#{index}: {exc}") from exc
                index += 1
                header = handle.readline().strip()
            return

    labels = parse_newera_labels(path)
    if labels is None:
        return
    wave, flux = _read_text_spectrum(path)
    yield str(path), labels, wave, flux


def read_spectrum(path: Path):
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return _read_hdf5(path)
    if path.suffix.lower() in {".txt", ".dat", ".csv", ".gz"}:
        return _read_text_spectrum(path)
    raise ValueError(f"Unsupported spectrum format: {path}")


def _gaussian_smooth(flux, sigma_pix: float):
    if sigma_pix <= 0.0:
        return flux
    try:
        from scipy.ndimage import gaussian_filter1d

        return gaussian_filter1d(flux, sigma_pix, mode="nearest")
    except Exception:
        # Lightweight fallback: a finite Gaussian kernel.
        import numpy as np

        radius = max(2, int(round(4.0 * sigma_pix)))
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma_pix) ** 2)
        kernel /= np.sum(kernel)
        return np.convolve(flux, kernel, mode="same")


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
    bad = ~np.isfinite(cont) | (cont <= 0)
    cont[bad] = np.nanmedian(values[np.isfinite(values) & (values > 0)])
    return cont


def preprocess_spectrum(
    wave,
    flux,
    target_wave,
    *,
    target_resolution: float,
    continuum_window: int,
):
    import numpy as np

    order = np.argsort(wave)
    wave = np.asarray(wave, dtype=float)[order]
    flux = np.asarray(flux, dtype=float)[order]
    finite = np.isfinite(wave) & np.isfinite(flux) & (flux > 0)
    wave = wave[finite]
    flux = flux[finite]
    if wave.size < 10:
        raise ValueError("Spectrum has too few finite positive points")

    interp = np.interp(target_wave, wave, flux, left=np.nan, right=np.nan)
    finite_interp = np.isfinite(interp) & (interp > 0)
    if np.count_nonzero(finite_interp) < 0.9 * target_wave.size:
        raise ValueError("Spectrum does not cover enough of the DESI wavelength grid")
    median = np.nanmedian(interp[finite_interp])
    interp[~finite_interp] = median

    if target_resolution > 0:
        mean_wave = float(np.nanmean(target_wave))
        sigma_angstrom = mean_wave / (target_resolution * 2.355)
        sigma_pix = sigma_angstrom / float(np.nanmedian(np.diff(target_wave)))
        interp = _gaussian_smooth(interp, sigma_pix)

    if continuum_window > 1:
        continuum = _running_median(interp, continuum_window)
        interp = interp / continuum
    else:
        interp = interp / np.nanmedian(interp)
    return interp.astype("float32")


def main() -> None:
    args = parse_args()
    import numpy as np

    from kan_payne.payne_data import PayneGrid, label_scaler, random_split, save_payne_grid, scale_labels

    input_root = Path(args.input_root)
    files = [path for path in input_root.glob(args.glob) if path.is_file()]
    target_wave = np.arange(args.wave_min, args.wave_max + 0.5 * args.wave_step, args.wave_step)
    ranges = {
        "Teff": _parse_range(args.teff_range),
        "logg": _parse_range(args.logg_range),
        "M_H": _parse_range(args.mh_range),
        "alpha_M": _parse_range(args.alpha_range),
    }

    labels = []
    spectra = []
    used_files = []
    skipped = []
    for path in files:
        try:
            iterator = iter_newera_spectra(path)
            n_from_file = 0
            for source_id, label, wave, flux in iterator:
                n_from_file += 1
                if not all(_in_range(label[name], ranges[name]) for name in DEFAULT_LABEL_NAMES):
                    skipped.append((source_id, "outside requested label range"))
                    continue
                spec = preprocess_spectrum(
                    wave,
                    flux,
                    target_wave,
                    target_resolution=args.target_resolution,
                    continuum_window=args.continuum_window,
                )
                labels.append([label[name] for name in DEFAULT_LABEL_NAMES])
                spectra.append(spec)
                used_files.append(source_id)
                if args.max_spectra and len(used_files) >= args.max_spectra:
                    break
            if n_from_file == 0:
                skipped.append((str(path), "no supported spectra or could not parse labels"))
        except Exception as exc:
            skipped.append((str(path), str(exc)))
            continue
        if args.max_spectra and len(used_files) >= args.max_spectra:
            break

    summary = {
        "input_root": str(input_root),
        "glob": args.glob,
        "n_files_seen": len(files),
        "n_used": len(used_files),
        "n_skipped": len(skipped),
        "target_wave_min": float(target_wave[0]),
        "target_wave_max": float(target_wave[-1]),
        "target_wave_step": float(args.wave_step),
        "label_names": list(DEFAULT_LABEL_NAMES),
        "first_used_files": used_files[:5],
        "first_skipped": skipped[:10],
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if not spectra:
        raise SystemExit(json.dumps(summary, indent=2, sort_keys=True))

    labels_np = np.asarray(labels, dtype=np.float32)
    spectra_np = np.asarray(spectra, dtype=np.float32)
    train_idx, valid_idx, test_idx = random_split(
        labels_np.shape[0],
        train_fraction=args.train_fraction,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
    )
    label_min, label_max = label_scaler(labels_np, train_idx)
    label_scaled = scale_labels(labels_np, label_min, label_max)
    mask = ~np.all(np.isfinite(spectra_np), axis=0)
    metadata = {
        **summary,
        "source": "NewEra PHOENIX/1D LTE grid converted for DESI-like KAN-Payne",
        "normalization": "running-median pseudo-continuum normalization",
        "target_resolution": float(args.target_resolution),
        "continuum_window": int(args.continuum_window),
        "used_files": used_files,
    }
    grid = PayneGrid(
        wavelength=target_wave.astype(np.float64),
        mask=mask.astype(bool),
        labels=labels_np,
        spectra=spectra_np,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        label_min=label_min,
        label_max=label_max,
        label_scaled=label_scaled,
        label_names=DEFAULT_LABEL_NAMES,
        metadata=metadata,
    )
    output = save_payne_grid(grid, args.output)
    print(f"Wrote DESI/NewEra PayneGrid to {output}")
    print(json.dumps({k: v for k, v in summary.items() if k != "first_skipped"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
