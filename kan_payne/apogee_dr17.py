"""Utilities for APOGEE DR17 spectra and ASPCAP comparison catalogs.

The functions in this module intentionally avoid pandas so the data pipeline can
run in minimal astronomy Python environments with only numpy and astropy.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.request import urlopen

import numpy as np


ASPCAP_ROOT = (
    "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1"
)
REDUX_STARS_ROOT = "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars"
ALLSTAR_URL = f"{ASPCAP_ROOT}/allStar-dr17-synspec_rev1.fits"

DEFAULT_BAD_ASPCAP_FLAGS = (
    "STAR_BAD",
    "CHI2_BAD",
    "NO_ASPCAP_RESULT",
    "TEFF_BAD",
    "LOGG_BAD",
    "M_H_BAD",
)
DEFAULT_BAD_STAR_FLAGS = (
    "BAD_PIXELS",
    "VERY_BRIGHT_NEIGHBOR",
    "LOW_SNR",
    "BAD_RV_COMBINATION",
    "RV_FAIL",
)

DEFAULT_REFERENCE_COLUMNS = (
    "TEFF",
    "TEFF_ERR",
    "TEFF_SPEC",
    "LOGG",
    "LOGG_ERR",
    "LOGG_SPEC",
    "M_H",
    "M_H_ERR",
    "FE_H",
    "FE_H_ERR",
    "ALPHA_M",
    "ALPHA_M_ERR",
    "C_FE",
    "C_FE_ERR",
    "N_FE",
    "N_FE_ERR",
    "O_FE",
    "O_FE_ERR",
    "MG_FE",
    "MG_FE_ERR",
    "SI_FE",
    "SI_FE_ERR",
    "CA_FE",
    "CA_FE_ERR",
    "TI_FE",
    "TI_FE_ERR",
    "NI_FE",
    "NI_FE_ERR",
)

DEFAULT_ALLSTAR_COLUMNS = (
    "APOGEE_ID",
    "ALT_ID",
    "TELESCOPE",
    "FIELD",
    "LOCATION_ID",
    "FILE",
    "SNR",
    "SNREV",
    "NVISITS",
    "EXTRATARG",
    "ASPCAPFLAGS",
    "STARFLAGS",
    "ASPCAPFLAG",
    "STARFLAG",
    *DEFAULT_REFERENCE_COLUMNS,
)


def clean_text(value: object) -> str:
    """Convert FITS bytes/scalars to a clean Python string."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    text = str(value)
    return text.strip()


def row_get(row: Mapping[str, object] | object, column: str, default: object = "") -> object:
    """Get a value from either a dict-like row or an astropy Row."""

    if isinstance(row, Mapping):
        return row.get(column, default)
    colnames = getattr(row, "colnames", None)
    if colnames is None and hasattr(row, "table"):
        colnames = getattr(row.table, "colnames", None)
    if colnames is not None and column not in colnames:
        return default
    try:
        return row[column]  # type: ignore[index]
    except Exception:
        return default


def as_bool_mask(flags: Sequence[object], bad_names: Iterable[str]) -> np.ndarray:
    """Return True where a verbose APOGEE flag string contains no bad flags."""

    bad_names = tuple(bad_names)
    keep = np.ones(len(flags), dtype=bool)
    for index, value in enumerate(flags):
        text = clean_text(value)
        keep[index] = not any(name in text for name in bad_names)
    return keep


def finite_column(table: Table, column: str) -> np.ndarray:
    """Return a finite-value mask for a numeric table column."""

    if column not in table.colnames:
        return np.zeros(len(table), dtype=bool)
    values = np.asarray(table[column], dtype=float)
    return np.isfinite(values) & (np.abs(values) < 1.0e5)


def column_range_mask(
    table: Table,
    column: str,
    low: float | None = None,
    high: float | None = None,
) -> np.ndarray:
    """Return a finite range mask for one numeric column."""

    mask = finite_column(table, column)
    if not np.any(mask):
        return mask
    values = np.asarray(table[column], dtype=float)
    if low is not None:
        mask &= values >= low
    if high is not None:
        mask &= values <= high
    return mask


def _isin_text(values: Sequence[object], allowed: Sequence[str]) -> np.ndarray:
    allowed = {item.strip() for item in allowed if item.strip()}
    return np.array([clean_text(value) in allowed for value in values], dtype=bool)


def select_clean_allstar(
    table: object,
    *,
    telescopes: Sequence[str] = ("apo25m", "lco25m"),
    snrev_min: float = 100.0,
    snr_min: float | None = None,
    nvisits_min: int = 2,
    teff_range: tuple[float, float] | None = (3500.0, 5500.0),
    logg_range: tuple[float, float] | None = (0.0, 3.8),
    feh_range: tuple[float, float] | None = (-2.0, 0.5),
    bad_aspcap_flags: Sequence[str] = DEFAULT_BAD_ASPCAP_FLAGS,
    bad_star_flags: Sequence[str] = DEFAULT_BAD_STAR_FLAGS,
    exclude_duplicates: bool = True,
) -> np.ndarray:
    """Build a conservative science mask for APOGEE DR17 allStar rows."""

    mask = np.ones(len(table), dtype=bool)

    if "TELESCOPE" in table.colnames:
        mask &= _isin_text(table["TELESCOPE"], telescopes)

    if "SNREV" in table.colnames:
        mask &= column_range_mask(table, "SNREV", snrev_min, None)
    elif "SNR" in table.colnames:
        mask &= column_range_mask(table, "SNR", snrev_min, None)

    if snr_min is not None and "SNR" in table.colnames:
        mask &= column_range_mask(table, "SNR", snr_min, None)

    if "NVISITS" in table.colnames:
        mask &= column_range_mask(table, "NVISITS", nvisits_min, None)

    if "ASPCAPFLAGS" in table.colnames:
        mask &= as_bool_mask(table["ASPCAPFLAGS"], bad_aspcap_flags)
    if "STARFLAGS" in table.colnames:
        mask &= as_bool_mask(table["STARFLAGS"], bad_star_flags)

    if teff_range is not None:
        mask &= column_range_mask(table, "TEFF", teff_range[0], teff_range[1])
    if logg_range is not None:
        mask &= column_range_mask(table, "LOGG", logg_range[0], logg_range[1])
    if feh_range is not None:
        mask &= column_range_mask(table, "FE_H", feh_range[0], feh_range[1])

    for column in ("TEFF", "LOGG", "FE_H", "M_H", "ALPHA_M"):
        if column in table.colnames:
            mask &= finite_column(table, column)

    if exclude_duplicates and "EXTRATARG" in table.colnames:
        duplicate_bit = 1 << 4
        mask &= (np.asarray(table["EXTRATARG"], dtype=np.int64) & duplicate_bit) == 0

    return mask


def star_name_for_files(row: Mapping[str, object] | object) -> str:
    """Return the star identifier used in per-star DR17 filenames."""

    telescope = clean_text(row_get(row, "TELESCOPE", ""))
    alt_id = clean_text(row_get(row, "ALT_ID", ""))
    apogee_id = clean_text(row_get(row, "APOGEE_ID", ""))
    if telescope == "apo1m" and alt_id:
        return alt_id
    return apogee_id


def aspcapstar_url(row: Mapping[str, object] | object) -> str:
    telescope = clean_text(row_get(row, "TELESCOPE"))
    field = clean_text(row_get(row, "FIELD"))
    star_name = star_name_for_files(row)
    return f"{ASPCAP_ROOT}/{telescope}/{field}/aspcapStar-dr17-{star_name}.fits"


def aspcapstar_local_path(row: Mapping[str, object] | object) -> str:
    telescope = clean_text(row_get(row, "TELESCOPE"))
    field = clean_text(row_get(row, "FIELD"))
    star_name = star_name_for_files(row)
    return f"aspcap/{telescope}/{field}/aspcapStar-dr17-{star_name}.fits"


def apstar_url(row: Mapping[str, object] | object) -> str:
    telescope = clean_text(row_get(row, "TELESCOPE"))
    field = clean_text(row_get(row, "FIELD"))
    file_name = clean_text(row_get(row, "FILE", ""))
    if not file_name:
        star_name = star_name_for_files(row)
        prefix = "asStar" if telescope == "lco25m" else "apStar"
        file_name = f"{prefix}-dr17-{star_name}.fits"
    return f"{REDUX_STARS_ROOT}/{telescope}/{field}/{file_name}"


def apstar_local_path(row: Mapping[str, object] | object) -> str:
    telescope = clean_text(row_get(row, "TELESCOPE"))
    field = clean_text(row_get(row, "FIELD"))
    file_name = clean_text(row_get(row, "FILE", ""))
    if not file_name:
        star_name = star_name_for_files(row)
        prefix = "asStar" if telescope == "lco25m" else "apStar"
        file_name = f"{prefix}-dr17-{star_name}.fits"
    return f"stars/{telescope}/{field}/{file_name}"


def wavelength_from_header(header: object, n_pixels: int) -> np.ndarray:
    """Construct a wavelength grid from an APOGEE log-linear FITS header."""

    crval = float(header["CRVAL1"])
    cdelt = float(header["CDELT1"])
    crpix = float(header.get("CRPIX1", 1.0))
    pixels = np.arange(n_pixels, dtype=float) + 1.0
    return 10.0 ** (crval + (pixels - crpix) * cdelt)


def load_payne_wavelength(path: str | os.PathLike[str]) -> np.ndarray:
    """Load a Payne wavelength npz file with a `wavelength` array."""

    with np.load(path) as payload:
        if "wavelength" not in payload:
            raise KeyError(f"{path} does not contain a 'wavelength' array")
        return np.asarray(payload["wavelength"], dtype=float)


def _parse_fits_value(raw_value: str) -> object:
    value = raw_value.split("/", 1)[0].strip()
    if not value:
        return ""
    if value.startswith("'"):
        end = value.rfind("'")
        if end > 0:
            return value[1:end].strip()
    if value in ("T", "F"):
        return value == "T"
    normalized = value.replace("D", "E")
    try:
        if any(char in normalized for char in (".", "E", "e")):
            return float(normalized)
        return int(normalized)
    except ValueError:
        return value


def _read_fits_header(handle) -> dict[str, object]:
    cards: list[str] = []
    while True:
        block = handle.read(2880)
        if len(block) != 2880:
            raise EOFError("Unexpected end of FITS file while reading header")
        for offset in range(0, 2880, 80):
            card = block[offset : offset + 80].decode("ascii", errors="ignore")
            cards.append(card)
            if card.startswith("END"):
                header: dict[str, object] = {}
                for item in cards:
                    key = item[:8].strip()
                    if not key or key in ("END", "COMMENT", "HISTORY"):
                        continue
                    if item[8:10] == "= ":
                        header[key] = _parse_fits_value(item[10:])
                return header


def _fits_dtype(bitpix: int) -> np.dtype:
    mapping = {
        8: np.dtype("u1"),
        16: np.dtype(">i2"),
        32: np.dtype(">i4"),
        64: np.dtype(">i8"),
        -32: np.dtype(">f4"),
        -64: np.dtype(">f8"),
    }
    if bitpix not in mapping:
        raise ValueError(f"Unsupported FITS BITPIX={bitpix}")
    return mapping[bitpix]


def _hdu_data_size(header: Mapping[str, object]) -> int:
    bitpix = int(header.get("BITPIX", 8))
    naxis = int(header.get("NAXIS", 0))
    if naxis == 0:
        return 0
    size = abs(bitpix) // 8
    for axis in range(1, naxis + 1):
        size *= int(header.get(f"NAXIS{axis}", 0))
    return size


def _skip_hdu_data(handle, header: Mapping[str, object]) -> None:
    data_size = _hdu_data_size(header)
    handle.seek(data_size + ((-data_size) % 2880), os.SEEK_CUR)


def read_fits_image_hdus(
    path: str | os.PathLike[str],
    hdu_indices: Sequence[int],
) -> dict[int, tuple[dict[str, object], np.ndarray | None]]:
    """Read selected FITS image HDUs without importing astropy.

    This minimal reader is intended for APOGEE `aspcapStar` image extensions.
    It is not a general replacement for astropy.io.fits.
    """

    wanted = set(hdu_indices)
    result: dict[int, tuple[dict[str, object], np.ndarray | None]] = {}
    max_hdu = max(wanted)
    with Path(path).open("rb") as handle:
        hdu_index = 0
        while hdu_index <= max_hdu:
            header = _read_fits_header(handle)
            bitpix = int(header.get("BITPIX", 8))
            naxis = int(header.get("NAXIS", 0))
            axes = [int(header.get(f"NAXIS{axis}", 0)) for axis in range(1, naxis + 1)]
            dtype = _fits_dtype(bitpix)
            data_size = int(np.prod(axes, dtype=np.int64)) * dtype.itemsize if axes else 0

            data = None
            if hdu_index in wanted:
                raw = handle.read(data_size)
                if len(raw) != data_size:
                    raise EOFError(f"Unexpected end of FITS file in HDU {hdu_index}")
                if data_size:
                    shape = tuple(reversed(axes))
                    data = np.frombuffer(raw, dtype=dtype).reshape(shape).astype(
                        dtype.newbyteorder("="), copy=False
                    )
                result[hdu_index] = (header, data)
            else:
                handle.seek(data_size, os.SEEK_CUR)

            padding = (-data_size) % 2880
            if padding:
                handle.seek(padding, os.SEEK_CUR)
            hdu_index += 1
    return result


def _table_format(tform: str) -> tuple[np.dtype, int]:
    match = re.match(r"^\s*(\d*)([A-Z])", tform)
    if match is None:
        raise ValueError(f"Unsupported FITS TFORM={tform!r}")
    repeat = int(match.group(1) or "1")
    code = match.group(2)
    base = {
        "A": np.dtype(f"S{repeat}"),
        "L": np.dtype(f"S{repeat}"),
        "B": np.dtype("u1"),
        "I": np.dtype(">i2"),
        "J": np.dtype(">i4"),
        "K": np.dtype(">i8"),
        "E": np.dtype(">f4"),
        "D": np.dtype(">f8"),
    }.get(code)
    if base is None:
        raise ValueError(f"Unsupported FITS TFORM={tform!r}")
    if code in ("A", "L"):
        return base, repeat
    if repeat == 1:
        return base, base.itemsize
    return np.dtype((base, (repeat,))), base.itemsize * repeat


class FitsColumnTable:
    """Memory-mapped selected columns from a FITS binary table."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        data_offset: int,
        n_rows: int,
        dtype: np.dtype,
    ) -> None:
        self.path = Path(path)
        self.colnames = list(dtype.names or ())
        self._data = np.memmap(
            self.path,
            mode="r",
            dtype=dtype,
            offset=data_offset,
            shape=(n_rows,),
        )

    def __len__(self) -> int:
        return int(self._data.shape[0])

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return {name: self._data[name][int(key)] for name in self.colnames}
        return self._data[key]


def read_fits_bintable_columns(
    path: str | os.PathLike[str],
    columns: Sequence[str] | None = None,
) -> FitsColumnTable:
    """Memory-map selected columns from the first FITS BINTABLE extension."""

    wanted = set(columns) if columns is not None else None
    with Path(path).open("rb") as handle:
        primary_header = _read_fits_header(handle)
        _skip_hdu_data(handle, primary_header)
        table_header = _read_fits_header(handle)
        data_offset = handle.tell()

    n_rows = int(table_header["NAXIS2"])
    row_size = int(table_header["NAXIS1"])
    n_fields = int(table_header["TFIELDS"])

    names: list[str] = []
    formats: list[np.dtype] = []
    offsets: list[int] = []
    running_offset = 0
    for field_index in range(1, n_fields + 1):
        name = clean_text(table_header.get(f"TTYPE{field_index}", ""))
        tform = clean_text(table_header.get(f"TFORM{field_index}", ""))
        dtype, n_bytes = _table_format(tform)
        if wanted is None or name in wanted:
            names.append(name)
            formats.append(dtype)
            offsets.append(running_offset)
        running_offset += n_bytes

    missing = sorted((wanted or set()) - set(names))
    if missing:
        print(f"Warning: missing allStar columns: {', '.join(missing)}")

    dtype = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": row_size}
    )
    return FitsColumnTable(path, data_offset=data_offset, n_rows=n_rows, dtype=dtype)


def read_aspcapstar(
    path: str | os.PathLike[str],
    *,
    target_wavelength: np.ndarray | None = None,
    error_floor: float = 0.005,
) -> dict[str, object]:
    """Read one aspcapStar file and optionally interpolate to a target grid."""

    hdus = read_fits_image_hdus(path, (0, 1, 2, 3))
    primary_header, _ = hdus[0]
    flux_header, flux_data = hdus[1]
    _, err_data = hdus[2]
    _, best_fit_data = hdus[3]
    if flux_data is None or err_data is None or best_fit_data is None:
        raise ValueError(f"{path} does not contain expected image HDUs 1, 2, and 3")

    flux = np.asarray(flux_data, dtype=np.float32)
    err = np.asarray(err_data, dtype=np.float32)
    best_fit = np.asarray(best_fit_data, dtype=np.float32)
    wave = wavelength_from_header(flux_header, flux.shape[-1])
    header = dict(primary_header)

    good = np.isfinite(flux) & np.isfinite(err) & (err > 0.0)
    if error_floor > 0:
        err = np.sqrt(err**2 + np.float32(error_floor) ** 2)

    if target_wavelength is not None:
        target_wavelength = np.asarray(target_wavelength, dtype=float)
        interp_flux = np.interp(target_wavelength, wave, flux, left=np.nan, right=np.nan)
        interp_err = np.interp(target_wavelength, wave, err, left=np.nan, right=np.nan)
        interp_model = np.interp(
            target_wavelength, wave, best_fit, left=np.nan, right=np.nan
        )
        interp_good = np.isfinite(interp_flux) & np.isfinite(interp_err) & (
            interp_err > 0.0
        )
        return {
            "wave": target_wavelength,
            "flux": interp_flux.astype(np.float32),
            "err": interp_err.astype(np.float32),
            "best_fit": interp_model.astype(np.float32),
            "mask": ~interp_good,
            "source_wave": wave,
            "source_mask": ~good,
            "header": header,
        }

    return {
        "wave": wave,
        "flux": flux,
        "err": err,
        "best_fit": best_fit,
        "mask": ~good,
        "header": header,
    }


def download_file(
    url: str,
    output_path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download one file with streaming I/O."""

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    with urlopen(url) as response, tmp_path.open("wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
    tmp_path.replace(output_path)
    return output_path


def write_manifest(
    rows: Iterable[Mapping[str, object]],
    output_path: str | os.PathLike[str],
    *,
    include_apstar: bool = True,
) -> Path:
    """Write a CSV manifest used by the downloader and preprocessor."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_columns = (
        "APOGEE_ID",
        "ALT_ID",
        "TELESCOPE",
        "FIELD",
        "LOCATION_ID",
        "FILE",
        "SNR",
        "SNREV",
        "NVISITS",
        "ASPCAPFLAGS",
        "STARFLAGS",
        "ASPCAPFLAG",
        "STARFLAG",
        "PARAMFLAG",
        "ELEMFLAG",
    )
    reference_columns = tuple(
        column for column in DEFAULT_REFERENCE_COLUMNS if column not in metadata_columns
    )
    path_columns = (
        "aspcapstar_url",
        "aspcapstar_path",
    )
    columns = list(metadata_columns + reference_columns + path_columns)
    if include_apstar:
        columns.extend(("apstar_url", "apstar_path"))

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            row_dict = {column: clean_text(row_get(row, column, "")) for column in columns}
            row_dict["aspcapstar_url"] = aspcapstar_url(row)
            row_dict["aspcapstar_path"] = aspcapstar_local_path(row)
            if include_apstar:
                row_dict["apstar_url"] = apstar_url(row)
                row_dict["apstar_path"] = apstar_local_path(row)
            writer.writerow(row_dict)
    return output_path


def read_manifest(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def save_preprocessed_npz(
    output_path: str | os.PathLike[str],
    *,
    wave: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    mask: np.ndarray,
    reference_labels: Mapping[str, np.ndarray],
    metadata: Sequence[Mapping[str, object]],
) -> Path:
    """Save a compact preprocessed APOGEE subset."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "wave": np.asarray(wave, dtype=np.float64),
        "flux": np.asarray(flux, dtype=np.float32),
        "err": np.asarray(err, dtype=np.float32),
        "mask": np.asarray(mask, dtype=bool),
        "metadata_json": np.asarray([json.dumps(item) for item in metadata]),
    }
    for key, value in reference_labels.items():
        payload[f"label_{key}"] = np.asarray(value)
    np.savez_compressed(output_path, **payload)
    return output_path
