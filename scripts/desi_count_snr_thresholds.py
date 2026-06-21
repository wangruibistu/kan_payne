#!/usr/bin/env python
"""Count DESI spectra above S/N thresholds from DR1 FITS catalogs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="append", required=True, help="Input FITS catalog; may repeat.")
    parser.add_argument("--hdu", default=None, help="Table HDU name or number. Default: scan all table HDUs.")
    parser.add_argument(
        "--snr-column",
        default="auto",
        help="S/N column. Use auto to choose the best SNR/TSNR2-like column in each HDU.",
    )
    parser.add_argument("--thresholds", default="10,20,30,50")
    parser.add_argument("--surveys", default=None, help="Optional comma-separated SURVEY filter.")
    parser.add_argument("--programs", default=None, help="Optional comma-separated PROGRAM filter.")
    parser.add_argument("--spectype", default=None, help="Optional SPECTYPE filter, e.g. STAR.")
    parser.add_argument("--primary-only", action="store_true", help="Require ZCAT_PRIMARY/PRIMARY-like column if present.")
    parser.add_argument("--zwarn-zero", action="store_true", help="Require ZWARN==0 if present.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def _norm_set(text: str | None) -> set[str] | None:
    if text is None:
        return None
    values = {item.strip().lower() for item in text.split(",") if item.strip()}
    return values or None


def _find_col(columns, *candidates):
    by_upper = {col.upper(): col for col in columns}
    for candidate in candidates:
        if candidate.upper() in by_upper:
            return by_upper[candidate.upper()]
    return None


def _snr_candidates(columns):
    cols = list(columns)
    upper = {col: col.upper() for col in cols}
    candidates = [col for col in cols if "SNR" in upper[col]]
    if not candidates:
        return []
    preferred_tokens = [
        "MEDIAN_CALIB_SNR",
        "MEDIAN_COADD_SNR",
        "SNR_MEDIAN",
        "SNR",
        "TSNR2_MWS",
        "TSNR2_BGS",
        "TSNR2_GPBBRIGHT",
        "TSNR2_GPBDARK",
        "TSNR2_LRG",
        "TSNR2_ELG",
    ]
    def score(col):
        u = upper[col]
        for i, token in enumerate(preferred_tokens):
            if token in u:
                return i
        return 100
    return sorted(candidates, key=score)


def _table_hdus(hdul, hdu_arg):
    if hdu_arg is not None:
        try:
            idx = int(hdu_arg)
            return [(idx, hdul[idx])]
        except ValueError:
            return [(hdu_arg, hdul[hdu_arg])]
    out = []
    for i, hdu in enumerate(hdul):
        if getattr(hdu, "columns", None) is not None and hdu.columns is not None:
            if hdu.columns.names:
                out.append((hdu.name or str(i), hdu))
    return out


def _as_str_array(values):
    import numpy as np

    if values.dtype.kind in {"S", "U", "O"}:
        return np.char.lower(np.asarray(values).astype(str))
    return np.char.lower(np.asarray(values).astype(str))


def _counts_for_hdu(hdu, *, snr_column, thresholds, surveys, programs, spectype, primary_only, zwarn_zero):
    import numpy as np

    columns = list(hdu.columns.names or [])
    data = hdu.data
    if data is None or len(data) == 0:
        return None

    if snr_column == "auto":
        candidates = _snr_candidates(columns)
        if not candidates:
            return {
                "available_snr_columns": [],
                "error": "no SNR/TSNR2-like column found",
                "rows_total": int(len(data)),
            }
        chosen = candidates[0]
    else:
        chosen = _find_col(columns, snr_column)
        if chosen is None:
            return {
                "available_snr_columns": _snr_candidates(columns),
                "error": f"requested S/N column {snr_column!r} not found",
                "rows_total": int(len(data)),
            }

    values = np.asarray(data.field(chosen), dtype=float)
    transform = "sqrt" if chosen.upper().startswith("TSNR2") else "identity"
    snr = np.sqrt(np.clip(values, 0.0, None)) if transform == "sqrt" else values
    keep = np.isfinite(snr)

    survey_col = _find_col(columns, "SURVEY")
    if surveys is not None and survey_col is not None:
        keep &= np.isin(_as_str_array(data.field(survey_col)), list(surveys))
    program_col = _find_col(columns, "PROGRAM")
    if programs is not None and program_col is not None:
        keep &= np.isin(_as_str_array(data.field(program_col)), list(programs))
    spectype_col = _find_col(columns, "SPECTYPE")
    if spectype is not None and spectype_col is not None:
        keep &= _as_str_array(data.field(spectype_col)) == spectype.lower()
    if primary_only:
        primary_col = _find_col(columns, "ZCAT_PRIMARY", "PRIMARY", "BEST")
        if primary_col is not None:
            keep &= np.asarray(data.field(primary_col)).astype(bool)
    if zwarn_zero:
        zwarn_col = _find_col(columns, "ZWARN")
        if zwarn_col is not None:
            keep &= np.asarray(data.field(zwarn_col)) == 0

    target_col = _find_col(columns, "TARGETID", "TARGET_ID")
    counts = {}
    for threshold in thresholds:
        mask = keep & (snr > threshold)
        if target_col is not None:
            unique_targets = int(np.unique(np.asarray(data.field(target_col))[mask]).size)
        else:
            unique_targets = None
        counts[str(threshold)] = {
            "rows": int(np.count_nonzero(mask)),
            "unique_targetid": unique_targets,
        }

    return {
        "rows_total": int(len(data)),
        "rows_after_basic_filters": int(np.count_nonzero(keep)),
        "chosen_snr_column": chosen,
        "snr_transform": transform,
        "available_snr_columns": _snr_candidates(columns),
        "counts_gt": counts,
    }


def main() -> None:
    args = parse_args()
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    surveys = _norm_set(args.surveys)
    programs = _norm_set(args.programs)

    from astropy.io import fits

    results = {}
    for catalog in args.catalog:
        path = Path(catalog)
        per_hdu = {}
        with fits.open(path, memmap=True) as hdul:
            for hdu_id, hdu in _table_hdus(hdul, args.hdu):
                per_hdu[str(hdu_id)] = _counts_for_hdu(
                    hdu,
                    snr_column=args.snr_column,
                    thresholds=thresholds,
                    surveys=surveys,
                    programs=programs,
                    spectype=args.spectype,
                    primary_only=args.primary_only,
                    zwarn_zero=args.zwarn_zero,
                )
        results[str(path)] = per_hdu

    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
