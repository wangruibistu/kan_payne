#!/usr/bin/env python
"""Build DESI DR1 healpix spectrum download manifests from MWS/zcatalog FITS files.

The script reads only a small set of FITS columns and writes URL manifests for
the coadd/spectra/redrock products needed by KAN-Payne inference. It is meant
to run on lily after the DESI DR1 zcatalog and MWS VAC seed files are present.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


DESI_DR1_BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        action="append",
        required=True,
        help="Input FITS catalog. May be repeated, e.g. MWS VAC and zpix main catalogs.",
    )
    parser.add_argument("--output-dir", default="/home/wangrui/data/desi/dr1/healpix_pilot")
    parser.add_argument("--base-url", default=DESI_DR1_BASE)
    parser.add_argument("--surveys", default="main")
    parser.add_argument("--programs", default="bright,dark")
    parser.add_argument(
        "--default-survey",
        default=None,
        help="Survey value to use when the selected FITS table has no SURVEY column.",
    )
    parser.add_argument(
        "--default-program",
        default=None,
        help="Program value to use when the selected FITS table has no PROGRAM column.",
    )
    parser.add_argument("--products", default="coadd,redrock")
    parser.add_argument("--max-targets", type=int, default=10000)
    parser.add_argument("--max-healpix", type=int, default=None)
    parser.add_argument(
        "--zero-column",
        action="append",
        default=[],
        help="Keep rows where this column is zero, if present. May be repeated.",
    )
    parser.add_argument(
        "--finite-column",
        action="append",
        default=[],
        help="Keep rows where this column is finite, if present. May be repeated.",
    )
    parser.add_argument(
        "--min-column",
        action="append",
        default=[],
        metavar="COLUMN:VALUE",
        help="Keep rows where COLUMN >= VALUE, if present. Useful for S/N cuts.",
    )
    parser.add_argument(
        "--max-column",
        action="append",
        default=[],
        metavar="COLUMN:VALUE",
        help="Keep rows where COLUMN <= VALUE, if present.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Inspect available columns and selected healpix groups without writing URLs.",
    )
    return parser.parse_args()


def _norm_set(text: str) -> set[str]:
    return {item.strip().lower() for item in text.split(",") if item.strip()}


def _parse_thresholds(items: Iterable[str]) -> dict[str, float]:
    thresholds = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Threshold must have COLUMN:VALUE form, got {item!r}")
        col, value = item.split(":", 1)
        thresholds[col.strip()] = float(value)
    return thresholds


def _find_col(columns: Iterable[str], *candidates: str) -> str | None:
    by_upper = {col.upper(): col for col in columns}
    for candidate in candidates:
        if candidate.upper() in by_upper:
            return by_upper[candidate.upper()]
    return None


def _read_relevant_columns(
    path: Path,
    extra_columns: Iterable[str],
    *,
    default_survey: str | None,
    default_program: str | None,
):
    import numpy as np
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        hdu_index = None
        for i, hdu in enumerate(hdul):
            if getattr(hdu, "columns", None) is not None and hdu.columns is not None:
                names = list(hdu.columns.names or [])
                if names:
                    hdu_index = i
                    break
        if hdu_index is None:
            raise ValueError(f"No table HDU found in {path}")

        data = hdul[hdu_index].data
        columns = list(hdul[hdu_index].columns.names or [])
        target_col = _find_col(columns, "TARGETID", "TARGET_ID")
        survey_col = _find_col(columns, "SURVEY")
        program_col = _find_col(columns, "PROGRAM")
        healpix_col = _find_col(columns, "HEALPIX", "HPXPIXEL", "HPXPIX")

        if survey_col is None and default_survey is None:
            missing = "SURVEY"
        elif program_col is None and default_program is None:
            missing = "PROGRAM"
        elif target_col is None:
            missing = "TARGETID"
        elif healpix_col is None:
            missing = "HEALPIX"
        else:
            missing = None
        if missing is not None:
            raise ValueError(
                f"{path} is missing required {missing}-like column; available columns: "
                + ", ".join(columns[:80])
            )

        selected_columns = {
            "targetid": target_col,
            "healpix": healpix_col,
        }
        if survey_col is not None:
            selected_columns["survey"] = survey_col
        if program_col is not None:
            selected_columns["program"] = program_col
        for extra in extra_columns:
            col = _find_col(columns, extra)
            if col is not None:
                selected_columns[extra] = col

        result = {}
        for key, col in selected_columns.items():
            values = data.field(col)
            if values.dtype.kind in {"S", "U", "O"}:
                values = np.asarray(values).astype(str)
            else:
                values = np.asarray(values)
            result[key] = values
        n_rows = len(result["targetid"])
        if "survey" not in result:
            result["survey"] = np.full(n_rows, default_survey, dtype=object)
        if "program" not in result:
            result["program"] = np.full(n_rows, default_program, dtype=object)
        return columns, result


def _apply_filters(
    rows,
    *,
    surveys: set[str],
    programs: set[str],
    zero_columns,
    finite_columns,
    min_columns: dict[str, float],
    max_columns: dict[str, float],
):
    import numpy as np

    n = len(rows["targetid"])
    keep = np.ones(n, dtype=bool)
    keep &= np.isin(np.char.lower(rows["survey"].astype(str)), list(surveys))
    keep &= np.isin(np.char.lower(rows["program"].astype(str)), list(programs))
    keep &= np.isfinite(rows["healpix"].astype(float))

    for col in zero_columns:
        if col in rows:
            keep &= rows[col] == 0
    for col in finite_columns:
        if col in rows:
            keep &= np.isfinite(rows[col].astype(float))
    for col, threshold in min_columns.items():
        if col in rows:
            values = rows[col].astype(float)
            keep &= np.isfinite(values) & (values >= threshold)
    for col, threshold in max_columns.items():
        if col in rows:
            values = rows[col].astype(float)
            keep &= np.isfinite(values) & (values <= threshold)
    return keep


def _product_filename(product: str, survey: str, program: str, healpix: int) -> str:
    if product == "coadd":
        return f"coadd-{survey}-{program}-{healpix}.fits"
    if product == "spectra":
        return f"spectra-{survey}-{program}-{healpix}.fits"
    if product == "redrock":
        return f"redrock-{survey}-{program}-{healpix}.fits"
    raise ValueError(f"Unsupported product: {product}")


def main() -> None:
    args = parse_args()

    import numpy as np

    surveys = _norm_set(args.surveys)
    programs = _norm_set(args.programs)
    products = [item.strip().lower() for item in args.products.split(",") if item.strip()]
    min_columns = _parse_thresholds(args.min_column)
    max_columns = _parse_thresholds(args.max_column)
    extra_columns = set(args.zero_column) | set(args.finite_column) | set(min_columns) | set(max_columns)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    catalog_summaries = []
    for catalog in args.catalog:
        path = Path(catalog)
        columns, rows = _read_relevant_columns(
            path,
            extra_columns,
            default_survey=args.default_survey,
            default_program=args.default_program,
        )
        keep = _apply_filters(
            rows,
            surveys=surveys,
            programs=programs,
            zero_columns=args.zero_column,
            finite_columns=args.finite_column,
            min_columns=min_columns,
            max_columns=max_columns,
        )
        idx = np.flatnonzero(keep)
        if args.max_targets is not None:
            remaining = max(0, args.max_targets - len(selected_rows))
            idx = idx[:remaining]
        for i in idx:
            selected_rows.append(
                {
                    "catalog": str(path),
                    "targetid": str(rows["targetid"][i]),
                    "survey": str(rows["survey"][i]).lower(),
                    "program": str(rows["program"][i]).lower(),
                    "healpix": int(rows["healpix"][i]),
                }
            )
        catalog_summaries.append(
            {
                "catalog": str(path),
                "columns_preview": columns[:80],
                "rows_selected": int(len(idx)),
                "rows_total": int(len(rows["targetid"])),
            }
        )
        if args.max_targets is not None and len(selected_rows) >= args.max_targets:
            break

    groups = sorted({(row["survey"], row["program"], row["healpix"]) for row in selected_rows})
    if args.max_healpix is not None:
        groups = groups[: args.max_healpix]

    urls = []
    for survey, program, healpix in groups:
        group = healpix // 100
        for product in products:
            filename = _product_filename(product, survey, program, healpix)
            urls.append(f"{args.base_url}/{survey}/{program}/{group}/{healpix}/{filename}")

    summary = {
        "catalogs": catalog_summaries,
        "n_targets": len(selected_rows),
        "n_healpix_groups": len(groups),
        "n_urls": len(urls),
        "products": products,
        "surveys": sorted(surveys),
        "programs": sorted(programs),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_only:
        return

    url_path = out_dir / "desi_dr1_healpix_urls.txt"
    group_path = out_dir / "desi_dr1_healpix_groups.csv"
    target_path = out_dir / "desi_dr1_selected_targets.csv"
    summary_path = out_dir / "desi_dr1_healpix_manifest_summary.json"

    url_path.write_text("\n".join(urls) + "\n")
    with group_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["survey", "program", "healpix"])
        writer.writeheader()
        for survey, program, healpix in groups:
            writer.writerow({"survey": survey, "program": program, "healpix": healpix})
    with target_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["catalog", "targetid", "survey", "program", "healpix"])
        writer.writeheader()
        writer.writerows(selected_rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {url_path}")
    print(f"Wrote {group_path}")
    print(f"Wrote {target_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
