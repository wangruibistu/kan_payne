#!/usr/bin/env python
"""Build a filtered NewEra HSR HDF5 download manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="list_of_available_NewEraV3_models.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teff-range", default="4500,7000")
    parser.add_argument("--logg-range", default="1.0,5.0")
    parser.add_argument("--mh-range", default="-2.5,0.5")
    parser.add_argument("--alpha-range", default="-0.2,0.6")
    parser.add_argument("--teff-step", type=float, default=250.0, help="Keep Teff multiples of this step; 0 disables.")
    parser.add_argument("--logg-step", type=float, default=0.5, help="Keep logg multiples of this step; 0 disables.")
    parser.add_argument("--mh-step", type=float, default=0.5, help="Keep [M/H] multiples of this step; 0 disables.")
    parser.add_argument("--alpha-values", default="-0.2,0.0,0.2,0.4,0.6")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth row after sorting; useful for smoke tests.")
    return parser.parse_args()


def _range(text: str) -> tuple[float, float]:
    left, right = text.split(",", 1)
    return float(left), float(right)


def _values(text: str) -> set[float]:
    return {round(float(item), 6) for item in text.split(",") if item.strip()}


def _in_range(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def _on_step(value: float, step: float) -> bool:
    if step <= 0:
        return True
    return abs(value / step - round(value / step)) < 1.0e-6


def parse_hsr_filename(filename: str):
    match = re.search(
        r"lte(?P<teff>[0-9]{4,5})(?P<neglogg>[+-][0-9]+(?:\.[0-9]+)?)"
        r"(?P<mh>[+-][0-9]+(?:\.[0-9]+)?)(?:\.alpha=(?P<alpha>[+-][0-9]+(?:\.[0-9]+)?))?"
        r"\.PHOENIX-NewEra-ACES-COND-2023\.HSR\.h5",
        filename,
    )
    if not match:
        return None
    return {
        "teff": float(match.group("teff")),
        "logg": -float(match.group("neglogg")),
        "m_h": float(match.group("mh")),
        "alpha_m": float(match.group("alpha") or 0.0),
    }


def iter_catalog_rows(path: Path):
    for line in path.read_text(errors="ignore").splitlines()[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        labels = parse_hsr_filename(parts[1])
        if labels is None:
            continue
        try:
            size = int(parts[3])
        except ValueError:
            size = None
        yield {
            "index": int(parts[0]),
            "filename": parts[1],
            "md5": parts[2],
            "filesize": size,
            "url": parts[4],
            **labels,
        }


def main() -> None:
    args = parse_args()
    teff_range = _range(args.teff_range)
    logg_range = _range(args.logg_range)
    mh_range = _range(args.mh_range)
    alpha_range = _range(args.alpha_range)
    alpha_values = _values(args.alpha_values)

    selected = []
    for row in iter_catalog_rows(Path(args.catalog)):
        if not _in_range(row["teff"], teff_range):
            continue
        if not _in_range(row["logg"], logg_range):
            continue
        if not _in_range(row["m_h"], mh_range):
            continue
        if not _in_range(row["alpha_m"], alpha_range):
            continue
        if alpha_values and round(row["alpha_m"], 6) not in alpha_values:
            continue
        if not _on_step(row["teff"], args.teff_step):
            continue
        if not _on_step(row["logg"], args.logg_step):
            continue
        if not _on_step(row["m_h"], args.mh_step):
            continue
        selected.append(row)

    selected.sort(key=lambda item: (item["teff"], item["logg"], item["m_h"], item["alpha_m"], item["filename"]))
    if args.stride > 1:
        selected = selected[:: args.stride]
    if args.max_files is not None:
        selected = selected[: args.max_files]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "newera_hsr_selected.csv"
    url_path = out_dir / "newera_hsr_urls.txt"
    summary_path = out_dir / "newera_hsr_manifest_summary.json"

    fields = ["index", "filename", "teff", "logg", "m_h", "alpha_m", "filesize", "md5", "url"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({field: row[field] for field in fields})
    with url_path.open("w") as handle:
        for row in selected:
            handle.write(f"{row['url']}?download=1\n")
    summary = {
        "catalog": args.catalog,
        "n_selected": len(selected),
        "total_size_gb": sum((row["filesize"] or 0) for row in selected) / 1.0e9,
        "teff_range": teff_range,
        "logg_range": logg_range,
        "mh_range": mh_range,
        "alpha_range": alpha_range,
        "alpha_values": sorted(alpha_values),
        "teff_step": args.teff_step,
        "logg_step": args.logg_step,
        "mh_step": args.mh_step,
        "max_files": args.max_files,
        "stride": args.stride,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
