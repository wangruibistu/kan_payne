#!/usr/bin/env python
"""Download APOGEE DR17 per-star spectra from a manifest CSV."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default="data/raw/apogee_dr17")
    parser.add_argument(
        "--product",
        choices=("aspcapstar", "apstar"),
        default="aspcapstar",
        help="Which URL/path columns to use from the manifest.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print one progress line every N completed files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue when one spectrum fails to download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from kan_payne.apogee_dr17 import download_file, read_manifest

    rows = read_manifest(args.manifest)
    if args.max_files is not None:
        rows = rows[: args.max_files]

    url_column = f"{args.product}_url"
    path_column = f"{args.product}_path"
    data_root = Path(args.data_root)

    def fetch(index: int, row: dict[str, str]) -> tuple[int, Path, str | None]:
        url = row[url_column]
        output_path = data_root / row[path_column]
        try:
            download_file(url, output_path, overwrite=args.overwrite)
        except (HTTPError, URLError, OSError) as exc:
            return index, output_path, f"Failed to download {url}: {exc}"
        return index, output_path, None

    failures = 0
    if args.workers <= 1:
        for index, row in enumerate(rows, start=1):
            if index == 1 or index == len(rows) or index % args.progress_every == 0:
                print(f"[{index}/{len(rows)}] {data_root / row[path_column]}", flush=True)
            _, _, error = fetch(index, row)
            if error is not None:
                failures += 1
                if not args.keep_going:
                    raise SystemExit(error)
                print(error, flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(fetch, index, row): index
                for index, row in enumerate(rows, start=1)
            }
            for done, future in enumerate(as_completed(futures), start=1):
                index, output_path, error = future.result()
                should_print = (
                    done == 1
                    or done == len(rows)
                    or done % args.progress_every == 0
                    or error is not None
                )
                if should_print:
                    print(f"[{done}/{len(rows)}] row={index} {output_path}", flush=True)
                if error is not None:
                    failures += 1
                    if not args.keep_going:
                        raise SystemExit(error)
                    print(error, flush=True)

    if failures:
        raise SystemExit(f"Completed with {failures} failed downloads")
    print("Download complete")


if __name__ == "__main__":
    main()
