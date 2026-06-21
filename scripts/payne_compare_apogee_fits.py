#!/usr/bin/env python
"""Compare APOGEE fitted Payne labels with ASPCAP reference labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _robust_stats(values):
    import numpy as np

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "median": float("nan"), "mad_sigma": float("nan"), "p16": float("nan"), "p84": float("nan")}
    median = np.median(values)
    mad_sigma = 1.4826 * np.median(np.abs(values - median))
    return {
        "n": int(values.size),
        "median": float(median),
        "mad_sigma": float(mad_sigma),
        "p16": float(np.percentile(values, 16)),
        "p84": float(np.percentile(values, 84)),
    }


def _model_name(path: Path, payload) -> str:
    if "model" in payload:
        value = payload["model"]
        return str(value.tolist() if hasattr(value, "tolist") else value)
    return path.stem


def main() -> None:
    args = parse_args()

    import numpy as np

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    label_pairs = {
        "Teff": "TEFF",
        "logg": "LOGG",
        "Fe_H": "FE_H",
    }
    for fit_path_text in args.fit:
        fit_path = Path(fit_path_text)
        with np.load(fit_path, allow_pickle=True) as payload:
            data = {key: payload[key] for key in payload.files}
        model = _model_name(fit_path, data)
        label_names = [str(item) for item in data["label_names"]]
        fitted = np.asarray(data["fitted_labels"], dtype=float)
        success = np.asarray(data["success"], dtype=bool)
        for label_name, aspcap_name in label_pairs.items():
            if label_name not in label_names:
                continue
            aspcap_key = f"aspcap_{aspcap_name}"
            if aspcap_key not in data:
                continue
            idx = label_names.index(label_name)
            reference = np.asarray(data[aspcap_key], dtype=float)
            residual = fitted[:, idx] - reference
            valid = success & np.isfinite(residual) & np.isfinite(reference)
            stats = _robust_stats(residual[valid])
            rows.append(
                {
                    "model": model,
                    "fit": str(fit_path),
                    "label": label_name,
                    "aspcap": aspcap_name,
                    **stats,
                }
            )
        rows.append(
            {
                "model": model,
                "fit": str(fit_path),
                "label": "fit_mae_x1e4",
                "aspcap": "",
                **_robust_stats(np.asarray(data["mae_x1e4"], dtype=float)[success]),
            }
        )
        rows.append(
            {
                "model": model,
                "fit": str(fit_path),
                "label": "fit_chi2",
                "aspcap": "",
                **_robust_stats(np.asarray(data["chi2"], dtype=float)[success]),
            }
        )

    csv_path = output_dir / "apogee_fit_aspcap_summary.csv"
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    json_path = output_dir / "apogee_fit_aspcap_summary.json"
    with json_path.open("w") as handle:
        json.dump({"summary": rows, "summary_csv": str(csv_path)}, handle, indent=2, sort_keys=True)
    print(f"Wrote APOGEE fit comparison to {output_dir}")


if __name__ == "__main__":
    main()
