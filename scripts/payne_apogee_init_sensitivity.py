#!/usr/bin/env python
"""Compare APOGEE fits initialized from ASPCAP labels and grid midpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LABELS = [
    ("Teff", "TEFF", "K"),
    ("logg", "LOGG", "dex"),
    ("Fe_H", "FE_H", "dex"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aspcap-init",
        nargs="+",
        required=True,
        help="Fit files initialized from ASPCAP labels.",
    )
    parser.add_argument(
        "--midpoint-init",
        nargs="+",
        required=True,
        help="Fit files initialized from the synthetic-grid midpoint.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _model_name(path: Path) -> str:
    name = path.name
    if "kan_payne" in name:
        return "KAN-Payne"
    if "transformer_payne" in name:
        return "TransformerPayne"
    if "payne_mlp" in name:
        return "Payne-MLP"
    return path.stem


def _mad_sigma(values):
    import numpy as np

    med = np.nanmedian(values)
    return 1.4826 * np.nanmedian(np.abs(values - med))


def _load_by_model(paths):
    import numpy as np

    loaded = {}
    for item in paths:
        path = Path(item)
        data = np.load(path, allow_pickle=True)
        loaded[_model_name(path)] = {"path": str(path), "data": data}
    return loaded


def main() -> None:
    args = parse_args()

    import numpy as np

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    aspcap = _load_by_model(args.aspcap_init)
    midpoint = _load_by_model(args.midpoint_init)
    models = sorted(set(aspcap) & set(midpoint))
    if not models:
        raise SystemExit("No matching models between --aspcap-init and --midpoint-init")

    summary_rows = []
    residual_rows = []
    for model in models:
        a = aspcap[model]["data"]
        m = midpoint[model]["data"]
        n = min(a["fitted_labels"].shape[0], m["fitted_labels"].shape[0])
        if "row_index" in a.files and "row_index" in m.files:
            if not np.array_equal(a["row_index"][:n], m["row_index"][:n]):
                raise SystemExit(f"row_index mismatch for {model}")
        names = [str(item) for item in a["label_names"]]
        idx = {name: i for i, name in enumerate(names)}
        good = a["success"][:n].astype(bool) & m["success"][:n].astype(bool)
        summary_rows.append(
            {
                "model": model,
                "n_common_success": int(good.sum()),
                "aspcap_init_chi2_median": float(np.nanmedian(a["chi2"][:n][good])),
                "midpoint_init_chi2_median": float(np.nanmedian(m["chi2"][:n][good])),
                "aspcap_init_mae_x1e4_median": float(np.nanmedian(a["mae_x1e4"][:n][good])),
                "midpoint_init_mae_x1e4_median": float(np.nanmedian(m["mae_x1e4"][:n][good])),
            }
        )
        for label, aspcap_key, unit in LABELS:
            ref = a[f"aspcap_{aspcap_key}"][:n]
            a_fit = a["fitted_labels"][:n, idx[label]]
            m_fit = m["fitted_labels"][:n, idx[label]]
            mask = good & np.isfinite(ref) & np.isfinite(a_fit) & np.isfinite(m_fit)
            a_res = a_fit[mask] - ref[mask]
            m_res = m_fit[mask] - ref[mask]
            delta = m_fit[mask] - a_fit[mask]
            residual_rows.append(
                {
                    "model": model,
                    "label": label,
                    "unit": unit,
                    "n": int(mask.sum()),
                    "aspcap_init_median": float(np.nanmedian(a_res)),
                    "aspcap_init_mad_sigma": float(_mad_sigma(a_res)),
                    "midpoint_init_median": float(np.nanmedian(m_res)),
                    "midpoint_init_mad_sigma": float(_mad_sigma(m_res)),
                    "midpoint_minus_aspcap_init_median": float(np.nanmedian(delta)),
                    "midpoint_minus_aspcap_init_mad_sigma": float(_mad_sigma(delta)),
                }
            )

    with (out / "apogee_init_sensitivity_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (out / "apogee_init_sensitivity_residuals.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(residual_rows[0].keys()))
        writer.writeheader()
        writer.writerows(residual_rows)
    (out / "apogee_init_sensitivity_summary.json").write_text(
        json.dumps({"summary": summary_rows, "residuals": residual_rows}, indent=2, sort_keys=True)
        + "\n"
    )
    print(json.dumps({"summary": summary_rows, "residuals": residual_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
