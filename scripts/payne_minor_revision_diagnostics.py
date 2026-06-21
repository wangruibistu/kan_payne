#!/usr/bin/env python
"""Compute small diagnostic tables requested during paper revision."""

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
    parser.add_argument("--fit", nargs="+", required=True, help="Main APOGEE fit NPZ files.")
    parser.add_argument("--observed", required=True, help="Preprocessed APOGEE observed spectra NPZ.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-npix", type=int, default=5000)
    parser.add_argument("--boundary", type=float, default=0.49)
    parser.add_argument("--chi2-quantile", type=float, default=0.99)
    parser.add_argument("--mae-quantile", type=float, default=0.99)
    return parser.parse_args()


def _model_name(path: Path) -> str:
    name = path.name
    if "kan_payne" in name:
        return "KAN-Payne"
    if "transformer_payne" in name:
        return "TransformerPayne-style"
    if "payne_mlp" in name:
        return "Payne-MLP"
    return path.stem


def _mad_sigma(values):
    import numpy as np

    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    med = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - med)))


def _quality_mask(data, label_index, args):
    import numpy as np

    success = data["success"].astype(bool)
    base = success & (data["npix"] >= args.min_npix)
    for label, _, _ in LABELS:
        base &= np.abs(data["fitted_label_scaled"][:, label_index[label]]) < args.boundary
    fit_base = success & (data["npix"] >= args.min_npix)
    chi2_cut = np.nanquantile(data["chi2"][fit_base], args.chi2_quantile)
    mae_cut = np.nanquantile(data["mae_x1e4"][fit_base], args.mae_quantile)
    return base & (data["chi2"] <= chi2_cut) & (data["mae_x1e4"] <= mae_cut)


def _spearman(x, y):
    import numpy as np

    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return float("nan"), float("nan")
    rx = _rank_average_ties(x[mask])
    ry = _rank_average_ties(y[mask])
    rho = np.corrcoef(rx, ry)[0, 1]
    return float(rho), float("nan")


def _rank_average_ties(values):
    import numpy as np

    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _linear_slope(x, y):
    import numpy as np

    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return float(slope), float(intercept)


def _observed_snr(path: Path, row_index):
    import numpy as np

    with np.load(path, allow_pickle=True) as obs:
        flux = np.asarray(obs["flux"], dtype=float)[row_index]
        err = np.asarray(obs["err"], dtype=float)[row_index]
        mask = np.asarray(obs["mask"], dtype=bool)[row_index]
    valid = (~mask) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    snr = np.full(flux.shape[0], np.nan, dtype=float)
    ratio = np.full_like(flux, np.nan, dtype=float)
    ratio[valid] = np.abs(flux[valid]) / err[valid]
    for i in range(flux.shape[0]):
        if np.any(np.isfinite(ratio[i])):
            snr[i] = np.nanmedian(ratio[i])
    return snr


def main() -> None:
    args = parse_args()

    import numpy as np

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fits = []
    common = None
    for fit in args.fit:
        path = Path(fit)
        data = np.load(path, allow_pickle=True)
        names = [str(item) for item in data["label_names"]]
        label_index = {name: i for i, name in enumerate(names)}
        quality = _quality_mask(data, label_index, args)
        common = quality.copy() if common is None else (common & quality)
        fits.append(
            {
                "model": _model_name(path),
                "path": str(path),
                "data": data,
                "label_index": label_index,
                "quality": quality,
            }
        )

    own_rows = []
    for item in fits:
        data = item["data"]
        label_index = item["label_index"]
        for label, aspcap_key, unit in LABELS:
            pred = data["fitted_labels"][:, label_index[label]]
            ref = data[f"aspcap_{aspcap_key}"]
            for sample_name, mask in [
                ("own_quality", item["quality"]),
                ("common_quality", common),
            ]:
                finite = mask & np.isfinite(pred) & np.isfinite(ref)
                residual = pred[finite] - ref[finite]
                own_rows.append(
                    {
                        "model": item["model"],
                        "sample": sample_name,
                        "label": label,
                        "unit": unit,
                        "n": int(np.count_nonzero(finite)),
                        "median": float(np.nanmedian(residual)),
                        "mad_sigma": _mad_sigma(residual),
                    }
                )

    kan = next(item for item in fits if item["model"] == "KAN-Payne")
    data = kan["data"]
    idx = kan["label_index"]
    mask = common.copy()
    logg_resid = data["fitted_labels"][:, idx["logg"]] - data["aspcap_LOGG"]
    row_index = data["row_index"] if "row_index" in data.files else np.arange(logg_resid.size)
    snr = _observed_snr(Path(args.observed), row_index)
    diagnostics = [
        ("ASPCAP_Teff", data["aspcap_TEFF"], "K"),
        ("ASPCAP_logg", data["aspcap_LOGG"], "dex"),
        ("ASPCAP_Fe_H", data["aspcap_FE_H"], "dex"),
        ("median_snr", snr, ""),
    ]
    trend_rows = []
    for name, values, unit in diagnostics:
        finite = mask & np.isfinite(logg_resid) & np.isfinite(values)
        rho, pvalue = _spearman(values[finite], logg_resid[finite])
        slope, intercept = _linear_slope(values[finite], logg_resid[finite])
        trend_rows.append(
            {
                "model": "KAN-Payne",
                "residual": "logg_fit_minus_aspcap",
                "x": name,
                "x_unit": unit,
                "n": int(np.count_nonzero(finite)),
                "spearman_rho": rho,
                "spearman_pvalue": pvalue,
                "linear_slope_dex_per_xunit": slope,
                "linear_intercept": intercept,
            }
        )

    with (output_dir / "apogee_own_vs_common_residual_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(own_rows[0].keys()))
        writer.writeheader()
        writer.writerows(own_rows)
    with (output_dir / "kan_payne_logg_residual_trends.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trend_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trend_rows)
    payload = {"own_vs_common": own_rows, "kan_payne_logg_trends": trend_rows}
    (output_dir / "minor_revision_diagnostics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
