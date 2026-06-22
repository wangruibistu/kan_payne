#!/usr/bin/env python
"""Fit one DESI spectrum with per-arm DESI resolution matrices."""

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
    parser.add_argument("--selected-targets", required=True)
    parser.add_argument("--coadd-root", required=True)
    parser.add_argument("--sp-catalog", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--targetid", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--continuum-window", type=int, default=301)
    parser.add_argument("--fit-pixels", type=int, default=0, help="0 uses all good pixels.")
    parser.add_argument("--error-floor", type=float, default=0.0)
    parser.add_argument("--no-resolution", action="store_true")
    parser.add_argument("--save-residuals", default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def _choose_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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


def _load_selected_targets(path: Path):
    rows = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "survey": str(row["survey"]),
                    "program": str(row["program"]),
                    "healpix": int(row["healpix"]),
                    "targetid": int(row["targetid"]),
                }
            )
    return rows


def _load_sp_record(paths, targetid: int):
    import numpy as np
    from astropy.io import fits

    wanted = [
        "TARGETID",
        "TEFF",
        "LOGG",
        "FEH",
        "ALPHAFE",
        "SNR_MED",
        "RV_ADOP",
        "RV_ERR",
        "SUCCESS",
        "HEALPIX",
    ]
    for path in paths:
        with fits.open(path, memmap=True) as hdul:
            table = hdul["SPTAB"].data
            cols = set(hdul["SPTAB"].columns.names)
            targetids = np.asarray(table["TARGETID"], dtype=np.int64)
            matches = np.flatnonzero(targetids == int(targetid))
            if matches.size == 0:
                continue
            i = int(matches[0])
            record = {}
            for col in wanted:
                if col in cols:
                    record[col.lower()] = table[col][i]
            return record
    return None


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


def _select_target(rows, target_index: int, targetid: int | None):
    if targetid is not None:
        for row in rows:
            if int(row["targetid"]) == int(targetid):
                return row
        raise KeyError(f"TARGETID {targetid} is not in selected target table")
    return rows[target_index]


def _read_arms(path: Path, targetid: int):
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
            arms.append(
                {
                    "band": band,
                    "wave_obs": np.asarray(hdul[f"{band}_WAVELENGTH"].data, dtype="float64"),
                    "flux": np.asarray(hdul[f"{band}_FLUX"].data[row], dtype="float64"),
                    "ivar": np.asarray(hdul[f"{band}_IVAR"].data[row], dtype="float64"),
                    "mask": np.asarray(hdul[f"{band}_MASK"].data[row]),
                    "resolution": np.asarray(hdul[f"{band}_RESOLUTION"].data[row], dtype="float32"),
                }
            )
    return arms


def _prepare_arm(arm, rv_kms: float, grid_wave, continuum_window: int, error_floor: float):
    import numpy as np

    flux = np.asarray(arm["flux"], dtype=float)
    ivar = np.asarray(arm["ivar"], dtype=float)
    mask = np.asarray(arm["mask"])
    wave_obs = np.asarray(arm["wave_obs"], dtype=float)
    wave_rest = wave_obs / (1.0 + float(rv_kms) / C_KMS)
    finite = np.isfinite(wave_rest) & np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0) & (mask == 0)
    in_grid = (wave_rest >= float(grid_wave[0])) & (wave_rest <= float(grid_wave[-1]))
    valid = finite & in_grid
    fallback = float(np.nanmedian(flux[finite])) if np.any(finite) else 1.0
    filled_flux = flux.copy()
    filled_flux[~np.isfinite(filled_flux)] = fallback
    continuum = _running_median(filled_flux, continuum_window)
    norm_flux = filled_flux / continuum
    err = np.full_like(norm_flux, np.inf, dtype=float)
    err[finite] = 1.0 / np.sqrt(ivar[finite]) / continuum[finite]
    if error_floor > 0:
        err[finite] = np.sqrt(err[finite] ** 2 + error_floor**2)
    good = valid & np.isfinite(norm_flux) & np.isfinite(err) & (err > 0)
    return {
        "band": arm["band"],
        "wave_rest": wave_rest.astype("float32"),
        "flux": norm_flux.astype("float32"),
        "err": err.astype("float32"),
        "good": good.astype(bool),
        "resolution": arm["resolution"].astype("float32"),
    }


def _checkpoint_model(checkpoint_path: str | Path, device):
    import torch

    from kan_payne.payne_emulators import build_emulator

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(checkpoint["model_config"])
    model, _ = build_emulator(
        config["model"],
        n_labels=int(config["n_labels"]),
        n_pixels=int(config["n_pixels"]),
        hidden_sizes=tuple(config.get("hidden_sizes", (300, 300))),
        activation=config.get("activation", "leaky_relu"),
        d_model=int(config.get("d_model", 128)),
        n_label_tokens=int(config.get("n_label_tokens", 16)),
        n_heads=int(config.get("n_heads", 4)),
        n_layers=int(config.get("n_layers", 4)),
        dim_feedforward=int(config.get("dim_feedforward", 256)),
        wave_frequencies=int(config.get("wave_frequencies", 16)),
        dropout=float(config.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, config


def _label_names(checkpoint, n_labels: int):
    if "label_names" in checkpoint:
        return [str(item) for item in checkpoint["label_names"]]
    if n_labels == 4:
        return ["Teff", "logg", "M_H", "alpha_M"]
    return [f"label_{i}" for i in range(n_labels)]


def _initial_labels(sp, label_min, label_max, label_names):
    import numpy as np

    labels = 0.5 * (label_min + label_max)
    aliases = {
        "Teff": "teff",
        "logg": "logg",
        "M_H": "feh",
        "alpha_M": "alphafe",
    }
    for i, name in enumerate(label_names):
        key = aliases.get(name)
        if key is None or key not in sp:
            continue
        value = float(sp[key])
        if np.isfinite(value):
            labels[i] = value
    return np.clip(labels, label_min, label_max).astype("float32")


def _predict_full(model, model_name, label_scaled, checkpoint):
    if model_name == "transformer_payne":
        raise NotImplementedError("This resolution-matrix pilot currently expects a vector-output emulator.")
    prediction = model(label_scaled[None, :]).squeeze(0)
    norm = checkpoint.get("spectra_normalization", {"mode": "none"})
    if norm and norm.get("mode") == "minmax":
        import torch

        y_min = torch.as_tensor(norm["y_min"], dtype=prediction.dtype, device=prediction.device)
        y_max = torch.as_tensor(norm["y_max"], dtype=prediction.dtype, device=prediction.device)
        prediction = prediction * (y_max - y_min) + y_min
    return prediction


def _interp_uniform_torch(y_grid, x_grid, x_new):
    import torch

    x0 = float(x_grid[0])
    dx = float(x_grid[1] - x_grid[0])
    t = (x_new - x0) / dx
    i0 = torch.floor(t).to(torch.long)
    i0 = torch.clamp(i0, 0, y_grid.numel() - 2)
    frac = (t - i0.to(t.dtype)).clamp(0.0, 1.0)
    return y_grid[i0] * (1.0 - frac) + y_grid[i0 + 1] * frac


def _apply_desi_resolution_torch(model_flux, resolution):
    import numpy as np
    import torch

    data = torch.as_tensor(resolution, dtype=model_flux.dtype, device=model_flux.device)
    n = model_flux.numel()
    ndiag = data.shape[0]
    offsets = np.arange(ndiag // 2, -(ndiag // 2) - 1, -1, dtype=int)
    out = torch.zeros_like(model_flux)
    for k, offset in enumerate(offsets):
        if offset >= 0:
            out[: n - offset] = out[: n - offset] + data[k, offset:n] * model_flux[offset:n]
        else:
            jmax = n + offset
            out[-offset:n] = out[-offset:n] + data[k, :jmax] * model_flux[:jmax]
    return out


def _select_fit_indices(prepared_arms, fit_pixels: int, seed: int = 42):
    import numpy as np

    selections = []
    counts = [int(np.count_nonzero(arm["good"])) for arm in prepared_arms]
    total = sum(counts)
    rng = np.random.default_rng(seed)
    for arm, count in zip(prepared_arms, counts):
        good_idx = np.flatnonzero(arm["good"])
        if fit_pixels and total > fit_pixels and count > 0:
            n_arm = max(1, int(round(fit_pixels * count / total)))
            good_idx = np.sort(rng.choice(good_idx, size=min(n_arm, good_idx.size), replace=False))
        selections.append(good_idx.astype("int64"))
    return selections


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch

    from kan_payne.payne_data import load_payne_grid

    rows = _load_selected_targets(Path(args.selected_targets))
    target_row = _select_target(rows, args.target_index, args.targetid)
    sp = _load_sp_record([Path(item) for item in args.sp_catalog], int(target_row["targetid"]))
    if sp is None:
        raise KeyError(f"TARGETID {target_row['targetid']} has no SP catalog row")
    rv = float(sp.get("rv_adop", np.nan))
    if not np.isfinite(rv):
        raise ValueError("Target has no finite RV_ADOP")

    coadd = _coadd_path(
        Path(args.coadd_root),
        str(target_row["survey"]),
        str(target_row["program"]),
        int(target_row["healpix"]),
    )
    raw_arms = _read_arms(coadd, int(target_row["targetid"]))
    grid = load_payne_grid(args.grid)
    grid_wave = np.asarray(grid.wavelength, dtype="float32")
    arms = [
        _prepare_arm(arm, rv, grid_wave, args.continuum_window, args.error_floor)
        for arm in raw_arms
    ]
    fit_indices = _select_fit_indices(arms, args.fit_pixels)

    device = _choose_device(args.device)
    model, checkpoint, config = _checkpoint_model(args.checkpoint, device)
    model_name = config["model"]
    label_min = np.asarray(checkpoint["label_min"], dtype="float32")
    label_max = np.asarray(checkpoint["label_max"], dtype="float32")
    label_names = _label_names(checkpoint, label_min.size)
    init_labels = _initial_labels(sp, label_min, label_max, label_names)
    init_scaled = (init_labels - label_min) / (label_max - label_min) - 0.5

    raw = torch.tensor(init_scaled, dtype=torch.float32, device=device, requires_grad=True)
    label_min_t = torch.from_numpy(label_min).to(device)
    label_max_t = torch.from_numpy(label_max).to(device)
    grid_wave_t = torch.from_numpy(grid_wave).to(device)
    optimizer = torch.optim.Adam([raw], lr=args.lr)

    arm_tensors = []
    for arm, idx in zip(arms, fit_indices):
        idx_t = torch.from_numpy(idx).to(device)
        arm_tensors.append(
            {
                "band": arm["band"],
                "wave_rest": torch.from_numpy(arm["wave_rest"]).to(device),
                "flux": torch.from_numpy(arm["flux"]).to(device),
                "err": torch.from_numpy(arm["err"]).to(device),
                "idx": idx_t,
                "resolution": torch.from_numpy(arm["resolution"]).to(device),
            }
        )

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        scaled = torch.clamp(raw, -0.5, 0.5)
        pred_grid = _predict_full(model, model_name, scaled, checkpoint)
        losses = []
        for arm in arm_tensors:
            pred_native = _interp_uniform_torch(pred_grid, grid_wave_t, arm["wave_rest"])
            if not args.no_resolution:
                pred_native = _apply_desi_resolution_torch(pred_native, arm["resolution"])
            idx = arm["idx"]
            losses.append(torch.mean(((pred_native[idx] - arm["flux"][idx]) / arm["err"][idx]) ** 2))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        if args.progress_every and (step + 1) % args.progress_every == 0:
            print(f"step {step + 1}/{args.steps} loss={float(loss.detach().cpu()):.6g}", flush=True)

    residual_payload = {}
    with torch.no_grad():
        scaled = torch.clamp(raw, -0.5, 0.5)
        fitted_labels = ((scaled.detach().cpu().numpy() + 0.5) * (label_max - label_min) + label_min).astype(
            "float32"
        )
        pred_grid = _predict_full(model, model_name, scaled, checkpoint)
        chi_terms = []
        abs_terms = []
        npix = 0
        per_arm = {}
        for arm in arm_tensors:
            pred_native = _interp_uniform_torch(pred_grid, grid_wave_t, arm["wave_rest"])
            if not args.no_resolution:
                pred_native = _apply_desi_resolution_torch(pred_native, arm["resolution"])
            idx = arm["idx"]
            residual = pred_native[idx] - arm["flux"][idx]
            chi = torch.square(residual / arm["err"][idx]).detach().cpu().numpy()
            abs_res = torch.abs(residual).detach().cpu().numpy()
            chi_terms.append(chi)
            abs_terms.append(abs_res)
            npix += int(idx.numel())
            per_arm[arm["band"]] = {
                "npix": int(idx.numel()),
                "chi2": float(np.mean(chi)),
                "mae_x1e4": float(np.mean(abs_res) * 1.0e4),
            }
            residual_payload[f"{arm['band']}_wave_rest"] = arm["wave_rest"].detach().cpu().numpy()
            residual_payload[f"{arm['band']}_fit_index"] = idx.detach().cpu().numpy()
            residual_payload[f"{arm['band']}_model"] = pred_native.detach().cpu().numpy()
            residual_payload[f"{arm['band']}_flux"] = arm["flux"].detach().cpu().numpy()
            residual_payload[f"{arm['band']}_err"] = arm["err"].detach().cpu().numpy()

    chi_all = np.concatenate(chi_terms)
    abs_all = np.concatenate(abs_terms)
    result = {
        "target": target_row,
        "coadd": str(coadd),
        "model": model_name,
        "checkpoint": args.checkpoint,
        "use_resolution": not args.no_resolution,
        "rv_adop": rv,
        "rv_err": float(sp.get("rv_err", np.nan)),
        "snr_med": float(sp.get("snr_med", np.nan)),
        "sp_success": bool(sp.get("success", False)),
        "steps": args.steps,
        "lr": args.lr,
        "fit_pixels": int(npix),
        "label_names": label_names,
        "initial_labels": {name: float(value) for name, value in zip(label_names, init_labels)},
        "fitted_labels": {name: float(value) for name, value in zip(label_names, fitted_labels)},
        "fitted_label_scaled": {name: float(value) for name, value in zip(label_names, scaled.detach().cpu().numpy())},
        "chi2": float(np.mean(chi_all)),
        "mae_x1e4": float(np.mean(abs_all) * 1.0e4),
        "per_arm": per_arm,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.save_residuals:
        np.savez_compressed(args.save_residuals, **residual_payload)


if __name__ == "__main__":
    main()
