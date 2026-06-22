#!/usr/bin/env python
"""Fit labels for observed spectra with a trained Payne-style emulator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PAYNE_25_LABELS = (
    "Teff",
    "logg",
    "Vturb",
    "C_H",
    "N_H",
    "O_H",
    "Na_H",
    "Mg_H",
    "Al_H",
    "Si_H",
    "P_H",
    "S_H",
    "K_H",
    "Ca_H",
    "Ti_H",
    "V_H",
    "Cr_H",
    "Mn_H",
    "Fe_H",
    "Co_H",
    "Ni_H",
    "Cu_H",
    "Ge_H",
    "C12_C13",
    "Vmacro",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", required=True, help="Observed NPZ with wave/flux/err/mask arrays.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-stars", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--fit-pixels", type=int, default=2048)
    parser.add_argument("--eval-wave-chunk", type=int, default=1024)
    parser.add_argument("--init", choices=("observed", "midpoint"), default="observed")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def _choose_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _wave_scaled(wavelength):
    import numpy as np

    wave = np.asarray(wavelength, dtype="float32")
    return (2.0 * (wave - np.nanmin(wave)) / (np.nanmax(wave) - np.nanmin(wave)) - 1.0).astype(
        "float32"
    )


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
    if n_labels == len(PAYNE_25_LABELS):
        return list(PAYNE_25_LABELS)
    return [f"label_{i}" for i in range(n_labels)]


def _find_observed_label(observed, name: str, row: int):
    import numpy as np

    candidates = [
        f"label_{name}",
        f"label_{name.upper()}",
        f"label_{name.lower()}",
        f"label_{name.replace('[', '').replace(']', '').replace('/', '_')}",
    ]
    aliases = {
        "Teff": ["label_TEFF", "label_teff"],
        "logg": ["label_LOGG", "label_logg"],
        "M_H": ["label_M_H", "label_FEH", "label_FE_H"],
        "Fe_H": ["label_FE_H", "label_FEH", "label_M_H"],
        "alpha_M": ["label_alpha_M", "label_ALPHA_M", "label_ALPHAFE"],
    }
    candidates.extend(aliases.get(name, []))
    for key in candidates:
        if key in observed:
            value = float(observed[key][row])
            if np.isfinite(value):
                return value
    return None


def _label_initial_guess(observed, label_min, label_max, label_names, row: int, mode: str):
    import numpy as np

    labels = 0.5 * (label_min + label_max)
    if mode != "observed":
        return labels.astype("float32")
    for i, name in enumerate(label_names):
        value = _find_observed_label(observed, name, row)
        if value is not None:
            labels[i] = value
    return np.clip(labels, label_min, label_max).astype("float32")


def _predict(model, model_name, label_scaled, wave_scaled, pixel_idx, *, wave_chunk: int):
    import torch

    if model_name == "transformer_payne":
        if pixel_idx is None:
            rows = []
            for start in range(0, wave_scaled.numel(), wave_chunk):
                rows.append(model(label_scaled[None, :], wave_scaled[start : start + wave_chunk]).squeeze(0))
            return torch.cat(rows)
        return model(label_scaled[None, :], wave_scaled[pixel_idx]).squeeze(0)
    full = model(label_scaled[None, :]).squeeze(0)
    return full if pixel_idx is None else full[pixel_idx]


def _inverse_prediction(prediction, checkpoint, pixel_idx=None):
    norm = checkpoint.get("spectra_normalization", {"mode": "none"})
    if not norm or norm.get("mode") != "minmax":
        return prediction
    import torch

    y_min = torch.as_tensor(norm["y_min"], dtype=prediction.dtype, device=prediction.device)
    y_max = torch.as_tensor(norm["y_max"], dtype=prediction.dtype, device=prediction.device)
    if pixel_idx is not None:
        y_min = y_min[pixel_idx]
        y_max = y_max[pixel_idx]
    return prediction * (y_max - y_min) + y_min


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch

    device = _choose_device(args.device)
    model, checkpoint, config = _checkpoint_model(args.checkpoint, device)
    model_name = config["model"]
    label_min = np.asarray(checkpoint["label_min"], dtype="float32")
    label_max = np.asarray(checkpoint["label_max"], dtype="float32")
    label_names = _label_names(checkpoint, label_min.size)

    with np.load(args.observed, allow_pickle=True) as payload:
        observed = {key: payload[key] for key in payload.files}
    wave = np.asarray(observed["wave"], dtype="float32")
    flux = np.asarray(observed["flux"], dtype="float32")
    err = np.asarray(observed["err"], dtype="float32")
    mask = np.asarray(observed["mask"], dtype=bool)
    n_rows = flux.shape[0]
    stop = n_rows if args.max_stars is None else min(n_rows, args.start + args.max_stars)
    rows = np.arange(args.start, stop, dtype=np.int64)

    label_min_t = torch.from_numpy(label_min).to(device)
    label_max_t = torch.from_numpy(label_max).to(device)
    wave_t = torch.from_numpy(_wave_scaled(wave)).to(device)
    rng = np.random.default_rng(42)

    fitted_scaled = np.empty((rows.size, label_min.size), dtype="float32")
    fitted_labels = np.empty_like(fitted_scaled)
    chi2 = np.empty(rows.size, dtype="float32")
    npix = np.empty(rows.size, dtype="int32")
    mae = np.empty(rows.size, dtype="float32")
    success = np.zeros(rows.size, dtype=bool)

    for out_index, row in enumerate(rows):
        good = np.isfinite(flux[row]) & np.isfinite(err[row]) & (err[row] > 0) & ~mask[row]
        good_idx = np.where(good)[0]
        if good_idx.size == 0:
            fitted_scaled[out_index] = np.nan
            fitted_labels[out_index] = np.nan
            chi2[out_index] = np.nan
            mae[out_index] = np.nan
            npix[out_index] = 0
            continue
        if args.fit_pixels and good_idx.size > args.fit_pixels:
            pixel_idx_np = np.sort(rng.choice(good_idx, size=args.fit_pixels, replace=False))
        else:
            pixel_idx_np = good_idx

        init_labels = _label_initial_guess(observed, label_min, label_max, label_names, int(row), args.init)
        init_scaled = (init_labels - label_min) / (label_max - label_min) - 0.5
        raw = torch.tensor(init_scaled, dtype=torch.float32, device=device, requires_grad=True)
        pixel_idx = torch.from_numpy(pixel_idx_np.astype("int64")).to(device)
        target = torch.from_numpy(flux[row, pixel_idx_np]).to(device)
        sigma = torch.from_numpy(err[row, pixel_idx_np]).to(device)
        optimizer = torch.optim.Adam([raw], lr=args.lr)

        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            scaled = torch.clamp(raw, -0.5, 0.5)
            pred = _predict(model, model_name, scaled, wave_t, pixel_idx, wave_chunk=args.eval_wave_chunk)
            pred = _inverse_prediction(pred, checkpoint, pixel_idx=pixel_idx)
            loss = torch.mean(((pred - target) / sigma) ** 2)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            scaled = torch.clamp(raw, -0.5, 0.5)
            pred_full = _predict(model, model_name, scaled, wave_t, None, wave_chunk=args.eval_wave_chunk)
            pred_full = _inverse_prediction(pred_full, checkpoint, pixel_idx=None)
            good_target = torch.from_numpy(flux[row, good_idx]).to(device)
            good_sigma = torch.from_numpy(err[row, good_idx]).to(device)
            good_pred = pred_full[torch.from_numpy(good_idx.astype("int64")).to(device)]
            residual = good_pred - good_target
            fitted_scaled[out_index] = scaled.detach().cpu().numpy()
            fitted_labels[out_index] = (
                (fitted_scaled[out_index] + 0.5) * (label_max - label_min) + label_min
            ).astype("float32")
            chi2[out_index] = float(torch.mean((residual / good_sigma) ** 2).detach().cpu())
            mae[out_index] = float(torch.mean(torch.abs(residual)).detach().cpu() * 1.0e4)
            npix[out_index] = int(good_idx.size)
            success[out_index] = True

        if args.progress_every and (out_index + 1) % args.progress_every == 0:
            print(
                f"Fitted {out_index + 1}/{rows.size} spectra; "
                f"last chi2={chi2[out_index]:.4f}, mae_x1e4={mae[out_index]:.4f}",
                flush=True,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "row_index": rows,
        "label_names": np.asarray(label_names),
        "fitted_labels": fitted_labels,
        "fitted_label_scaled": fitted_scaled,
        "chi2": chi2,
        "mae_x1e4": mae,
        "npix": npix,
        "success": success,
        "checkpoint": np.asarray(str(args.checkpoint)),
        "model": np.asarray(model_name),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "observed": args.observed,
                    "checkpoint": args.checkpoint,
                    "steps": args.steps,
                    "lr": args.lr,
                    "fit_pixels": args.fit_pixels,
                    "init": args.init,
                    "label_names": label_names,
                },
                sort_keys=True,
            )
        ),
    }
    for key, value in observed.items():
        if np.asarray(value).shape[:1] == (n_rows,) and key not in {"flux", "err", "mask"}:
            payload[f"observed_{key}"] = value[rows]
    np.savez_compressed(output, **payload)
    print(f"Wrote fitted labels for {rows.size} spectra to {output}")


if __name__ == "__main__":
    main()
