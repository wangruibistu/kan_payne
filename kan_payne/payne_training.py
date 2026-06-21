"""Training loops for Payne-MLP, KAN-Payne, and TransformerPayne emulators."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kan_payne.payne_data import PayneGrid, regression_metrics
from kan_payne.payne_emulators import build_emulator


def parse_hidden_sizes(value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse hidden layer sizes from CLI text or a sequence."""

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return ()
        return tuple(int(item.strip()) for item in cleaned.split(",") if item.strip())
    return tuple(int(item) for item in value)


def choose_device(device_name: str):
    """Choose a torch device lazily."""

    import torch

    requested = device_name.lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_random_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalise_spectra(
    grid: PayneGrid,
    *,
    normalization: str,
) -> tuple[np.ndarray, dict[str, np.ndarray | str]]:
    spectra = np.asarray(grid.spectra, dtype=np.float32)
    if normalization == "none":
        return spectra, {"mode": "none"}
    if normalization != "minmax":
        raise ValueError(f"Unsupported spectra normalization: {normalization}")

    train = spectra[grid.train_idx]
    y_min = np.nanmin(train, axis=0).astype(np.float32)
    y_max = np.nanmax(train, axis=0).astype(np.float32)
    span = y_max - y_min
    span[span <= 0.0] = 1.0
    scaled = (spectra - y_min) / span
    return scaled.astype(np.float32), {"mode": "minmax", "y_min": y_min, "y_max": y_max}


def _inverse_spectra(values: np.ndarray, norm: Mapping[str, Any]) -> np.ndarray:
    if norm.get("mode") == "minmax":
        return values * (norm["y_max"] - norm["y_min"]) + norm["y_min"]
    return values


def _masked_l1(predicted, target, good_pixels):
    import torch

    residual = torch.abs(predicted - target)
    if good_pixels is not None:
        residual = residual[:, good_pixels]
    return torch.mean(residual)


def _checkpoint_payload(
    *,
    model,
    model_config,
    optimizer,
    epoch: int,
    history: list[dict[str, float]],
    normalization: Mapping[str, Any],
    label_min: np.ndarray,
    label_max: np.ndarray,
    label_names: Sequence[str],
    wavelength: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "history": history,
        "spectra_normalization": {
            key: value for key, value in normalization.items() if isinstance(value, str)
        },
        "label_min": np.asarray(label_min, dtype=np.float32),
        "label_max": np.asarray(label_max, dtype=np.float32),
        "label_names": np.asarray(tuple(label_names)),
        "wavelength": np.asarray(wavelength, dtype=np.float64),
        "mask": np.asarray(mask, dtype=bool),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _save_checkpoint(path: str | Path, payload: Mapping[str, Any], normalization: Mapping[str, Any]) -> Path:
    import torch

    checkpoint = dict(payload)
    if normalization.get("mode") == "minmax":
        checkpoint["spectra_normalization"] = {
            "mode": "minmax",
            "y_min": np.asarray(normalization["y_min"], dtype=np.float32),
            "y_max": np.asarray(normalization["y_max"], dtype=np.float32),
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def _write_history(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _predict_vector_model(model, labels, *, batch_size: int, device: Any) -> np.ndarray:
    import torch

    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, labels.shape[0], batch_size):
            batch = torch.from_numpy(labels[start : start + batch_size]).to(device=device)
            rows.append(model(batch).detach().cpu().numpy().astype(np.float32))
    return np.vstack(rows)


def train_vector_emulator(
    grid: PayneGrid,
    *,
    model_name: str,
    output_dir: str | Path,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1.0e-4,
    weight_decay: float = 0.0,
    hidden_sizes: Sequence[int] = (300, 300),
    activation: str = "leaky_relu",
    spectra_normalization: str = "none",
    use_mask: bool = False,
    device_name: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    """Train a vector-output Payne-MLP or KAN-Payne model."""

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    set_random_seed(seed)
    device = choose_device(device_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets_np, norm = _normalise_spectra(grid, normalization=spectra_normalization)
    model, model_config = build_emulator(
        model_name,
        n_labels=grid.labels.shape[1],
        n_pixels=grid.spectra.shape[1],
        hidden_sizes=hidden_sizes,
        activation=activation,
    )
    model.to(device)

    labels = torch.from_numpy(grid.label_scaled.astype(np.float32))
    targets = torch.from_numpy(targets_np.astype(np.float32))
    train_dataset = TensorDataset(labels[grid.train_idx], targets[grid.train_idx])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    good_pixels = None
    if use_mask:
        good_pixels = torch.from_numpy(grid.good_pixel_mask).to(device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    best_valid_mae = float("inf")
    best_checkpoint = output_dir / f"{model_name}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_labels, batch_targets in train_loader:
            batch_labels = batch_labels.to(device=device)
            batch_targets = batch_targets.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(batch_labels)
            loss = _masked_l1(predicted, batch_targets, good_pixels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()) * 1.0e4)

        valid_labels = grid.label_scaled[grid.valid_idx]
        predicted_valid = _predict_vector_model(
            model,
            valid_labels.astype(np.float32),
            batch_size=batch_size,
            device=device,
        )
        predicted_valid = _inverse_spectra(predicted_valid, norm)
        valid_target = grid.spectra[grid.valid_idx]
        full_metrics = regression_metrics(predicted_valid, valid_target)
        good_metrics = regression_metrics(
            predicted_valid,
            valid_target,
            good_pixel_mask=grid.good_pixel_mask,
        )
        row = {
            "epoch": float(epoch),
            "train_l1_x1e4": float(np.mean(train_losses)),
            "valid_mae_x1e4": full_metrics["mae"],
            "valid_rmse_x1e4": full_metrics["rmse"],
            "valid_good_mae_x1e4": good_metrics["mae"],
            "valid_good_rmse_x1e4": good_metrics["rmse"],
        }
        history.append(row)
        if good_metrics["mae"] < best_valid_mae:
            best_valid_mae = good_metrics["mae"]
            payload = _checkpoint_payload(
                model=model,
                model_config=model_config,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                normalization=norm,
                label_min=grid.label_min,
                label_max=grid.label_max,
                label_names=grid.label_names,
                wavelength=grid.wavelength,
                mask=grid.mask,
            )
            _save_checkpoint(best_checkpoint, payload, norm)

        print(
            f"Epoch {epoch:04d} train={row['train_l1_x1e4']:.4f} "
            f"valid_good_mae={row['valid_good_mae_x1e4']:.4f} "
            f"valid_good_rmse={row['valid_good_rmse_x1e4']:.4f}"
        )

    final_checkpoint = output_dir / f"{model_name}_final.pt"
    payload = _checkpoint_payload(
        model=model,
        model_config=model_config,
        optimizer=optimizer,
        epoch=epochs,
        history=history,
        normalization=norm,
        label_min=grid.label_min,
        label_max=grid.label_max,
        label_names=grid.label_names,
        wavelength=grid.wavelength,
        mask=grid.mask,
    )
    _save_checkpoint(final_checkpoint, payload, norm)

    summary = {
        "model": model_name,
        "device": str(device),
        "epochs": epochs,
        "best_checkpoint": str(best_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "history": history,
        "spectra_normalization": spectra_normalization,
        "use_mask": use_mask,
    }
    _write_history(output_dir / f"{model_name}_history.json", summary)
    return summary


def _wave_scaled(wavelength: np.ndarray) -> np.ndarray:
    wave = np.asarray(wavelength, dtype=np.float32)
    return (2.0 * (wave - np.nanmin(wave)) / (np.nanmax(wave) - np.nanmin(wave)) - 1.0).astype(
        np.float32
    )


def _evaluate_transformer(
    model,
    grid: PayneGrid,
    targets_np: np.ndarray,
    norm: Mapping[str, Any],
    *,
    batch_size: int,
    wave_chunk: int,
    device: Any,
) -> dict[str, float]:
    import torch

    labels_np = grid.label_scaled[grid.valid_idx].astype(np.float32)
    targets_flux = grid.spectra[grid.valid_idx].astype(np.float32)
    wave_np = _wave_scaled(grid.wavelength)
    good = grid.good_pixel_mask

    abs_sum = 0.0
    sq_sum = 0.0
    count = 0
    abs_good_sum = 0.0
    sq_good_sum = 0.0
    good_count = 0

    model.eval()
    with torch.no_grad():
        for row_start in range(0, labels_np.shape[0], batch_size):
            row_slice = slice(row_start, row_start + batch_size)
            label_batch = torch.from_numpy(labels_np[row_slice]).to(device=device)
            for pix_start in range(0, wave_np.shape[0], wave_chunk):
                pix_slice = slice(pix_start, pix_start + wave_chunk)
                wave_batch = torch.from_numpy(wave_np[pix_slice]).to(device=device)
                predicted = model(label_batch, wave_batch)
                predicted_np = predicted.detach().cpu().numpy().astype(np.float32)
                predicted_flux = _inverse_spectra(predicted_np, {
                    key: value[pix_slice] if isinstance(value, np.ndarray) else value
                    for key, value in norm.items()
                })
                target = targets_flux[row_slice, pix_slice]
                residual = predicted_flux - target
                finite = np.isfinite(residual)
                abs_sum += float(np.sum(np.abs(residual[finite])))
                sq_sum += float(np.sum(residual[finite] ** 2))
                count += int(np.sum(finite))

                good_chunk = good[pix_slice]
                if np.any(good_chunk):
                    good_residual = residual[:, good_chunk]
                    good_finite = np.isfinite(good_residual)
                    abs_good_sum += float(np.sum(np.abs(good_residual[good_finite])))
                    sq_good_sum += float(np.sum(good_residual[good_finite] ** 2))
                    good_count += int(np.sum(good_finite))

    scale = 1.0e4
    return {
        "valid_mae_x1e4": abs_sum / max(count, 1) * scale,
        "valid_rmse_x1e4": float(np.sqrt(sq_sum / max(count, 1)) * scale),
        "valid_good_mae_x1e4": abs_good_sum / max(good_count, 1) * scale,
        "valid_good_rmse_x1e4": float(np.sqrt(sq_good_sum / max(good_count, 1)) * scale),
    }


def train_transformer_payne(
    grid: PayneGrid,
    *,
    output_dir: str | Path,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1.0e-4,
    weight_decay: float = 0.0,
    spectra_normalization: str = "none",
    use_mask: bool = False,
    device_name: str = "auto",
    seed: int = 42,
    d_model: int = 128,
    n_label_tokens: int = 16,
    n_heads: int = 4,
    n_layers: int = 4,
    dim_feedforward: int = 256,
    wave_frequencies: int = 16,
    dropout: float = 0.0,
    pixels_per_spectrum: int = 256,
    eval_wave_chunk: int = 512,
) -> dict[str, Any]:
    """Train the wavelength-conditioned TransformerPayne-style emulator."""

    import torch

    set_random_seed(seed)
    device = choose_device(device_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets_np, norm = _normalise_spectra(grid, normalization=spectra_normalization)
    model, model_config = build_emulator(
        "transformer_payne",
        n_labels=grid.labels.shape[1],
        n_pixels=grid.spectra.shape[1],
        d_model=d_model,
        n_label_tokens=n_label_tokens,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        wave_frequencies=wave_frequencies,
        dropout=dropout,
    )
    model.to(device)

    labels = torch.from_numpy(grid.label_scaled.astype(np.float32)).to(device=device)
    targets = torch.from_numpy(targets_np.astype(np.float32)).to(device=device)
    wave = torch.from_numpy(_wave_scaled(grid.wavelength)).to(device=device)
    pixel_pool_np = np.where(grid.good_pixel_mask if use_mask else np.ones_like(grid.mask, dtype=bool))[0]
    pixel_pool = torch.from_numpy(pixel_pool_np.astype(np.int64)).to(device=device)
    train_idx = torch.from_numpy(grid.train_idx.astype(np.int64)).to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    best_valid_mae = float("inf")
    best_checkpoint = output_dir / "transformer_payne_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        order = train_idx[torch.randperm(train_idx.numel(), device=device)]
        train_losses: list[float] = []
        for start in range(0, order.numel(), batch_size):
            row_idx = order[start : start + batch_size]
            label_batch = labels[row_idx]
            sample = torch.randint(
                0,
                pixel_pool.numel(),
                (row_idx.numel(), pixels_per_spectrum),
                device=device,
            )
            pixel_idx = pixel_pool[sample]
            wave_batch = wave[pixel_idx]
            target_batch = targets[row_idx[:, None], pixel_idx]

            optimizer.zero_grad(set_to_none=True)
            predicted = model(label_batch, wave_batch)
            loss = torch.mean(torch.abs(predicted - target_batch))
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()) * 1.0e4)

        metrics = _evaluate_transformer(
            model,
            grid,
            targets_np,
            norm,
            batch_size=batch_size,
            wave_chunk=eval_wave_chunk,
            device=device,
        )
        row = {
            "epoch": float(epoch),
            "train_l1_x1e4": float(np.mean(train_losses)),
            **metrics,
        }
        history.append(row)
        if metrics["valid_good_mae_x1e4"] < best_valid_mae:
            best_valid_mae = metrics["valid_good_mae_x1e4"]
            payload = _checkpoint_payload(
                model=model,
                model_config=model_config,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                normalization=norm,
                label_min=grid.label_min,
                label_max=grid.label_max,
                label_names=grid.label_names,
                wavelength=grid.wavelength,
                mask=grid.mask,
            )
            _save_checkpoint(best_checkpoint, payload, norm)

        print(
            f"Epoch {epoch:04d} train={row['train_l1_x1e4']:.4f} "
            f"valid_good_mae={row['valid_good_mae_x1e4']:.4f} "
            f"valid_good_rmse={row['valid_good_rmse_x1e4']:.4f}"
        )

    final_checkpoint = output_dir / "transformer_payne_final.pt"
    payload = _checkpoint_payload(
        model=model,
        model_config=model_config,
        optimizer=optimizer,
        epoch=epochs,
        history=history,
        normalization=norm,
        label_min=grid.label_min,
        label_max=grid.label_max,
        label_names=grid.label_names,
        wavelength=grid.wavelength,
        mask=grid.mask,
    )
    _save_checkpoint(final_checkpoint, payload, norm)

    summary = {
        "model": "transformer_payne",
        "device": str(device),
        "epochs": epochs,
        "best_checkpoint": str(best_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "history": history,
        "spectra_normalization": spectra_normalization,
        "use_mask": use_mask,
        "pixels_per_spectrum": pixels_per_spectrum,
    }
    _write_history(output_dir / "transformer_payne_history.json", summary)
    return summary


def train_emulator(
    grid: PayneGrid,
    *,
    model_name: str,
    output_dir: str | Path,
    **kwargs,
) -> dict[str, Any]:
    """Dispatch to the right training loop."""

    if model_name == "transformer_payne":
        return train_transformer_payne(grid, output_dir=output_dir, **kwargs)
    return train_vector_emulator(grid, model_name=model_name, output_dir=output_dir, **kwargs)
