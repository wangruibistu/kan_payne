"""Data utilities for The Payne APOGEE/Kurucz synthetic training set.

This module keeps the shared emulator dataset independent of PyTorch so data
preparation, validation, and pretrained Payne-MLP reproduction can run in a
minimal NumPy environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from urllib.request import urlretrieve

import numpy as np


THE_PAYNE_GITHUB_RAW = "https://raw.githubusercontent.com/tingyuansen/The_Payne/master"

THE_PAYNE_FILES: Mapping[str, tuple[str, str]] = {
    "training": (
        "other_data/kurucz_training_spectra.npz",
        f"{THE_PAYNE_GITHUB_RAW}/other_data/kurucz_training_spectra.npz",
    ),
    "wavelength": (
        "other_data/apogee_wavelength.npz",
        f"{THE_PAYNE_GITHUB_RAW}/other_data/apogee_wavelength.npz",
    ),
    "mask": (
        "other_data/apogee_mask.npz",
        f"{THE_PAYNE_GITHUB_RAW}/other_data/apogee_mask.npz",
    ),
    "pretrained_mlp": (
        "neural_nets/NN_normalized_spectra.npz",
        f"{THE_PAYNE_GITHUB_RAW}/neural_nets/NN_normalized_spectra.npz",
    ),
}

PAYNE_LABEL_NAMES: tuple[str, ...] = (
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


@dataclass(frozen=True)
class PayneGrid:
    """In-memory representation of the unified Payne training grid."""

    wavelength: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    spectra: np.ndarray
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    label_min: np.ndarray
    label_max: np.ndarray
    label_scaled: np.ndarray
    label_names: tuple[str, ...] = PAYNE_LABEL_NAMES
    metadata: Mapping[str, object] | None = None

    @property
    def good_pixel_mask(self) -> np.ndarray:
        """Return True where pixels should be used in masked metrics/losses."""

        return ~np.asarray(self.mask, dtype=bool)


def path_for_the_payne_file(root: str | Path, key: str) -> Path:
    """Return the expected local path for one The Payne data file."""

    if key not in THE_PAYNE_FILES:
        raise KeyError(f"Unknown The Payne file key: {key}")
    relative_path, _ = THE_PAYNE_FILES[key]
    return Path(root) / relative_path


def ensure_the_payne_files(
    root: str | Path,
    *,
    keys: Sequence[str] = ("training", "wavelength", "mask", "pretrained_mlp"),
    download_missing: bool = False,
) -> dict[str, Path]:
    """Check or download the small The Payne files used by this workflow."""

    root = Path(root)
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for key in keys:
        path = path_for_the_payne_file(root, key)
        if not path.exists():
            if download_missing:
                _, url = THE_PAYNE_FILES[key]
                path.parent.mkdir(parents=True, exist_ok=True)
                urlretrieve(url, path)
            else:
                missing.append(f"{key}: {path}")
        paths[key] = path

    if missing:
        formatted = "\n".join(missing)
        raise FileNotFoundError(
            "Missing The Payne data files. Re-run with --download-missing or "
            f"place them under {root}:\n{formatted}"
        )
    return paths


def _load_npz_array(path: str | Path, key: str) -> np.ndarray:
    with np.load(path) as payload:
        if key not in payload:
            raise KeyError(f"{path} does not contain array {key!r}")
        return np.asarray(payload[key])


def load_the_payne_raw(
    *,
    training_npz: str | Path,
    wavelength_npz: str | Path,
    mask_npz: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load The Payne raw arrays as labels(N,25), spectra(N,P), wave(P), mask(P)."""

    with np.load(training_npz) as payload:
        if "labels" not in payload or "spectra" not in payload:
            raise KeyError(f"{training_npz} must contain 'labels' and 'spectra'")
        labels = np.asarray(payload["labels"], dtype=np.float32)
        spectra = np.asarray(payload["spectra"], dtype=np.float32)

    if labels.ndim != 2 or spectra.ndim != 2:
        raise ValueError("Expected labels and spectra to be two-dimensional arrays")
    if labels.shape[0] == len(PAYNE_LABEL_NAMES):
        labels = labels.T
    if labels.shape[1] != len(PAYNE_LABEL_NAMES):
        raise ValueError(
            f"Expected {len(PAYNE_LABEL_NAMES)} labels, got shape {labels.shape}"
        )

    wavelength = _load_npz_array(wavelength_npz, "wavelength").astype(np.float64)
    mask = _load_npz_array(mask_npz, "apogee_mask").astype(bool)

    if spectra.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Spectrum/label row mismatch: {spectra.shape[0]} vs {labels.shape[0]}"
        )
    if spectra.shape[1] != wavelength.shape[0] or mask.shape[0] != wavelength.shape[0]:
        raise ValueError(
            "Wavelength/mask length does not match spectral pixel count: "
            f"spectra={spectra.shape}, wavelength={wavelength.shape}, mask={mask.shape}"
        )
    return labels, spectra, wavelength, mask


def official_payne_split(n_rows: int, train_size: int = 800) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return The Payne repository split: first 800 train, remaining validation."""

    if train_size <= 0 or train_size >= n_rows:
        raise ValueError(f"train_size must be in [1, {n_rows - 1}], got {train_size}")
    train_idx = np.arange(train_size, dtype=np.int64)
    valid_idx = np.arange(train_size, n_rows, dtype=np.int64)
    test_idx = np.empty(0, dtype=np.int64)
    return train_idx, valid_idx, test_idx


def random_split(
    n_rows: int,
    *,
    train_fraction: float = 0.8,
    valid_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a deterministic random train/valid/test split."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= valid_fraction < 1.0:
        raise ValueError("valid_fraction must be in [0, 1)")
    if train_fraction + valid_fraction > 1.0:
        raise ValueError("train_fraction + valid_fraction must be <= 1")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows).astype(np.int64)
    train_end = int(round(n_rows * train_fraction))
    valid_end = train_end + int(round(n_rows * valid_fraction))
    return order[:train_end], order[train_end:valid_end], order[valid_end:]


def label_scaler(labels: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Payne-style min/max scaling from the training labels."""

    train_labels = np.asarray(labels, dtype=np.float32)[train_idx]
    label_min = np.nanmin(train_labels, axis=0).astype(np.float32)
    label_max = np.nanmax(train_labels, axis=0).astype(np.float32)
    span = label_max - label_min
    if np.any(span <= 0.0) or not np.all(np.isfinite(span)):
        bad = np.where((span <= 0.0) | ~np.isfinite(span))[0]
        raise ValueError(f"Invalid label scaling span at columns {bad.tolist()}")
    return label_min, label_max


def scale_labels(
    labels: np.ndarray,
    label_min: np.ndarray,
    label_max: np.ndarray,
    *,
    center: bool = True,
) -> np.ndarray:
    """Scale labels to [0,1] or Payne's centered [-0.5,0.5] convention."""

    scaled = (np.asarray(labels, dtype=np.float32) - label_min) / (label_max - label_min)
    if center:
        scaled = scaled - np.float32(0.5)
    return scaled.astype(np.float32)


def build_unified_payne_grid(
    *,
    training_npz: str | Path,
    wavelength_npz: str | Path,
    mask_npz: str | Path,
    split: str = "official",
    train_size: int = 800,
    train_fraction: float = 0.8,
    valid_fraction: float = 0.2,
    random_seed: int = 42,
) -> PayneGrid:
    """Build the unified grid used by all emulator comparisons."""

    labels, spectra, wavelength, mask = load_the_payne_raw(
        training_npz=training_npz,
        wavelength_npz=wavelength_npz,
        mask_npz=mask_npz,
    )
    if split == "official":
        train_idx, valid_idx, test_idx = official_payne_split(labels.shape[0], train_size)
    elif split == "random":
        train_idx, valid_idx, test_idx = random_split(
            labels.shape[0],
            train_fraction=train_fraction,
            valid_fraction=valid_fraction,
            seed=random_seed,
        )
    else:
        raise ValueError(f"Unsupported split: {split}")

    label_min, label_max = label_scaler(labels, train_idx)
    label_scaled = scale_labels(labels, label_min, label_max)
    metadata = {
        "source": "The Payne GitHub Kurucz/APOGEE training set",
        "split": split,
        "train_size": int(train_idx.size),
        "valid_size": int(valid_idx.size),
        "test_size": int(test_idx.size),
        "label_scaling": "scaled = (label - train_min) / (train_max - train_min) - 0.5",
        "mask_semantics": "True means omitted/bad Payne APOGEE pixel",
    }
    return PayneGrid(
        wavelength=wavelength,
        mask=mask,
        labels=labels.astype(np.float32),
        spectra=spectra.astype(np.float32),
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        label_min=label_min,
        label_max=label_max,
        label_scaled=label_scaled,
        metadata=metadata,
    )


def save_payne_grid(grid: PayneGrid, output_path: str | Path) -> Path:
    """Persist a unified Payne grid as a compressed NPZ."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(grid.metadata or {})
    metadata["n_spectra"] = int(grid.spectra.shape[0])
    metadata["n_pixels"] = int(grid.spectra.shape[1])
    np.savez_compressed(
        output_path,
        wavelength=np.asarray(grid.wavelength, dtype=np.float64),
        mask=np.asarray(grid.mask, dtype=bool),
        labels=np.asarray(grid.labels, dtype=np.float32),
        label_scaled=np.asarray(grid.label_scaled, dtype=np.float32),
        spectra=np.asarray(grid.spectra, dtype=np.float32),
        train_idx=np.asarray(grid.train_idx, dtype=np.int64),
        valid_idx=np.asarray(grid.valid_idx, dtype=np.int64),
        test_idx=np.asarray(grid.test_idx, dtype=np.int64),
        label_min=np.asarray(grid.label_min, dtype=np.float32),
        label_max=np.asarray(grid.label_max, dtype=np.float32),
        label_names=np.asarray(grid.label_names),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return output_path


def load_payne_grid(path: str | Path) -> PayneGrid:
    """Load a unified Payne grid produced by :func:`save_payne_grid`."""

    with np.load(path) as payload:
        metadata: Mapping[str, object] | None = None
        if "metadata_json" in payload:
            metadata = json.loads(str(payload["metadata_json"]))
        return PayneGrid(
            wavelength=np.asarray(payload["wavelength"], dtype=np.float64),
            mask=np.asarray(payload["mask"], dtype=bool),
            labels=np.asarray(payload["labels"], dtype=np.float32),
            spectra=np.asarray(payload["spectra"], dtype=np.float32),
            train_idx=np.asarray(payload["train_idx"], dtype=np.int64),
            valid_idx=np.asarray(payload["valid_idx"], dtype=np.int64),
            test_idx=np.asarray(payload.get("test_idx", np.empty(0)), dtype=np.int64),
            label_min=np.asarray(payload["label_min"], dtype=np.float32),
            label_max=np.asarray(payload["label_max"], dtype=np.float32),
            label_scaled=np.asarray(payload["label_scaled"], dtype=np.float32),
            label_names=tuple(str(item) for item in payload["label_names"]),
            metadata=metadata,
        )


def grid_summary(grid: PayneGrid) -> dict[str, object]:
    """Return a compact JSON-serializable summary for logs and CLI output."""

    return {
        "n_spectra": int(grid.spectra.shape[0]),
        "n_pixels": int(grid.spectra.shape[1]),
        "n_labels": int(grid.labels.shape[1]),
        "wavelength_min": float(np.nanmin(grid.wavelength)),
        "wavelength_max": float(np.nanmax(grid.wavelength)),
        "bad_pixel_fraction": float(np.mean(grid.mask)),
        "train_size": int(grid.train_idx.size),
        "valid_size": int(grid.valid_idx.size),
        "test_size": int(grid.test_idx.size),
        "spectra_min": float(np.nanmin(grid.spectra)),
        "spectra_max": float(np.nanmax(grid.spectra)),
        "label_names": list(grid.label_names),
    }


def load_pretrained_payne_mlp(path: str | Path) -> tuple[np.ndarray, ...]:
    """Load The Payne repository MLP coefficient NPZ."""

    with np.load(path) as payload:
        keys = (
            "w_array_0",
            "w_array_1",
            "w_array_2",
            "b_array_0",
            "b_array_1",
            "b_array_2",
            "x_min",
            "x_max",
        )
        missing = [key for key in keys if key not in payload]
        if missing:
            raise KeyError(f"{path} missing arrays: {missing}")
        return tuple(np.asarray(payload[key], dtype=np.float32) for key in keys)


def leaky_relu_numpy(values: np.ndarray, negative_slope: float = 0.01) -> np.ndarray:
    return np.where(values > 0.0, values, negative_slope * values)


def predict_pretrained_payne_mlp(labels: np.ndarray, coeffs: tuple[np.ndarray, ...]) -> np.ndarray:
    """Predict spectra with The Payne repository MLP coefficients."""

    w0, w1, w2, b0, b1, b2, x_min, x_max = coeffs
    scaled = (np.asarray(labels, dtype=np.float32) - x_min) / (x_max - x_min) - 0.5
    inside = scaled @ w0.T + b0
    outside = leaky_relu_numpy(inside) @ w1.T + b1
    spectra = leaky_relu_numpy(outside) @ w2.T + b2
    return np.asarray(spectra, dtype=np.float32)


def regression_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    good_pixel_mask: np.ndarray | None = None,
    scale: float = 1.0e4,
) -> dict[str, float]:
    """Compute MAE/RMSE metrics, optionally over good pixels only."""

    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if predicted.shape != target.shape:
        raise ValueError(f"Shape mismatch: predicted={predicted.shape}, target={target.shape}")

    if good_pixel_mask is not None:
        good_pixel_mask = np.asarray(good_pixel_mask, dtype=bool)
        residual = predicted[:, good_pixel_mask] - target[:, good_pixel_mask]
    else:
        residual = predicted - target
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n_values": 0.0}
    return {
        "mae": float(np.mean(np.abs(residual)) * scale),
        "rmse": float(np.sqrt(np.mean(residual**2)) * scale),
        "n_values": float(residual.size),
    }
