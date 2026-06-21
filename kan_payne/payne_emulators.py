"""PyTorch emulator definitions for Payne-style spectral models.

The module intentionally imports PyTorch inside builder functions. This lets
data-preparation and NumPy reproduction scripts run even when a local PyTorch
installation is unavailable or slow to initialize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EmulatorConfig:
    """Serializable model configuration used in checkpoints."""

    model: str
    n_labels: int
    n_pixels: int
    hidden_sizes: tuple[int, ...] = (300, 300)
    activation: str = "leaky_relu"
    d_model: int = 128
    n_label_tokens: int = 16
    n_heads: int = 4
    n_layers: int = 4
    dim_feedforward: int = 256
    wave_frequencies: int = 16
    dropout: float = 0.0


def _activation_module(name: str):
    import torch.nn as nn

    normalized = name.lower()
    if normalized == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


def build_payne_mlp(config: EmulatorConfig):
    """Build the original Payne-style vector-output MLP."""

    import torch.nn as nn

    layers: list[nn.Module] = []
    previous = config.n_labels
    for hidden_size in config.hidden_sizes:
        layers.append(nn.Linear(previous, hidden_size))
        layers.append(_activation_module(config.activation))
        previous = hidden_size
    layers.append(nn.Linear(previous, config.n_pixels))
    return nn.Sequential(*layers)


def build_kan_payne(config: EmulatorConfig):
    """Build a vector-output KAN emulator with the efficient-kan package."""

    try:
        from efficient_kan import KAN
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "KAN-Payne requires the optional efficient_kan package. "
            "Install it with `pip install efficient_kan`."
        ) from exc

    layers_hidden = [config.n_labels, *config.hidden_sizes, config.n_pixels]
    import torch.nn as nn

    activation_map = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
    }
    activation = activation_map.get(config.activation)
    if activation is None:
        raise ValueError(f"Unsupported KAN base activation: {config.activation}")
    return KAN(layers_hidden=layers_hidden, base_activation=activation)


def build_transformer_payne(config: EmulatorConfig):
    """Build a practical TransformerPayne-style scalar-flux emulator."""

    import math
    import torch
    import torch.nn as nn

    class CrossAttentionBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cross_attn = nn.MultiheadAttention(
                config.d_model,
                config.n_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.norm_attn = nn.LayerNorm(config.d_model)
            self.ffn = nn.Sequential(
                nn.Linear(config.d_model, config.dim_feedforward),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.dim_feedforward, config.d_model),
            )
            self.norm_ffn = nn.LayerNorm(config.d_model)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, query, label_tokens):
            attended, _ = self.cross_attn(
                query,
                label_tokens,
                label_tokens,
                need_weights=False,
            )
            query = self.norm_attn(query + self.dropout(attended))
            return self.norm_ffn(query + self.dropout(self.ffn(query)))

    class TransformerPayneTorch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_wave_features = 1 + 2 * config.wave_frequencies
            self.label_encoder = nn.Sequential(
                nn.Linear(config.n_labels, config.n_label_tokens * config.d_model),
                nn.GELU(),
            )
            self.wave_encoder = nn.Sequential(
                nn.Linear(self.n_wave_features, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.blocks = nn.ModuleList([CrossAttentionBlock() for _ in range(config.n_layers)])
            self.head = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
            )

        def encode_wavelength(self, wave_scaled):
            wave_scaled = wave_scaled.to(dtype=torch.float32)
            if wave_scaled.ndim == 1:
                wave_scaled = wave_scaled[:, None]
            freqs = torch.arange(
                1,
                config.wave_frequencies + 1,
                device=wave_scaled.device,
                dtype=wave_scaled.dtype,
            )
            phase = wave_scaled[..., None] * freqs * math.pi
            features = [wave_scaled[..., None], torch.sin(phase), torch.cos(phase)]
            return torch.cat(features, dim=-1)

        def forward(self, labels_scaled, wave_scaled):
            if wave_scaled.ndim == 1:
                wave_scaled = wave_scaled.unsqueeze(0).expand(labels_scaled.shape[0], -1)
            label_tokens = self.label_encoder(labels_scaled).reshape(
                labels_scaled.shape[0],
                config.n_label_tokens,
                config.d_model,
            )
            wave_features = self.encode_wavelength(wave_scaled)
            query = self.wave_encoder(wave_features)
            for block in self.blocks:
                query = block(query, label_tokens)
            return self.head(query).squeeze(-1)

    return TransformerPayneTorch()


def build_emulator(
    model: str,
    *,
    n_labels: int,
    n_pixels: int,
    hidden_sizes: Sequence[int] = (300, 300),
    activation: str = "leaky_relu",
    d_model: int = 128,
    n_label_tokens: int = 16,
    n_heads: int = 4,
    n_layers: int = 4,
    dim_feedforward: int = 256,
    wave_frequencies: int = 16,
    dropout: float = 0.0,
):
    """Build one of the three emulator families."""

    config = EmulatorConfig(
        model=model,
        n_labels=n_labels,
        n_pixels=n_pixels,
        hidden_sizes=tuple(int(item) for item in hidden_sizes),
        activation=activation,
        d_model=d_model,
        n_label_tokens=n_label_tokens,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        wave_frequencies=wave_frequencies,
        dropout=dropout,
    )
    normalized = model.lower()
    if normalized == "payne_mlp":
        return build_payne_mlp(config), config
    if normalized == "kan_payne":
        return build_kan_payne(config), config
    if normalized == "transformer_payne":
        return build_transformer_payne(config), config
    raise ValueError(f"Unsupported emulator model: {model}")
