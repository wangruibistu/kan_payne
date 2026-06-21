#!/usr/bin/env python
"""Summarize checkpoint parameter counts and best validation metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--history", nargs="*", default=[])
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _model_label(model: str) -> str:
    return {
        "payne_mlp": "Payne-MLP",
        "kan_payne": "KAN-Payne",
        "transformer_payne": "TransformerPayne",
    }.get(model, model)


def _history_lookup(paths):
    lookup = {}
    for path in paths:
        p = Path(path)
        with p.open() as handle:
            payload = json.load(handle)
        model = payload.get("model") or p.parent.name
        history = payload.get("history", [])
        best = min(history, key=lambda row: row.get("valid_good_mae_x1e4", float("inf")))
        lookup[model] = {
            "history_path": str(p),
            "epochs": int(payload.get("epochs", len(history))),
            "best_epoch_from_history": int(best.get("epoch", -1)),
            "best_valid_good_mae_x1e4": float(best.get("valid_good_mae_x1e4", float("nan"))),
            "best_valid_good_rmse_x1e4": float(best.get("valid_good_rmse_x1e4", float("nan"))),
            "device": payload.get("device", ""),
        }
    return lookup


def main():
    args = parse_args()

    import torch

    histories = _history_lookup(args.history)
    rows = []
    for checkpoint_path in args.checkpoint:
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location="cpu")
        config = dict(checkpoint["model_config"])
        state = checkpoint["model_state_dict"]
        n_tensors = len(state)
        n_params = int(sum(t.numel() for t in state.values()))
        trainable_mb = n_params * 4 / 1024**2
        model = config["model"]
        row = {
            "model": model,
            "label": _model_label(model),
            "checkpoint": str(path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "n_labels": int(config.get("n_labels", -1)),
            "n_pixels": int(config.get("n_pixels", -1)),
            "n_tensors": n_tensors,
            "n_parameters": n_params,
            "parameter_memory_float32_mb": round(trainable_mb, 3),
            "hidden_sizes": ",".join(str(v) for v in config.get("hidden_sizes", [])),
            "activation": config.get("activation", ""),
            "d_model": int(config.get("d_model", 0)),
            "n_label_tokens": int(config.get("n_label_tokens", 0)),
            "n_heads": int(config.get("n_heads", 0)),
            "n_layers": int(config.get("n_layers", 0)),
            "wave_frequencies": int(config.get("wave_frequencies", 0)),
        }
        row.update(histories.get(model, {}))
        rows.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    Path(args.output_json).write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
