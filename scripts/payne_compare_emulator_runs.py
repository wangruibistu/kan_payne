#!/usr/bin/env python
"""Summarize training histories and validation residuals for emulator runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write PNG loss/residual plots when matplotlib is available.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _history_path(run_dir: Path, model: str) -> Path:
    return run_dir / "emulators" / model / f"{model}_history.json"


def _evaluation_path(run_dir: Path, model: str) -> Path:
    return run_dir / "evaluations" / f"{model}_valid.json"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = ("payne_mlp", "kan_payne", "transformer_payne")
    summary_rows = []
    history_rows = []
    histories = {}
    for model in models:
        history_file = _history_path(run_dir, model)
        if not history_file.exists():
            print(f"Missing history for {model}: {history_file}")
            continue
        history_payload = _load_json(history_file)
        history = history_payload["history"]
        histories[model] = history
        best_row = min(history, key=lambda row: row["valid_good_mae_x1e4"])
        final_row = history[-1]

        eval_payload = {}
        evaluation_file = _evaluation_path(run_dir, model)
        if evaluation_file.exists():
            eval_payload = _load_json(evaluation_file)

        summary_rows.append(
            {
                "model": model,
                "epochs": int(history_payload["epochs"]),
                "best_epoch_by_history": int(best_row["epoch"]),
                "best_valid_good_mae_x1e4": best_row["valid_good_mae_x1e4"],
                "best_valid_good_rmse_x1e4": best_row["valid_good_rmse_x1e4"],
                "final_valid_good_mae_x1e4": final_row["valid_good_mae_x1e4"],
                "final_valid_good_rmse_x1e4": final_row["valid_good_rmse_x1e4"],
                "eval_checkpoint_epoch": eval_payload.get("checkpoint_epoch", ""),
                "eval_valid_good_mae_x1e4": eval_payload.get("metrics_good_pixels", {}).get(
                    "mae", ""
                ),
                "eval_valid_good_rmse_x1e4": eval_payload.get("metrics_good_pixels", {}).get(
                    "rmse", ""
                ),
                "star_mae_median_x1e4": eval_payload.get("star_mae_x1e4", {}).get(
                    "median", ""
                ),
                "pixel_mae_median_x1e4": eval_payload.get("pixel_mae_x1e4", {}).get(
                    "median", ""
                ),
            }
        )
        for row in history:
            history_rows.append({"model": model, **row})

    summary_csv = output_dir / "emulator_validation_summary.csv"
    if summary_rows:
        with summary_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    history_csv = output_dir / "emulator_training_history.csv"
    if history_rows:
        with history_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)

    summary_json = output_dir / "emulator_validation_summary.json"
    with summary_json.open("w") as handle:
        json.dump(
            {
                "run_dir": str(run_dir),
                "summary": summary_rows,
                "history_csv": str(history_csv),
                "summary_csv": str(summary_csv),
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    if args.plot and histories:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"Skipping plots; matplotlib unavailable: {exc}")
        else:
            plt.figure(figsize=(8, 5))
            for model, history in histories.items():
                x = [row["epoch"] for row in history]
                y = [row["valid_good_mae_x1e4"] for row in history]
                plt.plot(x, y, label=model)
            plt.xlabel("Epoch")
            plt.ylabel("Validation MAE on unmasked pixels (x1e-4)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "validation_mae_history.png", dpi=180)
            plt.close()

            plt.figure(figsize=(8, 5))
            for model, history in histories.items():
                x = [row["epoch"] for row in history]
                y = [row["train_l1_x1e4"] for row in history]
                plt.plot(x, y, label=model)
            plt.xlabel("Epoch")
            plt.ylabel("Training L1 (x1e-4)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "training_l1_history.png", dpi=180)
            plt.close()

    print(f"Wrote comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
