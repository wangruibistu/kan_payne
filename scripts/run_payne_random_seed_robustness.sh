#!/usr/bin/env bash
set -euo pipefail

# Random-split robustness experiment for the Payne synthetic grid.
# Intended for lily, from ~/kan_payne_a100. It waits until the current
# cuda:1 APOGEE midpoint fit has finished, then runs sequentially on cuda:1.

PYTHON="${PYTHON:-/home/wangrui/miniconda3/envs/py311/bin/python}"
THE_PAYNE_ROOT="${THE_PAYNE_ROOT:-data/external/the_payne}"
RUN_DIR="${RUN_DIR:-data/processed/payne_random_seed_robustness_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda:1}"
EPOCHS="${EPOCHS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-4}"
SEEDS="${SEEDS:-11 22 33}"

mkdir -p "${RUN_DIR}"/{grids,emulators,eval,logs}

while pgrep -af "payne_fit_apogee_spectra.py .*--device cuda:1 .*1k_midpoint" >/dev/null; do
  echo "$(date) waiting for cuda:1 APOGEE midpoint fit to finish" | tee -a "${RUN_DIR}/logs/wait.log"
  sleep 300
done

for seed in ${SEEDS}; do
  GRID="${RUN_DIR}/grids/payne_grid_random_seed_${seed}.npz"
  "${PYTHON}" scripts/payne_prepare_training_grid.py \
    --the-payne-root "${THE_PAYNE_ROOT}" \
    --output "${GRID}" \
    --split random \
    --train-fraction 0.8 \
    --valid-fraction 0.2 \
    --random-seed "${seed}" \
    2>&1 | tee "${RUN_DIR}/logs/prepare_seed_${seed}.log"

  "${PYTHON}" scripts/payne_train_emulator.py \
    --grid "${GRID}" \
    --model payne_mlp \
    --output-dir "${RUN_DIR}/emulators/seed_${seed}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --hidden-sizes 300,300 \
    --activation leaky_relu \
    --spectra-normalization none \
    --device "${DEVICE}" \
    --seed "${seed}" \
    2>&1 | tee "${RUN_DIR}/logs/train_payne_mlp_seed_${seed}.log"

  "${PYTHON}" scripts/payne_train_emulator.py \
    --grid "${GRID}" \
    --model kan_payne \
    --output-dir "${RUN_DIR}/emulators/seed_${seed}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --hidden-sizes 64,256 \
    --activation gelu \
    --spectra-normalization none \
    --device "${DEVICE}" \
    --seed "${seed}" \
    2>&1 | tee "${RUN_DIR}/logs/train_kan_payne_seed_${seed}.log"

  "${PYTHON}" scripts/payne_train_emulator.py \
    --grid "${GRID}" \
    --model transformer_payne \
    --output-dir "${RUN_DIR}/emulators/seed_${seed}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --d-model 128 \
    --n-label-tokens 16 \
    --n-heads 4 \
    --n-layers 4 \
    --pixels-per-spectrum 256 \
    --eval-wave-chunk 1024 \
    --spectra-normalization none \
    --device "${DEVICE}" \
    --seed "${seed}" \
    2>&1 | tee "${RUN_DIR}/logs/train_transformer_payne_seed_${seed}.log"

  for model in payne_mlp kan_payne transformer_payne; do
    "${PYTHON}" scripts/payne_evaluate_checkpoint.py \
      --grid "${GRID}" \
      --checkpoint "${RUN_DIR}/emulators/seed_${seed}/${model}/${model}_best.pt" \
      --split valid \
      --output-json "${RUN_DIR}/eval/${model}_seed_${seed}_valid_metrics.json" \
      --device "${DEVICE}" \
      --wave-chunk 1024 \
      2>&1 | tee "${RUN_DIR}/logs/eval_${model}_seed_${seed}.log"
  done
done

echo "DONE ${RUN_DIR}"
