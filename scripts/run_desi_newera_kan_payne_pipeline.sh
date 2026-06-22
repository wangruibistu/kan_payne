#!/usr/bin/env bash
set -euo pipefail

# End-to-end NewEra V3 LowRes + DESI DR1 MWS KAN-Payne pipeline.
# Intended to run from ~/kan_payne_a100 on lily.

PYTHON="${PYTHON:-/home/wangrui/miniconda3/envs/py311/bin/python}"
DATA_DIR="${DATA_DIR:-/home/wangrui/data}"
NEWERA_TAR="${NEWERA_TAR:-${DATA_DIR}/newera/lowres/PHOENIX-NewEraV3-LowRes-SPECTRA.tar.gz}"
NEWERA_ADD="${NEWERA_ADD:-${DATA_DIR}/newera/lowres/PHOENIX-NewEraV3-add001-LowRes-SPECTRA.Z+0.5.txt}"
NEWERA_EXTRACT="${NEWERA_EXTRACT:-${DATA_DIR}/newera/lowres/extracted_v3}"
NEWERA_GLOB="${NEWERA_GLOB:-**/*.txt}"
RUN_DIR="${RUN_DIR:-data/processed/desi_newera_kan_payne_$(date +%Y%m%d_%H%M%S)}"
GRID_NAME="${GRID_NAME:-desi_newera_v3_grid.npz}"
MODEL="${MODEL:-kan_payne}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-4}"
HIDDEN_SIZES="${HIDDEN_SIZES:-128,128}"
MAX_NEWERA_SPECTRA="${MAX_NEWERA_SPECTRA:-}"
WAVE_MIN="${WAVE_MIN:-3600}"
WAVE_MAX="${WAVE_MAX:-9800}"
WAVE_STEP="${WAVE_STEP:-1.0}"
TARGET_RESOLUTION="${TARGET_RESOLUTION:-0}"
CONTINUUM_WINDOW="${CONTINUUM_WINDOW:-301}"
SMOOTH_OBSERVED_SIGMA_PIX="${SMOOTH_OBSERVED_SIGMA_PIX:-0}"
DESI_SELECTED="${DESI_SELECTED:-${DATA_DIR}/desi/dr1/healpix_snr30_all/pilot_top_healpix_10k/selected_targets.csv}"
DESI_COADD_ROOT="${DESI_COADD_ROOT:-${DATA_DIR}/desi/dr1/healpix_snr30_all/pilot_top_healpix_10k/files}"
DESI_SP_BRIGHT="${DESI_SP_BRIGHT:-${DATA_DIR}/desi/dr1/vac/mws/iron/v1.0/sp_output/230211/sppix-main-bright.fits}"
DESI_SP_DARK="${DESI_SP_DARK:-${DATA_DIR}/desi/dr1/vac/mws/iron/v1.0/sp_output/230211/sppix-main-dark.fits}"
MAX_DESI_STARS="${MAX_DESI_STARS:-10000}"
FIT_STEPS="${FIT_STEPS:-500}"
FIT_PIXELS="${FIT_PIXELS:-2048}"

mkdir -p "${RUN_DIR}"/{logs,emulators,eval,desi}

if [[ ! -d "${NEWERA_EXTRACT}" && -f "${NEWERA_TAR}" ]]; then
  mkdir -p "${NEWERA_EXTRACT}"
  tar -xzf "${NEWERA_TAR}" -C "${NEWERA_EXTRACT}"
fi
if [[ -n "${NEWERA_ADD}" && -f "${NEWERA_ADD}" && "${NEWERA_GLOB}" == *".txt"* ]]; then
  cp -n "${NEWERA_ADD}" "${NEWERA_EXTRACT}/" || true
fi

GRID="${RUN_DIR}/${GRID_NAME}"
GRID_ARGS=(
  scripts/desi_prepare_newera_grid.py
  --input-root "${NEWERA_EXTRACT}"
  --glob "${NEWERA_GLOB}"
  --output "${GRID}"
  --wave-min "${WAVE_MIN}"
  --wave-max "${WAVE_MAX}"
  --wave-step "${WAVE_STEP}"
  --target-resolution "${TARGET_RESOLUTION}"
  --continuum-window "${CONTINUUM_WINDOW}"
  --teff-range="3500,8000"
  --logg-range="0,5"
  --mh-range="-3.0,0.5"
  --alpha-range="-0.2,0.8"
  --train-fraction 0.8
  --valid-fraction 0.1
  --seed 42
)
if [[ -n "${MAX_NEWERA_SPECTRA}" ]]; then
  GRID_ARGS+=(--max-spectra "${MAX_NEWERA_SPECTRA}")
fi
"${PYTHON}" "${GRID_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/logs/prepare_newera_grid.log"

"${PYTHON}" scripts/payne_train_emulator.py \
  --grid "${GRID}" \
  --model "${MODEL}" \
  --output-dir "${RUN_DIR}/emulators" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --hidden-sizes "${HIDDEN_SIZES}" \
  --device "${DEVICE}" \
  --spectra-normalization none \
  2>&1 | tee "${RUN_DIR}/logs/train_${MODEL}.log"

CHECKPOINT="${RUN_DIR}/emulators/${MODEL}/${MODEL}_best.pt"
"${PYTHON}" scripts/payne_evaluate_checkpoint.py \
  --grid "${GRID}" \
  --checkpoint "${CHECKPOINT}" \
  --split valid \
  --output-json "${RUN_DIR}/eval/${MODEL}_valid_metrics.json" \
  --output-npz "${RUN_DIR}/eval/${MODEL}_valid_residuals.npz" \
  --device "${DEVICE}" \
  2>&1 | tee "${RUN_DIR}/logs/evaluate_${MODEL}.log"

OBSERVED="${RUN_DIR}/desi/desi_dr1_mws_snr30_pilot_observed.npz"
"${PYTHON}" scripts/desi_prepare_observed_spectra.py \
  --selected-targets "${DESI_SELECTED}" \
  --coadd-root "${DESI_COADD_ROOT}" \
  --sp-catalog "${DESI_SP_BRIGHT}" \
  --sp-catalog "${DESI_SP_DARK}" \
  --grid "${GRID}" \
  --output "${OBSERVED}" \
  --max-stars "${MAX_DESI_STARS}" \
  --snr-min 30 \
  --continuum-window "${CONTINUUM_WINDOW}" \
  --normalize-per-arm \
  --smooth-observed-sigma-pix "${SMOOTH_OBSERVED_SIGMA_PIX}" \
  2>&1 | tee "${RUN_DIR}/logs/prepare_desi_observed.log"

"${PYTHON}" scripts/payne_fit_observed_spectra.py \
  --observed "${OBSERVED}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${RUN_DIR}/desi/desi_dr1_mws_snr30_pilot_${MODEL}_fit.npz" \
  --device "${DEVICE}" \
  --steps "${FIT_STEPS}" \
  --fit-pixels "${FIT_PIXELS}" \
  --progress-every 50 \
  2>&1 | tee "${RUN_DIR}/logs/fit_desi_${MODEL}.log"

echo "DONE ${RUN_DIR}"
