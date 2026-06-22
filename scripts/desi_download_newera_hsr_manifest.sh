#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:?Usage: desi_download_newera_hsr_manifest.sh URLS_TXT OUTPUT_DIR [JOBS]}"
OUTPUT_DIR="${2:?Usage: desi_download_newera_hsr_manifest.sh URLS_TXT OUTPUT_DIR [JOBS]}"
JOBS="${3:-2}"

mkdir -p "${OUTPUT_DIR}"

download_one() {
  local url="$1"
  local output_dir="$2"
  (
    cd "${output_dir}"
    curl -L -J -O --fail --retry 10 --retry-delay 10 \
      --connect-timeout 60 --max-time 900 "${url}"
  )
}

export -f download_one

grep -vE '^\s*(#|$)' "${MANIFEST}" | \
  xargs -n 1 -P "${JOBS}" -I {} bash -c 'download_one "$1" "$2"' _ {} "${OUTPUT_DIR}"
