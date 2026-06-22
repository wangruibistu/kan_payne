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
    wget -c --content-disposition --trust-server-names \
      --tries=0 --timeout=60 --waitretry=15 --read-timeout=120 \
      --progress=dot:giga "${url}"
  )
}

export -f download_one

grep -vE '^\s*(#|$)' "${MANIFEST}" | \
  xargs -n 1 -P "${JOBS}" -I {} bash -c 'download_one "$1" "$2"' _ {} "${OUTPUT_DIR}"
