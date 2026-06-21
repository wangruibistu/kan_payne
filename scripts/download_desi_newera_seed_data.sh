#!/usr/bin/env bash
set -euo pipefail

# Seed data downloader for the DESI + NewEra KAN-Payne project.
# Run this on lily. It intentionally downloads metadata/catalog products first;
# DESI healpix spectra should be selected from these catalogs before bulk fetches.

BASE_DIR="${BASE_DIR:-/home/wangrui/data}"
NEWERA_DIR="${NEWERA_DIR:-${BASE_DIR}/newera}"
DESI_DIR="${DESI_DIR:-${BASE_DIR}/desi/dr1}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/logs}"
JOBS="${JOBS:-1}"

mkdir -p \
  "${NEWERA_DIR}"/{metadata,lowres} \
  "${DESI_DIR}"/{zcatalog,vac/mws/iron/v1.0/rv_output/240520,vac/mws/iron/v1.0/sp_output/230211} \
  "${LOG_DIR}"

download_one() {
  local url="$1"
  local out_dir="$2"
  local log_name="$3"
  mkdir -p "${out_dir}"
  (
    cd "${out_dir}"
    wget -c --tries=0 --timeout=60 --waitretry=15 --read-timeout=120 \
      --progress=dot:giga "${url}" \
      >> "${LOG_DIR}/${log_name}.log" 2>&1
  )
}

download_list() {
  local manifest="$1"
  local out_dir="$2"
  local log_name="$3"
  if [[ "${JOBS}" == "1" ]]; then
    while IFS= read -r url; do
      [[ -z "${url}" || "${url}" =~ ^# ]] && continue
      download_one "${url}" "${out_dir}" "${log_name}"
    done < "${manifest}"
  else
    grep -vE '^\s*(#|$)' "${manifest}" | \
      xargs -n 1 -P "${JOBS}" -I {} bash -c \
        'url="$1"; out="$2"; log="$3"; mkdir -p "$out"; cd "$out"; wget -c --tries=0 --timeout=60 --waitretry=15 --read-timeout=120 --progress=dot:giga "$url" >> "'"${LOG_DIR}"'/${log}.log" 2>&1' \
        _ {} "${out_dir}" "${log_name}"
  fi
}

NEWERA_BUCKET="https://www.fdr.uni-hamburg.de/api/files/17d647e6-a771-4c68-88b6-23bf6ca0029b"

cat > "${NEWERA_DIR}/metadata/newera_small_files.urls" <<EOF
${NEWERA_BUCKET}/example_read_gaia_fmt.py
${NEWERA_BUCKET}/example_read_HSR_H5.py
${NEWERA_BUCKET}/example_read_structure_from_HSR_H5.py
${NEWERA_BUCKET}/get_NewEra_from_FDR.py
${NEWERA_BUCKET}/list_of_available_NewEra_models.txt
${NEWERA_BUCKET}/list_of_available_NewEraV2_models.txt
${NEWERA_BUCKET}/list_of_available_NewEraV3_models.txt
${NEWERA_BUCKET}/list_of_available_additional_NewEra_models.txt
${NEWERA_BUCKET}/Readme.PHOENIX.gaia_fmt.txt
${NEWERA_BUCKET}/PHOENIX-Vega+SunV3-GAIA-DR4-SPECTRA-Z-0.0.txt
${NEWERA_BUCKET}/PHOENIX-Vega+SunV3-GAIA-DR4-PHOTOMETRY-Z-0.0.txt
EOF

cat > "${NEWERA_DIR}/lowres/newera_lowres_v3.urls" <<EOF
${NEWERA_BUCKET}/PHOENIX-NewEraV3-LowRes-SPECTRA.tar.gz
${NEWERA_BUCKET}/PHOENIX-NewEraV3-add001-LowRes-SPECTRA.Z+0.5.txt
EOF

DESI_ZCAT="https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/zcatalog/v1"
cat > "${DESI_DIR}/zcatalog/desi_dr1_zcatalog_main.urls" <<EOF
${DESI_ZCAT}/redux_iron_zcatalog_v1.sha256sum
${DESI_ZCAT}/zpix-main-bright.fits
${DESI_ZCAT}/zpix-main-dark.fits
EOF

DESI_MWS="https://data.desi.lbl.gov/public/dr1/vac/dr1/mws/iron/v1.0"
cat > "${DESI_DIR}/vac/mws/iron/v1.0/desi_dr1_mws_core.urls" <<EOF
${DESI_MWS}/README.md
${DESI_MWS}/dr1_vac_dr1_mws_iron_v1.0.sha256sum
${DESI_MWS}/mwsall-pix-iron.fits
EOF

cat > "${DESI_DIR}/vac/mws/iron/v1.0/rv_output/240520/desi_dr1_mws_rv_main.urls" <<EOF
${DESI_MWS}/rv_output/240520/dr1_vac_dr1_mws_iron_v1.0_rv_output_240520.sha256sum
${DESI_MWS}/rv_output/240520/rvpix-main-bright.fits
${DESI_MWS}/rv_output/240520/rvpix-main-dark.fits
EOF

cat > "${DESI_DIR}/vac/mws/iron/v1.0/sp_output/230211/desi_dr1_mws_sp_main.urls" <<EOF
${DESI_MWS}/sp_output/230211/dr1_vac_dr1_mws_iron_v1.0_sp_output_230211.sha256sum
${DESI_MWS}/sp_output/230211/sppix-main-bright.fits
${DESI_MWS}/sp_output/230211/sppix-main-dark.fits
EOF

download_list "${NEWERA_DIR}/metadata/newera_small_files.urls" "${NEWERA_DIR}/metadata" "newera_small_files"
download_list "${NEWERA_DIR}/lowres/newera_lowres_v3.urls" "${NEWERA_DIR}/lowres" "newera_lowres_v3"
download_list "${DESI_DIR}/zcatalog/desi_dr1_zcatalog_main.urls" "${DESI_DIR}/zcatalog" "desi_dr1_zcatalog_main"
download_list "${DESI_DIR}/vac/mws/iron/v1.0/desi_dr1_mws_core.urls" "${DESI_DIR}/vac/mws/iron/v1.0" "desi_dr1_mws_core"
download_list "${DESI_DIR}/vac/mws/iron/v1.0/rv_output/240520/desi_dr1_mws_rv_main.urls" "${DESI_DIR}/vac/mws/iron/v1.0/rv_output/240520" "desi_dr1_mws_rv_main"
download_list "${DESI_DIR}/vac/mws/iron/v1.0/sp_output/230211/desi_dr1_mws_sp_main.urls" "${DESI_DIR}/vac/mws/iron/v1.0/sp_output/230211" "desi_dr1_mws_sp_main"

du -sh "${NEWERA_DIR}" "${DESI_DIR}" | tee "${LOG_DIR}/desi_newera_seed_download_sizes.txt"
