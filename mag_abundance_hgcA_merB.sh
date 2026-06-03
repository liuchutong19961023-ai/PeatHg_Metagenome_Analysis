#!/bin/bash
# ============================================================
# MAG abundance pipeline for hgcA-MAGs and merB-MAGs
# Steps: QC (Trimmomatic) -> Bowtie2 mapping (stream) -> BAM -> CoverM genome -> cleanup
#
# Input:
#   RAW fastqs in ${RAW_DIR}, named as one of:
#     XXX_1.fq.gz / XXX_2.fq.gz
#     XXX_1.fastq.gz / XXX_2.fastq.gz
#     XXX_R1.fq.gz / XXX_R2.fq.gz
#     XXX_R1.fastq.gz / XXX_R2.fastq.gz
#
# MAG inputs:
#   hgcA-MAG representative genomes in ${HGCA_MAG_DIR}
#   merB-MAG representative genomes in ${MERB_MAG_DIR}
#
# Outputs:
#   ${RESULTS_DIR}/hgcA_MAGs/${SAMPLE_ID}.hgcA_MAG_abundance.tsv
#   ${RESULTS_DIR}/merB_MAGs/${SAMPLE_ID}.merB_MAG_abundance.tsv
#
# CoverM metrics:
#   relative_abundance, rpkm, covered_fraction, mean
# ============================================================

set -euo pipefail

# ----------------
# User parameters
# ----------------

WORKDIR="/path/to/MAGs_Abun"
THREADS=16

RAW_DIR="${WORKDIR}/RAW"

HGCA_MAG_DIR="${WORKDIR}/hgcA_MAGs_representatives"
MERB_MAG_DIR="${WORKDIR}/merB_MAGs_representatives"
MAG_EXT="fa"

TRIMMOMATIC_PATH="${WORKDIR}/Trimmomatic-0.39"
TRIMMOMATIC_JAR="${TRIMMOMATIC_PATH}/trimmomatic-0.39.jar"
ADAPTERS="${TRIMMOMATIC_PATH}/adapters/TruSeq3-PE.fa"

READS_DIR="${WORKDIR}/reads"
RESULTS_DIR="${WORKDIR}/results"
LOG_DIR="${WORKDIR}/logs"
INDEX_DIR="${WORKDIR}/index"
TMP_DIR="${WORKDIR}/tmp"

KEEP_READS=false
KEEP_BAM=false
REMOVE_RAW=false

mkdir -p "${RAW_DIR}" "${READS_DIR}" "${RESULTS_DIR}" "${LOG_DIR}" "${INDEX_DIR}" "${TMP_DIR}"
mkdir -p "${RESULTS_DIR}/hgcA_MAGs" "${RESULTS_DIR}/merB_MAGs"
mkdir -p "${INDEX_DIR}/hgcA_MAGs" "${INDEX_DIR}/merB_MAGs"

# ----------------
# Tool checks
# ----------------

need_cmd () {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: '$1' not found in PATH"
    exit 127
  }
}

need_file () {
  [[ -f "$1" ]] || {
    echo "ERROR: file not found: $1"
    exit 2
  }
}

need_dir () {
  [[ -d "$1" ]] || {
    echo "ERROR: directory not found: $1"
    exit 2
  }
}

echo "[check] checking required tools..."
need_cmd java
need_cmd bowtie2
need_cmd bowtie2-build
need_cmd samtools
need_cmd coverm
need_cmd gzip
need_cmd awk

need_file "${TRIMMOMATIC_JAR}"
need_file "${ADAPTERS}"
need_dir "${RAW_DIR}"
need_dir "${HGCA_MAG_DIR}"
need_dir "${MERB_MAG_DIR}"

echo "[info] WORKDIR       = ${WORKDIR}"
echo "[info] RAW_DIR       = ${RAW_DIR}"
echo "[info] HGCA_MAG_DIR  = ${HGCA_MAG_DIR}"
echo "[info] MERB_MAG_DIR  = ${MERB_MAG_DIR}"
echo "[info] THREADS       = ${THREADS}"
echo "[info] KEEP_READS    = ${KEEP_READS}"
echo "[info] KEEP_BAM      = ${KEEP_BAM}"
echo "[info] REMOVE_RAW    = ${REMOVE_RAW}"

# ----------------
# Build MAG indexes
# ----------------

build_mag_index () {
  local mag_type="$1"
  local mag_dir="$2"
  local index_subdir="${INDEX_DIR}/${mag_type}"
  local combined_fa="${index_subdir}/all_${mag_type}.prefixed.fa"
  local bt2_prefix="${index_subdir}/${mag_type}_bt2_prefixed"

  mkdir -p "${index_subdir}"

  if [[ -s "${combined_fa}" ]] && { [[ -f "${bt2_prefix}.1.bt2" ]] || [[ -f "${bt2_prefix}.1.bt2l" ]]; }; then
    echo "[index] ${mag_type}: Bowtie2 index already present. Skip building."
    return 0
  fi

  echo "[index] ${mag_type}: building combined FASTA and Bowtie2 index from ${mag_dir}..."

  : > "${combined_fa}"

  shopt -s nullglob
  local files=( "${mag_dir}"/*."${MAG_EXT}" )
  shopt -u nullglob

  if (( ${#files[@]} == 0 )); then
    echo "[index][ERROR] ${mag_type}: no MAG files (*.${MAG_EXT}) found in ${mag_dir}"
    exit 2
  fi

  for f in "${files[@]}"; do
    cat "$f" >> "${combined_fa}"
  done

  [[ -s "${combined_fa}" ]] || {
    echo "[index][ERROR] ${mag_type}: combined FASTA is empty"
    exit 2
  }

  bowtie2-build "${combined_fa}" "${bt2_prefix}" \
    1>"${LOG_DIR}/${mag_type}.bowtie2-build.log" \
    2>&1

  echo "[index] ${mag_type}: Bowtie2 index built."
}

# ----------------
# Process one MAG set
# ----------------

run_coverm_for_mag_set () {
  local sample_id="$1"
  local mag_type="$2"
  local mag_dir="$3"
  local bt2_prefix="${INDEX_DIR}/${mag_type}/${mag_type}_bt2_prefixed"
  local r1_trim="$4"
  local r2_trim="$5"

  local bam_sorted="${RESULTS_DIR}/${mag_type}/${sample_id}.${mag_type}.sorted.bam"
  local out_tsv="${RESULTS_DIR}/${mag_type}/${sample_id}.${mag_type}_abundance.tsv"
  local coverm_log="${LOG_DIR}/${sample_id}.${mag_type}.coverm.log"

  echo "[${sample_id}] ${mag_type}: mapping with Bowtie2 -> BAM..."

  bowtie2 --fast-local --mm --no-unal --no-mixed --no-discordant \
    -p "${THREADS}" \
    -x "${bt2_prefix}" \
    -1 "${r1_trim}" \
    -2 "${r2_trim}" \
    2>"${LOG_DIR}/${sample_id}.${mag_type}.bowtie2.stderr.log" \
  | samtools view -@ "${THREADS}" -u - \
  | samtools sort -@ "${THREADS}" -m 2G -T "${TMP_DIR}/${sample_id}.${mag_type}.tmp" -o "${bam_sorted}" -

  samtools index -@ "${THREADS}" "${bam_sorted}"

  echo "[${sample_id}] ${mag_type}: CoverM genome..."

  coverm genome \
    --bam-files "${bam_sorted}" \
    --genome-fasta-directory "${mag_dir}" \
    --genome-fasta-extension "${MAG_EXT}" \
    --methods relative_abundance rpkm covered_fraction mean \
    --min-read-aligned-length 50 \
    --min-read-percent-identity 0.95 \
    --min-covered-fraction 0.10 \
    --proper-pairs-only \
    --exclude-supplementary \
    --threads "${THREADS}" \
    --output-file "${out_tsv}" \
    2>"${coverm_log}"

  echo "[${sample_id}] ${mag_type}: output written to ${out_tsv}"

  if [[ "${KEEP_BAM}" == "false" ]]; then
    echo "[${sample_id}] ${mag_type}: removing BAM and BAI..."
    rm -f "${bam_sorted}" "${bam_sorted}.bai" || true
  fi
}

# ----------------
# Process one sample
# ----------------

process_sample () {
  local sample_id="$1"
  local raw_r1="$2"
  local raw_r2="$3"

  echo "================ [${sample_id}] START ================"

  echo "[${sample_id}] trimming with Trimmomatic..."

  local r1_trim="${READS_DIR}/${sample_id}_1.trim.fastq.gz"
  local r1_unp="${READS_DIR}/${sample_id}_1.unp.fastq.gz"
  local r2_trim="${READS_DIR}/${sample_id}_2.trim.fastq.gz"
  local r2_unp="${READS_DIR}/${sample_id}_2.unp.fastq.gz"

  java -jar "${TRIMMOMATIC_JAR}" PE -threads "${THREADS}" -phred33 \
    "${raw_r1}" "${raw_r2}" \
    "${r1_trim}" "${r1_unp}" \
    "${r2_trim}" "${r2_unp}" \
    ILLUMINACLIP:"${ADAPTERS}":2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36 \
    2>"${LOG_DIR}/${sample_id}.trimmomatic.log"

  run_coverm_for_mag_set "${sample_id}" "hgcA_MAGs" "${HGCA_MAG_DIR}" "${r1_trim}" "${r2_trim}"
  run_coverm_for_mag_set "${sample_id}" "merB_MAGs" "${MERB_MAG_DIR}" "${r1_trim}" "${r2_trim}"

  if [[ "${KEEP_READS}" == "false" ]]; then
    echo "[${sample_id}] cleanup: removing trimmed FASTQ..."
    rm -f "${r1_trim}" "${r2_trim}" "${r1_unp}" "${r2_unp}" || true
  fi

  if [[ "${REMOVE_RAW}" == "true" ]]; then
    echo "[${sample_id}] cleanup: removing original RAW fastq..."
    rm -f "${raw_r1}" "${raw_r2}" || true
  fi

  echo "================ [${sample_id}] END =================="
}

# ----------------
# Main workflow
# ----------------

build_mag_index "hgcA_MAGs" "${HGCA_MAG_DIR}"
build_mag_index "merB_MAGs" "${MERB_MAG_DIR}"

shopt -s nullglob
CANDIDATES=( \
  "${RAW_DIR}"/*_1.fq.gz \
  "${RAW_DIR}"/*_1.fastq.gz \
  "${RAW_DIR}"/*_R1.fq.gz \
  "${RAW_DIR}"/*_R1.fastq.gz \
)
shopt -u nullglob

if (( ${#CANDIDATES[@]} == 0 )); then
  echo "[ERROR] No files matched ${RAW_DIR}/*_1.fq.gz, *_1.fastq.gz, *_R1.fq.gz, or *_R1.fastq.gz"
  exit 3
fi

declare -A seen_sample

for raw_r1 in "${CANDIDATES[@]}"; do
  base="$(basename "${raw_r1}")"
  sample_id=""
  raw_r2=""

  case "${base}" in
    *_1.fastq.gz)
      sample_id="${base%_1.fastq.gz}"
      raw_r2="${RAW_DIR}/${sample_id}_2.fastq.gz"
      ;;
    *_1.fq.gz)
      sample_id="${base%_1.fq.gz}"
      raw_r2="${RAW_DIR}/${sample_id}_2.fq.gz"
      ;;
    *_R1.fastq.gz)
      sample_id="${base%_R1.fastq.gz}"
      raw_r2="${RAW_DIR}/${sample_id}_R2.fastq.gz"
      ;;
    *_R1.fq.gz)
      sample_id="${base%_R1.fq.gz}"
      raw_r2="${RAW_DIR}/${sample_id}_R2.fq.gz"
      ;;
    *)
      continue
      ;;
  esac

  if [[ -n "${seen_sample[$sample_id]+x}" ]]; then
    continue
  fi
  seen_sample["${sample_id}"]=1

  if [[ ! -f "${raw_r2}" ]]; then
    echo "[WARN] Missing pair for ${raw_r1}; expected ${raw_r2}; skipping."
    continue
  fi

  process_sample "${sample_id}" "${raw_r1}" "${raw_r2}"
done

echo "[all done] MAG abundance tables are in:"
echo "  ${RESULTS_DIR}/hgcA_MAGs"
echo "  ${RESULTS_DIR}/merB_MAGs"
