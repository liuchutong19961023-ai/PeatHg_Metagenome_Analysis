#!/bin/bash
set -euo pipefail

# Gene relative abundance pipeline
#
# This script follows the original workflow:
# existing raw paired-end reads -> decompression -> Trimmomatic -> MEGAHIT assembly
# -> Prodigal ORF prediction -> HMMsearch -> extraction of target-gene-containing contigs
# -> Bowtie2 mapping using decompressed raw reads -> samtools/bedtools coverage
# -> total coverage / raw total reads normalization
#
# The only added step relative to the original coverage workflow is:
# GeneRelativeAbundance = (TotalCoverage / RawTotalReads) × 10^6
#
# Input files expected in WORKDIR:
#   sample_1.fastq.gz
#   sample_2.fastq.gz
#
# Final output:
#   gene_relative_abundance.tsv

WORKDIR="/path/to/project"
MEGAHIT_BASE="${WORKDIR}/megahit_output"
TRIMMOMATIC_PATH="/path/to/Trimmomatic-0.39"
ADAPTERS="${TRIMMOMATIC_PATH}/adapters/TruSeq3-PE.fa"
THREADS=16

# HMM model paths
amoA_A_MODEL="${WORKDIR}/amoA_AOA.hmm"
amoA_B_MODEL="${WORKDIR}/amoA_AOB.hmm"
HAO_MODEL="${WORKDIR}/hao.hmm"
nifH_MODEL="${WORKDIR}/nifH.hmm"
nirB_MODEL="${WORKDIR}/nirB.hmm"
nirK_MODEL="${WORKDIR}/nirK.hmm"
nirS_MODEL="${WORKDIR}/nirS.hmm"
norB_MODEL="${WORKDIR}/norB.hmm"
nosZ_MODEL="${WORKDIR}/nosZ.hmm"
nrfA_MODEL="${WORKDIR}/nrfA.hmm"
HGC_A_MODEL="${WORKDIR}/HgcA_654.hmm"
MER_B_MODEL="${WORKDIR}/merB.hmm"

OUTPUT_TABLE="${WORKDIR}/gene_relative_abundance.tsv"

echo -e "SampleID\tGene\tTotalCoverage\tRawTotalReads\tGeneRelativeAbundance" > "${OUTPUT_TABLE}"

count_reads() {
    local fq="$1"
    if [[ "${fq}" == *.gz ]]; then
        zcat "${fq}" | awk 'END {print NR/4}'
    else
        awk 'END {print NR/4}' "${fq}"
    fi
}

run_hmmsearch() {
    local sample_id="$1"
    local gene="$2"
    local hmm_model="$3"
    local inc_evalue="$4"

    hmmsearch \
        --tblout "${WORKDIR}/${sample_id}_${gene}_output.txt" \
        -o "${WORKDIR}/${sample_id}_${gene}_outputfile" \
        --incE "${inc_evalue}" \
        -A "${WORKDIR}/${sample_id}_${gene}_alignment" \
        "${hmm_model}" \
        "${WORKDIR}/${sample_id}_final.contigs.faa"
}

filter_hmm_hits() {
    local sample_id="$1"
    local gene="$2"
    local evalue_cutoff="$3"

    awk -v cutoff="${evalue_cutoff}" '{if ($5 < cutoff) print $1}' \
        "${WORKDIR}/${sample_id}_${gene}_output.txt" \
        > "${WORKDIR}/${sample_id}_${gene}_contigs.txt"
}

extract_contigs() {
    local sample_id="$1"
    local gene="$2"

    local input_ids="${WORKDIR}/${sample_id}_${gene}_contigs.txt"
    local output_fa="${WORKDIR}/${sample_id}_${gene}_final.contigs.fa"

    rm -f "${output_fa}"

    # This follows the original script:
    # extract the first two underscore-separated fields from the Prodigal/HMM hit ID.
    awk -F'_' '{print $1"_"$2}' "${input_ids}" | sort -u | while read -r contig_id; do
        [[ -z "${contig_id}" ]] && continue
        awk -v contig_id="${contig_id}" 'BEGIN {RS=">"; ORS=""} {
            if ($1 == contig_id) print ">"$0
        }' "${WORKDIR}/${sample_id}_final.contigs.fa" >> "${output_fa}"
    done
}

extract_proteins() {
    local sample_id="$1"
    local gene="$2"

    local input_ids="${WORKDIR}/${sample_id}_${gene}_contigs.txt"
    local output_faa="${WORKDIR}/${sample_id}_${gene}_final.contigs.faa"

    rm -f "${output_faa}"

    while read -r contig_id; do
        [[ -z "${contig_id}" ]] && continue
        awk -v contig_id="${contig_id}" 'BEGIN {RS=">"; ORS=""} {
            gsub(/^ +| +$/, "", $1)
            if ($1 == contig_id) print ">"$0
        }' "${WORKDIR}/${sample_id}_final.contigs.faa" >> "${output_faa}"
    done < "${input_ids}"
}

map_raw_reads_and_calculate_relative_abundance() {
    local sample_id="$1"
    local gene="$2"
    local raw_read1="$3"
    local raw_read2="$4"
    local raw_total_reads="$5"

    local target_fa="${WORKDIR}/${sample_id}_${gene}_final.contigs.fa"
    local index_prefix="${WORKDIR}/${sample_id}_${gene}_final_contigs"
    local sam="${WORKDIR}/${sample_id}_${gene}_alignment.sam"
    local bam="${WORKDIR}/${sample_id}_${gene}_alignment.sorted.bam"
    local coverage_txt="${WORKDIR}/${sample_id}_${gene}_coverage.txt"
    local contig_cov_txt="${WORKDIR}/${sample_id}_${gene}_contig_coverage.txt"

    if [[ ! -s "${target_fa}" ]]; then
        echo -e "${sample_id}\t${gene}\t0\t${raw_total_reads}\t0" >> "${OUTPUT_TABLE}"
        return
    fi

    bowtie2-build "${target_fa}" "${index_prefix}"

    # This follows the original coverage workflow:
    # mapping uses decompressed raw reads, not trimmed reads.
    bowtie2 \
        -x "${index_prefix}" \
        -1 "${raw_read1}" \
        -2 "${raw_read2}" \
        -S "${sam}" \
        -p "${THREADS}"

    samtools view -Sb "${sam}" | samtools sort -o "${bam}"
    samtools index "${bam}"

    bedtools genomecov -ibam "${bam}" -d > "${coverage_txt}"

    awk '{cov[$1]+=$3; len[$1]++} END {for (id in cov) print id, cov[id]/len[id]}' \
        "${coverage_txt}" > "${contig_cov_txt}"

    # Added normalization step:
    # TotalCoverage is the sum of mean coverage values of target-gene-containing contigs.
   # GeneRelativeAbundance is calculated as:
   # (TotalCoverage / RawTotalReads) × 10^6
    total_coverage=$(awk '{sum += $2} END {if (sum == "") sum=0; print sum}' "${contig_cov_txt}")

    gene_relative_abundance=$(awk -v cov="${total_coverage}" -v reads="${raw_total_reads}" \
        'BEGIN {if (reads > 0) print (cov / reads) * 1000000; else print 0}')

    echo -e "${sample_id}\t${gene}\t${total_coverage}\t${raw_total_reads}\t${gene_relative_abundance}" \
        >> "${OUTPUT_TABLE}"

    rm -f "${sam}" "${bam}" "${bam}.bai" "${coverage_txt}"
    rm -rf "${index_prefix}"*
}

process_sample() {
    local SAMPLE_ID="$1"
    local RAW_READ1_GZ="$2"
    local RAW_READ2_GZ="$3"

    echo "Processing sample: ${SAMPLE_ID}"

    local RAW_READ1_FASTQ="${WORKDIR}/${SAMPLE_ID}_1.fastq"
    local RAW_READ2_FASTQ="${WORKDIR}/${SAMPLE_ID}_2.fastq"

    echo "Decompressing raw reads..."
    if [[ "${RAW_READ1_GZ}" == *.gz ]]; then
        gunzip -c "${RAW_READ1_GZ}" > "${RAW_READ1_FASTQ}"
    else
        cp "${RAW_READ1_GZ}" "${RAW_READ1_FASTQ}"
    fi

    if [[ "${RAW_READ2_GZ}" == *.gz ]]; then
        gunzip -c "${RAW_READ2_GZ}" > "${RAW_READ2_FASTQ}"
    else
        cp "${RAW_READ2_GZ}" "${RAW_READ2_FASTQ}"
    fi

    RAW_READS_1=$(count_reads "${RAW_READ1_FASTQ}")
    RAW_READS_2=$(count_reads "${RAW_READ2_FASTQ}")
    RAW_TOTAL_READS=$(awk -v r1="${RAW_READS_1}" -v r2="${RAW_READS_2}" 'BEGIN {print r1 + r2}')
    echo "Raw total reads for ${SAMPLE_ID}: ${RAW_TOTAL_READS}"

    echo "Running Trimmomatic..."
    java -jar "${TRIMMOMATIC_PATH}/trimmomatic-0.39.jar" PE -threads "${THREADS}" -phred33 \
        "${RAW_READ1_FASTQ}" "${RAW_READ2_FASTQ}" \
        "${WORKDIR}/${SAMPLE_ID}_1_trimmed.fastq" "${WORKDIR}/${SAMPLE_ID}_1_unpaired.fastq" \
        "${WORKDIR}/${SAMPLE_ID}_2_trimmed.fastq" "${WORKDIR}/${SAMPLE_ID}_2_unpaired.fastq" \
        ILLUMINACLIP:"${ADAPTERS}":2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36

    TRIMMED_READ1="${WORKDIR}/${SAMPLE_ID}_1_trimmed.fastq"
    TRIMMED_READ2="${WORKDIR}/${SAMPLE_ID}_2_trimmed.fastq"

    echo "Running MEGAHIT..."
    MEGAHIT_OUTPUT="${MEGAHIT_BASE}/${SAMPLE_ID}"
    rm -rf "${MEGAHIT_OUTPUT}"

    megahit \
        -1 "${TRIMMED_READ1}" \
        -2 "${TRIMMED_READ2}" \
        -r "${WORKDIR}/${SAMPLE_ID}_1_unpaired.fastq,${WORKDIR}/${SAMPLE_ID}_2_unpaired.fastq" \
        --out-dir "${MEGAHIT_OUTPUT}" \
        --min-contig-len 500 \
        --k-min 27 \
        --k-max 87 \
        -t "${THREADS}"

    mv "${MEGAHIT_OUTPUT}/final.contigs.fa" "${WORKDIR}/${SAMPLE_ID}_final.contigs.fa"

    echo "Running Prodigal..."
    prodigal \
        -i "${WORKDIR}/${SAMPLE_ID}_final.contigs.fa" \
        -a "${WORKDIR}/${SAMPLE_ID}_final.contigs.faa" \
        -o "${WORKDIR}/${SAMPLE_ID}_final.contigs.gff" \
        -p meta

    echo "Running HMMsearch..."
    run_hmmsearch "${SAMPLE_ID}" "amoA_AOA" "${amoA_A_MODEL}" "1E-5"
    run_hmmsearch "${SAMPLE_ID}" "amoA_AOB" "${amoA_B_MODEL}" "1E-5"
    run_hmmsearch "${SAMPLE_ID}" "hao" "${HAO_MODEL}" "1E-5"
    run_hmmsearch "${SAMPLE_ID}" "nifH" "${nifH_MODEL}" "1E-5"
    run_hmmsearch "${SAMPLE_ID}" "nirB" "${nirB_MODEL}" "1E-5"
    run_hmmsearch "${SAMPLE_ID}" "nirK" "${nirK_MODEL}" "1E-5"
    run_hmmsearch "${SAMPLE_ID}" "nirS" "${nirS_MODEL}" "1E-5"
    run_hmmsearch "${SAMPLE_ID}" "norB" "${norB_MODEL}" "1E-5"
    run_hmmsearch "${SAMPLE_ID}" "nosZ" "${nosZ_MODEL}" "1E-5"
    run_hmmsearch "${SAMPLE_ID}" "nrfA" "${nrfA_MODEL}" "1E-5"
    run_hmmsearch "${SAMPLE_ID}" "hgcA" "${HGC_A_MODEL}" "1E-40"
    run_hmmsearch "${SAMPLE_ID}" "merB" "${MER_B_MODEL}" "1E-7"

    echo "Filtering HMM hits..."
    filter_hmm_hits "${SAMPLE_ID}" "amoA_AOA" "1E-5"
    filter_hmm_hits "${SAMPLE_ID}" "amoA_AOB" "1E-5"
    filter_hmm_hits "${SAMPLE_ID}" "hao" "1E-5"
    filter_hmm_hits "${SAMPLE_ID}" "nifH" "1E-5"
    filter_hmm_hits "${SAMPLE_ID}" "nirB" "1E-5"
    filter_hmm_hits "${SAMPLE_ID}" "nirK" "1E-5"
    filter_hmm_hits "${SAMPLE_ID}" "nirS" "1E-5"
    filter_hmm_hits "${SAMPLE_ID}" "norB" "1E-5"
    filter_hmm_hits "${SAMPLE_ID}" "nosZ" "1E-5"
    filter_hmm_hits "${SAMPLE_ID}" "nrfA" "1E-5"
    filter_hmm_hits "${SAMPLE_ID}" "hgcA" "1E-40"
    filter_hmm_hits "${SAMPLE_ID}" "merB" "1E-7"

    for GENE in amoA_AOA amoA_AOB hao nifH nirB nirK nirS norB nosZ nrfA hgcA merB; do
        echo "Extracting ${GENE}-containing contigs and proteins..."
        extract_contigs "${SAMPLE_ID}" "${GENE}"
        extract_proteins "${SAMPLE_ID}" "${GENE}"
    done

    for GENE in amoA_AOA amoA_AOB hao nifH nirB nirK nirS norB nosZ nrfA hgcA merB; do
        echo "Mapping raw reads and calculating relative abundance for ${GENE}..."
        map_raw_reads_and_calculate_relative_abundance \
            "${SAMPLE_ID}" \
            "${GENE}" \
            "${RAW_READ1_FASTQ}" \
            "${RAW_READ2_FASTQ}" \
            "${RAW_TOTAL_READS}"
    done

    echo "Cleaning intermediate files for ${SAMPLE_ID}..."
    rm -f "${WORKDIR}/${SAMPLE_ID}"*_outputfile
    rm -f "${WORKDIR}/${SAMPLE_ID}"*_alignment.sam
    rm -f "${WORKDIR}/${SAMPLE_ID}"*_alignment.sorted.bam
    rm -f "${WORKDIR}/${SAMPLE_ID}"*_alignment.sorted.bam.bai
    rm -f "${WORKDIR}/${SAMPLE_ID}"*_coverage.txt
    rm -rf "${MEGAHIT_OUTPUT}"

    echo "Sample ${SAMPLE_ID} completed."
}

for RAW_READ1_GZ in "${WORKDIR}"/*_1.fastq.gz; do
    SAMPLE_ID=$(basename "${RAW_READ1_GZ}" "_1.fastq.gz")
    RAW_READ2_GZ="${WORKDIR}/${SAMPLE_ID}_2.fastq.gz"

    if [[ ! -f "${RAW_READ2_GZ}" ]]; then
        echo "Warning: paired read file not found for ${SAMPLE_ID}; skipping." >&2
        continue
    fi

    process_sample "${SAMPLE_ID}" "${RAW_READ1_GZ}" "${RAW_READ2_GZ}"
done

echo "All samples processed successfully."
echo "Final output: ${OUTPUT_TABLE}"
