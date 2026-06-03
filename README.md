# PeatHg_Metagenome_Analysis

Code and workflows 

---

## Overview

This repository contains the scripts, HMM profiles, and statistical workflows used to quantify Hg-cycling genes, nitrogen-cycling genes, and Hg-transforming metagenome-assembled genomes (MAGs) from peat-soil metagenomes.

The repository includes:

| File                         | Description                                                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `gene_relative_abundance.sh` | Pipeline for estimating sequencing-depth-normalized relative abundances of Hg-cycling and nitrogen-cycling genes |
| `mag_abundance_hgcA_merB.sh` | Pipeline for quantifying hgcA-MAG and merB-MAG relative abundances using CoverM                                  |
| `code.R`                     | Statistical analyses                                                                                             |
| `HMM/`                       | HMM profiles used for functional gene identification                                                             |

---

## Functional Gene Abundance Estimation

The script `gene_relative_abundance.sh` performs:

1. Read quality control using Trimmomatic
2. Metagenome assembly using MEGAHIT
3. ORF prediction using Prodigal
4. Functional gene identification using HMMER
5. Extraction of target-gene-containing contigs
6. Read mapping using Bowtie2
7. Coverage calculation using Samtools and Bedtools
8. Relative abundance normalization

The following functional genes were analyzed:

### Hg-cycling genes

* hgcA
* merB

### Nitrogen-cycling genes

* nifH
* amoA (AOA)
* amoA (AOB)
* hao
* nirB
* nirK
* nirS
* norB
* nosZ
* nrfA

Relative gene abundance was calculated as:

[(Total gene coverage / Total reads) × 10^6]


---

## MAG Abundance Estimation

The script `mag_abundance_hgcA_merB.sh` quantifies abundances of dereplicated hgcA-MAGs and merB-MAGs.

Workflow:

1. Read quality control using Trimmomatic
2. Mapping against representative MAG genomes using Bowtie2
3. BAM processing using Samtools
4. MAG abundance estimation using CoverM genome mode

Output metrics include:

* Relative abundance
* RPKM
* Covered fraction
* Mean coverage

RPKM values were used for downstream MAG-level analyses.

---

## HMM Profiles

The directory `HMM/` contains the HMM profiles used for functional gene detection.

These profiles were used with HMMER (`hmmsearch`) to identify Hg-cycling and nitrogen-cycling genes from predicted protein sequences.

---

## Statistical Analyses

The script `code.R` contains the statistical analyses used in this study, including:

- Random forest analysis
- Partial least-squares path modeling (PLS-PM)
- Cross-validation analyses
- Data visualization

---

## Software

Major software used in this study:

* Trimmomatic
* MEGAHIT
* Prodigal
* HMMER
* Bowtie2
* Samtools
* Bedtools
* CoverM
* R
