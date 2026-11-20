#!/bin/bash
#SBATCH --job-name=f2p_dl_fgdb2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
set -euo pipefail
cd ${F2P_ROOT:?set F2P_ROOT}/data/raw/fusiongdb2
F=combinedFGDB2genes_ORF_analyzed_gencode_h19v19_real_Inframe_only_transcript_seq_with_orffinder_result.txt
URL="https://web.archive.org/web/20250525203821id_/https://compbio.uth.edu/FusionGDB2/combined_tables/$F"
curl -sSL --retry 5 --retry-delay 30 -C - -o "$F" "$URL"
ls -l "$F"
sha256sum "$F" > "$F.sha256"
printf "Source: %s\nOriginal: https://compbio.uth.edu/FusionGDB2/combined_tables/%s (offline since late 2026)\nOriginal last-modified: 2022-03-24\nDownloaded: %s\n" "$URL" "$F" "$(date -I)" > SOURCE.txt
