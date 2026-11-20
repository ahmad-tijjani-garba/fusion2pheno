#!/bin/bash
#SBATCH --job-name=f2p_embed_v2new
#SBATCH --partition=gpul40,gpua4000,gpu4070ti,gpu3060,gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# mean-pooled embeddings of the 2,303 new strict-label sequences (inference only, no training).
set -euo pipefail
P=${F2P_ROOT:?set F2P_ROOT}
export HF_HOME=$P/models/hf_cache HF_HUB_OFFLINE=1
source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
cd "$P"
F="UNEXPECTED|MISSING|Loading weights|^Key|^---|^Notes|can be ignored|newly initialized|LOAD REPORT|^$"
FA=data/labels_v2/strict/new_sequences.fasta
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u scripts/embed_sequences.py facebook/esm2_t33_650M_UR50D esm2_650m_v2new --fasta $FA 2>&1 | grep --line-buffered -v -E "$F"
python -u scripts/embed_sequences.py ChatterjeeLab/FusOn-pLM fuson_plm_v2new --fasta $FA 2>&1 | grep --line-buffered -v -E "$F"
ls -l data/embeddings/*_v2new
