#!/bin/bash
#SBATCH --job-name=f2p_embed
#SBATCH --partition=gpul40,gpua4000,gpu4070ti,gpu3060,gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# Inference only: frozen embeddings, no training.
set -euo pipefail
P=${F2P_ROOT:?set F2P_ROOT}
export HF_HOME=$P/models/hf_cache HF_HUB_OFFLINE=1
source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
cd "$P"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/embed_sequences.py facebook/esm2_t33_650M_UR50D esm2_650m 2>&1 | grep --line-buffered -v -E "UNEXPECTED|MISSING|Loading weights|^Key|^---|^Notes|can be ignored|newly initialized|LOAD REPORT|^$"
python scripts/embed_sequences.py ChatterjeeLab/FusOn-pLM fuson_plm 2>&1 | grep --line-buffered -v -E "UNEXPECTED|MISSING|Loading weights|^Key|^---|^Notes|can be ignored|newly initialized|LOAD REPORT|^$"
ls -l data/embeddings/*
