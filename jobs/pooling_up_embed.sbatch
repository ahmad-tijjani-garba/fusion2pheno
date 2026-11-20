#!/bin/bash
#SBATCH --job-name=f2p_pool_up_embed
#SBATCH --partition=gpul40,gpua4000,gpu4070ti,gpu3060,gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# re-embed pooled features with MANE + UniProt junctions (inference only).
set -euo pipefail
P=${F2P_ROOT:?set F2P_ROOT}
export HF_HOME=$P/models/hf_cache HF_HUB_OFFLINE=1
cd "$P"; source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
F="UNEXPECTED|MISSING|Loading weights|^Key|^---|^Notes|can be ignored|newly initialized|LOAD REPORT|^$|warnings.warn"
J=data/junctions/junctions_mane_uniprot.csv
nvidia-smi --query-gpu=name --format=csv,noheader
python -u scripts/embed_pooled.py facebook/esm2_t33_650M_UR50D esm2_650m_up --junctions $J 2>&1 | grep --line-buffered -v -E "$F"
python -u scripts/embed_pooled.py ChatterjeeLab/FusOn-pLM fuson_plm_up --junctions $J 2>&1 | grep --line-buffered -v -E "$F"
