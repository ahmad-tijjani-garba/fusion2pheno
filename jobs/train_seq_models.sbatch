#!/bin/bash
#SBATCH --job-name=f2p_train_seq
#SBATCH --partition=gpul40,gpua4000,gpu4070ti,gpu3060,gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# train logreg + small MLP on frozen ESM-2 / FusOn-pLM embeddings.
set -euo pipefail
cd ${F2P_ROOT:?set F2P_ROOT}
source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
python scripts/train_seq_models.py
