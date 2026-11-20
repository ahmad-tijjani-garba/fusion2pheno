#!/bin/bash
#SBATCH --job-name=f2p_train_v2_matched
#SBATCH --partition=gpul40,gpua4000,gpu4070ti,gpu3060,gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# baselines + logreg/MLP on frozen ESM-2 / FusOn-pLM embeddings,
# both classes from one source at a time, 21 gene-held-out assignments. No pLM fine-tuning.
set -euo pipefail
cd ${F2P_ROOT:?set F2P_ROOT}/src
source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -u train_v2_matched.py --seeds 42 $(seq 1 20) 2>&1 | grep --line-buffered -v -E "Warning|warnings.warn|_methods|ret = "
