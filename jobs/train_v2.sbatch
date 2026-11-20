#!/bin/bash
#SBATCH --job-name=f2p_train_v2
#SBATCH --partition=gpul40,gpua4000,gpu4070ti,gpu3060,gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# shortcut baselines + logreg/MLP on frozen ESM-2 / FusOn-pLM embeddings,
# strict label set v2, 11 gene-held-out assignments, subsets AB_all / AB_nonRT / A_only. No pLM fine-tuning.
set -euo pipefail
cd ${F2P_ROOT:?set F2P_ROOT}
source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -u scripts/train_v2.py 2>&1 | grep --line-buffered -v -E "ConvergenceWarning|warnings.warn"
