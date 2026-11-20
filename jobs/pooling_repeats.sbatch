#!/bin/bash
#SBATCH --job-name=f2p_pool_reps
#SBATCH --partition=gpul40,gpua4000,gpu4070ti,gpu3060,gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# retrain MLPs for pooling robustness (10 gene-disjoint seeds, 5 calibration repeats).
set -euo pipefail
cd ${F2P_ROOT:?set F2P_ROOT}; source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
python -u scripts/pooling_repeats.py 2>&1 | grep --line-buffered -v -E "ConvergenceWarning|warnings.warn"
