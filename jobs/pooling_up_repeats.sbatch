#!/bin/bash
#SBATCH --job-name=f2p_pool_up_reps
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# rerun pooling robustness (10 gene-disjoint seeds, 5 calibration repeats) on MANE + UniProt junction pools.
set -euo pipefail
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
cd ${F2P_ROOT:?set F2P_ROOT}; source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
python -u scripts/pooling_repeats.py _up 2>&1 | grep --line-buffered -v -E "ConvergenceWarning|warnings.warn"
