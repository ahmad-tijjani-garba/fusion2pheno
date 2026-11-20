#!/bin/bash
#SBATCH --job-name=f2p_pool_reps_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# retrain MLPs for pooling robustness (10 gene-disjoint seeds, 5 calibration repeats).
set -euo pipefail
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
cd ${F2P_ROOT:?set F2P_ROOT}; source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
python -u scripts/pooling_repeats.py 2>&1 | grep --line-buffered -v -E "ConvergenceWarning|warnings.warn"
