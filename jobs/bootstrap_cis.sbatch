#!/bin/bash
#SBATCH --job-name=f2p_bootstrap
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# bootstrap 95% CIs from saved predictions and per-split results (no training).
set -euo pipefail
cd ${F2P_ROOT:?set F2P_ROOT}; source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
python -u scripts/bootstrap_cis.py 2>&1 | grep --line-buffered -v -E "UndefinedMetricWarning|warnings.warn"
