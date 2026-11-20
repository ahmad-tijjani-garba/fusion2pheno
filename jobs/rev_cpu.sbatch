#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# revision runs.
# Usage: sbatch -J <name> rev_job.sbatch "<command line run in src/>" ["<second command>" ...]
set -euo pipefail
cd ${F2P_ROOT:?set F2P_ROOT}/src
source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
export HF_HOME=${F2P_ROOT:?set F2P_ROOT}/models/hf_cache HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
for c in "$@"; do echo ">>> $c"; eval "$c"; done
