#!/bin/bash
#SBATCH --partition=gpul40,gpua4000,gpu4070ti,gpu3060,gpu4090
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# revision runs.
set -euo pipefail
cd ${F2P_ROOT:?set F2P_ROOT}/src
source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
export HF_HOME=${F2P_ROOT:?set F2P_ROOT}/models/hf_cache HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -u pooling_repeats.py
python -u pooling_repeats.py _up
